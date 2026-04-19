import io
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.schemas import VoiceAnalysisResponse
from core.config import RESPONSE_STRUCTURE
from engines.psyche import psyche_engine
import engines.anthropic_client as ac

router = APIRouter()


def _analyze_audio(audio_bytes: bytes) -> dict:
    import soundfile as sf
    import librosa

    buffer = io.BytesIO(audio_bytes)
    try:
        audio_data, sr = sf.read(buffer)
    except Exception:
        buffer.seek(0)
        audio_data, sr = librosa.load(buffer, sr=None)

    if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
        y = np.mean(audio_data, axis=1).astype(np.float32)
    else:
        y = np.array(audio_data, dtype=np.float32)

    if len(y) == 0:
        return {
            "tempo": 0.0,
            "avg_volume": 0.0,
            "pause_count": 0,
            "speech_rate_wpm": 0.0,
            "duration_seconds": 0.0,
        }

    duration = len(y) / sr

    rms = librosa.feature.rms(y=y)[0]
    avg_volume = float(np.mean(rms))

    intervals = librosa.effects.split(y, top_db=25)
    pause_count = max(0, len(intervals) - 1)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        tempo_val = 0.0

    speech_duration = sum((end - start) / sr for start, end in intervals)
    syllables_per_second = 4.0
    estimated_words = speech_duration * syllables_per_second / 1.5
    speech_rate_wpm = (estimated_words / duration * 60.0) if duration > 0 else 0.0

    return {
        "tempo": round(tempo_val, 2),
        "avg_volume": round(avg_volume, 4),
        "pause_count": pause_count,
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "duration_seconds": round(duration, 2),
    }


@router.post("/voice/analyze", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    transcript: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")

    try:
        audio_metrics = _analyze_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio analysis failed: {e}")

    try:
        psyche_context = await psyche_engine.get_context_block(session_id)
    except Exception:
        psyche_context = ""

    transcript_section = f"\nTranscript:\n{transcript}" if transcript else "\n(No transcript provided)"

    prompt = (
        f"Analyse this voice interaction and generate a Soul Report.\n\n"
        f"Audio metrics:\n"
        f"- Tempo/rhythm: {audio_metrics['tempo']} BPM\n"
        f"- Average volume: {audio_metrics['avg_volume']:.4f}\n"
        f"- Pause count: {audio_metrics['pause_count']}\n"
        f"- Estimated speech rate: {audio_metrics['speech_rate_wpm']:.1f} WPM\n"
        f"- Duration: {audio_metrics['duration_seconds']:.1f} seconds\n"
        f"{transcript_section}\n\n"
        "Generate a Soul Report that:\n"
        "1. Interprets what the voice patterns reveal about current cognitive state\n"
        "2. Identifies confidence indicators from pacing and pauses\n"
        "3. Detects emotional undercurrents in the delivery\n"
        "4. Connects vocal patterns to learning state\n"
        "5. Offers one insight the student likely doesn't know about their own voice\n\n"
        f"{psyche_context}"
    )

    system = (
        "You are the Soul Reader — an expert in reading cognitive and emotional states from voice patterns. "
        "Generate an empathetic, insightful Soul Report. Be specific about what the metrics reveal."
    )
    api_messages = [{"role": "user", "content": prompt}]

    try:
        soul_report = await ac.complete_response(api_messages, system, max_tokens=1024)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Soul report generation failed: {e}")

    confidence_indicators = {
        "high_pause_density": audio_metrics["pause_count"] > 10,
        "slow_speech": audio_metrics["speech_rate_wpm"] < 100,
        "fast_speech": audio_metrics["speech_rate_wpm"] > 180,
        "low_volume": audio_metrics["avg_volume"] < 0.05,
        "estimated_confidence": "high" if audio_metrics["speech_rate_wpm"] > 140 and audio_metrics["pause_count"] < 8 else "medium" if audio_metrics["speech_rate_wpm"] > 100 else "low",
    }

    return VoiceAnalysisResponse(
        session_id=session_id,
        soul_report=soul_report,
        tempo=audio_metrics["tempo"],
        avg_volume=audio_metrics["avg_volume"],
        pause_count=audio_metrics["pause_count"],
        speech_rate_wpm=audio_metrics["speech_rate_wpm"],
        confidence_indicators=confidence_indicators,
    )
