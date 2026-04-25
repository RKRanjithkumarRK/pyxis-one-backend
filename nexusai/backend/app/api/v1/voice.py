"""Voice API — Whisper STT + ElevenLabs TTS."""
from __future__ import annotations
import asyncio
import io

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/voice", tags=["voice"])

VOICES = [
    {"id": "rachel", "name": "Rachel", "preview": None},
    {"id": "adam", "name": "Adam", "preview": None},
    {"id": "bella", "name": "Bella", "preview": None},
    {"id": "elli", "name": "Elli", "preview": None},
    {"id": "josh", "name": "Josh", "preview": None},
    {"id": "sam", "name": "Sam", "preview": None},
    {"id": "sarah", "name": "Sarah", "preview": None},
    {"id": "thomas", "name": "Thomas", "preview": None},
    {"id": "charlie", "name": "Charlie", "preview": None},
]


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "rachel"
    model_id: str = "eleven_turbo_v2"


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    current_user: User = Depends(get_current_user),
):
    """Transcribe audio chunk via Whisper (openai/whisper-1)."""
    content = await audio.read()
    text = await _whisper_transcribe(content, audio.filename or "audio.webm", language)
    return {"text": text, "language": language}


@router.post("/tts")
async def text_to_speech(
    body: TTSRequest,
    current_user: User = Depends(get_current_user),
):
    """Stream TTS audio from ElevenLabs."""
    audio_stream = _elevenlabs_stream(body.text, body.voice_id, body.model_id)
    return StreamingResponse(audio_stream, media_type="audio/mpeg")


@router.get("/voices")
async def list_voices():
    return {"voices": VOICES}


async def _whisper_transcribe(content: bytes, filename: str, language: str) -> str:
    import openai
    client = openai.AsyncOpenAI()
    file_tuple = (filename, io.BytesIO(content), "audio/webm")
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        language=language if language != "auto" else None,
    )
    return response.text


async def _elevenlabs_stream(text: str, voice_id: str, model_id: str):
    from app.core.config import settings
    import httpx

    api_key = getattr(settings, "ELEVENLABS_API_KEY", "")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                yield chunk
