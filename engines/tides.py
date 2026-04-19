import re
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import TideReading, ConceptMastery
import engines.anthropic_client as ac

_HEDGE_WORDS = re.compile(
    r"\b(maybe|perhaps|possibly|i think|i guess|sort of|kind of|not sure|"
    r"probably|might|could be|i believe|approximately|roughly|unclear)\b",
    re.IGNORECASE,
)
_CONFIDENCE_WORDS = re.compile(
    r"\b(definitely|certainly|clearly|obviously|precisely|exactly|"
    r"always|never|must|will|is defined as|by definition)\b",
    re.IGNORECASE,
)
_TECHNICAL_WORDS = re.compile(
    r"\b[A-Za-z][a-z]*(?:tion|ism|ity|ance|ence|ology|ary|ive|ous)\b"
)


def _score_vocabulary_precision(text: str) -> float:
    words = text.split()
    if not words:
        return 0.5
    technical_count = len(_TECHNICAL_WORDS.findall(text))
    avg_word_len = sum(len(w) for w in words) / len(words)
    vocab_score = min(1.0, (technical_count / max(len(words), 1)) * 5 + (avg_word_len - 3) / 10)
    return max(0.0, min(1.0, round(vocab_score, 4)))


def _score_confidence(text: str) -> float:
    hedge_count = len(_HEDGE_WORDS.findall(text))
    confidence_count = len(_CONFIDENCE_WORDS.findall(text))
    words = text.split()
    if not words:
        return 0.5
    hedge_ratio = hedge_count / len(words)
    confidence_ratio = confidence_count / len(words)
    score = 0.5 + (confidence_ratio - hedge_ratio) * 10
    return max(0.0, min(1.0, round(score, 4)))


class TideEngine:
    async def record_reading(self, session_id: str, concept: str, message: str) -> None:
        vocab_precision = _score_vocabulary_precision(message)
        confidence_score = _score_confidence(message)

        async with AsyncSessionLocal() as db:
            reading = TideReading(
                session_id=session_id,
                concept=concept,
                vocabulary_precision=vocab_precision,
                confidence_score=confidence_score,
                reading_date=datetime.utcnow(),
            )
            db.add(reading)

            cm_result = await db.execute(
                select(ConceptMastery).where(
                    ConceptMastery.session_id == session_id,
                    ConceptMastery.concept == concept,
                )
            )
            cm = cm_result.scalar_one_or_none()
            if cm is None:
                cm = ConceptMastery(
                    session_id=session_id,
                    concept=concept,
                    mastery_score=(vocab_precision + confidence_score) / 2,
                    last_encounter=datetime.utcnow(),
                )
                db.add(cm)
            else:
                cm.tide_data = {
                    "last_vocab_precision": vocab_precision,
                    "last_confidence": confidence_score,
                }
                cm.last_encounter = datetime.utcnow()
                cm.mastery_score = round(
                    0.8 * cm.mastery_score + 0.2 * (vocab_precision + confidence_score) / 2, 4
                )

            await db.commit()

    async def get_chart(self, session_id: str, concept: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TideReading)
                .where(
                    TideReading.session_id == session_id,
                    TideReading.concept == concept,
                )
                .order_by(TideReading.reading_date)
            )
            readings = result.scalars().all()

        if not readings:
            return {
                "session_id": session_id,
                "concept": concept,
                "readings": [],
                "trend": "no_data",
                "alert": None,
            }

        reading_dicts = [
            {
                "date": r.reading_date.isoformat(),
                "vocabulary_precision": r.vocabulary_precision,
                "confidence_score": r.confidence_score,
                "composite": round((r.vocabulary_precision + r.confidence_score) / 2, 4),
            }
            for r in readings
        ]

        if len(readings) >= 3:
            recent_avg = sum(r.vocabulary_precision + r.confidence_score for r in readings[-3:]) / 6
            older_avg = sum(r.vocabulary_precision + r.confidence_score for r in readings[:-3]) / max(1, len(readings[:-3]) * 2)
            if recent_avg < older_avg - 0.1:
                trend = "receding"
            elif recent_avg > older_avg + 0.1:
                trend = "rising"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        alert = await self.generate_alert(concept, session_id) if trend == "receding" else None

        return {
            "session_id": session_id,
            "concept": concept,
            "readings": reading_dicts,
            "trend": trend,
            "alert": alert,
        }

    async def detect_receding(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(ConceptMastery.session_id == session_id)
            )
            masteries = result.scalars().all()

        receding = []
        for m in masteries:
            chart = await self.get_chart(session_id, m.concept)
            if chart.get("trend") == "receding":
                receding.append({"concept": m.concept, "current_mastery": m.mastery_score})

        return receding

    async def generate_alert(self, concept: str, session_id: str) -> str:
        prompt = (
            f"The student's understanding of '{concept}' is showing signs of decay — "
            "vocabulary precision and confidence are declining over recent sessions. "
            "Generate a brief, motivating alert message (2-3 sentences) that:\n"
            "1. Names the decay without being discouraging\n"
            "2. Suggests one immediate action\n"
            "3. Reframes it as normal and recoverable"
        )
        system = "You are a supportive tide monitor. Be direct and encouraging."
        messages = [{"role": "user", "content": prompt}]
        return await ac.complete_response(messages, system, max_tokens=256)


tide_engine = TideEngine()
