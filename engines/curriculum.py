import json
from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import Message, ConceptMastery
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class CurriculumEngine:
    async def _get_recent_messages(self, session_id: str, hours: int = 72) -> list[dict]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.timestamp >= cutoff)
                .order_by(Message.timestamp)
                .limit(40)
            )
            msgs = result.scalars().all()
        return [{"role": m.role, "content": m.content[:400]} for m in msgs]

    async def generate_next_moves(self, session_id: str, topic: str = "") -> list[dict]:
        history = await self._get_recent_messages(session_id)
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-10:])

        prompt = (
            f"Based on this learning session history, generate exactly 3 prioritised next learning moves.\n\n"
            f"Recent history:\n{history_text}\n\n"
            f"Current topic focus: {topic or 'general'}\n\n"
            "Return JSON array of 3 objects, each with:\n"
            '{"move": "...", "reasoning": "...", "priority": 1|2|3, "estimated_minutes": N}\n'
            "Return ONLY valid JSON array."
        )
        system = (
            "You are a master curriculum architect. Analyse learning patterns and prescribe the "
            "optimal next three learning moves. Return only valid JSON."
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            moves: list = json.loads(raw)
            if not isinstance(moves, list):
                moves = [moves]
        except Exception:
            moves = [
                {"move": f"Review core concepts of {topic or 'the subject'}", "reasoning": "Foundation solidification", "priority": 1, "estimated_minutes": 15},
                {"move": "Apply concepts to a novel problem", "reasoning": "Transfer practice", "priority": 2, "estimated_minutes": 20},
                {"move": "Identify one unresolved confusion and ask a precise question", "reasoning": "Targeted gap closure", "priority": 3, "estimated_minutes": 10},
            ]

        return moves[:3]

    async def rewrite_curriculum(self, session_id: str) -> dict:
        history = await self._get_recent_messages(session_id)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score)
            )
            masteries = result.scalars().all()

        weak = [m.concept for m in masteries if m.mastery_score < 0.4]
        strong = [m.concept for m in masteries if m.mastery_score >= 0.7]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-8:])

        prompt = (
            f"Rewrite the complete learning curriculum for this student.\n\n"
            f"Weak concepts (mastery < 40%): {weak}\n"
            f"Strong concepts (mastery > 70%): {strong}\n"
            f"Recent session:\n{history_text}\n\n"
            "Return JSON with keys: sequence (ordered list of concepts), "
            "immediate_priority, weekly_goals, avoid_until_ready\n"
            "Return ONLY valid JSON."
        )
        system = "You are a curriculum architect. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            curriculum: dict = json.loads(raw)
        except Exception:
            curriculum = {
                "sequence": weak + strong,
                "immediate_priority": weak[:2] if weak else [],
                "weekly_goals": [],
                "avoid_until_ready": [],
            }

        return curriculum

    async def get_sequence(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score, ConceptMastery.last_encounter)
            )
            masteries = result.scalars().all()

        return [
            {
                "concept": m.concept,
                "mastery_score": m.mastery_score,
                "next_encounter": m.next_encounter.isoformat() if m.next_encounter else None,
            }
            for m in masteries
        ]


curriculum_engine = CurriculumEngine()
