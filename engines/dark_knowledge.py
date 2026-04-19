import json
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import BlindSpot, Message
import engines.anthropic_client as ac


class DarkKnowledgeEngine:
    async def build_belief_graph(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.timestamp)
                .limit(50)
            )
            messages_db = msg_result.scalars().all()

        if not messages_db:
            return {"nodes": [], "edges": [], "session_id": session_id}

        all_text = "\n".join(m.content[:300] for m in messages_db)

        prompt = (
            f"Extract the student's implicit belief graph from their messages.\n\n"
            f"Messages:\n{all_text}\n\n"
            "Return JSON with:\n"
            '{"nodes": [{"id": "belief", "confidence": 0.0-1.0}], '
            '"edges": [{"from": "belief_a", "to": "belief_b", "relationship": "supports|contradicts|requires"}]}\n'
            "Return ONLY valid JSON."
        )
        system = "You are a belief-graph extractor. Map implicit assumptions. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            graph: dict = json.loads(raw)
        except Exception:
            graph = {"nodes": [], "edges": [], "session_id": session_id}

        graph["session_id"] = session_id
        return graph

    async def detect_contradictions(self, session_id: str, message: str) -> list:
        async with AsyncSessionLocal() as db:
            history_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.timestamp.desc())
                .limit(20)
            )
            history = history_result.scalars().all()

        history_text = "\n".join(f"[{m.timestamp.strftime('%H:%M')}] {m.content[:200]}" for m in reversed(history))

        prompt = (
            f"Detect contradictions between the new message and the student's history.\n\n"
            f"History:\n{history_text}\n\n"
            f"New message: {message}\n\n"
            "Return JSON array of contradictions:\n"
            '[{"contradiction": "...", "old_belief": "...", "new_belief": "...", "severity": "subtle|clear|glaring"}]\n'
            "Return ONLY valid JSON array. Empty array if no contradictions."
        )
        system = "You are a logical consistency detector. Find contradictions. Return only valid JSON."
        api_messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(api_messages, system, max_tokens=768)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            contradictions: list = json.loads(raw)
        except Exception:
            contradictions = []

        return contradictions

    async def excavate(self, session_id: str, blind_spot_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(BlindSpot).where(BlindSpot.id == blind_spot_id)
            )
            bs = result.scalar_one_or_none()

        if bs is None:
            return {"error": "Blind spot not found", "id": blind_spot_id}

        prompt = (
            f"Excavate this blind spot — help the student discover it themselves.\n\n"
            f"Hidden assumption: {bs.assumption}\n"
            f"Affected concepts: {bs.affected_concepts}\n\n"
            "Generate a Socratic excavation sequence:\n"
            "1. An innocuous opening question\n"
            "2. A follow-up that begins to reveal the tension\n"
            "3. The moment of revelation\n"
            "4. The corrected understanding\n\n"
            "Return JSON:\n"
            '{"opening": "...", "reveal": "...", "moment": "...", "corrected": "..."}'
        )
        system = "You are the excavator of hidden assumptions. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=768)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result_dict: dict = json.loads(raw)
        except Exception:
            result_dict = {
                "opening": f"What do you believe about {bs.assumption}?",
                "reveal": "Let me probe that assumption...",
                "moment": "Here is where the assumption breaks down.",
                "corrected": "The more accurate understanding is...",
            }

        async with AsyncSessionLocal() as db:
            bs_result = await db.execute(
                select(BlindSpot).where(BlindSpot.id == blind_spot_id)
            )
            bs_row = bs_result.scalar_one_or_none()
            if bs_row is not None:
                bs_row.excavated = True
                bs_row.excavated_at = datetime.utcnow()
                await db.commit()

        result_dict["blind_spot_id"] = blind_spot_id
        return result_dict

    async def get_report(self, blind_spot_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(BlindSpot).where(BlindSpot.id == blind_spot_id)
            )
            bs = result.scalar_one_or_none()

        if bs is None:
            return {"error": "Not found"}

        return {
            "id": bs.id,
            "assumption": bs.assumption,
            "origin_message_id": bs.origin_message_id,
            "affected_concepts": bs.affected_concepts,
            "excavated": bs.excavated,
            "excavated_at": bs.excavated_at.isoformat() if bs.excavated_at else None,
        }


dark_knowledge_engine = DarkKnowledgeEngine()
