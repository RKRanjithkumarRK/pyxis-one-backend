import json
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import BlindSpot, Message
import engines.anthropic_client as ac


class BlindSpotEngine:
    async def analyze(self, session_id: str, message: str) -> list:
        async with AsyncSessionLocal() as db:
            history_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.role == "user")
                .order_by(Message.timestamp.desc())
                .limit(15)
            )
            history = history_result.scalars().all()

        history_text = "\n".join(f"{m.content[:200]}" for m in reversed(history))

        prompt = (
            f"Identify implicit assumptions and blind spots in this student message.\n\n"
            f"Message history context:\n{history_text}\n\n"
            f"Current message: {message}\n\n"
            "Find assumptions the student makes without realising — gaps in understanding they "
            "don't know they have.\n\n"
            "Return JSON array:\n"
            '[{"assumption": "...", "why_its_wrong": "...", "affected_concepts": ["c1", "c2"], '
            '"severity": "low|medium|high"}]\n'
            "Return ONLY valid JSON array. Empty array if no blind spots."
        )
        system = "You are a blind spot detector. Find hidden assumptions. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            spots: list = json.loads(raw)
        except Exception:
            spots = []

        async with AsyncSessionLocal() as db:
            last_msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(1)
            )
            last_msg = last_msg_result.scalar_one_or_none()
            origin_id = last_msg.id if last_msg else None

            for spot in spots:
                existing_result = await db.execute(
                    select(BlindSpot).where(
                        BlindSpot.session_id == session_id,
                        BlindSpot.assumption == spot.get("assumption", ""),
                    )
                )
                existing = existing_result.scalar_one_or_none()
                if existing is None:
                    bs = BlindSpot(
                        session_id=session_id,
                        assumption=spot.get("assumption", ""),
                        origin_message_id=origin_id,
                        affected_concepts=spot.get("affected_concepts", []),
                        excavated=False,
                    )
                    db.add(bs)
            await db.commit()

        return spots

    async def build_assumption_tree(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(BlindSpot).where(BlindSpot.session_id == session_id)
            )
            blind_spots = result.scalars().all()

        if not blind_spots:
            return {"nodes": [], "root_assumptions": [], "session_id": session_id}

        spots_text = "\n".join(
            f"- {bs.assumption} (affects: {bs.affected_concepts})" for bs in blind_spots
        )

        prompt = (
            f"Build an assumption dependency tree from these blind spots.\n\n"
            f"Blind spots:\n{spots_text}\n\n"
            "Which assumptions depend on or reinforce others?\n"
            "Return JSON:\n"
            '{"root_assumptions": ["fundamental ones"], "tree": {'
            '"assumption": {"depends_on": [], "reinforces": []}}}\n'
            "Return ONLY valid JSON."
        )
        system = "You are an assumption archaeologist. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            tree: dict = json.loads(raw)
        except Exception:
            tree = {
                "root_assumptions": [bs.assumption for bs in blind_spots[:3]],
                "tree": {bs.assumption: {"depends_on": [], "reinforces": []} for bs in blind_spots},
            }

        tree["session_id"] = session_id
        tree["total_blind_spots"] = len(blind_spots)
        tree["excavated"] = sum(1 for bs in blind_spots if bs.excavated)
        return tree

    async def detect_new(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(BlindSpot).where(
                    BlindSpot.session_id == session_id,
                    BlindSpot.excavated == False,  # noqa: E712
                )
            )
            unexcavated = result.scalars().all()

        return [
            {
                "id": bs.id,
                "assumption": bs.assumption,
                "affected_concepts": bs.affected_concepts,
                "origin_message_id": bs.origin_message_id,
            }
            for bs in unexcavated
        ]

    async def generate_reveal(self, blind_spot_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(BlindSpot).where(BlindSpot.id == blind_spot_id)
            )
            bs = result.scalar_one_or_none()

        if bs is None:
            return {"error": "Blind spot not found"}

        prompt = (
            f"Generate a reveal sequence for this hidden assumption.\n\n"
            f"Hidden assumption: {bs.assumption}\n"
            f"Affected concepts: {bs.affected_concepts}\n\n"
            "Create a 3-step Socratic reveal:\n"
            "Step 1: An innocent question that primes the student\n"
            "Step 2: A question that creates cognitive dissonance\n"
            "Step 3: The reveal question that makes the assumption visible\n\n"
            "Return JSON:\n"
            '{"step_1": "...", "step_2": "...", "step_3": "...", "corrected_understanding": "..."}\n'
            "Return ONLY valid JSON."
        )
        system = "You are a master of Socratic revelation. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=768)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            reveal: dict = json.loads(raw)
        except Exception:
            reveal = {
                "step_1": f"What do you understand about {bs.assumption}?",
                "step_2": "What if the opposite were true?",
                "step_3": "Where does your original assumption break down?",
                "corrected_understanding": "The accurate view is more nuanced.",
            }

        reveal["blind_spot_id"] = blind_spot_id
        reveal["assumption"] = bs.assumption
        return reveal


blind_spot_engine = BlindSpotEngine()
