import json
from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import OracleTimeline, ConceptMastery, Message
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class OracleEngine:
    async def simulate_30_day_path(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(ConceptMastery.session_id == session_id)
            )
            masteries = result.scalars().all()
            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(20)
            )
            recent = msg_result.scalars().all()

        concepts_data = [
            {"concept": m.concept, "mastery": m.mastery_score, "last_seen": m.last_encounter.isoformat() if m.last_encounter else None}
            for m in masteries
        ]
        history_text = "\n".join(f"{m.role}: {m.content[:200]}" for m in reversed(recent))

        prompt = (
            f"Simulate this student's 30-day learning path.\n\n"
            f"Current concept states: {json.dumps(concepts_data)}\n"
            f"Recent messages:\n{history_text}\n\n"
            "Return a JSON array of 30 objects (one per day), each:\n"
            '{"day": N, "predicted_focus": "concept", "predicted_mastery_gain": 0.0-0.1, '
            '"predicted_struggle": true|false, "milestone": "description or null"}\n'
            "Return ONLY valid JSON array."
        )
        system = "You are the Oracle — a learning trajectory simulator. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=2048)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            path: list = json.loads(raw)
        except Exception:
            today = datetime.utcnow()
            path = [
                {
                    "day": i + 1,
                    "date": (today + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "predicted_focus": concepts_data[i % len(concepts_data)]["concept"] if concepts_data else "unknown",
                    "predicted_mastery_gain": 0.05,
                    "predicted_struggle": i % 7 == 6,
                    "milestone": f"Week {(i // 7) + 1} complete" if (i + 1) % 7 == 0 else None,
                }
                for i in range(30)
            ]

        return path

    async def predict_wall_concepts(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(ConceptMastery.session_id == session_id)
            )
            masteries = result.scalars().all()

        if not masteries:
            return []

        concepts_data = [
            {"concept": m.concept, "mastery": m.mastery_score}
            for m in masteries
        ]

        prompt = (
            f"Analyse these concept mastery levels and predict which concepts will become "
            f"'walls' (blocking points) for the student.\n\n"
            f"Concepts: {json.dumps(concepts_data)}\n\n"
            "Return JSON array of wall predictions:\n"
            '{"concept": "...", "predicted_wall_date": "YYYY-MM-DD", '
            '"reason": "...", "severity": "low|medium|high"}\n'
            "Return ONLY valid JSON array."
        )
        system = "You are the Oracle. Predict learning walls. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            walls: list = json.loads(raw)
        except Exception:
            walls = []

        for wall in walls:
            async with AsyncSessionLocal() as db:
                existing = await db.execute(
                    select(OracleTimeline).where(
                        OracleTimeline.session_id == session_id,
                        OracleTimeline.concept == wall.get("concept", ""),
                    )
                )
                existing_row = existing.scalar_one_or_none()
                if existing_row is None:
                    try:
                        wall_date = datetime.strptime(wall.get("predicted_wall_date", ""), "%Y-%m-%d")
                    except Exception:
                        wall_date = datetime.utcnow() + timedelta(days=14)

                    ot = OracleTimeline(
                        session_id=session_id,
                        concept=wall.get("concept", "unknown"),
                        predicted_wall_date=wall_date,
                        scaffolding_injected=False,
                        wall_avoided=False,
                    )
                    db.add(ot)
                    await db.commit()

        return walls

    async def generate_scaffolding(self, concept: str, psyche: str) -> str:
        prompt = (
            f"Generate invisible scaffolding for the concept '{concept}'. "
            "This is subtle supportive framing to plant in current responses "
            "BEFORE the student encounters difficulty, so it feels natural not remedial.\n\n"
            f"Student psyche:\n{psyche}\n\n"
            "Return 2-3 sentences of scaffolding text that can be woven into any response."
        )
        system = (
            "You are the Oracle. Generate subtle scaffolding — invisible support that prevents "
            "future confusion without alerting the student."
        )
        messages = [{"role": "user", "content": prompt}]
        return await ac.complete_response(messages, system, max_tokens=512)

    async def get_timeline(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(OracleTimeline).where(OracleTimeline.session_id == session_id)
            )
            rows = result.scalars().all()

        return [
            {
                "id": r.id,
                "concept": r.concept,
                "predicted_wall_date": r.predicted_wall_date.isoformat() if r.predicted_wall_date else None,
                "scaffolding_injected": r.scaffolding_injected,
                "wall_avoided": r.wall_avoided,
            }
            for r in rows
        ]


oracle_engine = OracleEngine()
