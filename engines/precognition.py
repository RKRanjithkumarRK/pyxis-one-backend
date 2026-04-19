import json
from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import ConceptMastery, Message, OracleTimeline
import engines.anthropic_client as ac


class PrecognitionEngine:
    async def simulate_trajectory(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            mastery_result = await db.execute(
                select(ConceptMastery).where(ConceptMastery.session_id == session_id)
            )
            masteries = mastery_result.scalars().all()

            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(20)
            )
            recent = msg_result.scalars().all()

        if not masteries:
            return []

        mastery_data = [
            {"concept": m.concept, "mastery": m.mastery_score, "revolution": m.helix_revolution}
            for m in masteries
        ]
        history_preview = "\n".join(f"{m.role}: {m.content[:150]}" for m in reversed(recent))

        prompt = (
            f"Simulate this student's learning trajectory for the next 90 days.\n\n"
            f"Current mastery states: {json.dumps(mastery_data)}\n"
            f"Recent session:\n{history_preview}\n\n"
            "Return JSON array of trajectory milestones:\n"
            '[{"day": N, "concept": "...", "predicted_mastery": 0.0-1.0, '
            '"milestone": "description", "probability": 0.0-1.0}]\n'
            "Return ONLY valid JSON array. Include ~15 milestones."
        )
        system = "You are the Precognition engine. Simulate futures. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=2048)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            trajectory: list = json.loads(raw)
        except Exception:
            today = datetime.utcnow()
            trajectory = [
                {
                    "day": (i + 1) * 7,
                    "date": (today + timedelta(days=(i + 1) * 7)).strftime("%Y-%m-%d"),
                    "concept": mastery_data[i % len(mastery_data)]["concept"] if mastery_data else "unknown",
                    "predicted_mastery": min(1.0, mastery_data[i % len(mastery_data)]["mastery"] + 0.1 * (i + 1)) if mastery_data else 0.5,
                    "milestone": f"Week {i + 1} checkpoint",
                    "probability": 0.8 - i * 0.05,
                }
                for i in range(12)
            ]

        return trajectory

    async def identify_future_struggles(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            mastery_result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score)
            )
            masteries = mastery_result.scalars().all()

        if not masteries:
            return []

        weak = [m for m in masteries if m.mastery_score < 0.5]
        mastery_data = [{"concept": m.concept, "mastery": m.mastery_score} for m in weak[:10]]

        prompt = (
            f"Identify which future concepts this student will struggle with based on current gaps.\n\n"
            f"Current weak concepts: {json.dumps(mastery_data)}\n\n"
            "Which future topics depend on these weak foundations?\n"
            "Return JSON array:\n"
            '[{"concept": "...", "depends_on_weak": ["c1"], "struggle_probability": 0.0-1.0, '
            '"estimated_arrival_days": N, "prevention_hint": "..."}]\n'
            "Return ONLY valid JSON array."
        )
        system = "You are the Precognition engine. Identify future struggles. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            struggles: list = json.loads(raw)
        except Exception:
            struggles = [
                {
                    "concept": f"Advanced {m['concept']}",
                    "depends_on_weak": [m["concept"]],
                    "struggle_probability": 0.8,
                    "estimated_arrival_days": 21,
                    "prevention_hint": f"Solidify {m['concept']} before advancing",
                }
                for m in mastery_data[:3]
            ]

        return struggles

    async def seed_scaffolding(self, session_id: str, concept: str) -> str:
        prompt = (
            f"Generate invisible scaffolding for the concept '{concept}' to plant now.\n\n"
            "This scaffolding should be woven naturally into responses today so that when "
            "the student encounters this concept as a wall in 2-4 weeks, they already have "
            "mental hooks. It must feel natural, not remedial.\n\n"
            "Return 2-4 sentences of natural scaffolding text."
        )
        system = "You are the Precognition engine. Seed invisible scaffolding."
        messages = [{"role": "user", "content": prompt}]
        return await ac.complete_response(messages, system, max_tokens=512)

    async def get_constellation_map(self, session_id: str) -> dict:
        trajectory = await self.simulate_trajectory(session_id)
        struggles = await self.identify_future_struggles(session_id)

        async with AsyncSessionLocal() as db:
            oracle_result = await db.execute(
                select(OracleTimeline).where(OracleTimeline.session_id == session_id)
            )
            oracle_timelines = oracle_result.scalars().all()

        return {
            "session_id": session_id,
            "trajectory": trajectory,
            "predicted_struggles": struggles,
            "oracle_walls": [
                {
                    "concept": ot.concept,
                    "predicted_wall_date": ot.predicted_wall_date.isoformat() if ot.predicted_wall_date else None,
                    "avoided": ot.wall_avoided,
                }
                for ot in oracle_timelines
            ],
            "generated_at": datetime.utcnow().isoformat(),
        }


precognition_engine = PrecognitionEngine()
