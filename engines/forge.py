from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import ForgeProgress
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class ForgeEngine:
    STAGES = [
        "RAW_ORE",
        "HEATING",
        "HAMMERING",
        "QUENCHING",
        "TEMPERING",
        "POLISHING",
        "ASSAYING",
    ]

    def _next_stage(self, current: str) -> str:
        idx = self.STAGES.index(current) if current in self.STAGES else 0
        if idx + 1 < len(self.STAGES):
            return self.STAGES[idx + 1]
        return "ASSAYING"

    async def _get_or_create(self, session_id: str, concept: str) -> ForgeProgress:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ForgeProgress).where(
                    ForgeProgress.session_id == session_id,
                    ForgeProgress.concept == concept,
                )
            )
            fp = result.scalar_one_or_none()
            if fp is None:
                fp = ForgeProgress(session_id=session_id, concept=concept, stage="RAW_ORE")
                db.add(fp)
                await db.commit()
                await db.refresh(fp)
            return fp

    async def get_stage(self, session_id: str, concept: str) -> str:
        fp = await self._get_or_create(session_id, concept)
        return fp.stage

    async def advance_stage(self, session_id: str, concept: str) -> str:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ForgeProgress).where(
                    ForgeProgress.session_id == session_id,
                    ForgeProgress.concept == concept,
                )
            )
            fp = result.scalar_one_or_none()
            if fp is None:
                fp = ForgeProgress(session_id=session_id, concept=concept, stage="RAW_ORE")
                db.add(fp)
                await db.commit()
                await db.refresh(fp)
                return fp.stage

            if fp.stage == "ASSAYING":
                fp.completed_at = datetime.utcnow()
                await db.commit()
                return "ASSAYING"

            fp.stage = self._next_stage(fp.stage)
            fp.stage_entered_at = datetime.utcnow()
            if fp.stage == "ASSAYING":
                fp.completed_at = datetime.utcnow()
            await db.commit()
            return fp.stage

    async def get_stage_prompt(self, stage: str, concept: str, psyche: str) -> str:
        stage_instructions = {
            "RAW_ORE": (
                f"The student is encountering '{concept}' for the first time.\n"
                "Provide a gentle, compelling first exposure. Spark curiosity. "
                "Do NOT overwhelm — give the single most important intuition. "
                "End with one question that makes them want to know more."
            ),
            "HEATING": (
                f"The student has a basic model of '{concept}'. "
                "Deliberately surface the FLAWS in their initial understanding. "
                "Break their naive mental model with a carefully chosen counter-example. "
                "Make them feel the productive discomfort of a challenged assumption."
            ),
            "HAMMERING": (
                f"Apply Socratic pressure to '{concept}'. "
                "Ask hard, probing questions that expose weak points. "
                "Do not give answers — force the student to reason through contradictions. "
                "Challenge every assertion. Stay relentless but supportive."
            ),
            "QUENCHING": (
                f"The student's model of '{concept}' is being tested in novel contexts. "
                "Present three completely different real-world applications that require "
                "adapting the concept. Each context should feel surprisingly different."
            ),
            "TEMPERING": (
                f"The student returns to '{concept}' after a gap. "
                "Begin with a gentle retrieval prompt. Then introduce ONE new nuance "
                "they have not seen yet. Integrate it with what they already know."
            ),
            "POLISHING": (
                f"The student must now teach '{concept}' back to you. "
                "Prompt them to explain it as if teaching a curious 12-year-old. "
                "After they explain, find the ONE subtle inaccuracy in their explanation "
                "and help them refine it."
            ),
            "ASSAYING": (
                f"Final forge assessment for '{concept}'. "
                "Present the hardest possible question under maximum cognitive load. "
                "Combine the concept with two other unrelated domains. "
                "Require precise formal language. Evaluate mercilessly but fairly. "
                "Provide a detailed verdict on mastery."
            ),
        }

        instruction = stage_instructions.get(stage, stage_instructions["RAW_ORE"])
        system = (
            f"You are the Forge — a master metallurgist of the mind.\n"
            f"Stage: {stage}\n\n"
            f"{instruction}\n\n"
            f"{RESPONSE_STRUCTURE}\n\n"
            f"{psyche}"
        )
        messages = [{"role": "user", "content": f"Forge stage {stage} for concept: {concept}"}]
        return await ac.complete_response(messages, system)

    async def check_tempering_ready(self, session_id: str, concept: str) -> bool:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ForgeProgress).where(
                    ForgeProgress.session_id == session_id,
                    ForgeProgress.concept == concept,
                )
            )
            fp = result.scalar_one_or_none()

        if fp is None or fp.stage != "TEMPERING":
            return False

        elapsed = datetime.utcnow() - fp.stage_entered_at
        return elapsed >= timedelta(hours=24)


forge_engine = ForgeEngine()
