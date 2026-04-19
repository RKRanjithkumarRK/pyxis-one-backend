from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import HelixRevolution, ConceptMastery
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class HelixEngine:
    REVOLUTIONS = [
        "SURFACE",
        "STRUCTURAL",
        "EDGE_CASES",
        "FIRST_PRINCIPLES",
        "FIND_FLAWS",
        "SYNTHESIS",
    ]

    # SM-2 base intervals in days per revolution level
    _BASE_INTERVALS = [1, 3, 7, 14, 30, 60]

    def _sm2_next(self, revolution_idx: int, easiness: float, quality: int) -> tuple[int, float]:
        """Returns (days_until_next, new_easiness_factor)."""
        new_ef = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        if quality < 3:
            days = 1
        else:
            base = self._BASE_INTERVALS[min(revolution_idx, len(self._BASE_INTERVALS) - 1)]
            days = max(1, round(base * new_ef))
        return days, new_ef

    async def _get_or_create(self, session_id: str, concept: str) -> HelixRevolution:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(HelixRevolution).where(
                    HelixRevolution.session_id == session_id,
                    HelixRevolution.concept == concept,
                )
            )
            hr = result.scalar_one_or_none()
            if hr is None:
                hr = HelixRevolution(
                    session_id=session_id,
                    concept=concept,
                    revolution="SURFACE",
                    next_due=datetime.utcnow(),
                )
                db.add(hr)
                await db.commit()
                await db.refresh(hr)
            return hr

    async def get_revolution(self, session_id: str, concept: str) -> str:
        hr = await self._get_or_create(session_id, concept)
        return hr.revolution

    async def schedule_next(self, session_id: str, concept: str, quality: int = 4) -> None:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(HelixRevolution).where(
                    HelixRevolution.session_id == session_id,
                    HelixRevolution.concept == concept,
                )
            )
            hr = result.scalar_one_or_none()
            if hr is None:
                return

            current_idx = self.REVOLUTIONS.index(hr.revolution) if hr.revolution in self.REVOLUTIONS else 0
            days, _ = self._sm2_next(current_idx, 2.5, quality)

            hr.completed_at = datetime.utcnow()
            hr.next_due = datetime.utcnow() + timedelta(days=days)

            if quality >= 3 and current_idx + 1 < len(self.REVOLUTIONS):
                hr.revolution = self.REVOLUTIONS[current_idx + 1]

            cm_result = await db.execute(
                select(ConceptMastery).where(
                    ConceptMastery.session_id == session_id,
                    ConceptMastery.concept == concept,
                )
            )
            cm = cm_result.scalar_one_or_none()
            if cm is not None:
                cm.helix_revolution = hr.revolution
                cm.next_encounter = hr.next_due

            await db.commit()

    async def get_prompt(self, revolution: str, concept: str, psyche: str) -> str:
        revolution_prompts = {
            "SURFACE": (
                f"Engage the student with '{concept}' at surface level.\n"
                "Provide the most intuitive, accessible explanation possible. "
                "Use one powerful analogy. Avoid jargon. Make it memorable."
            ),
            "STRUCTURAL": (
                f"Engage the student with '{concept}' at structural level.\n"
                "Explain the internal mechanics — how it actually works step by step. "
                "What are the moving parts? What does each component do?"
            ),
            "EDGE_CASES": (
                f"Probe the edge cases of '{concept}'.\n"
                "Where does the standard explanation break down? "
                "What are the boundary conditions? What happens at the limits?"
            ),
            "FIRST_PRINCIPLES": (
                f"Deconstruct '{concept}' to first principles.\n"
                "What is the minimal set of axioms from which this concept emerges? "
                "Could you rebuild it from scratch? What assumptions are hidden?"
            ),
            "FIND_FLAWS": (
                f"Challenge the student to find flaws in '{concept}'.\n"
                "What is wrong or incomplete about standard explanations? "
                "What do textbooks get wrong or oversimplify? "
                "Where has this concept misled practitioners?"
            ),
            "SYNTHESIS": (
                f"Synthesize '{concept}' across domains.\n"
                "Connect it to three unexpected fields. Show how understanding it "
                "transforms understanding of something completely different. "
                "Produce a unified insight that transcends the concept itself."
            ),
        }

        instruction = revolution_prompts.get(revolution, revolution_prompts["SURFACE"])
        system = (
            f"You are the Helix — a spiral learning engine.\n"
            f"Revolution: {revolution}\n\n"
            f"{instruction}\n\n"
            f"{RESPONSE_STRUCTURE}\n\n"
            f"{psyche}"
        )
        messages = [{"role": "user", "content": f"Helix revolution {revolution} for: {concept}"}]
        return await ac.complete_response(messages, system)

    async def get_due_concepts(self, session_id: str) -> list:
        now = datetime.utcnow()
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(HelixRevolution).where(
                    HelixRevolution.session_id == session_id,
                    HelixRevolution.next_due <= now,
                )
            )
            due = result.scalars().all()

        return [
            {
                "concept": hr.concept,
                "revolution": hr.revolution,
                "overdue_hours": round((now - hr.next_due).total_seconds() / 3600, 1) if hr.next_due else 0,
            }
            for hr in due
        ]


helix_engine = HelixEngine()
