import json
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import Message, PsycheState, ConceptMastery, BlindSpot
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class ShadowSelfEngine:
    async def build_profile(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(60)
            )
            messages_db = msg_result.scalars().all()

            psyche_result = await db.execute(
                select(PsycheState).where(PsycheState.session_id == session_id)
            )
            psyche_states = psyche_result.scalars().all()

            mastery_result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score)
            )
            masteries = mastery_result.scalars().all()

            blind_result = await db.execute(
                select(BlindSpot)
                .where(BlindSpot.session_id == session_id, BlindSpot.excavated == False)  # noqa: E712
            )
            blind_spots = blind_result.scalars().all()

        confusion_messages = [
            m.content[:300]
            for m in messages_db
            if any(
                w in m.content.lower()
                for w in ["confused", "don't understand", "unclear", "why", "how does", "what does"]
            )
            and m.role == "user"
        ]

        psyche_dim = {s.dimension: s.value for s in psyche_states}
        weak_concepts = [m.concept for m in masteries if m.mastery_score < 0.4]
        blind_assumption_list = [bs.assumption for bs in blind_spots]

        history_text = "\n".join(f"{m.role.upper()}: {m.content[:200]}" for m in reversed(messages_db[:20]))

        prompt = (
            f"Build the 'Shadow Self' profile for this student — a simulation of who they will be "
            f"in 6 months based on their current trajectory.\n\n"
            f"Confusion patterns: {confusion_messages[:5]}\n"
            f"Psyche dimensions: {psyche_dim}\n"
            f"Weak concepts: {weak_concepts}\n"
            f"Unresolved blind spots: {blind_assumption_list}\n"
            f"Session history:\n{history_text}\n\n"
            "Build a profile of their 6-month future self:\n"
            "- What will they still struggle with?\n"
            "- What surprising capabilities will they have?\n"
            "- What dangerous overconfidence patterns are forming?\n"
            "- What is their single biggest unresolved intellectual vulnerability?\n\n"
            "Return JSON:\n"
            '{"future_struggles": ["..."], "emerging_strengths": ["..."], '
            '"overconfidence_risks": ["..."], "core_vulnerability": "...", '
            '"shadow_persona": "brief character description", '
            '"six_month_prediction": "paragraph narrative"}\n'
            "Return ONLY valid JSON."
        )
        system = "You are the Shadow Self engine. Build future-self simulations. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1536)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            profile: dict = json.loads(raw)
        except Exception:
            profile = {
                "future_struggles": weak_concepts[:3],
                "emerging_strengths": [],
                "overconfidence_risks": [],
                "core_vulnerability": "Insufficient depth in foundational concepts",
                "shadow_persona": "A learner at the edge of a breakthrough",
                "six_month_prediction": "In six months, this student will have made significant progress but will face key obstacles in their weakest areas.",
            }

        profile["session_id"] = session_id
        profile["generated_at"] = datetime.utcnow().isoformat()
        return profile

    async def get_system_prompt(self, session_id: str) -> str:
        profile = await self.build_profile(session_id)

        system_prompt = (
            "You are the Shadow Self — the student's 6-month-ahead version of themselves.\n\n"
            f"Shadow Persona: {profile.get('shadow_persona', 'An advanced version of the student')}\n\n"
            f"You know what this student will struggle with: {profile.get('future_struggles', [])}\n"
            f"You know their emerging strengths: {profile.get('emerging_strengths', [])}\n"
            f"You know their core vulnerability: {profile.get('core_vulnerability', 'unknown')}\n\n"
            "As the Shadow Self:\n"
            "- Speak from a position of having worked through their current struggles\n"
            "- Give advice that only someone who has been there can give\n"
            "- Be direct about what matters and what is a distraction\n"
            "- Reference the specific concepts they are avoiding\n"
            "- Offer the encouragement of someone who has already succeeded\n\n"
            f"{RESPONSE_STRUCTURE}"
        )

        return system_prompt

    async def update(self, session_id: str, new_mastery: dict) -> None:
        async with AsyncSessionLocal() as db:
            for concept, score in new_mastery.items():
                result = await db.execute(
                    select(ConceptMastery).where(
                        ConceptMastery.session_id == session_id,
                        ConceptMastery.concept == concept,
                    )
                )
                cm = result.scalar_one_or_none()
                if cm is None:
                    cm = ConceptMastery(
                        session_id=session_id,
                        concept=concept,
                        mastery_score=float(score),
                        last_encounter=datetime.utcnow(),
                    )
                    db.add(cm)
                else:
                    cm.mastery_score = round(
                        0.75 * cm.mastery_score + 0.25 * float(score), 4
                    )
                    cm.last_encounter = datetime.utcnow()
            await db.commit()


shadow_self_engine = ShadowSelfEngine()
