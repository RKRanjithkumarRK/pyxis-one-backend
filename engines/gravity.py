from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import ConceptMastery, Message

CRITICAL_MASS_THRESHOLD = 0.75
ORBITAL_AFFINITY_WINDOW = 10  # messages to consider for co-occurrence


class GravityEngine:
    async def update_mass(self, session_id: str, concept: str, engagement: float) -> None:
        engagement = max(0.0, min(1.0, engagement))
        async with AsyncSessionLocal() as db:
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
                    mastery_score=engagement * 0.1,
                    last_encounter=datetime.utcnow(),
                )
                db.add(cm)
            else:
                delta = engagement * 0.08
                cm.mastery_score = round(min(1.0, cm.mastery_score + delta), 4)
                cm.last_encounter = datetime.utcnow()
            await db.commit()

    async def check_critical_mass(self, session_id: str, concept: str) -> bool:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(
                    ConceptMastery.session_id == session_id,
                    ConceptMastery.concept == concept,
                )
            )
            cm = result.scalar_one_or_none()

        if cm is None:
            return False
        return cm.mastery_score >= CRITICAL_MASS_THRESHOLD

    async def get_orbital_system(self, session_id: str, concept: str) -> dict:
        async with AsyncSessionLocal() as db:
            all_result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score.desc())
                .limit(15)
            )
            all_concepts = all_result.scalars().all()

            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(ORBITAL_AFFINITY_WINDOW)
            )
            recent_messages = msg_result.scalars().all()

        recent_text = " ".join(m.content for m in recent_messages).lower()

        center = next((c for c in all_concepts if c.concept == concept), None)
        center_mass = center.mastery_score if center else 0.0

        satellites = []
        for c in all_concepts:
            if c.concept == concept:
                continue
            affinity = 0.3
            if c.concept.lower() in recent_text:
                affinity = 0.8
            satellites.append(
                {
                    "concept": c.concept,
                    "mass": c.mastery_score,
                    "distance": round(1.0 - affinity * c.mastery_score, 4),
                    "revolution": c.helix_revolution or "SURFACE",
                    "affinity": affinity,
                }
            )

        satellites.sort(key=lambda x: x["distance"])

        return {
            "center": concept,
            "center_mass": center_mass,
            "critical_mass_reached": center_mass >= CRITICAL_MASS_THRESHOLD,
            "satellites": satellites[:10],
        }

    async def get_universe_map(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(ConceptMastery.session_id == session_id)
            )
            concepts = result.scalars().all()

        total_mass = sum(c.mastery_score for c in concepts)
        galaxies = [
            {
                "concept": c.concept,
                "mass": c.mastery_score,
                "stage": c.helix_revolution or "SURFACE",
                "critical": c.mastery_score >= CRITICAL_MASS_THRESHOLD,
                "last_seen": c.last_encounter.isoformat() if c.last_encounter else None,
            }
            for c in sorted(concepts, key=lambda x: x.mastery_score, reverse=True)
        ]

        return {
            "session_id": session_id,
            "galaxies": galaxies,
            "total_mass": round(total_mass, 4),
            "universe_age_concepts": len(concepts),
            "critical_mass_concepts": sum(1 for c in concepts if c.mastery_score >= CRITICAL_MASS_THRESHOLD),
        }


gravity_engine = GravityEngine()
