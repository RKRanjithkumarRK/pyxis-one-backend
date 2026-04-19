import json
from datetime import datetime, timedelta
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import MirrorReport, Message, PsycheState, ConceptMastery
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac

MIRROR_INTERVAL_DAYS = 7


class MirrorEngine:
    async def check_due(self, session_id: str) -> bool:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MirrorReport)
                .where(MirrorReport.session_id == session_id)
                .order_by(MirrorReport.generated_at.desc())
                .limit(1)
            )
            last = result.scalar_one_or_none()

        if last is None:
            async with AsyncSessionLocal() as db:
                msg_result = await db.execute(
                    select(Message).where(Message.session_id == session_id).limit(5)
                )
                msgs = msg_result.scalars().all()
            return len(msgs) >= 5

        return datetime.utcnow() - last.generated_at >= timedelta(days=MIRROR_INTERVAL_DAYS)

    async def generate_report(self, session_id: str) -> str:
        async with AsyncSessionLocal() as db:
            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(100)
            )
            messages_db = msg_result.scalars().all()

            psyche_result = await db.execute(
                select(PsycheState).where(PsycheState.session_id == session_id)
            )
            psyche_states = psyche_result.scalars().all()

            mastery_result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score.desc())
            )
            masteries = mastery_result.scalars().all()

        history_text = "\n".join(
            f"{m.role.upper()}: {m.content[:250]}" for m in reversed(messages_db[:40])
        )
        dim_summary = "\n".join(
            f"  {s.dimension}: {s.value:.2f}" for s in psyche_states
        )
        mastery_summary = "\n".join(
            f"  {m.concept}: {m.mastery_score:.2f}" for m in masteries[:10]
        )

        prompt = (
            f"Generate a deep personal mirror report for this student.\n\n"
            f"Session history (recent):\n{history_text}\n\n"
            f"Psyche dimensions:\n{dim_summary}\n\n"
            f"Concept mastery (top 10):\n{mastery_summary}\n\n"
            "Write a personal narrative (500-700 words) that:\n"
            "1. Describes the student's unique learning personality with precision\n"
            "2. Names their 3 greatest intellectual strengths with specific evidence\n"
            "3. Names their 2 deepest patterns of avoidance or confusion\n"
            "4. Describes a turning point moment from their sessions\n"
            "5. Offers one profound insight about how they learn that they probably don't know\n"
            "6. Ends with a challenge that only they specifically should take on\n\n"
            "Write in second person. Be compassionate, precise, and honest."
        )
        system = (
            "You are the Mirror — a truthful, compassionate observer of learning. "
            "Write a personal narrative that feels like a wise mentor reflecting back the student's journey."
        )
        messages = [{"role": "user", "content": prompt}]
        report_content = await ac.complete_response(messages, system, max_tokens=2048)

        insights_prompt = (
            f"From this mirror report, extract 5 key insights as a JSON array of strings:\n\n{report_content}"
        )
        insights_system = "Extract insights. Return ONLY valid JSON array of strings."
        try:
            raw = await ac.complete_response(
                [{"role": "user", "content": insights_prompt}],
                insights_system,
                max_tokens=512,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            insights: list = json.loads(raw)
        except Exception:
            insights = ["Unique learning pattern identified", "Strengths recognised", "Growth areas mapped"]

        async with AsyncSessionLocal() as db:
            report = MirrorReport(
                session_id=session_id,
                generated_at=datetime.utcnow(),
                report_content=report_content,
                key_insights=insights,
            )
            db.add(report)
            await db.commit()
            await db.refresh(report)

        return report_content

    async def get_last_report(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MirrorReport)
                .where(MirrorReport.session_id == session_id)
                .order_by(MirrorReport.generated_at.desc())
                .limit(1)
            )
            report = result.scalar_one_or_none()

        if report is None:
            return {
                "session_id": session_id,
                "report": None,
                "key_insights": [],
                "generated_at": None,
            }

        return {
            "session_id": session_id,
            "report": report.report_content,
            "key_insights": report.key_insights or [],
            "generated_at": report.generated_at.isoformat(),
        }


mirror_engine = MirrorEngine()
