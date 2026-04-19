import json
from uuid import uuid4
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import NemesisRecord, Message, PsycheState
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class NemesisEngine:
    async def analyze_weaknesses(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            msg_result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp.desc())
                .limit(30)
            )
            messages_db = msg_result.scalars().all()

            psyche_result = await db.execute(
                select(PsycheState).where(PsycheState.session_id == session_id)
            )
            psyche_states = psyche_result.scalars().all()

        history_text = "\n".join(
            f"{m.role.upper()}: {m.content[:300]}" for m in reversed(messages_db)
        )
        low_dims = [
            f"{s.dimension}: {s.value:.2f}"
            for s in psyche_states
            if s.value < 0.4
        ]

        prompt = (
            f"Identify this student's core intellectual weaknesses from their session history.\n\n"
            f"Session history:\n{history_text}\n\n"
            f"Low psyche dimensions: {', '.join(low_dims) if low_dims else 'none identified yet'}\n\n"
            "Return JSON array of weaknesses:\n"
            '[{"weakness": "...", "evidence": "...", "severity": "low|medium|high"}]\n'
            "Return ONLY valid JSON array. Maximum 5 weaknesses."
        )
        system = "You are the Nemesis — relentless identifier of intellectual weaknesses. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            weaknesses: list = json.loads(raw)
        except Exception:
            weaknesses = [{"weakness": "Precision in formal definitions", "evidence": "Vague language in responses", "severity": "medium"}]

        return weaknesses

    async def generate_challenge(self, session_id: str, weakness: str) -> str:
        async with AsyncSessionLocal() as db:
            existing = await db.execute(
                select(NemesisRecord).where(
                    NemesisRecord.session_id == session_id,
                    NemesisRecord.weakness == weakness,
                )
            )
            record = existing.scalar_one_or_none()
            if record is None:
                record = NemesisRecord(
                    session_id=session_id,
                    weakness=weakness,
                    challenges_issued=0,
                    challenges_passed=0,
                    challenges_failed=0,
                )
                db.add(record)

            record.challenges_issued += 1
            await db.commit()

        prompt = (
            f"Generate a targeted challenge that forces the student to confront this weakness:\n"
            f"Weakness: {weakness}\n\n"
            "The challenge must:\n"
            "1. Be impossible to answer correctly with the weakness intact\n"
            "2. Expose the weakness without naming it\n"
            "3. Require precise, rigorous thinking\n"
            "4. Have a clear right/wrong evaluable answer\n\n"
            f"{RESPONSE_STRUCTURE}\n"
            "End with the challenge question clearly marked as NEMESIS CHALLENGE:"
        )
        system = "You are the Nemesis — relentless intellectual adversary. Your challenges cannot be bluffed."
        messages = [{"role": "user", "content": prompt}]
        return await ac.complete_response(messages, system, max_tokens=1024)

    async def record_outcome(self, session_id: str, challenge_id: str, passed: bool) -> None:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(NemesisRecord).where(NemesisRecord.id == challenge_id)
            )
            record = result.scalar_one_or_none()
            if record is None:
                result2 = await db.execute(
                    select(NemesisRecord).where(NemesisRecord.session_id == session_id)
                    .order_by(NemesisRecord.challenges_issued.desc())
                    .limit(1)
                )
                record = result2.scalar_one_or_none()

            if record is not None:
                if passed:
                    record.challenges_passed += 1
                else:
                    record.challenges_failed += 1
                await db.commit()

    async def get_record(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(NemesisRecord).where(NemesisRecord.session_id == session_id)
            )
            records = result.scalars().all()

        return {
            "session_id": session_id,
            "weaknesses": [
                {
                    "id": r.id,
                    "weakness": r.weakness,
                    "issued": r.challenges_issued,
                    "passed": r.challenges_passed,
                    "failed": r.challenges_failed,
                    "pass_rate": (r.challenges_passed / r.challenges_issued) if r.challenges_issued > 0 else 0.0,
                }
                for r in records
            ],
        }


nemesis_engine = NemesisEngine()
