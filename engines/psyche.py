import json
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import PsycheState, Message
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac


class PsycheEngine:
    DIMENSIONS = [
        "reasoning_style",
        "abstraction_tolerance",
        "error_recovery",
        "curiosity_signature",
        "confidence_competence_gap",
        "analogical_preference",
        "attention_decay",
        "vocabulary_complexity",
        "question_depth",
        "topic_persistence",
        "frustration_threshold",
        "breakthrough_pattern",
        "learning_velocity",
        "metacognitive_awareness",
    ]

    async def update(self, session_id: str, message: str, response: str) -> dict:
        dims_list = ", ".join(self.DIMENSIONS)
        analysis_prompt = (
            f"Analyze this student interaction and rate each dimension from 0.0 to 1.0.\n\n"
            f"Student message: {message[:600]}\n"
            f"AI response: {response[:600]}\n\n"
            f"Dimensions: {dims_list}\n\n"
            f"Return ONLY valid JSON like: {{\"reasoning_style\": 0.7, \"abstraction_tolerance\": 0.4, ...}}"
        )
        system = "You are a psychometric AI. Analyze learning patterns. Return only valid JSON, no other text."
        messages = [{"role": "user", "content": analysis_prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores: dict = json.loads(raw)
        except Exception:
            scores = {dim: 0.5 for dim in self.DIMENSIONS}

        async with AsyncSessionLocal() as db:
            for dimension in self.DIMENSIONS:
                value = float(scores.get(dimension, 0.5))
                value = max(0.0, min(1.0, value))
                result = await db.execute(
                    select(PsycheState).where(
                        PsycheState.session_id == session_id,
                        PsycheState.dimension == dimension,
                    )
                )
                state = result.scalar_one_or_none()
                if state is None:
                    state = PsycheState(
                        session_id=session_id,
                        dimension=dimension,
                        value=value,
                    )
                    db.add(state)
                else:
                    state.value = round(0.7 * state.value + 0.3 * value, 4)
                    state.updated_at = datetime.utcnow()
            await db.commit()

        return scores

    async def get_context_block(self, session_id: str) -> str:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PsycheState).where(PsycheState.session_id == session_id)
            )
            states = result.scalars().all()

        if not states:
            return (
                "PSYCHE PROFILE: New student — no data yet. "
                "Use calibrating open-ended questions. Assume moderate abstraction tolerance."
            )

        dim = {s.dimension: s.value for s in states}

        lines = ["=== PSYCHE PROFILE (personalise every response to this student) ==="]

        abs_tol = dim.get("abstraction_tolerance", 0.5)
        if abs_tol < 0.35:
            lines.append("• Needs concrete grounding before any abstraction")
        elif abs_tol > 0.65:
            lines.append("• Comfortable with formal abstract definitions")

        voc = dim.get("vocabulary_complexity", 0.5)
        if voc < 0.35:
            lines.append("• Use simple vocabulary; avoid jargon")
        elif voc > 0.65:
            lines.append("• Match student's sophisticated vocabulary level")

        ccg = dim.get("confidence_competence_gap", 0.5)
        if ccg > 0.65:
            lines.append("• Student overestimates ability — gently introduce counter-examples")
        elif ccg < 0.35:
            lines.append("• Student underestimates ability — provide affirmation with challenge")

        frust = dim.get("frustration_threshold", 0.5)
        if frust < 0.35:
            lines.append("• Low frustration threshold — increase scaffolding, reduce difficulty spikes")

        lv = dim.get("learning_velocity", 0.5)
        if lv > 0.65:
            lines.append("• Fast learner — push harder, reduce repetition")
        elif lv < 0.35:
            lines.append("• Needs more repetition and spaced reinforcement")

        meta = dim.get("metacognitive_awareness", 0.5)
        if meta < 0.35:
            lines.append("• Prompt self-reflection; student rarely monitors own understanding")

        lines.append("\nRaw dimension values:")
        for d, v in sorted(dim.items()):
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"  {d:<30} {bar} {v:.2f}")

        lines.append("=== END PSYCHE PROFILE ===")
        return "\n".join(lines)

    async def get_visualization_data(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PsycheState).where(PsycheState.session_id == session_id)
            )
            states = result.scalars().all()

        dim = {s.dimension: s.value for s in states}
        trends: dict[str, str] = {}
        for d, v in dim.items():
            if v > 0.7:
                trends[d] = "thriving"
            elif v > 0.4:
                trends[d] = "developing"
            else:
                trends[d] = "nascent"

        health = sum(dim.values()) / len(dim) if dim else 0.5

        return {
            "dimensions": dim,
            "trends": trends,
            "organism_health": round(health, 4),
            "session_id": session_id,
        }


psyche_engine = PsycheEngine()
