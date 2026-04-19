import json
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import CivilizationState
from core.config import RESPONSE_STRUCTURE
import engines.anthropic_client as ac

ERAS = [
    "Stone Age", "Bronze Age", "Iron Age", "Classical Antiquity",
    "Medieval", "Renaissance", "Industrial", "Modern", "Information Age",
    "Post-Singularity",
]


class CivilizationEngine:
    async def initialize(self, session_id: str, subject: str) -> dict:
        async with AsyncSessionLocal() as db:
            existing = await db.execute(
                select(CivilizationState).where(
                    CivilizationState.session_id == session_id,
                    CivilizationState.subject == subject,
                )
            )
            state = existing.scalar_one_or_none()
            if state is not None:
                return self._state_to_dict(state)

            prompt = (
                f"Initialize a civilization game for learning '{subject}'.\n\n"
                "The student governs a civilization that advances by correctly applying "
                f"concepts from {subject}. "
                "Generate the starting state:\n"
                '{"initial_crisis": {"title": "...", "description": "...", "concept_tested": "...", "options": ["A: ...", "B: ...", "C: ..."]}, '
                '"resources": {"knowledge": 100, "stability": 80, "population": 1000, "innovation": 50}}\n'
                "Return ONLY valid JSON."
            )
            system = "You are a civilization game master for learning. Return only valid JSON."
            messages = [{"role": "user", "content": prompt}]

            try:
                raw = await ac.complete_response(messages, system, max_tokens=768)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                init_data: dict = json.loads(raw)
            except Exception:
                init_data = {
                    "initial_crisis": {
                        "title": f"The {subject} Crisis",
                        "description": f"Your civilization faces a challenge requiring knowledge of {subject}.",
                        "concept_tested": subject,
                        "options": ["A: Apply basic principles", "B: Seek more information", "C: Trial and error"],
                    },
                    "resources": {"knowledge": 100, "stability": 80, "population": 1000, "innovation": 50},
                }

            state = CivilizationState(
                session_id=session_id,
                subject=subject,
                era=ERAS[0],
                population=init_data["resources"].get("population", 1000),
                resources=init_data["resources"],
                decisions_history=[],
                current_crisis=init_data.get("initial_crisis"),
                turn_number=0,
            )
            db.add(state)
            await db.commit()
            await db.refresh(state)
            return self._state_to_dict(state)

    async def make_decision(self, session_id: str, decision: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CivilizationState)
                .where(CivilizationState.session_id == session_id)
                .order_by(CivilizationState.turn_number.desc())
                .limit(1)
            )
            state = result.scalar_one_or_none()

        if state is None:
            return {"error": "No civilization found. Call /civilization/initialize first."}

        prompt = (
            f"Evaluate this civilization decision for learning '{state.subject}'.\n\n"
            f"Current era: {state.era}\n"
            f"Current crisis: {json.dumps(state.current_crisis)}\n"
            f"Resources: {json.dumps(state.resources)}\n"
            f"Student decision: {decision}\n\n"
            "Evaluate the decision's correctness based on the subject knowledge required.\n"
            "Return JSON:\n"
            '{"correct": true|false, "consequences": "narrative of what happens", '
            '"resource_changes": {"knowledge": +/-N, "stability": +/-N, "population": +/-N, "innovation": +/-N}, '
            '"lesson": "what concept this tested", '
            '"new_crisis": {"title": "...", "description": "...", "concept_tested": "...", "options": ["A: ...", "B: ...", "C: ..."]}}\n'
            "Return ONLY valid JSON."
        )
        system = "You are a civilization game master. Evaluate decisions for learning correctness. Return only valid JSON."
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = await ac.complete_response(messages, system, max_tokens=1024)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result_data: dict = json.loads(raw)
        except Exception:
            result_data = {
                "correct": True,
                "consequences": "Your decision has mixed results.",
                "resource_changes": {"knowledge": 5, "stability": 0, "population": 0, "innovation": 5},
                "lesson": state.subject,
                "new_crisis": state.current_crisis,
            }

        async with AsyncSessionLocal() as db:
            state_result = await db.execute(
                select(CivilizationState)
                .where(CivilizationState.session_id == session_id)
                .order_by(CivilizationState.turn_number.desc())
                .limit(1)
            )
            state_row = state_result.scalar_one_or_none()
            if state_row is not None:
                resources = dict(state_row.resources or {})
                changes = result_data.get("resource_changes", {})
                for k, v in changes.items():
                    resources[k] = max(0, resources.get(k, 0) + v)

                history = list(state_row.decisions_history or [])
                history.append({
                    "turn": state_row.turn_number,
                    "decision": decision,
                    "correct": result_data.get("correct"),
                    "lesson": result_data.get("lesson"),
                })

                state_row.resources = resources
                state_row.decisions_history = history
                state_row.current_crisis = result_data.get("new_crisis")
                state_row.turn_number += 1
                state_row.population = resources.get("population", state_row.population)

                if state_row.turn_number % 5 == 0 and state_row.turn_number > 0:
                    era_idx = min(state_row.turn_number // 5, len(ERAS) - 1)
                    state_row.era = ERAS[era_idx]

                await db.commit()

        return {
            "session_id": session_id,
            "consequences": result_data.get("consequences"),
            "new_state": {
                "era": state_row.era if state_row else state.era,
                "resources": resources if state_row else state.resources,
                "turn_number": (state_row.turn_number if state_row else state.turn_number),
                "current_crisis": result_data.get("new_crisis"),
            },
            "turn_number": state_row.turn_number if state_row else state.turn_number,
            "correct": result_data.get("correct"),
            "lesson": result_data.get("lesson"),
        }

    async def advance_turn(self, session_id: str) -> dict:
        return await self.make_decision(session_id, "SKIP_TURN: advance without decision")

    async def get_crisis(self, session_id: str) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CivilizationState)
                .where(CivilizationState.session_id == session_id)
                .order_by(CivilizationState.turn_number.desc())
                .limit(1)
            )
            state = result.scalar_one_or_none()

        if state is None:
            return {"error": "No civilization found"}

        return {
            "session_id": session_id,
            "era": state.era,
            "turn_number": state.turn_number,
            "current_crisis": state.current_crisis,
            "resources": state.resources,
        }

    def _state_to_dict(self, state: CivilizationState) -> dict:
        return {
            "id": state.id,
            "session_id": state.session_id,
            "subject": state.subject,
            "era": state.era,
            "population": state.population,
            "resources": state.resources,
            "turn_number": state.turn_number,
            "current_crisis": state.current_crisis,
        }


civilization_engine = CivilizationEngine()
