import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.schemas import TridentRequest, TridentResponse
from core.config import RESPONSE_STRUCTURE
from engines.psyche import psyche_engine
import engines.anthropic_client as ac

router = APIRouter()

_ARCHITECT_SYSTEM = (
    "You are the Architect — a first-principles theoretician.\n"
    "Build rigorous theoretical frameworks from the ground up.\n"
    "Start from axioms. Derive everything. Miss nothing structurally important.\n"
    "Be precise, dense, and uncompromising in intellectual rigour.\n\n"
    "{psyche}\n\n"
    "{structure}"
)

_STREET_FIGHTER_SYSTEM = (
    "You are the Street Fighter — a brutal pragmatist.\n"
    "Give the fastest, most effective path to actually using this knowledge.\n"
    "Cut every abstraction that doesn't immediately produce results.\n"
    "What do you do first? What do you do next? What do you never do?\n"
    "Be ruthlessly practical. No fluff. Only what works.\n\n"
    "{psyche}\n\n"
    "{structure}"
)

_HERETIC_SYSTEM = (
    "You are the Heretic — a radical contrarian and epistemic insurgent.\n"
    "Systematically attack the mainstream consensus view.\n"
    "What do textbooks get catastrophically wrong?\n"
    "What hidden assumptions is everyone making?\n"
    "What inconvenient truths does the establishment suppress?\n"
    "Be provocative, precise, and well-sourced in your contrarianism.\n\n"
    "{psyche}\n\n"
    "{structure}"
)


@router.post("/trident/stream", response_model=TridentResponse)
async def trident_stream(request: TridentRequest, db: AsyncSession = Depends(get_db)):
    try:
        psyche_context = await psyche_engine.get_context_block(request.session_id)
    except Exception:
        psyche_context = ""

    messages = [{"role": "user", "content": request.question}]

    architect_system = _ARCHITECT_SYSTEM.format(psyche=psyche_context, structure=RESPONSE_STRUCTURE)
    street_system = _STREET_FIGHTER_SYSTEM.format(psyche=psyche_context, structure=RESPONSE_STRUCTURE)
    heretic_system = _HERETIC_SYSTEM.format(psyche=psyche_context, structure=RESPONSE_STRUCTURE)

    try:
        architect, street_fighter, heretic = await asyncio.gather(
            ac.complete_response(messages, architect_system),
            ac.complete_response(messages, street_system),
            ac.complete_response(messages, heretic_system),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trident engine error: {e}")

    return TridentResponse(
        session_id=request.session_id,
        architect=architect,
        street_fighter=street_fighter,
        heretic=heretic,
    )
