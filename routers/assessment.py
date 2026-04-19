import json
from datetime import datetime, timedelta
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.models import Message, PsycheState
from core.schemas import (
    AssessmentGenerateRequest, AssessmentGenerateResponse,
    AssessmentAutopsyRequest, AssessmentAutopsyResponse,
)
from core.config import RESPONSE_STRUCTURE
from engines.psyche import psyche_engine
import engines.anthropic_client as ac

router = APIRouter()


@router.post("/assessment/generate", response_model=AssessmentGenerateResponse)
async def generate_assessment(request: AssessmentGenerateRequest, db: AsyncSession = Depends(get_db)):
    cutoff = datetime.utcnow() - timedelta(hours=72)
    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id == request.session_id, Message.timestamp >= cutoff)
        .order_by(Message.timestamp)
        .limit(60)
    )
    messages_db = msg_result.scalars().all()

    psyche_result = await db.execute(
        select(PsycheState).where(PsycheState.session_id == request.session_id)
    )
    psyche_states = psyche_result.scalars().all()

    weak_dims = [s.dimension for s in psyche_states if s.value < 0.4]
    history_text = "\n".join(f"{m.role.upper()}: {m.content[:300]}" for m in messages_db)

    prompt = (
        f"Generate a procedural assessment based on this 72-hour learning session.\n\n"
        f"Session history:\n{history_text}\n\n"
        f"Weak psyche dimensions: {weak_dims}\n"
        f"Topic focus: {request.topic or 'all covered topics'}\n\n"
        "Generate exactly 5 exam questions that:\n"
        "1. Test understanding, not memorisation\n"
        "2. Target the weakest concepts from the session\n"
        "3. Require transfer to novel contexts\n"
        "4. Have unambiguous correct answers\n"
        "5. Are calibrated to the student's demonstrated level\n\n"
        "Return JSON array:\n"
        '[{"question_id": "q1", "question": "...", "type": "open|mcq|proof", '
        '"expected_answer": "...", "concept_tested": "...", '
        '"origin_message_index": N, "difficulty": "easy|medium|hard"}]\n'
        "Return ONLY valid JSON array."
    )
    system = (
        "You are a master examiner. Generate precise, deep assessment questions. "
        "Return only valid JSON."
    )
    api_messages = [{"role": "user", "content": prompt}]

    try:
        raw = await ac.complete_response(api_messages, system, max_tokens=2048)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        questions: list = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment generation failed: {e}")

    assessment_id = str(uuid4())
    return AssessmentGenerateResponse(
        session_id=request.session_id,
        questions=questions,
        assessment_id=assessment_id,
    )


@router.post("/assessment/autopsy", response_model=AssessmentAutopsyResponse)
async def assessment_autopsy(request: AssessmentAutopsyRequest, db: AsyncSession = Depends(get_db)):
    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id == request.session_id)
        .order_by(Message.timestamp)
        .limit(80)
    )
    messages_db = msg_result.scalars().all()
    history_text = "\n".join(
        f"[{i}] {m.role.upper()}: {m.content[:250]}" for i, m in enumerate(messages_db)
    )

    wrong_answers = [a for a in request.answers if not a.get("correct", True)]
    answers_text = json.dumps(request.answers)

    prompt = (
        f"Perform a forensic autopsy of this assessment.\n\n"
        f"Session history (indexed):\n{history_text}\n\n"
        f"Assessment answers: {answers_text}\n\n"
        "For each wrong answer:\n"
        "1. Trace the confusion to its origin message in the session (use the index)\n"
        "2. Identify the exact moment the wrong model was formed\n"
        "3. Explain what went wrong at that moment\n"
        "4. Prescribe the minimal correction\n\n"
        "Return JSON:\n"
        '{"overall_score": 0.0-1.0, "forensic_analysis": '
        '[{"question_id": "...", "correct": true|false, "origin_message_index": N, '
        '"confusion_source": "...", "correction": "..."}], '
        '"pattern": "...", "prescription": "..."}\n'
        "Return ONLY valid JSON."
    )
    system = "You are a forensic learning analyst. Trace every error to its origin. Return only valid JSON."
    api_messages = [{"role": "user", "content": prompt}]

    try:
        raw = await ac.complete_response(api_messages, system, max_tokens=2048)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        autopsy: dict = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autopsy failed: {e}")

    score = float(autopsy.get("overall_score", 0.0))
    wrong_origins = autopsy.get("forensic_analysis", [])

    return AssessmentAutopsyResponse(
        session_id=request.session_id,
        forensic_report=autopsy,
        wrong_answer_origins=[a for a in wrong_origins if not a.get("correct", True)],
        score=score,
    )
