import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.models import Session as PyxisSession, Message
from core.schemas import ChatRequest
from core.config import RESPONSE_STRUCTURE
from engines.psyche import psyche_engine
from engines.curriculum import curriculum_engine
from engines.oracle import oracle_engine
from engines.tides import tide_engine
from engines.gravity import gravity_engine
import engines.anthropic_client as ac

router = APIRouter()


async def _ensure_session(session_id: str, student_name: str | None, db: AsyncSession) -> PyxisSession:
    result = await db.execute(select(PyxisSession).where(PyxisSession.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        session = PyxisSession(id=session_id, student_name=student_name)
        db.add(session)
        await db.commit()
        await db.refresh(session)
    else:
        session.last_active = datetime.utcnow()
        await db.commit()
    return session


async def _save_message(session_id: str, role: str, content: str, feature_mode: str, psyche_snapshot: dict | None, db: AsyncSession) -> Message:
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        feature_mode=feature_mode,
        psyche_snapshot=psyche_snapshot,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def _load_history(session_id: str, db: AsyncSession, limit: int = 20) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.timestamp.desc())
        .limit(limit)
    )
    msgs = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(msgs)]


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        await _ensure_session(request.session_id, request.student_name, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session error: {e}")

    psyche_context = await psyche_engine.get_context_block(request.session_id)
    history = await _load_history(request.session_id, db)

    system = (
        f"You are Pyxis One — an advanced AI learning companion.\n"
        f"Feature mode: {request.feature_mode or 'standard'}\n\n"
        f"{RESPONSE_STRUCTURE}\n\n"
        f"{psyche_context}"
    )

    await _save_message(request.session_id, "user", request.message, request.feature_mode or "standard", None, db)

    history.append({"role": "user", "content": request.message})

    collected_response: list[str] = []

    async def event_generator():
        try:
            async for chunk in ac.stream_response(history, system):
                collected_response.append(chunk)
                yield f"data: {json.dumps({'content': chunk, 'type': 'text'})}\n\n"

            full_response = "".join(collected_response)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            asyncio.create_task(
                _post_chat_tasks(request.session_id, request.message, full_response, request.feature_mode or "standard")
            )

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _post_chat_tasks(session_id: str, user_message: str, ai_response: str, feature_mode: str) -> None:
    try:
        from core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            msg = Message(
                session_id=session_id,
                role="assistant",
                content=ai_response,
                feature_mode=feature_mode,
            )
            db.add(msg)
            await db.commit()
    except Exception:
        pass

    try:
        await psyche_engine.update(session_id, user_message, ai_response)
    except Exception:
        pass

    try:
        await tide_engine.record_reading(session_id, "general", user_message)
    except Exception:
        pass

    try:
        await gravity_engine.update_mass(session_id, "general", 0.5)
    except Exception:
        pass

    try:
        await oracle_engine.predict_wall_concepts(session_id)
    except Exception:
        pass
