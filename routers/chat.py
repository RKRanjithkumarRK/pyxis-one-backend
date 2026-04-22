"""
Main chat streaming endpoint.

POST /api/chat/stream
  - Runs full pipeline: intent → route → context → stream → tools → memory
  - Returns SSE stream of StreamEvent JSON lines
  - Supports: model selection, regeneration, branching, file attachments, tool use
"""

from __future__ import annotations
import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db, AsyncSessionLocal
from core.models import (
    Session as PyxisSession,
    Message,
    Conversation,
    ConversationMessage,
    FileUpload,
)
from core.schemas import ChatRequest
from core.pipeline.stream_orchestrator import orchestrate
from core.middleware.rate_limiter import check as rate_check, RateLimitError
from engines.psyche import psyche_engine
from engines.curriculum import curriculum_engine
from engines.oracle import oracle_engine
from engines.tides import tide_engine
from engines.gravity import gravity_engine
from routers.conversations import generate_title, save_message

router = APIRouter()


# ── Main streaming endpoint ───────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    # Ensure session
    session = await _ensure_session(request.session_id, request.student_name, db)
    user_tier = session.tier

    # Rate limiting
    try:
        await rate_check(request.session_id, user_tier)
    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail={"message": "Rate limit exceeded", "retry_after": e.retry_after},
            headers={"Retry-After": str(e.retry_after)},
        )

    # Load or create conversation
    conv = await _ensure_conversation(
        request.session_id,
        request.conversation_id,
        request.model or "claude-sonnet-4-6",
        request.feature_mode or "standard",
        db,
    )

    # Load history for this conversation + branch
    history = await _load_conversation_history(
        conv.id, request.branch_index or 0, db
    )

    # Load psyche context
    psyche_context = await psyche_engine.get_context_block(request.session_id)

    # Load file contexts if any
    file_contexts: list[dict] = []
    has_attachments = False
    if request.file_ids:
        for fid in request.file_ids:
            res = await db.execute(select(FileUpload).where(FileUpload.id == fid))
            f = res.scalar_one_or_none()
            if f:
                file_contexts.append({
                    "filename": f.filename,
                    "content": f.extracted_text or "",
                    "image_b64": f.image_b64,
                })
                has_attachments = True

    # Save user message immediately (before streaming starts)
    user_msg = await save_message(
        db,
        conversation_id=conv.id,
        role="user",
        content=request.message,
        branch_index=request.branch_index or 0,
        feature_mode=request.feature_mode,
    )

    collected_chunks: list[str] = []
    tool_events: list[dict] = []
    final_usage: dict | None = None
    selected_model: str = request.model or "claude-sonnet-4-6"

    async def event_generator():
        nonlocal final_usage, selected_model

        try:
            async for event in orchestrate(
                session_id=request.session_id,
                message=request.message,
                history=history,
                feature_mode=request.feature_mode or "standard",
                psyche_context=psyche_context,
                rag_chunks=None,
                user_tier=user_tier,
                manual_model=request.model,
                enable_web_search=request.enable_web_search or False,
                has_attachments=has_attachments,
                file_contexts=file_contexts if file_contexts else None,
                temperature_boost=(request.regeneration_attempt or 0) * 0.1,
            ):
                if event.type == "model_selected":
                    try:
                        info = json.loads(event.content)
                        selected_model = info.get("model", selected_model)
                    except Exception:
                        pass
                    yield event.to_sse()

                elif event.type == "text":
                    collected_chunks.append(event.content)
                    yield event.to_sse()

                elif event.type in ("tool_start", "tool_done", "tool_result"):
                    tool_events.append(event.__dict__)
                    yield event.to_sse()

                elif event.type == "tool_delta":
                    yield event.to_sse()

                elif event.type == "thinking":
                    yield event.to_sse()

                elif event.type == "done":
                    final_usage = event.usage
                    yield event.to_sse()

                    # Schedule background tasks without blocking
                    full_response = "".join(collected_chunks)
                    asyncio.create_task(
                        _post_chat_tasks(
                            session_id=request.session_id,
                            conversation_id=conv.id,
                            user_message=request.message,
                            ai_response=full_response,
                            model=selected_model,
                            feature_mode=request.feature_mode or "standard",
                            branch_index=request.branch_index or 0,
                            usage=final_usage,
                            tool_events=tool_events,
                            is_first_message=len(history) == 0,
                        )
                    )

                elif event.type == "system":
                    yield event.to_sse()

                elif event.type == "error":
                    yield event.to_sse()

        except HTTPException:
            raise
        except Exception as e:
            err_event = {
                "type": "error",
                "content": str(e)[:300],
                "error_code": "stream_error",
            }
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Models list endpoint ───────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    from core.config import AVAILABLE_MODELS, settings
    models = []
    for m in AVAILABLE_MODELS:
        available = True
        if m["provider"] == "openai" and not settings.OPENAI_API_KEY:
            available = False
        if m["provider"] == "anthropic" and not settings.ANTHROPIC_API_KEY:
            available = False
        models.append({**m, "available": available})
    return {"models": models}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _ensure_session(
    session_id: str, student_name: str | None, db: AsyncSession
) -> PyxisSession:
    result = await db.execute(
        select(PyxisSession).where(PyxisSession.id == session_id)
    )
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


async def _ensure_conversation(
    session_id: str,
    conversation_id: str | None,
    model: str,
    feature_mode: str,
    db: AsyncSession,
) -> Conversation:
    if conversation_id:
        res = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = res.scalar_one_or_none()
        if conv:
            return conv

    # Create new conversation
    conv = Conversation(
        session_id=session_id,
        title="New conversation",
        model=model,
        feature_mode=feature_mode,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def _load_conversation_history(
    conversation_id: str,
    branch_index: int,
    db: AsyncSession,
    limit: int = 50,
) -> list[dict]:
    result = await db.execute(
        select(ConversationMessage)
        .where(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.branch_index == branch_index,
        )
        .order_by(ConversationMessage.created_at.desc())
        .limit(limit)
    )
    msgs = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(msgs)]


async def _post_chat_tasks(
    session_id: str,
    conversation_id: str,
    user_message: str,
    ai_response: str,
    model: str,
    feature_mode: str,
    branch_index: int,
    usage: dict | None,
    tool_events: list[dict],
    is_first_message: bool,
) -> None:
    async with AsyncSessionLocal() as db:
        # Save assistant message
        try:
            await save_message(
                db,
                conversation_id=conversation_id,
                role="assistant",
                content=ai_response,
                model=model,
                branch_index=branch_index,
                finish_reason="stop",
                usage=usage,
                feature_mode=feature_mode,
                tool_results=tool_events if tool_events else None,
            )
        except Exception:
            pass

        # Auto-generate title on first message
        if is_first_message and ai_response:
            try:
                title = await generate_title(conversation_id, user_message)
                res = await db.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )
                conv = res.scalar_one_or_none()
                if conv and conv.title in ("New conversation", None):
                    conv.title = title
                    await db.commit()
            except Exception:
                pass

        # Update conversation.updated_at
        try:
            res = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = res.scalar_one_or_none()
            if conv:
                conv.updated_at = datetime.utcnow()
                await db.commit()
        except Exception:
            pass

    # Legacy message save for existing engines
    try:
        async with AsyncSessionLocal() as db:
            for role, content in [("user", user_message), ("assistant", ai_response)]:
                msg = Message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    feature_mode=feature_mode,
                )
                db.add(msg)
            await db.commit()
    except Exception:
        pass

    # Run all learning engines in parallel
    await asyncio.gather(
        _safe(psyche_engine.update(session_id, user_message, ai_response)),
        _safe(tide_engine.record_reading(session_id, "general", user_message)),
        _safe(gravity_engine.update_mass(session_id, "general", 0.5)),
        _safe(oracle_engine.predict_wall_concepts(session_id)),
    )


async def _safe(coro):
    try:
        await coro
    except Exception:
        pass
