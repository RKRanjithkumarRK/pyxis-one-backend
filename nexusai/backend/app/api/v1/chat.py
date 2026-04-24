from __future__ import annotations
import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_bearer, decode_token
from app.services.llm.router import stream_chat, available_models, get_model_latency, ROUTES
from app.services.conversation.service import ConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/models")
async def list_models():
    models = available_models()
    # Attach rolling latency from Redis
    result = []
    for m in models:
        latency = await get_model_latency(m["id"])
        result.append({**m, "latency_p50_ms": latency})
    return {"models": result}


@router.post("/stream")
async def chat_stream(
    payload: dict,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    token_payload = decode_token(credentials.credentials)
    user_id = uuid.UUID(token_payload["sub"])
    is_guest = token_payload.get("is_guest", False)

    conversation_id_str: str | None = payload.get("conversation_id")
    model = payload.get("model", "claude-sonnet-4")
    user_message: str = payload.get("message", "")
    attachments: list = payload.get("attachments", [])
    agent_id: str | None = payload.get("agent_id")
    project_id: str | None = payload.get("project_id")
    kb_ids: list[str] = payload.get("knowledge_base_ids", [])
    use_web_search: bool = payload.get("web_search", False)
    use_memory: bool = payload.get("use_memory", True) and not is_guest

    if not user_message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    # Guest rate check
    if is_guest:
        from app.services.auth.service import AuthService
        remaining = await AuthService.get_guest_messages_remaining(str(user_id))
        if remaining <= 0:
            raise HTTPException(
                status_code=402,
                detail="Guest message limit reached. Sign up to continue.",
            )

    conv = await ConversationService.get_or_create(
        db,
        conversation_id_str,
        user_id,
        model_id=model,
        agent_id=agent_id,
        project_id=project_id,
    )
    conversation_id = conv.id

    # Build system prompt
    system_parts: list[str] = []
    if agent_id:
        agent_prompt = await ConversationService.get_agent_prompt(db, agent_id, user_id)
        if agent_prompt:
            system_parts.append(agent_prompt)
    elif project_id:
        project_prompt = await ConversationService.get_project_prompt(db, project_id, user_id)
        if project_prompt:
            system_parts.append(project_prompt)

    # Memory retrieval (Phase 8 will fully implement; stub here)
    if use_memory:
        pass  # MemoryService.retrieve_for wired in Phase 8

    # RAG retrieval (Phase 10 will fully implement; stub here)
    rag_context: list = []
    if kb_ids:
        pass  # RAGService.hybrid_search wired in Phase 10

    # Web search (Phase 6 will fully implement; stub here)
    web_results: list = []
    if use_web_search:
        pass  # Serper wired in Phase 6

    system_prompt = "\n\n".join(system_parts) if system_parts else None

    # Compose message content with attachments
    content: Any = user_message
    if attachments:
        parts: list[dict] = [{"type": "text", "text": user_message}]
        for att in attachments:
            if att.get("type") == "image":
                parts.append({"type": "image_url", "image_url": {"url": att["url"]}})
            elif att.get("type") == "document":
                parts.append({"type": "text", "text": f"[File: {att['name']}]\n{att.get('extracted_text', '')}"})
        content = parts

    # Build message history
    history = await ConversationService.get_history(db, conversation_id, user_id)
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    # Persist user message before streaming
    await ConversationService.append(
        db, conversation_id, "user", user_message,
        attachments=attachments or None,
        user_id=user_id,
    )

    if is_guest:
        from app.services.auth.service import AuthService
        await AuthService.increment_guest_message_count(str(user_id))

    async def event_stream():
        assistant_buf = ""
        model_used = model
        usage_final: dict | None = None

        yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': str(conversation_id)})}\n\n"

        try:
            async for event in stream_chat(
                model_id=model,
                messages=messages,
                user_id=str(user_id),
                conversation_id=str(conversation_id),
            ):
                if await request.is_disconnected():
                    break

                if event["type"] == "token":
                    assistant_buf += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "done":
                    model_used = event["model"]
                    usage_final = event["usage"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] in ("error", "fallback"):
                    yield f"data: {json.dumps(event)}\n\n"

        finally:
            if assistant_buf:
                msg_id = await ConversationService.append(
                    db, conversation_id, "assistant", assistant_buf,
                    model=model_used, usage=usage_final,
                    citations=(
                        [{"type": "web", **r} for r in web_results]
                        if web_results else None
                    ),
                    user_id=user_id,
                )
                yield f"data: {json.dumps({'type': 'message_id', 'message_id': str(msg_id)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/compare")
async def chat_compare(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    """Stream 2-3 models simultaneously in multiplexed SSE columns."""
    token_payload = decode_token(credentials.credentials)
    user_id = str(token_payload["sub"])

    models: list[str] = payload.get("models", [])
    if not (2 <= len(models) <= 3):
        raise HTTPException(status_code=422, detail="Compare requires 2-3 models")

    prompt: str = payload.get("message", "")
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    messages = [{"role": "user", "content": prompt}]

    async def multiplexed():
        queue: asyncio.Queue[dict] = asyncio.Queue()
        done_count = 0

        async def run(idx: int, model: str):
            async for ev in stream_chat(
                model_id=model,
                messages=messages,
                user_id=user_id,
                conversation_id=f"compare-{idx}",
            ):
                await queue.put({"column": idx, "model": model, **ev})
            await queue.put({"column": idx, "type": "column_done"})

        tasks = [asyncio.create_task(run(i, m)) for i, m in enumerate(models)]

        while done_count < len(models):
            ev = await queue.get()
            if ev.get("type") == "column_done":
                done_count += 1
                continue
            yield f"data: {json.dumps(ev)}\n\n"

        for t in tasks:
            t.cancel()

    return StreamingResponse(multiplexed(), media_type="text/event-stream")


@router.post("/{message_id}/feedback")
async def submit_feedback(
    message_id: str,
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    feedback = payload.get("feedback")
    if feedback not in ("good", "bad"):
        raise HTTPException(status_code=422, detail="feedback must be 'good' or 'bad'")
    await MessageRepository.set_feedback(db, uuid.UUID(message_id), feedback)
    return {"ok": True}
