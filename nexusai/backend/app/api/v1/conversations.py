from __future__ import annotations
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import bearer, decode_token
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationResponse(BaseModel):
    id: str
    title: str
    model_id: str
    active_branch_id: str | None
    project_id: str | None
    agent_id: str | None
    pinned_at: str | None
    archived_at: str | None
    is_shared: bool
    memory_enabled: bool
    web_search_enabled: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: str
    branch_id: str
    sequence: int
    role: str
    content: str
    model_id: str | None
    usage: dict | None
    citations: list | None
    attachments: list | None
    feedback: str | None
    created_at: str

    model_config = {"from_attributes": True}


class UpdateConversationRequest(BaseModel):
    title: str | None = None
    model_id: str | None = None
    pinned: bool | None = None
    archived: bool | None = None
    memory_enabled: bool | None = None
    web_search_enabled: bool | None = None


class EditMessageRequest(BaseModel):
    new_content: str


def _to_str(v) -> str | None:
    return str(v) if v else None


def _fmt(dt) -> str | None:
    return dt.isoformat() if dt else None


def _conv_dict(conv) -> dict:
    return {
        "id": str(conv.id),
        "title": conv.title,
        "model_id": conv.model_id,
        "active_branch_id": _to_str(conv.active_branch_id),
        "project_id": _to_str(conv.project_id),
        "agent_id": _to_str(conv.agent_id),
        "pinned_at": _fmt(conv.pinned_at),
        "archived_at": _fmt(conv.archived_at),
        "is_shared": conv.is_shared,
        "memory_enabled": conv.memory_enabled,
        "web_search_enabled": conv.web_search_enabled,
        "created_at": _fmt(conv.created_at),
        "updated_at": _fmt(conv.updated_at),
    }


def _msg_dict(msg) -> dict:
    return {
        "id": str(msg.id),
        "branch_id": str(msg.branch_id),
        "sequence": msg.sequence,
        "role": msg.role,
        "content": msg.content,
        "model_id": msg.model_id,
        "usage": msg.usage,
        "citations": msg.citations,
        "attachments": msg.attachments,
        "feedback": msg.feedback,
        "created_at": _fmt(msg.created_at),
    }


@router.get("")
async def list_conversations(
    include_archived: bool = Query(False),
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    convs = await ConversationRepository.list_for_user(
        db, user_id, include_archived=include_archived
    )
    return {"conversations": [_conv_dict(c) for c in convs]}


@router.post("", status_code=201)
async def create_conversation(
    payload: dict = {},
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    conv = await ConversationRepository.create(
        db,
        user_id,
        model_id=payload.get("model_id", "claude-sonnet-4"),
        project_id=uuid.UUID(payload["project_id"]) if payload.get("project_id") else None,
        agent_id=uuid.UUID(payload["agent_id"]) if payload.get("agent_id") else None,
    )
    return _conv_dict(conv)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    conv = await ConversationRepository.get(db, uuid.UUID(conversation_id), user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conv_dict(conv)


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    now = datetime.now(timezone.utc)
    updates: dict = {}
    if body.title is not None:
        updates["title"] = body.title
    if body.model_id is not None:
        updates["model_id"] = body.model_id
    if body.pinned is True:
        updates["pinned_at"] = now
    elif body.pinned is False:
        updates["pinned_at"] = None
    if body.archived is True:
        updates["archived_at"] = now
    elif body.archived is False:
        updates["archived_at"] = None
    if body.memory_enabled is not None:
        updates["memory_enabled"] = body.memory_enabled
    if body.web_search_enabled is not None:
        updates["web_search_enabled"] = body.web_search_enabled
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")
    conv = await ConversationRepository.update(db, uuid.UUID(conversation_id), user_id, **updates)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conv_dict(conv)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    deleted = await ConversationRepository.delete(db, uuid.UUID(conversation_id), user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.get("/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    conv = await ConversationRepository.get(db, uuid.UUID(conversation_id), user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conv.active_branch_id:
        return {"messages": []}
    msgs = await MessageRepository.list_branch(
        db, conv.id, conv.active_branch_id
    )
    return {"messages": [_msg_dict(m) for m in msgs]}


@router.post("/{conversation_id}/messages/{message_id}/edit")
async def edit_message(
    conversation_id: str,
    message_id: str,
    body: EditMessageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    from app.services.conversation.service import ConversationService
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    new_branch_id = await ConversationService.edit_message(
        db,
        uuid.UUID(conversation_id),
        uuid.UUID(message_id),
        body.new_content,
        user_id,
    )
    return {"branch_id": str(new_branch_id)}


@router.get("/search")
async def search_conversations(
    q: str = Query(..., min_length=1),
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    db: AsyncSession = Depends(get_db),
):
    tp = decode_token(credentials.credentials)
    user_id = uuid.UUID(tp["sub"])
    convs = await ConversationRepository.search(db, user_id, q)
    return {"conversations": [_conv_dict(c) for c in convs]}
