"""
Conversation management — ChatGPT-style history sidebar.

Endpoints:
  POST   /api/conversations                  create new conversation
  GET    /api/conversations/{session_id}     list all conversations (paginated)
  GET    /api/conversations/{id}/messages    load full message history
  PATCH  /api/conversations/{id}             rename / pin
  DELETE /api/conversations/{id}             delete
  GET    /api/conversations/search           full-text search
  POST   /api/conversations/{id}/branch      create a branch from a message
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.models import Conversation, ConversationMessage, Session as PyxisSession
from engines import unified_client as uc

router = APIRouter()


# ── Create ────────────────────────────────────────────────────────────────────

@router.post("/conversations")
async def create_conversation(body: dict, db: AsyncSession = Depends(get_db)):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")

    # Ensure session exists
    res = await db.execute(select(PyxisSession).where(PyxisSession.id == session_id))
    if res.scalar_one_or_none() is None:
        session = PyxisSession(id=session_id)
        db.add(session)

    conv = Conversation(
        session_id=session_id,
        title=body.get("title") or "New conversation",
        model=body.get("model", "claude-sonnet-4-6"),
        feature_mode=body.get("feature_mode"),
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return _conv_to_dict(conv)


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/conversations/list/{session_id}")
async def list_conversations(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
        .offset(offset)
    )
    convs = result.scalars().all()
    return {"conversations": [_conv_to_dict(c) for c in convs], "offset": offset, "limit": limit}


# ── Load messages ─────────────────────────────────────────────────────────────

@router.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    branch_index: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    # Verify conversation exists
    res = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = res.scalar_one_or_none()
    if not conv:
        raise HTTPException(404, "Conversation not found")

    # Load messages for the requested branch
    result = await db.execute(
        select(ConversationMessage)
        .where(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.branch_index == branch_index,
        )
        .order_by(ConversationMessage.created_at)
    )
    msgs = result.scalars().all()

    # Count how many branches exist
    branch_result = await db.execute(
        select(ConversationMessage.branch_index)
        .where(ConversationMessage.conversation_id == conversation_id)
        .distinct()
    )
    branch_indices = [r[0] for r in branch_result.all()]

    return {
        "conversation": _conv_to_dict(conv),
        "messages": [_msg_to_dict(m) for m in msgs],
        "branch_index": branch_index,
        "branch_count": len(branch_indices),
        "branch_indices": sorted(branch_indices),
    }


# ── Update (rename / pin) ─────────────────────────────────────────────────────

@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = res.scalar_one_or_none()
    if not conv:
        raise HTTPException(404, "Conversation not found")

    if "title" in body:
        conv.title = body["title"][:255]
    if "pinned" in body:
        conv.pinned = bool(body["pinned"])
    if "model" in body:
        conv.model = body["model"]

    conv.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(conv)
    return _conv_to_dict(conv)


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = res.scalar_one_or_none()
    if not conv:
        raise HTTPException(404, "Conversation not found")

    await db.delete(conv)
    await db.commit()
    return {"deleted": conversation_id}


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/conversations/search/{session_id}")
async def search_conversations(
    session_id: str,
    q: str = Query(..., min_length=1),
    db: AsyncSession = Depends(get_db),
):
    query_lower = f"%{q.lower()}%"

    # Search in titles
    title_result = await db.execute(
        select(Conversation)
        .where(
            Conversation.session_id == session_id,
            Conversation.title.ilike(query_lower),
        )
        .order_by(desc(Conversation.updated_at))
        .limit(20)
    )
    title_matches = title_result.scalars().all()

    # Search in message content
    msg_result = await db.execute(
        select(ConversationMessage)
        .join(Conversation, Conversation.id == ConversationMessage.conversation_id)
        .where(
            Conversation.session_id == session_id,
            ConversationMessage.content.ilike(query_lower),
        )
        .order_by(desc(ConversationMessage.created_at))
        .limit(20)
    )
    msg_matches = msg_result.scalars().all()

    # Gather conversation IDs from message matches
    msg_conv_ids = {m.conversation_id for m in msg_matches}
    extra_convs: list[Conversation] = []
    if msg_conv_ids:
        extra_result = await db.execute(
            select(Conversation).where(Conversation.id.in_(msg_conv_ids))
        )
        extra_convs = extra_result.scalars().all()

    # Deduplicate
    seen: set[str] = set()
    all_convs: list[Conversation] = []
    for c in list(title_matches) + extra_convs:
        if c.id not in seen:
            seen.add(c.id)
            all_convs.append(c)

    return {
        "query": q,
        "results": [_conv_to_dict(c) for c in all_convs[:20]],
    }


# ── Branch ────────────────────────────────────────────────────────────────────

@router.post("/conversations/{conversation_id}/branch")
async def create_branch(
    conversation_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new branch from a specific message.
    Used when user edits a prompt — everything after that point branches.
    """
    parent_message_id = body.get("parent_message_id")
    if not parent_message_id:
        raise HTTPException(400, "parent_message_id required")

    # Find the highest existing branch index for this conversation
    existing = await db.execute(
        select(ConversationMessage.branch_index)
        .where(ConversationMessage.conversation_id == conversation_id)
        .distinct()
    )
    indices = [r[0] for r in existing.all()]
    new_branch_index = (max(indices) + 1) if indices else 1

    return {
        "conversation_id": conversation_id,
        "new_branch_index": new_branch_index,
        "parent_message_id": parent_message_id,
    }


# ── Auto-generate title (called after first exchange) ────────────────────────

async def generate_title(conversation_id: str, first_message: str) -> str:
    """Generate a 5-word title from the first user message. Silent background task."""
    try:
        title = await uc.complete(
            messages=[{"role": "user", "content": f"Summarize this in 5 words or less (title case, no punctuation): {first_message[:300]}"}],
            system="You generate conversation titles. Reply with ONLY the title. No quotes, no explanation.",
            model="gpt-4o-mini",
            max_tokens=20,
            temperature=0.3,
        )
        return title.strip()[:80] or first_message[:60]
    except Exception:
        return first_message[:60]


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    model: str | None = None,
    branch_index: int = 0,
    parent_id: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    feature_mode: str | None = None,
    tool_calls: list | None = None,
    tool_results: list | None = None,
) -> ConversationMessage:
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model=model,
        branch_index=branch_index,
        parent_id=parent_id,
        finish_reason=finish_reason,
        usage=usage,
        feature_mode=feature_mode,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


# ── Serializers ───────────────────────────────────────────────────────────────

def _conv_to_dict(c: Conversation) -> dict:
    return {
        "id": c.id,
        "session_id": c.session_id,
        "title": c.title,
        "model": c.model,
        "feature_mode": c.feature_mode,
        "pinned": c.pinned,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def _msg_to_dict(m: ConversationMessage) -> dict:
    return {
        "id": m.id,
        "conversation_id": m.conversation_id,
        "parent_id": m.parent_id,
        "branch_index": m.branch_index,
        "role": m.role,
        "content": m.content,
        "model": m.model,
        "finish_reason": m.finish_reason,
        "tool_calls": m.tool_calls,
        "tool_results": m.tool_results,
        "usage": m.usage,
        "feature_mode": m.feature_mode,
        "created_at": m.created_at.isoformat(),
    }
