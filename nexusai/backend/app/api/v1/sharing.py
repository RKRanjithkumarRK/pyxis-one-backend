"""Sharing API — immutable conversation share links with snapshots."""
from __future__ import annotations
import hashlib
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user, get_db
from app.core.redis import redis_client
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message

router = APIRouter(prefix="/share", tags=["sharing"])

SHARE_TTL = 86400 * 365  # 1 year


def _share_key(token: str) -> str:
    return f"share:snapshot:{token}"


@router.post("/conversations/{conversation_id}")
async def create_share_link(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Snapshot a conversation and return a public share link token."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == uuid.UUID(conversation_id),
            Conversation.user_id == current_user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.sequence)
        .limit(500)
    )
    messages = msg_result.scalars().all()

    snapshot = {
        "conversation_id": str(conv.id),
        "title": conv.title or "Shared conversation",
        "model": conv.model,
        "created_at": conv.created_at.isoformat(),
        "owner": current_user.name or current_user.email or "Anonymous",
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }

    token = hashlib.sha256(f"{conversation_id}:{current_user.id}:{uuid.uuid4()}".encode()).hexdigest()[:32]
    await redis_client.setex(_share_key(token), SHARE_TTL, json.dumps(snapshot))

    frontend_url = __import__("app.core.config", fromlist=["settings"]).settings.FRONTEND_URL
    return {"token": token, "url": f"{frontend_url}/shared/{token}"}


@router.get("/{token}")
async def get_shared_snapshot(token: str):
    """Public endpoint — returns the snapshot for a share token."""
    data = await redis_client.get(_share_key(token))
    if data is None:
        raise HTTPException(status_code=404, detail="Share link not found or expired")
    return json.loads(data)
