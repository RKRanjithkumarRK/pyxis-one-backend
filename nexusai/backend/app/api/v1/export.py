"""Data Export API — Celery ZIP export + email link."""
from __future__ import annotations
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.core.redis import redis_client

router = APIRouter(prefix="/export", tags=["export"])

EXPORT_STATUS_TTL = 86400  # 24 hours


@router.post("/request")
async def request_export(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a Celery task to build a ZIP of all user data. Returns task_id."""
    from app.services.export.tasks import build_user_export
    task = build_user_export.delay(str(current_user.id), current_user.email)
    export_id = str(uuid.uuid4())
    await redis_client.setex(f"export:{export_id}:task", EXPORT_STATUS_TTL, task.id)
    await redis_client.setex(f"export:{export_id}:status", EXPORT_STATUS_TTL, "queued")
    return {"export_id": export_id, "task_id": task.id, "status": "queued"}


@router.get("/{export_id}/status")
async def get_export_status(
    export_id: str,
    current_user: User = Depends(get_current_user),
):
    status = await redis_client.get(f"export:{export_id}:status")
    if status is None:
        raise HTTPException(status_code=404, detail="Export not found")
    download_url = await redis_client.get(f"export:{export_id}:url")
    return {
        "export_id": export_id,
        "status": status.decode(),
        "download_url": download_url.decode() if download_url else None,
    }
