from __future__ import annotations
import asyncio
import json
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import redis_client
from app.core.security import require_bearer, decode_token
from app.repositories.research import ResearchRepository
from app.services.research.tasks import deep_research, CHANNEL_PREFIX

router = APIRouter(prefix="/research", tags=["research"])

STREAM_TIMEOUT = 600  # 10 min max for a research task


def _user_id(credentials: HTTPAuthorizationCredentials) -> uuid.UUID:
    return uuid.UUID(decode_token(credentials.credentials)["sub"])


class StartResearchRequest(BaseModel):
    query: str = Field(min_length=5, max_length=1000)
    depth: Literal["quick", "standard", "deep"] = "standard"


class ReportOut(BaseModel):
    id: str
    query: str
    depth: str
    status: str
    title: str | None
    report: dict | None
    error: str | None
    sources_count: int
    task_id: str | None
    created_at: str
    completed_at: str | None

    @classmethod
    def from_orm(cls, r) -> "ReportOut":
        return cls(
            id=str(r.id),
            query=r.query,
            depth=r.depth,
            status=r.status,
            title=r.title,
            report=r.report,
            error=r.error,
            sources_count=r.sources_count,
            task_id=r.task_id,
            created_at=r.created_at.isoformat(),
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
        )


@router.post("", response_model=ReportOut, status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    payload: StartResearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    report = await ResearchRepository.create(db, user_id, payload.query, payload.depth)
    await db.commit()
    await db.refresh(report)

    task = deep_research.apply_async(
        args=[str(report.id), payload.query, str(user_id), payload.depth],
        countdown=0,
    )
    await ResearchRepository.set_task_id(db, report, task.id)
    await db.commit()
    return ReportOut.from_orm(report)


@router.get("", response_model=list[ReportOut])
async def list_research(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    reports = await ResearchRepository.list_for_user(db, user_id)
    return [ReportOut.from_orm(r) for r in reports]


@router.get("/{report_id}", response_model=ReportOut)
async def get_report(
    report_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    report = await ResearchRepository.get(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return ReportOut.from_orm(report)


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    report = await ResearchRepository.get(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    await db.delete(report)
    await db.commit()


@router.get("/{report_id}/stream")
async def stream_progress(
    report_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    """SSE endpoint that streams research progress from Redis pub/sub."""
    user_id = _user_id(credentials)
    report = await ResearchRepository.get(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if report.status in ("complete", "error"):
        async def already_done():
            event = {
                "stage": report.status,
                "progress": 100 if report.status == "complete" else 0,
                "message": "Research complete!" if report.status == "complete" else report.error,
                "report_id": str(report_id),
            }
            yield f"data: {json.dumps(event)}\n\n"
        return StreamingResponse(already_done(), media_type="text/event-stream")

    channel = f"{CHANNEL_PREFIX}{report_id}:progress"

    async def event_stream():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        elapsed = 0.0
        try:
            while elapsed < STREAM_TIMEOUT:
                msg = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=1.0)
                if msg and msg["type"] == "message":
                    data = msg["data"]
                    yield f"data: {data}\n\n"
                    try:
                        parsed = json.loads(data)
                        if parsed.get("stage") in ("complete", "error"):
                            break
                    except Exception:
                        pass
                else:
                    yield ": keep-alive\n\n"
                    elapsed += 1.0
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
