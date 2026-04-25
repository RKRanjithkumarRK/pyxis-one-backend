"""Computer Use API — Playwright-in-E2B screenshot→model→action loop."""
from __future__ import annotations
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.project import Project
from app.services.sandbox import e2b_service as sb
from app.services.computer_use.browser import run_computer_use_loop, execute_approved_action

router = APIRouter(prefix="/computer-use", tags=["computer-use"])


class StartComputerUseRequest(BaseModel):
    project_id: str
    task: str = Field(min_length=1, max_length=2000)
    model: str = "claude-opus-4-20250514"
    max_steps: int = Field(default=10, ge=1, le=30)
    approval_required: bool = True


class ApproveActionRequest(BaseModel):
    project_id: str
    action: dict


@router.post("/start")
async def start_computer_use(
    body: StartComputerUseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a Computer Use session — returns SSE stream of steps/screenshots/actions."""
    sandbox_id = sb._registry.get(body.project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running — start it first")

    async def event_stream():
        async for event in run_computer_use_loop(
            sandbox_id=sandbox_id,
            task=body.task,
            model=body.model,
            max_steps=body.max_steps,
            approval_required=body.approval_required,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/approve")
async def approve_action(
    body: ApproveActionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute an action that was approved by the user."""
    sandbox_id = sb._registry.get(body.project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running")

    result = await execute_approved_action(sandbox_id, body.action)
    return result
