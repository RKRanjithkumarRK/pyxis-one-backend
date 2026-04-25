"""Workflows REST API — CRUD, trigger, run history."""
from __future__ import annotations
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.workflow import Workflow, WorkflowRun

router = APIRouter(prefix="/workflows", tags=["workflows"])


def _wf_dict(wf: Workflow) -> dict:
    return {
        "id": str(wf.id),
        "name": wf.name,
        "description": wf.description,
        "trigger_type": wf.trigger_type,
        "trigger_config": wf.trigger_config,
        "nodes": wf.nodes,
        "edges": wf.edges,
        "is_active": wf.is_active,
        "last_run_at": wf.last_run_at,
        "created_at": wf.created_at.isoformat(),
        "updated_at": wf.updated_at.isoformat(),
    }


def _run_dict(r: WorkflowRun) -> dict:
    return {
        "id": str(r.id),
        "workflow_id": str(r.workflow_id),
        "status": r.status,
        "trigger": r.trigger,
        "inputs": r.inputs,
        "outputs": r.outputs,
        "error": r.error,
        "node_results": r.node_results,
        "duration_ms": r.duration_ms,
        "created_at": r.created_at.isoformat(),
    }


async def _get_workflow(workflow_id: str, user: User, db: AsyncSession) -> Workflow:
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == uuid.UUID(workflow_id),
            Workflow.owner_id == user.id,
        )
    )
    wf = result.scalar_one_or_none()
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


# ─── Schemas ──────────────────────────────────────────────

class CreateWorkflowRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    trigger_type: str = Field(default="manual", pattern="^(manual|schedule|webhook)$")
    trigger_config: dict = Field(default_factory=dict)
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)


class UpdateWorkflowRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None
    trigger_type: str | None = Field(default=None, pattern="^(manual|schedule|webhook)$")
    trigger_config: dict | None = None
    nodes: list[dict] | None = None
    edges: list[dict] | None = None
    is_active: bool | None = None


class TriggerWorkflowRequest(BaseModel):
    inputs: dict = Field(default_factory=dict)


# ─── Endpoints ────────────────────────────────────────────

@router.get("")
async def list_workflows(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Workflow)
        .where(Workflow.owner_id == current_user.id)
        .order_by(Workflow.updated_at.desc())
    )
    return [_wf_dict(wf) for wf in result.scalars().all()]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_workflow(
    payload: CreateWorkflowRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = Workflow(
        owner_id=current_user.id,
        name=payload.name,
        description=payload.description,
        trigger_type=payload.trigger_type,
        trigger_config=payload.trigger_config,
        nodes=payload.nodes,
        edges=payload.edges,
    )
    db.add(wf)
    await db.commit()
    await db.refresh(wf)
    return _wf_dict(wf)


@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    return _wf_dict(wf)


@router.patch("/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    payload: UpdateWorkflowRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(wf, field, value)
    await db.commit()
    await db.refresh(wf)
    return _wf_dict(wf)


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    await db.delete(wf)
    await db.commit()


@router.post("/{workflow_id}/trigger")
async def trigger_workflow(
    workflow_id: str,
    payload: TriggerWorkflowRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    from app.services.workflows.tasks import run_workflow_task
    task = run_workflow_task.delay(str(wf.id), payload.inputs)
    return {"task_id": task.id, "workflow_id": str(wf.id), "status": "queued"}


@router.get("/{workflow_id}/runs")
async def list_runs(
    workflow_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    result = await db.execute(
        select(WorkflowRun)
        .where(WorkflowRun.workflow_id == wf.id)
        .order_by(WorkflowRun.created_at.desc())
        .limit(limit)
    )
    return [_run_dict(r) for r in result.scalars().all()]


@router.get("/{workflow_id}/runs/{run_id}")
async def get_run(
    workflow_id: str,
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wf = await _get_workflow(workflow_id, current_user, db)
    result = await db.execute(
        select(WorkflowRun).where(
            WorkflowRun.id == uuid.UUID(run_id),
            WorkflowRun.workflow_id == wf.id,
        )
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_dict(run)


@router.post("/webhook/{workflow_id}")
async def webhook_trigger(
    workflow_id: str,
    body: dict = None,
    db: AsyncSession = Depends(get_db),
):
    """Webhook endpoint — no auth, anyone with the URL can trigger."""
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == uuid.UUID(workflow_id),
            Workflow.trigger_type == "webhook",
            Workflow.is_active == True,
        )
    )
    wf = result.scalar_one_or_none()
    if wf is None:
        raise HTTPException(status_code=404, detail="Workflow not found or not webhook-enabled")
    from app.services.workflows.tasks import run_workflow_task
    task = run_workflow_task.delay(str(wf.id), body or {})
    return {"task_id": task.id, "status": "queued"}
