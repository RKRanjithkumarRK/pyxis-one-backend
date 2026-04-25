"""Sandbox REST API — E2B container lifecycle, file ops, shell execution."""
from __future__ import annotations
import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.models.project import Project, ProjectMember
from app.services.sandbox import e2b_service as sb
from app.services.storage import project_storage as ps

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


async def _assert_project_access(project_id: str, user: User, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == uuid.UUID(project_id)))
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id == user.id:
        return project
    member_result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == user.id,
        )
    )
    if member_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Access denied")
    return project


# ─── Sandbox lifecycle ────────────────────────────────────

@router.post("/projects/{project_id}/start")
async def start_sandbox(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start (or reconnect to) the sandbox for this project."""
    project = await _assert_project_access(project_id, current_user, db)
    sandbox_id = await sb.get_or_create_sandbox(project_id)
    synced = await ps.sync_from_gcs_to_sandbox(str(current_user.id), project_id, sandbox_id)
    return {"sandbox_id": sandbox_id, "files_synced": synced}


@router.delete("/projects/{project_id}/stop")
async def stop_sandbox(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    await sb.terminate_sandbox(project_id)
    return {"status": "stopped"}


# ─── File operations ─────────────────────────────────────

@router.get("/projects/{project_id}/files")
async def list_files(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    return await ps.list_project_files(str(current_user.id), project_id)


@router.get("/projects/{project_id}/files/{path:path}")
async def get_file(
    project_id: str,
    path: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    content = await ps.read_project_file(str(current_user.id), project_id, path)
    return {"path": path, "content": content.decode("utf-8", errors="replace")}


class WriteFileRequest(BaseModel):
    content: str


@router.put("/projects/{project_id}/files/{path:path}")
async def write_file(
    project_id: str,
    path: str,
    body: WriteFileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    await ps.write_project_file(str(current_user.id), project_id, path, body.content.encode("utf-8"))
    # Also write into running sandbox if one exists
    sandbox_id = sb._registry.get(project_id)
    if sandbox_id:
        await sb.write_file(sandbox_id, f"/workspace/{path}", body.content)
    return {"status": "saved", "path": path}


@router.delete("/projects/{project_id}/files/{path:path}")
async def delete_file(
    project_id: str,
    path: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    await ps.delete_project_file(str(current_user.id), project_id, path)
    return {"status": "deleted", "path": path}


# ─── Shell execution ─────────────────────────────────────

class RunCommandRequest(BaseModel):
    command: str
    workdir: str = "/workspace"


@router.post("/projects/{project_id}/exec")
async def exec_command(
    project_id: str,
    body: RunCommandRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run a command and return stdout/stderr synchronously."""
    await _assert_project_access(project_id, current_user, db)
    sandbox_id = sb._registry.get(project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running — call /start first")
    result = await sb.execute_command(sandbox_id, body.command, body.workdir)
    return result


@router.post("/projects/{project_id}/exec/stream")
async def stream_exec_command(
    project_id: str,
    body: RunCommandRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream command output as SSE."""
    await _assert_project_access(project_id, current_user, db)
    sandbox_id = sb._registry.get(project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running")

    async def event_stream():
        async for line in sb.stream_command(sandbox_id, body.command, body.workdir):
            yield f"data: {json.dumps({'output': line})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Port exposure (preview) ──────────────────────────────

@router.post("/projects/{project_id}/expose/{port}")
async def expose_port(
    project_id: str,
    port: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_project_access(project_id, current_user, db)
    sandbox_id = sb._registry.get(project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running")
    preview_url = await sb.expose_port(sandbox_id, port)
    return {"url": preview_url, "port": port}


# ─── AI shell helper ─────────────────────────────────────

class AIShellRequest(BaseModel):
    prompt: str
    model: str = "groq/llama-3.3-70b-versatile"


@router.post("/projects/{project_id}/ai-shell")
async def ai_shell(
    project_id: str,
    body: AIShellRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """AI generates and executes shell commands inside the sandbox."""
    await _assert_project_access(project_id, current_user, db)
    sandbox_id = sb._registry.get(project_id)
    if not sandbox_id:
        raise HTTPException(status_code=400, detail="Sandbox not running")

    from app.services.llm.router import litellm_complete
    system = (
        "You are a shell assistant inside a Linux container at /workspace. "
        "Respond ONLY with a JSON object: {\"command\": \"<bash command>\"}. "
        "No explanation, just the JSON."
    )
    response = await litellm_complete(
        model=body.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": body.prompt},
        ],
    )
    import json as _json
    try:
        cmd = _json.loads(response)["command"]
    except Exception:
        raise HTTPException(status_code=422, detail="Model did not return valid JSON")

    result = await sb.execute_command(sandbox_id, cmd)
    return {"command": cmd, **result}
