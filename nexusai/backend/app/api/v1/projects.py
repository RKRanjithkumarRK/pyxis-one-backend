from __future__ import annotations
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import require_bearer, decode_token
from app.repositories.project import ProjectRepository, VALID_ROLES, ROLE_OWNER, ROLE_EDITOR
from app.models.user import User

router = APIRouter(prefix="/projects", tags=["projects"])


def _user_id(creds: HTTPAuthorizationCredentials) -> uuid.UUID:
    return uuid.UUID(decode_token(creds.credentials)["sub"])


# ─── Serialisation helpers ────────────────────────────────

def _project_dict(p, role: str | None = None) -> dict:
    return {
        "id": str(p.id),
        "owner_id": str(p.owner_id),
        "name": p.name,
        "description": p.description,
        "system_prompt": p.system_prompt,
        "icon_url": p.icon_url,
        "role": role,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def _member_dict(m, email: str | None = None) -> dict:
    return {
        "user_id": str(m.user_id),
        "project_id": str(m.project_id),
        "role": m.role,
        "email": email,
    }


# ─── Pydantic schemas ─────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    system_prompt: str | None = Field(default=None, max_length=8000)
    icon_url: str | None = Field(default=None, max_length=2048)


class UpdateProjectRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2000)
    system_prompt: str | None = Field(default=None, max_length=8000)
    icon_url: str | None = Field(default=None, max_length=2048)


class InviteMemberRequest(BaseModel):
    email: str = Field(min_length=3)
    role: str = Field(default="viewer")


class UpdateMemberRoleRequest(BaseModel):
    role: str


# ─── Endpoints ───────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: CreateProjectRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.create(
        db,
        owner_id=user_id,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        icon_url=payload.icon_url,
    )
    await db.commit()
    await db.refresh(project)
    return _project_dict(project, role=ROLE_OWNER)


@router.get("")
async def list_projects(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    projects = await ProjectRepository.list_for_user(db, user_id)
    result = []
    for p in projects:
        member = await ProjectRepository.get_member(db, p.id, user_id)
        result.append(_project_dict(p, role=member.role if member else None))
    return result


@router.get("/{project_id}")
async def get_project(
    project_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    member = await ProjectRepository.get_member(db, project_id, user_id)
    if not member:
        raise HTTPException(status_code=403, detail="Access denied")
    return _project_dict(project, role=member.role)


@router.patch("/{project_id}")
async def update_project(
    project_id: uuid.UUID,
    payload: UpdateProjectRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    member = await ProjectRepository.get_member(db, project_id, user_id)
    if not member or member.role not in (ROLE_OWNER, ROLE_EDITOR):
        raise HTTPException(status_code=403, detail="Requires editor role")

    project = await ProjectRepository.update(
        db, project,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        icon_url=payload.icon_url,
    )
    await db.commit()
    await db.refresh(project)
    return _project_dict(project, role=member.role)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.owner_id != user_id:
        raise HTTPException(status_code=403, detail="Only the owner can delete a project")
    await ProjectRepository.delete(db, project)
    await db.commit()


# ─── Members ─────────────────────────────────────────────

@router.get("/{project_id}/members")
async def list_members(
    project_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not await ProjectRepository.is_member(db, project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    members = await ProjectRepository.list_members(db, project_id)
    result = []
    for m in members:
        user_result = await db.execute(select(User).where(User.id == m.user_id))
        user = user_result.scalar_one_or_none()
        result.append(_member_dict(m, email=user.email if user else None))
    return result


@router.post("/{project_id}/members")
async def invite_member(
    project_id: uuid.UUID,
    payload: InviteMemberRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(VALID_ROLES)}")

    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    member = await ProjectRepository.get_member(db, project_id, user_id)
    if not member or member.role not in (ROLE_OWNER, ROLE_EDITOR):
        raise HTTPException(status_code=403, detail="Requires editor role to invite")

    # Look up invitee by email
    result = await db.execute(select(User).where(User.email == payload.email))
    invitee = result.scalar_one_or_none()
    if not invitee:
        raise HTTPException(status_code=404, detail="User not found — they must have a NexusAI account")

    new_member = await ProjectRepository.add_member(db, project_id, invitee.id, payload.role)
    await db.commit()
    return _member_dict(new_member, email=invitee.email)


@router.patch("/{project_id}/members/{member_user_id}")
async def update_member_role(
    project_id: uuid.UUID,
    member_user_id: uuid.UUID,
    payload: UpdateMemberRoleRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {sorted(VALID_ROLES)}")
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    caller = await ProjectRepository.get_member(db, project_id, user_id)
    if not caller or caller.role != ROLE_OWNER:
        raise HTTPException(status_code=403, detail="Only the owner can change roles")
    if member_user_id == user_id:
        raise HTTPException(status_code=422, detail="Cannot change your own role")

    target = await ProjectRepository.get_member(db, project_id, member_user_id)
    if not target:
        raise HTTPException(status_code=404, detail="Member not found")
    target.role = payload.role
    db.add(target)
    await db.commit()
    return _member_dict(target)


@router.delete("/{project_id}/members/{member_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    project_id: uuid.UUID,
    member_user_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    caller = await ProjectRepository.get_member(db, project_id, user_id)
    if not caller or caller.role not in (ROLE_OWNER, ROLE_EDITOR):
        raise HTTPException(status_code=403, detail="Requires editor role")
    if member_user_id == project.owner_id:
        raise HTTPException(status_code=422, detail="Cannot remove the project owner")

    await ProjectRepository.remove_member(db, project_id, member_user_id)
    await db.commit()


# ─── Conversations scoped to project ─────────────────────

@router.get("/{project_id}/conversations")
async def list_project_conversations(
    project_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _user_id(credentials)
    project = await ProjectRepository.get(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not await ProjectRepository.is_member(db, project_id, user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    from app.models.conversation import Conversation
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.project_id == project_id,
            Conversation.user_id == user_id,
        )
        .order_by(Conversation.updated_at.desc())
        .limit(100)
    )
    convs = result.scalars().all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "model_id": c.model_id,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]
