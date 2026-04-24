from __future__ import annotations
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import require_bearer, decode_token, optional_bearer
from app.repositories.agent import AgentRepository
from app.schemas.agent import (
    AgentCreate,
    AgentListResponse,
    AgentOut,
    AgentUpdate,
    AgentVersionOut,
    RateAgentRequest,
)
from app.services.agents.service import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])


def _current_user(credentials: HTTPAuthorizationCredentials) -> uuid.UUID:
    payload = decode_token(credentials.credentials)
    return uuid.UUID(payload["sub"])


@router.get("", response_model=AgentListResponse)
async def list_agents(
    category: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
    sort: str = Query(default="popular", pattern="^(popular|newest|rating|name)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id: uuid.UUID | None = None
    if credentials:
        try:
            user_id = _current_user(credentials)
        except Exception:
            pass

    agents, total = await AgentRepository.list_store(
        db,
        category=category,
        search=search,
        sort=sort,
        page=page,
        page_size=page_size,
        user_id=user_id,
    )
    return AgentListResponse(
        agents=[AgentOut.model_validate(a) for a in agents],
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/mine", response_model=list[AgentOut])
async def list_my_agents(
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agents = await AgentRepository.list_user_agents(db, user_id)
    return [AgentOut.model_validate(a) for a in agents]


@router.post("", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
async def create_agent(
    payload: AgentCreate,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agent = await AgentService.create(db, user_id, payload)
    return AgentOut.model_validate(agent)


@router.get("/{agent_ref}", response_model=AgentOut)
async def get_agent(
    agent_ref: str,
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    db: AsyncSession = Depends(get_db),
):
    try:
        agent_id = uuid.UUID(agent_ref)
        agent = await AgentRepository.get_by_id(db, agent_id)
    except ValueError:
        agent = await AgentRepository.get_by_slug(db, agent_ref)

    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    if agent.visibility != "public":
        if not credentials:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        user_id = _current_user(credentials)
        if agent.creator_id != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return AgentOut.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentOut)
async def update_agent(
    agent_id: uuid.UUID,
    payload: AgentUpdate,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agent = await AgentService.update(db, agent_id, user_id, payload)
    return AgentOut.model_validate(agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    await AgentService.delete(db, agent_id, user_id)


@router.post("/{agent_id}/publish", response_model=AgentOut)
async def publish_agent(
    agent_id: uuid.UUID,
    public: bool = Query(default=True),
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agent = await AgentService.publish(db, agent_id, user_id, public=public)
    return AgentOut.model_validate(agent)


@router.get("/{agent_id}/versions", response_model=list[AgentVersionOut])
async def get_agent_versions(
    agent_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agent = await AgentRepository.get_by_id(db, agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.creator_id != user_id and not agent.is_builtin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    versions = await AgentRepository.list_versions(db, agent_id)
    return [AgentVersionOut.model_validate(v) for v in versions]


@router.post("/{agent_id}/restore/{version}", response_model=AgentOut)
async def restore_agent_version(
    agent_id: uuid.UUID,
    version: int,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    user_id = _current_user(credentials)
    agent = await AgentService.restore_version(db, agent_id, user_id, version)
    return AgentOut.model_validate(agent)


@router.post("/{agent_id}/rate", response_model=AgentOut)
async def rate_agent(
    agent_id: uuid.UUID,
    payload: RateAgentRequest,
    credentials: HTTPAuthorizationCredentials = Depends(require_bearer),
    db: AsyncSession = Depends(get_db),
):
    agent = await AgentRepository.get_by_id(db, agent_id)
    if not agent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.visibility != "public":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Can only rate public agents")
    agent = await AgentRepository.update_rating(db, agent, payload.rating)
    await db.commit()
    return AgentOut.model_validate(agent)
