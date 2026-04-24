from __future__ import annotations
import re
import uuid

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent
from app.repositories.agent import AgentRepository
from app.schemas.agent import AgentCreate, AgentUpdate


def _auto_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:128]


class AgentService:

    @staticmethod
    async def create(db: AsyncSession, user_id: uuid.UUID, payload: AgentCreate) -> Agent:
        slug = payload.slug or _auto_slug(payload.name)
        suffix = 0
        base = slug
        while await AgentRepository.get_by_slug(db, slug):
            suffix += 1
            slug = f"{base}-{suffix}"

        capabilities = payload.capabilities.model_dump() if payload.capabilities else None
        agent = await AgentRepository.create(
            db,
            creator_id=user_id,
            slug=slug,
            name=payload.name,
            description=payload.description,
            icon=payload.icon,
            category=payload.category,
            instructions=payload.instructions,
            starters=payload.starters,
            capabilities=capabilities,
            default_model=payload.default_model,
            visibility=payload.visibility,
            version=1,
            is_builtin=False,
        )
        await db.commit()
        await db.refresh(agent)
        return agent

    @staticmethod
    async def update(
        db: AsyncSession, agent_id: uuid.UUID, user_id: uuid.UUID, payload: AgentUpdate
    ) -> Agent:
        agent = await AgentRepository.get_by_id(db, agent_id)
        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        if agent.is_builtin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Built-in agents cannot be modified")
        if agent.creator_id != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not the owner")

        await AgentRepository.save_version(db, agent)

        updates: dict = {}
        if payload.name is not None:
            updates["name"] = payload.name
        if payload.description is not None:
            updates["description"] = payload.description
        if payload.icon is not None:
            updates["icon"] = payload.icon
        if payload.category is not None:
            updates["category"] = payload.category
        if payload.instructions is not None:
            updates["instructions"] = payload.instructions
        if payload.starters is not None:
            updates["starters"] = payload.starters
        if payload.capabilities is not None:
            updates["capabilities"] = payload.capabilities.model_dump()
        if payload.default_model is not None:
            updates["default_model"] = payload.default_model
        if payload.visibility is not None:
            updates["visibility"] = payload.visibility

        updates["version"] = agent.version + 1
        agent = await AgentRepository.update(db, agent, **updates)
        await db.commit()
        await db.refresh(agent)
        return agent

    @staticmethod
    async def delete(db: AsyncSession, agent_id: uuid.UUID, user_id: uuid.UUID) -> None:
        agent = await AgentRepository.get_by_id(db, agent_id)
        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        if agent.is_builtin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Built-in agents cannot be deleted")
        if agent.creator_id != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not the owner")
        await AgentRepository.delete(db, agent)
        await db.commit()

    @staticmethod
    async def publish(
        db: AsyncSession, agent_id: uuid.UUID, user_id: uuid.UUID, *, public: bool
    ) -> Agent:
        agent = await AgentRepository.get_by_id(db, agent_id)
        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        if agent.is_builtin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Built-in agents cannot be modified")
        if agent.creator_id != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not the owner")
        agent = await AgentRepository.update(
            db, agent, visibility="public" if public else "private"
        )
        await db.commit()
        await db.refresh(agent)
        return agent

    @staticmethod
    async def restore_version(
        db: AsyncSession, agent_id: uuid.UUID, user_id: uuid.UUID, version: int
    ) -> Agent:
        agent = await AgentRepository.get_by_id(db, agent_id)
        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
        if agent.is_builtin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Built-in agents cannot be modified")
        if agent.creator_id != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not the owner")

        av = await AgentRepository.get_version(db, agent_id, version)
        if not av:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

        await AgentRepository.save_version(db, agent)
        snap = av.snapshot
        agent = await AgentRepository.update(
            db,
            agent,
            name=snap.get("name", agent.name),
            description=snap.get("description"),
            icon=snap.get("icon"),
            category=snap.get("category", agent.category),
            instructions=snap.get("instructions"),
            starters=snap.get("starters"),
            capabilities=snap.get("capabilities"),
            default_model=snap.get("default_model", agent.default_model),
            version=agent.version + 1,
        )
        await db.commit()
        await db.refresh(agent)
        return agent
