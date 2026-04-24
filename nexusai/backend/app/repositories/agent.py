from __future__ import annotations
import uuid
from typing import Optional
from sqlalchemy import select, func, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent, AgentVersion


class AgentRepository:

    @staticmethod
    async def get_by_id(db: AsyncSession, agent_id: uuid.UUID) -> Optional[Agent]:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_slug(db: AsyncSession, slug: str) -> Optional[Agent]:
        result = await db.execute(select(Agent).where(Agent.slug == slug))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_store(
        db: AsyncSession,
        *,
        category: str | None = None,
        search: str | None = None,
        sort: str = "popular",
        page: int = 1,
        page_size: int = 24,
        user_id: uuid.UUID | None = None,
    ) -> tuple[list[Agent], int]:
        q = select(Agent).where(
            or_(
                Agent.visibility == "public",
                Agent.creator_id == user_id if user_id else False,
            )
        )
        if category:
            q = q.where(Agent.category == category)
        if search:
            pattern = f"%{search.lower()}%"
            q = q.where(
                or_(
                    func.lower(Agent.name).like(pattern),
                    func.lower(Agent.description).like(pattern),
                )
            )
        if sort == "popular":
            q = q.order_by(Agent.usage_count.desc(), Agent.rating.desc().nulls_last())
        elif sort == "newest":
            q = q.order_by(Agent.created_at.desc())
        elif sort == "rating":
            q = q.order_by(Agent.rating.desc().nulls_last(), Agent.rating_count.desc())
        else:
            q = q.order_by(Agent.name.asc())

        count_q = select(func.count()).select_from(q.subquery())
        total_result = await db.execute(count_q)
        total = total_result.scalar_one()

        q = q.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(q)
        return list(result.scalars().all()), total

    @staticmethod
    async def list_user_agents(db: AsyncSession, user_id: uuid.UUID) -> list[Agent]:
        result = await db.execute(
            select(Agent)
            .where(Agent.creator_id == user_id)
            .order_by(Agent.updated_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def create(db: AsyncSession, **kwargs) -> Agent:
        agent = Agent(**kwargs)
        db.add(agent)
        await db.flush()
        await db.refresh(agent)
        return agent

    @staticmethod
    async def update(db: AsyncSession, agent: Agent, **kwargs) -> Agent:
        for k, v in kwargs.items():
            setattr(agent, k, v)
        db.add(agent)
        await db.flush()
        await db.refresh(agent)
        return agent

    @staticmethod
    async def delete(db: AsyncSession, agent: Agent) -> None:
        await db.delete(agent)
        await db.flush()

    @staticmethod
    async def increment_usage(db: AsyncSession, agent_id: uuid.UUID) -> None:
        await db.execute(
            update(Agent).where(Agent.id == agent_id).values(usage_count=Agent.usage_count + 1)
        )

    @staticmethod
    async def upsert_builtin(db: AsyncSession, data: dict) -> Agent:
        existing = await AgentRepository.get_by_slug(db, data["slug"])
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
            db.add(existing)
            await db.flush()
            return existing
        agent = Agent(**data, is_builtin=True, visibility="public")
        db.add(agent)
        await db.flush()
        return agent

    @staticmethod
    async def save_version(db: AsyncSession, agent: Agent) -> AgentVersion:
        from datetime import datetime, timezone
        snapshot = {
            "name": agent.name,
            "description": agent.description,
            "icon": agent.icon,
            "category": agent.category,
            "instructions": agent.instructions,
            "starters": agent.starters,
            "capabilities": agent.capabilities,
            "default_model": agent.default_model,
            "visibility": agent.visibility,
        }
        av = AgentVersion(
            agent_id=agent.id,
            version=agent.version,
            snapshot=snapshot,
            created_at=datetime.now(timezone.utc),
        )
        db.add(av)
        await db.flush()
        await db.refresh(av)
        return av

    @staticmethod
    async def list_versions(db: AsyncSession, agent_id: uuid.UUID) -> list[AgentVersion]:
        result = await db.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.version.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_version(
        db: AsyncSession, agent_id: uuid.UUID, version: int
    ) -> Optional[AgentVersion]:
        result = await db.execute(
            select(AgentVersion).where(
                AgentVersion.agent_id == agent_id,
                AgentVersion.version == version,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_rating(db: AsyncSession, agent: Agent, new_rating: float) -> Agent:
        total = (agent.rating or 0.0) * agent.rating_count + new_rating
        agent.rating_count += 1
        agent.rating = round(total / agent.rating_count, 2)
        db.add(agent)
        await db.flush()
        await db.refresh(agent)
        return agent
