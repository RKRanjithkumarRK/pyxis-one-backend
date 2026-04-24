from __future__ import annotations
import uuid
from typing import Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.conversation import Conversation


class ConversationRepository:

    @staticmethod
    async def get(db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Conversation]:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_user(
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        include_archived: bool = False,
        limit: int = 200,
    ) -> list[Conversation]:
        q = select(Conversation).where(Conversation.user_id == user_id)
        if not include_archived:
            q = q.where(Conversation.archived_at.is_(None))
        q = q.order_by(
            Conversation.pinned_at.desc().nulls_last(),
            Conversation.updated_at.desc(),
        ).limit(limit)
        result = await db.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        title: str = "New conversation",
        model_id: str = "claude-sonnet-4",
        project_id: uuid.UUID | None = None,
        agent_id: uuid.UUID | None = None,
    ) -> Conversation:
        branch_id = uuid.uuid4()
        conv = Conversation(
            user_id=user_id,
            title=title,
            model_id=model_id,
            active_branch_id=branch_id,
            project_id=project_id,
            agent_id=agent_id,
        )
        db.add(conv)
        await db.flush()
        await db.refresh(conv)
        return conv

    @staticmethod
    async def update(
        db: AsyncSession,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        **kwargs,
    ) -> Optional[Conversation]:
        await db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .values(**kwargs)
        )
        return await ConversationRepository.get(db, conversation_id, user_id)

    @staticmethod
    async def delete(db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        result = await db.execute(
            delete(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return (result.rowcount or 0) > 0

    @staticmethod
    async def search(
        db: AsyncSession,
        user_id: uuid.UUID,
        query: str,
        limit: int = 20,
    ) -> list[Conversation]:
        result = await db.execute(
            select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.title.ilike(f"%{query}%"),
                Conversation.archived_at.is_(None),
            ).order_by(Conversation.updated_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
