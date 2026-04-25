from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import UserMemory


class MemoryRepository:

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: uuid.UUID,
        fact: str,
        embedding: list[float] | None = None,
        source_message_id: uuid.UUID | None = None,
    ) -> UserMemory:
        mem = UserMemory(
            user_id=user_id,
            fact=fact,
            embedding=embedding,
            source_message_id=source_message_id,
        )
        db.add(mem)
        await db.flush()
        await db.refresh(mem)
        return mem

    @staticmethod
    async def get(
        db: AsyncSession,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[UserMemory]:
        result = await db.execute(
            select(UserMemory).where(
                UserMemory.id == memory_id,
                UserMemory.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_user(
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        limit: int = 200,
    ) -> list[UserMemory]:
        result = await db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def search_similar(
        db: AsyncSession,
        user_id: uuid.UUID,
        query_embedding: list[float],
        *,
        limit: int = 8,
        max_distance: float = 0.5,
    ) -> list[UserMemory]:
        """Cosine-distance nearest-neighbor search via pgvector."""
        result = await db.execute(
            select(UserMemory)
            .where(
                UserMemory.user_id == user_id,
                UserMemory.embedding.isnot(None),
            )
            .order_by(UserMemory.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        rows = list(result.scalars().all())
        return rows

    @staticmethod
    async def deduplicate_check(
        db: AsyncSession,
        user_id: uuid.UUID,
        embedding: list[float],
        *,
        threshold: float = 0.12,  # cosine distance < 0.12 → duplicate
    ) -> bool:
        """Return True if a very similar memory already exists."""
        result = await db.execute(
            select(UserMemory.id)
            .where(
                UserMemory.user_id == user_id,
                UserMemory.embedding.isnot(None),
            )
            .order_by(UserMemory.embedding.cosine_distance(embedding))
            .limit(1)
        )
        row = result.first()
        if row is None:
            return False
        # Check distance of closest neighbour
        dist_result = await db.execute(
            select(UserMemory.embedding.cosine_distance(embedding).label("d"))
            .where(
                UserMemory.user_id == user_id,
                UserMemory.embedding.isnot(None),
            )
            .order_by(UserMemory.embedding.cosine_distance(embedding))
            .limit(1)
        )
        dist_row = dist_result.first()
        if dist_row is None:
            return False
        return float(dist_row.d) < threshold

    @staticmethod
    async def delete(db: AsyncSession, memory: UserMemory) -> None:
        await db.delete(memory)
        await db.flush()

    @staticmethod
    async def delete_all_for_user(db: AsyncSession, user_id: uuid.UUID) -> int:
        result = await db.execute(
            delete(UserMemory)
            .where(UserMemory.user_id == user_id)
            .returning(UserMemory.id)
        )
        rows = result.fetchall()
        await db.flush()
        return len(rows)

    @staticmethod
    async def mark_used(db: AsyncSession, memory: UserMemory) -> None:
        memory.last_used_at = datetime.now(timezone.utc)
        memory.use_count = (memory.use_count or 0) + 1
        db.add(memory)
        await db.flush()

    @staticmethod
    async def count(db: AsyncSession, user_id: uuid.UUID) -> int:
        result = await db.execute(
            select(func.count()).where(UserMemory.user_id == user_id)
        )
        return result.scalar_one() or 0
