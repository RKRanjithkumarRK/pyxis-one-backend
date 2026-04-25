"""Repository for KnowledgeBase, KBFile, KBChunk."""
from __future__ import annotations
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.models.knowledge_base import KnowledgeBase, KBFile, KBChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class KBRepository:
    # ── KnowledgeBase ────────────────────────────────────────────────────────

    @staticmethod
    async def create(
        db: "AsyncSession",
        user_id: uuid.UUID,
        name: str,
        description: str | None = None,
        project_id: uuid.UUID | None = None,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(user_id=user_id, name=name, description=description, project_id=project_id)
        db.add(kb)
        await db.commit()
        await db.refresh(kb)
        return kb

    @staticmethod
    async def get(db: "AsyncSession", kb_id: uuid.UUID, user_id: uuid.UUID) -> KnowledgeBase | None:
        return (
            await db.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user_id)
                .options(selectinload(KnowledgeBase.files))
            )
        ).scalar_one_or_none()

    @staticmethod
    async def list_for_user(db: "AsyncSession", user_id: uuid.UUID) -> list[KnowledgeBase]:
        rows = await db.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.user_id == user_id)
            .order_by(KnowledgeBase.created_at.desc())
            .options(selectinload(KnowledgeBase.files))
        )
        return list(rows.scalars())

    @staticmethod
    async def update(
        db: "AsyncSession",
        kb: KnowledgeBase,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBase:
        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        await db.commit()
        await db.refresh(kb)
        return kb

    @staticmethod
    async def delete(db: "AsyncSession", kb: KnowledgeBase) -> None:
        await db.delete(kb)
        await db.commit()

    # ── KBFile ───────────────────────────────────────────────────────────────

    @staticmethod
    async def create_file(
        db: "AsyncSession",
        kb_id: uuid.UUID,
        filename: str,
        file_type: str,
        file_size: int,
        storage_path: str,
    ) -> KBFile:
        f = KBFile(
            kb_id=kb_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            storage_path=storage_path,
            status="pending",
        )
        db.add(f)
        await db.commit()
        await db.refresh(f)
        return f

    @staticmethod
    async def get_file(db: "AsyncSession", file_id: uuid.UUID, kb_id: uuid.UUID) -> KBFile | None:
        return (
            await db.execute(
                select(KBFile).where(KBFile.id == file_id, KBFile.kb_id == kb_id)
            )
        ).scalar_one_or_none()

    @staticmethod
    async def delete_file(db: "AsyncSession", f: KBFile) -> None:
        await db.delete(f)
        await db.commit()

    # ── KBChunk ──────────────────────────────────────────────────────────────

    @staticmethod
    async def chunk_count(db: "AsyncSession", kb_id: uuid.UUID) -> int:
        result = await db.execute(
            select(func.count()).where(KBChunk.kb_id == kb_id)
        )
        return result.scalar_one()
