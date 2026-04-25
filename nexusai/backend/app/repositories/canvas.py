from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.canvas import CanvasDocument, CanvasVersion


class CanvasRepository:

    @staticmethod
    async def create(db: AsyncSession, user_id: uuid.UUID, title: str = "Untitled") -> CanvasDocument:
        doc = CanvasDocument(user_id=user_id, title=title)
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc

    @staticmethod
    async def get(db: AsyncSession, doc_id: uuid.UUID) -> Optional[CanvasDocument]:
        result = await db.execute(select(CanvasDocument).where(CanvasDocument.id == doc_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def list_for_user(
        db: AsyncSession, user_id: uuid.UUID, *, limit: int = 100
    ) -> list[CanvasDocument]:
        result = await db.execute(
            select(CanvasDocument)
            .where(CanvasDocument.user_id == user_id)
            .order_by(CanvasDocument.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(
        db: AsyncSession,
        doc: CanvasDocument,
        *,
        title: str | None = None,
        content: dict | None = None,
        content_text: str | None = None,
        save_version: bool = False,
    ) -> CanvasDocument:
        if save_version and doc.content is not None:
            await CanvasRepository.save_version(db, doc)
            doc.version += 1

        if title is not None:
            doc.title = title
        if content is not None:
            doc.content = content
        if content_text is not None:
            doc.content_text = content_text

        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc

    @staticmethod
    async def delete(db: AsyncSession, doc: CanvasDocument) -> None:
        await db.delete(doc)
        await db.flush()

    @staticmethod
    async def save_version(db: AsyncSession, doc: CanvasDocument) -> CanvasVersion:
        ver = CanvasVersion(
            document_id=doc.id,
            version=doc.version,
            title=doc.title,
            content=doc.content,
            created_at=datetime.now(timezone.utc),
        )
        db.add(ver)
        await db.flush()
        return ver

    @staticmethod
    async def list_versions(db: AsyncSession, doc_id: uuid.UUID) -> list[CanvasVersion]:
        result = await db.execute(
            select(CanvasVersion)
            .where(CanvasVersion.document_id == doc_id)
            .order_by(CanvasVersion.version.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_version(
        db: AsyncSession, doc_id: uuid.UUID, version: int
    ) -> Optional[CanvasVersion]:
        result = await db.execute(
            select(CanvasVersion).where(
                CanvasVersion.document_id == doc_id,
                CanvasVersion.version == version,
            )
        )
        return result.scalar_one_or_none()
