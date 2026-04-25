from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class CanvasDocument(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "canvas_documents"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="Untitled")
    content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    versions: Mapped[list["CanvasVersion"]] = relationship(
        "CanvasVersion", back_populates="document", cascade="all, delete-orphan", lazy="select"
    )


class CanvasVersion(Base, UUIDPrimaryKey):
    __tablename__ = "canvas_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("canvas_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    document: Mapped["CanvasDocument"] = relationship("CanvasDocument", back_populates="versions")
