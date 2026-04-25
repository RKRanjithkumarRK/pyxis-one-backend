from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class KnowledgeBase(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "knowledge_bases"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    files: Mapped[list["KBFile"]] = relationship(
        "KBFile", back_populates="kb", cascade="all, delete-orphan", lazy="select"
    )


class KBFile(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "kb_files"

    kb_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)

    kb: Mapped["KnowledgeBase"] = relationship("KnowledgeBase", back_populates="files")
    chunks: Mapped[list["KBChunk"]] = relationship(
        "KBChunk", back_populates="file", cascade="all, delete-orphan", lazy="select"
    )


class KBChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kb_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("kb_files.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    file: Mapped["KBFile"] = relationship("KBFile", back_populates="chunks")
