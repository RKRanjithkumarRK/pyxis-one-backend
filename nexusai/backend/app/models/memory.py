from __future__ import annotations
import uuid
from sqlalchemy import String, Integer, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy import DateTime, func
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class UserMemory(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "user_memories"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    last_used_at: Mapped[str | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    use_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="memories")  # noqa: F821

    __table_args__ = (
        Index("idx_mem_user", "user_id", "last_used_at"),
        # ivfflat index created in raw SQL in migration
    )
