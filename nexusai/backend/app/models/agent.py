from __future__ import annotations
import uuid
from sqlalchemy import String, Text, Boolean, Integer, Float, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class Agent(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "agents"

    creator_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    icon_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    starters: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    capabilities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    default_model: Mapped[str] = mapped_column(String(128), nullable=False, default="claude-sonnet-4")
    visibility: Mapped[str] = mapped_column(String(16), nullable=False, default="private")
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Stats
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("idx_agents_category_visibility", "category", "visibility"),
        Index("idx_agents_usage", "visibility", "usage_count"),
    )
