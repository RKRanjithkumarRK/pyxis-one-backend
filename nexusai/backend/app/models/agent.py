from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, Float, ForeignKey, Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
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
    icon: Mapped[str | None] = mapped_column(String(256), nullable=True)
    icon_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    starters: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    capabilities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    default_model: Mapped[str] = mapped_column(String(128), nullable=False, default="claude-sonnet-4")
    visibility: Mapped[str] = mapped_column(String(16), nullable=False, default="private")
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    versions: Mapped[list["AgentVersion"]] = relationship(
        "AgentVersion", back_populates="agent", cascade="all, delete-orphan", lazy="select"
    )

    __table_args__ = (
        Index("idx_agents_category_visibility", "category", "visibility"),
        Index("idx_agents_usage", "visibility", "usage_count"),
    )


class AgentVersion(Base, UUIDPrimaryKey):
    __tablename__ = "agent_versions"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    agent: Mapped["Agent"] = relationship("Agent", back_populates="versions")
