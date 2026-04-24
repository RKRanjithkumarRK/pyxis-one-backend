from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, ForeignKey, DateTime, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class ResearchReport(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "research_reports"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[str] = mapped_column(String(16), nullable=False, default="standard")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sources_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('pending','running','complete','error')", name="ck_research_status"),
        CheckConstraint("depth IN ('quick','standard','deep')", name="ck_research_depth"),
    )
