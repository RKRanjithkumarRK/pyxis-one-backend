from __future__ import annotations
import uuid
from sqlalchemy import String, Text, ForeignKey, Boolean, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class Workflow(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "workflows"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    trigger_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    nodes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    edges: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_run_at: Mapped[str | None] = mapped_column(String(64), nullable=True)

    runs: Mapped[list["WorkflowRun"]] = relationship(
        "WorkflowRun", back_populates="workflow", cascade="all, delete-orphan"
    )


class WorkflowRun(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "workflow_runs"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    trigger: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    inputs: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    outputs: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    node_results: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="runs")
