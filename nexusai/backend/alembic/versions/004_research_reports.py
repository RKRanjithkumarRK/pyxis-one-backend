"""Research reports table

Revision ID: 004
Revises: 003
Create Date: 2026-04-24

"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_reports",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("depth", sa.String(16), nullable=False, server_default="standard"),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column("report", postgresql.JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("task_id", sa.String(128), nullable=True),
        sa.Column("sources_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending','running','complete','error')",
            name="ck_research_status",
        ),
        sa.CheckConstraint(
            "depth IN ('quick','standard','deep')",
            name="ck_research_depth",
        ),
    )
    op.create_index("idx_research_user", "research_reports", ["user_id", "created_at"])
    op.create_index("idx_research_status", "research_reports", ["status", "created_at"])
    op.execute("""
        CREATE TRIGGER trg_research_reports_updated_at
        BEFORE UPDATE ON research_reports
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.drop_table("research_reports")
