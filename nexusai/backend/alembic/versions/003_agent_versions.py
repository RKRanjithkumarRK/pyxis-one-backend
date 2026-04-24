"""Agent versioning table

Revision ID: 003
Revises: 002
Create Date: 2026-04-24

"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("snapshot", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("idx_agent_versions_agent", "agent_versions", ["agent_id", "version"])

    # Add icon column (emoji or URL) to agents — was previously only in YAML
    op.add_column("agents", sa.Column("icon", sa.String(256), nullable=True))


def downgrade() -> None:
    op.drop_table("agent_versions")
    op.drop_column("agents", "icon")
