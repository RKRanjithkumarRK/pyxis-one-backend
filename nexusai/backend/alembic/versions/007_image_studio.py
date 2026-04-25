"""Image Studio — image_requests table

Revision ID: 007
Revises: 006
Create Date: 2026-04-25
"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("negative_prompt", sa.Text, nullable=True),
        sa.Column("model", sa.String(64), nullable=False, server_default="flux-schnell"),
        sa.Column("width", sa.Integer, nullable=False, server_default="1024"),
        sa.Column("height", sa.Integer, nullable=False, server_default="1024"),
        sa.Column("num_images", sa.Integer, nullable=False, server_default="4"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("result_urls", postgresql.JSONB, nullable=True),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_image_requests_user", "image_requests", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("image_requests")
