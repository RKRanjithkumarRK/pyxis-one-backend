"""Canvas documents and version history

Revision ID: 005
Revises: 004
Create Date: 2026-04-24

"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "canvas_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False, server_default="Untitled"),
        sa.Column("content", postgresql.JSONB, nullable=True),
        sa.Column("content_text", sa.Text, nullable=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_public", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_canvas_user", "canvas_documents", ["user_id", "updated_at"])
    op.execute("CREATE INDEX idx_canvas_fts ON canvas_documents USING GIN (to_tsvector('english', COALESCE(content_text, '')))")

    op.create_table(
        "canvas_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("canvas_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column("content", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_canvas_versions_doc", "canvas_versions", ["document_id", "version"])

    op.execute("""
        CREATE TRIGGER trg_canvas_documents_updated_at
        BEFORE UPDATE ON canvas_documents
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.drop_table("canvas_versions")
    op.drop_table("canvas_documents")
