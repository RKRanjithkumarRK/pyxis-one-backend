"""Initial schema: users, conversations, messages, memories, projects, agents

Revision ID: 001
Revises:
Create Date: 2026-04-24

"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ─── Enums ───────────────────────────────────────────────
    auth_provider = postgresql.ENUM(
        "email", "google", "github", "apple", "microsoft", "magic_link", "guest",
        name="auth_provider",
    )
    auth_provider.create(op.get_bind(), checkfirst=True)

    subscription_plan = postgresql.ENUM(
        "free", "plus", "team", "enterprise",
        name="subscription_plan",
    )
    subscription_plan.create(op.get_bind(), checkfirst=True)

    # ─── users ───────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.String(320), nullable=True, unique=True),
        sa.Column("name", sa.String(256), nullable=True),
        sa.Column("avatar_url", sa.String(2048), nullable=True),
        sa.Column("hashed_password", sa.String(256), nullable=True),
        sa.Column("provider", sa.Enum("email","google","github","apple","microsoft","magic_link","guest", name="auth_provider"), nullable=False, server_default="email"),
        sa.Column("provider_id", sa.String(256), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("plan", sa.Enum("free","plus","team","enterprise", name="subscription_plan"), nullable=False, server_default="free"),
        sa.Column("stripe_customer_id", sa.String(128), nullable=True),
        sa.Column("custom_instructions", sa.String(8000), nullable=True),
        sa.Column("memory_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_users_email_active", "users", ["email", "is_active"])
    op.create_index("idx_users_provider", "users", ["provider", "provider_id"])

    # ─── projects ────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("icon_url", sa.String(2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_projects_owner", "projects", ["owner_id", "created_at"])

    op.create_table(
        "project_members",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(16), nullable=False, server_default="viewer"),
        sa.CheckConstraint("role IN ('owner','editor','viewer')", name="ck_project_member_role"),
    )

    # ─── agents ──────────────────────────────────────────────
    op.create_table(
        "agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("creator_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("slug", sa.String(128), nullable=False, unique=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("icon_url", sa.String(2048), nullable=True),
        sa.Column("category", sa.String(64), nullable=False, server_default="general"),
        sa.Column("instructions", sa.Text, nullable=True),
        sa.Column("starters", postgresql.JSONB, nullable=True),
        sa.Column("capabilities", postgresql.JSONB, nullable=True),
        sa.Column("default_model", sa.String(128), nullable=False, server_default="claude-sonnet-4"),
        sa.Column("visibility", sa.String(16), nullable=False, server_default="private"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_builtin", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("usage_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("rating", sa.Float, nullable=True),
        sa.Column("rating_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_agents_slug", "agents", ["slug"], unique=True)
    op.create_index("idx_agents_category_visibility", "agents", ["category", "visibility"])
    op.create_index("idx_agents_usage", "agents", ["visibility", "usage_count"])

    # ─── conversations ────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False, server_default="New conversation"),
        sa.Column("model_id", sa.String(128), nullable=False, server_default="claude-sonnet-4"),
        sa.Column("active_branch_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="SET NULL"), nullable=True),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_shared", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("share_id", sa.String(32), nullable=True, unique=True),
        sa.Column("memory_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("web_search_enabled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("meta", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_conv_user_created", "conversations", ["user_id", "created_at"])
    op.create_index("idx_conv_pinned", "conversations", ["user_id", "pinned_at"])
    op.create_index("idx_conv_project", "conversations", ["project_id", "created_at"])

    # ─── messages ─────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("branch_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("parent_branch_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("sequence", sa.Integer, nullable=False, server_default="0"),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("model_id", sa.String(128), nullable=True),
        sa.Column("usage", postgresql.JSONB, nullable=True),
        sa.Column("citations", postgresql.JSONB, nullable=True),
        sa.Column("attachments", postgresql.JSONB, nullable=True),
        sa.Column("feedback", sa.String(8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_messages_branch", "messages", ["conversation_id", "branch_id", "sequence"])
    # Full-text search GIN index
    op.execute(
        "CREATE INDEX idx_messages_fts ON messages USING GIN (to_tsvector('english', content))"
    )

    # ─── user_memories ────────────────────────────────────────
    op.create_table(
        "user_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fact", sa.Text, nullable=False),
        sa.Column("embedding", sa.Text, nullable=True),  # stored as vector(1536); raw DDL below
        sa.Column("source_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("use_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    # Alter column type to vector after table creation
    op.execute("ALTER TABLE user_memories ALTER COLUMN embedding TYPE vector(1536) USING NULL")
    op.create_index("idx_mem_user", "user_memories", ["user_id", "last_used_at"])
    op.execute(
        "CREATE INDEX idx_mem_vec ON user_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # ─── Trigger: updated_at auto-maintenance ─────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
        $$ LANGUAGE plpgsql;
    """)
    for tbl in ("users", "conversations", "messages", "user_memories", "projects", "agents"):
        op.execute(f"""
            CREATE TRIGGER trg_{tbl}_updated_at
            BEFORE UPDATE ON {tbl}
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    for tbl in ("user_memories", "messages", "conversations", "agents", "project_members", "projects", "users"):
        op.drop_table(tbl)
    op.execute("DROP TYPE IF EXISTS auth_provider")
    op.execute("DROP TYPE IF EXISTS subscription_plan")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at() CASCADE")
