from __future__ import annotations
import uuid
import enum
from sqlalchemy import String, Integer, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy import Computed
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class Message(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Branch support (for edit-message → create-branch feature)
    branch_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        default=uuid.uuid4,
    )
    parent_branch_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Content
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # MessageRole values
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Metadata from provider
    usage: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    citations: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    attachments: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # User feedback
    feedback: Mapped[str | None] = mapped_column(String(8), nullable=True)  # "good"|"bad"

    # Relationships
    conversation: Mapped["Conversation"] = relationship(  # noqa: F821
        "Conversation", back_populates="messages"
    )

    __table_args__ = (
        # Main access pattern: fetch messages for a conversation branch in order
        Index("idx_messages_branch", "conversation_id", "branch_id", "sequence"),
        # Full-text search via GIN index (created in migration with to_tsvector)
        # Done in raw SQL in migration: CREATE INDEX idx_messages_fts ON messages USING GIN(...)
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} seq={self.sequence}>"
