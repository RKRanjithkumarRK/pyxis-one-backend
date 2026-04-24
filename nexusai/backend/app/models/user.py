from __future__ import annotations
import uuid
import enum
from sqlalchemy import String, Boolean, Enum as SAEnum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDPrimaryKey


class AuthProvider(str, enum.Enum):
    email = "email"
    google = "google"
    github = "github"
    apple = "apple"
    microsoft = "microsoft"
    magic_link = "magic_link"
    guest = "guest"


class SubscriptionPlan(str, enum.Enum):
    free = "free"
    plus = "plus"
    team = "team"
    enterprise = "enterprise"


class User(Base, UUIDPrimaryKey, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True, index=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider: Mapped[AuthProvider] = mapped_column(
        SAEnum(AuthProvider, name="auth_provider"),
        default=AuthProvider.email,
        nullable=False,
    )
    provider_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Subscription
    plan: Mapped[SubscriptionPlan] = mapped_column(
        SAEnum(SubscriptionPlan, name="subscription_plan"),
        default=SubscriptionPlan.free,
        nullable=False,
    )
    stripe_customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Preferences (JSON-serialized as text for simplicity; will store JSONB in migration)
    custom_instructions: Mapped[str | None] = mapped_column(String(8000), nullable=True)
    memory_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    conversations: Mapped[list["Conversation"]] = relationship(  # noqa: F821
        "Conversation", back_populates="user", lazy="noload"
    )
    memories: Mapped[list["UserMemory"]] = relationship(  # noqa: F821
        "UserMemory", back_populates="user", lazy="noload"
    )

    __table_args__ = (
        Index("idx_users_email_active", "email", "is_active"),
        Index("idx_users_provider", "provider", "provider_id"),
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} plan={self.plan}>"
