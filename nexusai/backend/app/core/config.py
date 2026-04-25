from __future__ import annotations
from typing import Literal
from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Core ─────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ─── Database ─────────────────────────────────────────────
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40

    # ─── Redis ────────────────────────────────────────────────
    REDIS_URL: str

    # ─── Qdrant ───────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # ─── AI Providers ─────────────────────────────────────────
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    CEREBRAS_API_KEY: str | None = None
    SAMBANOVA_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None

    # ─── OAuth ────────────────────────────────────────────────
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GITHUB_CLIENT_ID: str | None = None
    GITHUB_CLIENT_SECRET: str | None = None

    # ─── External Services ────────────────────────────────────
    SENDGRID_API_KEY: str | None = None
    SERPER_API_KEY: str | None = None
    REPLICATE_API_TOKEN: str | None = None
    ELEVENLABS_API_KEY: str | None = None
    E2B_API_KEY: str | None = None

    # ─── Stripe ───────────────────────────────────────────────
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None
    STRIPE_PUBLISHABLE_KEY: str | None = None
    STRIPE_PLUS_PRICE_ID: str | None = None
    STRIPE_TEAM_PRICE_ID: str | None = None

    # ─── GCS ──────────────────────────────────────────────────
    GCS_BUCKET_NAME: str = "nexusai-files"
    GOOGLE_CLOUD_PROJECT: str | None = None

    # ─── Frontend ─────────────────────────────────────────────
    NEXT_PUBLIC_API_URL: str = "http://localhost:8000"
    FRONTEND_URL: str = "http://localhost:3000"

    # ─── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ─── Observability ────────────────────────────────────────
    SENTRY_DSN: str | None = None
    POSTHOG_API_KEY: str | None = None
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    # ─── Rate limiting ────────────────────────────────────────
    RATE_LIMIT_UNAUTHENTICATED: str = "20/minute"
    RATE_LIMIT_FREE: str = "120/minute"
    RATE_LIMIT_PLUS: str = "600/minute"

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_min_length(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def warn_missing_providers(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            required = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
            missing = [k for k in required if not getattr(self, k)]
            if missing:
                raise ValueError(f"Production requires: {missing}")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def available_providers(self) -> list[str]:
        mapping = {
            "anthropic": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "groq": self.GROQ_API_KEY,
            "mistral": self.MISTRAL_API_KEY,
            "cerebras": self.CEREBRAS_API_KEY,
            "sambanova": self.SAMBANOVA_API_KEY,
        }
        return [k for k, v in mapping.items() if v]


settings = Settings()  # type: ignore[call-arg]  # loaded from env
