from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    DATABASE_URL: str = "sqlite+aiosqlite:///./pyxis.db"
    SECRET_KEY: str = "changeme-in-production"
    ENVIRONMENT: str = "production"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"

    @field_validator("ANTHROPIC_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        placeholders = {"your_key_here", "placeholder", "sk-ant-test-placeholder-replace-with-real-key", ""}
        if not v or v.strip().lower() in placeholders or "placeholder" in v.lower():
            raise ValueError(
                "\n\n  ❌  ANTHROPIC_API_KEY is not set.\n"
                "  1. Get your key: https://console.anthropic.com/settings/keys\n"
                "  2. Open pyxis-one-backend/.env\n"
                "  3. Replace the placeholder with your real key\n"
                "  4. Restart the server\n"
            )
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

RESPONSE_STRUCTURE = """
Structure every response exactly as:
ORIENTATION: one sentence stating what this accomplishes
CORE EXPLANATION: three collapsible tiers
  - SURFACE: simple intuitive explanation
  - STRUCTURAL: how it actually works mechanically
  - EXPERT: edge cases, nuance, and formal precision
EXAMPLES: 2-3 concrete examples from student context
VISUAL: describe an SVG diagram representing this concept
EDGE CASES: what breaks or limits this explanation
CROSS DOMAIN: one unexpected connection to a different field
FRONTIER: what is still unknown or actively debated
TEST YOURSELF: one precisely calibrated challenge question
NEXT MOVE: one recommended next action

Tag every factual claim with exactly one of:
[VERIFIED] [CONSENSUS] [DEBATED] [SPECULATIVE]
"""

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
