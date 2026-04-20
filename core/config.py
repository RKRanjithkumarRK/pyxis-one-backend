from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # At least one provider key must be set. Checked at runtime in ai_client.
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""       # Free: https://console.groq.com
    GEMINI_API_KEY: str = ""     # Free: https://aistudio.google.com/app/apikey

    DATABASE_URL: str = "sqlite+aiosqlite:///./pyxis.db"
    SECRET_KEY: str = "changeme-in-production"
    ENVIRONMENT: str = "production"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "https://pyxis-one-frontend.vercel.app,http://localhost:3000"

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

DEFAULT_MODEL = "claude-sonnet-4-6"
