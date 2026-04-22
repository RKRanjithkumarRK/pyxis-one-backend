from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── AI Provider Keys ─────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # ── Tool Provider Keys ───────────────────────────────────────────────────
    BRAVE_SEARCH_API_KEY: str = ""
    E2B_API_KEY: str = ""

    # ── Infrastructure ───────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./pyxis.db"
    REDIS_URL: str = ""           # Upstash Redis: rediss://...
    SECRET_KEY: str = "changeme-in-production"
    ENCRYPTION_KEY: str = ""      # Fernet key for vault encryption

    # ── App Config ───────────────────────────────────────────────────────────
    ENVIRONMENT: str = "production"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "https://pyxis-one-frontend.vercel.app,http://localhost:3000"

    # ── Model Defaults ───────────────────────────────────────────────────────
    DEFAULT_CLAUDE_MODEL: str = "claude-sonnet-4-6"
    DEFAULT_OPENAI_MODEL: str = "gpt-4o"
    DEFAULT_FAST_MODEL: str = "gpt-4o-mini"
    DEFAULT_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# ── Model registry ─────────────────────────────────────────────────────────────
AVAILABLE_MODELS = [
    {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet",
        "provider": "anthropic",
        "context_window": 200000,
        "description": "Deep reasoning, long-form coherence, nuanced analysis",
        "tier": "pro",
        "strengths": ["reasoning", "writing", "analysis"],
    },
    {
        "id": "claude-opus-4-7",
        "name": "Claude Opus",
        "provider": "anthropic",
        "context_window": 200000,
        "description": "Most capable Claude model for complex tasks",
        "tier": "enterprise",
        "strengths": ["reasoning", "math", "research"],
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku",
        "provider": "anthropic",
        "context_window": 200000,
        "description": "Fast and efficient for simple tasks",
        "tier": "free",
        "strengths": ["speed", "summarization"],
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        "context_window": 128000,
        "description": "Structured, direct responses with strong tool use",
        "tier": "pro",
        "strengths": ["coding", "tools", "vision"],
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "openai",
        "context_window": 128000,
        "description": "Fast and cost-efficient for straightforward tasks",
        "tier": "free",
        "strengths": ["speed", "simple_qa"],
    },
]

MODEL_CONTEXT_LIMITS = {
    "claude-sonnet-4-6": 200000,
    "claude-opus-4-7": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "llama-3.3-70b-versatile": 32768,
    "gemini-1.5-flash": 1000000,
}

# ── Cost per 1M tokens (USD) ────────────────────────────────────────────────────
MODEL_COSTS = {
    "claude-sonnet-4-6":        {"input": 3.00,  "output": 15.00, "cache_read": 0.30},
    "claude-opus-4-7":          {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    "claude-haiku-4-5-20251001":{"input": 0.80,  "output": 4.00,  "cache_read": 0.08},
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
}

# ── Rate limits per tier ─────────────────────────────────────────────────────────
TIER_LIMITS = {
    "free":       {"rpm": 10,  "tpd": 50_000},
    "pro":        {"rpm": 60,  "tpd": 1_000_000},
    "enterprise": {"rpm": 600, "tpd": 10_000_000},
}

# ── Legacy: kept for existing engines that import this ──────────────────────────
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

DEFAULT_MODEL = settings.DEFAULT_CLAUDE_MODEL
