from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── AI Provider Keys ─────────────────────────────────────────────────────
    # Anthropic intentionally removed — use free providers below
    GROQ_API_KEY:   str = ""   # Free: https://console.groq.com
    GEMINI_API_KEY: str = ""   # Free: https://aistudio.google.com/app/apikey
    OPENAI_API_KEY: str = ""   # Paid: https://platform.openai.com

    # ── Tool Provider Keys ───────────────────────────────────────────────────
    BRAVE_SEARCH_API_KEY: str = ""
    E2B_API_KEY:          str = ""

    # ── Infrastructure ───────────────────────────────────────────────────────
    DATABASE_URL:    str = "sqlite+aiosqlite:///./pyxis.db"
    REDIS_URL:       str = ""
    SECRET_KEY:      str = "changeme-in-production"
    ENCRYPTION_KEY:  str = ""

    # ── App Config ───────────────────────────────────────────────────────────
    ENVIRONMENT:     str = "production"
    PORT:            int = 8000
    FRONTEND_URL:    str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "https://pyxis-one-frontend.vercel.app,http://localhost:3000"

    # ── Model Defaults ───────────────────────────────────────────────────────
    DEFAULT_MODEL:      str = "gemini-2.0-flash"   # free, fast, 1M ctx
    DEFAULT_FAST_MODEL: str = "llama-3.1-8b-instant"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# ── Model registry — shown in /api/models endpoint and frontend dropdown ───────

AVAILABLE_MODELS = [
    # ── Groq (free, no credit card) ──────────────────────────────────────────
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B",
        "provider": "groq",
        "context_window": 128_000,
        "description": "Meta's best open model — balanced speed & quality",
        "tier": "free",
        "strengths": ["reasoning", "coding"],
        "available": True,
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B",
        "provider": "groq",
        "context_window": 128_000,
        "description": "Lightning-fast — best for quick answers",
        "tier": "free",
        "strengths": ["speed"],
        "available": True,
    },
    {
        "id": "mixtral-8x7b-32768",
        "name": "Mixtral 8x7B",
        "provider": "groq",
        "context_window": 32_768,
        "description": "Mixture-of-experts — great for long documents",
        "tier": "free",
        "strengths": ["reasoning", "long-context"],
        "available": True,
    },
    {
        "id": "gemma2-9b-it",
        "name": "Gemma 2 9B",
        "provider": "groq",
        "context_window": 8_192,
        "description": "Google's efficient open model via Groq",
        "tier": "free",
        "strengths": ["speed"],
        "available": True,
    },
    # ── Gemini (Google — generous free tier) ─────────────────────────────────
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "provider": "gemini",
        "context_window": 1_048_576,
        "description": "Google's fastest 2.0 model — 1M context, free",
        "tier": "free",
        "strengths": ["speed", "vision", "long-context"],
        "available": True,
    },
    {
        "id": "gemini-1.5-pro",
        "name": "Gemini 1.5 Pro",
        "provider": "gemini",
        "context_window": 2_097_152,
        "description": "2M context — strongest reasoning & analysis",
        "tier": "pro",
        "strengths": ["reasoning", "vision", "coding", "long-context"],
        "available": True,
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "provider": "gemini",
        "context_window": 1_048_576,
        "description": "Fast and efficient — 1M context",
        "tier": "free",
        "strengths": ["speed", "long-context"],
        "available": True,
    },
    # ── OpenAI (paid) ─────────────────────────────────────────────────────────
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        "context_window": 128_000,
        "description": "OpenAI's flagship — best tools & coding",
        "tier": "pro",
        "strengths": ["coding", "tools", "vision"],
        "available": True,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "openai",
        "context_window": 128_000,
        "description": "Cost-efficient with strong reasoning",
        "tier": "free",
        "strengths": ["speed", "coding"],
        "available": True,
    },
]

MODEL_CONTEXT_LIMITS: dict[str, int] = {m["id"]: m["context_window"] for m in AVAILABLE_MODELS}

# Cost per 1M tokens (USD); Groq/Gemini free-tier models cost $0
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Groq (free / negligible)
    "llama-3.3-70b-versatile": {"input": 0.0,  "output": 0.0},
    "llama-3.1-8b-instant":    {"input": 0.0,  "output": 0.0},
    "mixtral-8x7b-32768":      {"input": 0.0,  "output": 0.0},
    "gemma2-9b-it":            {"input": 0.0,  "output": 0.0},
    # Gemini (free up to quota)
    "gemini-2.0-flash":        {"input": 0.0,  "output": 0.0},
    "gemini-1.5-flash":        {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":          {"input": 1.25,  "output": 5.00},
    # OpenAI (paid)
    "gpt-4o":                  {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":             {"input": 0.15,  "output": 0.60},
}

TIER_LIMITS: dict[str, dict[str, int]] = {
    "free":       {"rpm": 10,  "tpd": 100_000},
    "pro":        {"rpm": 60,  "tpd": 2_000_000},
    "enterprise": {"rpm": 600, "tpd": 20_000_000},
}

RESPONSE_STRUCTURE = """
Structure every response exactly as:
ORIENTATION: one sentence stating what this accomplishes
CORE EXPLANATION: three collapsible tiers
  - SURFACE: simple intuitive explanation
  - STRUCTURAL: how it actually works mechanically
  - EXPERT: edge cases, nuance, and formal precision
EXAMPLES: 2-3 concrete examples from student context
EDGE CASES: what breaks or limits this explanation
NEXT MOVE: one recommended next action

Tag every factual claim with exactly one of:
[VERIFIED] [CONSENSUS] [DEBATED] [SPECULATIVE]
"""

DEFAULT_MODEL = settings.DEFAULT_MODEL
