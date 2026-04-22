from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Free AI Providers ────────────────────────────────────────────────────
    GROQ_API_KEY:      str = ""   # Free: https://console.groq.com
    GEMINI_API_KEY:    str = ""   # Free: https://aistudio.google.com/app/apikey
    CEREBRAS_API_KEY:  str = ""   # Free: https://cloud.cerebras.ai
    MISTRAL_API_KEY:   str = ""   # Free: https://console.mistral.ai
    SAMBANOVA_API_KEY: str = ""   # Free: https://cloud.sambanova.ai

    # ── Paid (optional) ──────────────────────────────────────────────────────
    OPENAI_API_KEY:    str = ""   # Paid: https://platform.openai.com

    # ── Tool Providers ───────────────────────────────────────────────────────
    BRAVE_SEARCH_API_KEY: str = ""
    E2B_API_KEY:          str = ""

    # ── Infrastructure ───────────────────────────────────────────────────────
    DATABASE_URL:    str = "sqlite+aiosqlite:///./pyxis.db"
    REDIS_URL:       str = ""
    SECRET_KEY:      str = "changeme-in-production"
    ENCRYPTION_KEY:  str = ""

    # ── App ──────────────────────────────────────────────────────────────────
    ENVIRONMENT:     str = "production"
    PORT:            int = 8000
    FRONTEND_URL:    str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "https://pyxis-one-frontend.vercel.app,http://localhost:3000"

    # ── Defaults ─────────────────────────────────────────────────────────────
    DEFAULT_MODEL:      str = "gemini-2.0-flash"
    DEFAULT_FAST_MODEL: str = "llama-3.1-8b-instant"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# ── Model registry  (frontend dropdown + /api/models endpoint) ─────────────────
#
# provider values must match AIModel.provider in lib/types.ts
# For disambiguating same-arch models across providers, prefix the id with
# "<provider>/" (e.g. "cerebras/llama-3.3-70b").  The backend strips the
# prefix before calling the API.

AVAILABLE_MODELS: list[dict] = [

    # ── Groq — free, ~500 tokens/s ────────────────────────────────────────
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B",
        "provider": "groq",
        "context_window": 128_000,
        "description": "Meta's best open model — strong reasoning & coding",
        "tier": "free",
        "strengths": ["reasoning", "coding"],
        "available": True,
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B",
        "provider": "groq",
        "context_window": 128_000,
        "description": "Fastest responses — ideal for quick queries",
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
        "description": "Google open model — compact & efficient",
        "tier": "free",
        "strengths": ["speed"],
        "available": True,
    },

    # ── Gemini — free tier, massive context ───────────────────────────────
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "provider": "gemini",
        "context_window": 1_048_576,
        "description": "Google's flagship free model — 1M context, vision",
        "tier": "free",
        "strengths": ["speed", "vision", "long-context"],
        "available": True,
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "provider": "gemini",
        "context_window": 1_048_576,
        "description": "Stable & efficient — 1M context window",
        "tier": "free",
        "strengths": ["speed", "long-context"],
        "available": True,
    },

    # ── Cerebras — free, 1,000+ tokens/s (fastest on Earth) ───────────────
    {
        "id": "cerebras/llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "provider": "cerebras",
        "context_window": 128_000,
        "description": "1,000 tok/s — Llama 70B at unprecedented speed",
        "tier": "free",
        "strengths": ["speed", "reasoning"],
        "available": True,
    },
    {
        "id": "cerebras/llama3.1-8b",
        "name": "Llama 3.1 8B",
        "provider": "cerebras",
        "context_window": 128_000,
        "description": "2,000 tok/s — the fastest 8B model available",
        "tier": "free",
        "strengths": ["speed"],
        "available": True,
    },

    # ── Mistral — free La Plateforme tier ─────────────────────────────────
    {
        "id": "mistral-small-3.1-24b-instruct",
        "name": "Mistral Small 3.1",
        "provider": "mistral",
        "context_window": 128_000,
        "description": "Mistral's best free model — vision + function calling",
        "tier": "free",
        "strengths": ["reasoning", "coding", "vision"],
        "available": True,
    },
    {
        "id": "codestral-2501",
        "name": "Codestral",
        "provider": "mistral",
        "context_window": 256_000,
        "description": "Specialised code model — 256K context",
        "tier": "free",
        "strengths": ["coding"],
        "available": True,
    },
    {
        "id": "open-mistral-nemo",
        "name": "Mistral Nemo 12B",
        "provider": "mistral",
        "context_window": 128_000,
        "description": "Compact multilingual model",
        "tier": "free",
        "strengths": ["speed", "multilingual"],
        "available": True,
    },

    # ── SambaNova — free, runs DeepSeek & Qwen ────────────────────────────
    {
        "id": "sambanova/deepseek-v3",
        "name": "DeepSeek V3",
        "provider": "sambanova",
        "context_window": 64_000,
        "description": "Top-tier coding & reasoning — rivals GPT-4o free",
        "tier": "free",
        "strengths": ["coding", "reasoning", "math"],
        "available": True,
    },
    {
        "id": "sambanova/qwen2.5-72b",
        "name": "Qwen 2.5 72B",
        "provider": "sambanova",
        "context_window": 32_768,
        "description": "Alibaba 72B — excellent multilingual & math",
        "tier": "free",
        "strengths": ["reasoning", "math", "multilingual"],
        "available": True,
    },
    {
        "id": "sambanova/llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "provider": "sambanova",
        "context_window": 8_192,
        "description": "Meta 70B on SambaNova silicon",
        "tier": "free",
        "strengths": ["reasoning"],
        "available": True,
    },

    # ── OpenAI — paid (optional) ──────────────────────────────────────────
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        "context_window": 128_000,
        "description": "Best tool use, vision & coding — requires API key",
        "tier": "pro",
        "strengths": ["coding", "tools", "vision"],
        "available": True,
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "openai",
        "context_window": 128_000,
        "description": "Cost-efficient OpenAI model",
        "tier": "free",
        "strengths": ["speed", "coding"],
        "available": True,
    },
]

# Auto-built lookups
MODEL_CONTEXT_LIMITS: dict[str, int] = {m["id"]: m["context_window"] for m in AVAILABLE_MODELS}

# Cost per 1M tokens (USD) — free providers are $0
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Groq / Cerebras / Mistral-free / SambaNova — all $0
    **{m["id"]: {"input": 0.0, "output": 0.0}
       for m in AVAILABLE_MODELS if m["provider"] in ("groq", "cerebras", "sambanova")},
    "gemini-2.0-flash":                  {"input": 0.0,  "output": 0.0},
    "gemini-1.5-flash":                  {"input": 0.075, "output": 0.30},
    "mistral-small-3.1-24b-instruct":    {"input": 0.0,  "output": 0.0},
    "codestral-2501":                    {"input": 0.0,  "output": 0.0},
    "open-mistral-nemo":                 {"input": 0.0,  "output": 0.0},
    "gpt-4o":                            {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":                       {"input": 0.15, "output": 0.60},
}

TIER_LIMITS: dict[str, dict[str, int]] = {
    "free":       {"rpm": 10,  "tpd": 200_000},
    "pro":        {"rpm": 60,  "tpd": 2_000_000},
    "enterprise": {"rpm": 600, "tpd": 20_000_000},
}

RESPONSE_STRUCTURE = """Structure: ORIENTATION → CORE (surface/structural/expert) → EXAMPLES → EDGE CASES → NEXT MOVE.
Tag every factual claim: [VERIFIED] [CONSENSUS] [DEBATED] [SPECULATIVE]"""

DEFAULT_MODEL = settings.DEFAULT_MODEL
