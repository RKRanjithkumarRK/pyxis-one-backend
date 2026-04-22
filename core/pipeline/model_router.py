"""
Model router — selects the best model given intent + user tier + manual override.

Provider lineup (6 providers, no Anthropic):
  Groq       — free, ultra-fast (Llama, Mixtral, Gemma)
  Gemini     — free generous tier, huge context (Gemini 2.0/1.5)
  Cerebras   — free, 1,000+ tok/s on WSE silicon
  Mistral    — free La Plateforme tier (Mistral Small, Codestral, Nemo)
  SambaNova  — free (DeepSeek V3, Qwen 2.5, Llama 70B)
  OpenAI     — paid, best tools & coding (GPT-4o)
"""

from __future__ import annotations
from dataclasses import dataclass
from core.config import settings
from core.pipeline.intent_classifier import RouterDecision


@dataclass
class ModelSelection:
    model:        str
    provider:     str          # openai | groq | gemini | cerebras | mistral | sambanova
    persona:      str          # "structured" | "analytical"
    inject_tools: list[str]
    max_tokens:   int
    temperature:  float


# ── Routing table ──────────────────────────────────────────────────────────────
# Default: free-tier — all free models so everyone gets a great experience.
# Pro: bump quality where it matters (Gemini 1.5 Pro / GPT-4o).

_INTENT_TO_MODEL: dict[str, dict[str, str]] = {
    "free": {
        "coding":    "sambanova/deepseek-v3",      # DeepSeek rivals GPT-4o at code, free
        "math":      "sambanova/deepseek-v3",      # Top math on SambaNova
        "reasoning": "gemini-2.0-flash",           # 1M ctx, strong reasoning
        "creative":  "llama-3.3-70b-versatile",    # Llama is fluid and creative
        "vision":    "gemini-2.0-flash",           # Gemini has vision
        "fast":      "cerebras/llama3.1-8b",       # 2,000 tok/s — fastest 8B
        "default":   "gemini-2.0-flash",
    },
    "pro": {
        "coding":    "gpt-4o",                     # Best tool-use for coding
        "math":      "gemini-1.5-pro",             # 2M ctx, top reasoning
        "reasoning": "gemini-1.5-pro",
        "creative":  "gemini-1.5-pro",
        "vision":    "gpt-4o",                     # OpenAI vision quality
        "fast":      "cerebras/llama3.1-8b",
        "default":   "gemini-1.5-pro",
    },
    "enterprise": {
        "coding":    "gpt-4o",
        "math":      "gemini-1.5-pro",
        "reasoning": "gemini-1.5-pro",
        "creative":  "gemini-1.5-pro",
        "vision":    "gpt-4o",
        "fast":      "cerebras/llama3.1-8b",
        "default":   "gemini-1.5-pro",
    },
}

_TOOL_MAP: dict[str, list[str]] = {
    "coding":    ["code_interpreter"],
    "math":      ["code_interpreter"],
    "reasoning": [],
    "creative":  [],
    "vision":    [],
    "fast":      [],
    "default":   [],
}

# provider + persona for each model
_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    # OpenAI
    "gpt-4o":                            ("openai",     "structured"),
    "gpt-4o-mini":                       ("openai",     "structured"),
    # Gemini
    "gemini-2.0-flash":                  ("gemini",     "analytical"),
    "gemini-1.5-pro":                    ("gemini",     "analytical"),
    "gemini-1.5-flash":                  ("gemini",     "analytical"),
    # Groq
    "llama-3.3-70b-versatile":           ("groq",       "structured"),
    "llama-3.1-8b-instant":              ("groq",       "structured"),
    "mixtral-8x7b-32768":                ("groq",       "structured"),
    "gemma2-9b-it":                      ("groq",       "structured"),
    # Cerebras
    "cerebras/llama-3.3-70b":            ("cerebras",   "structured"),
    "cerebras/llama3.1-8b":              ("cerebras",   "structured"),
    # Mistral
    "mistral-small-3.1-24b-instruct":    ("mistral",    "analytical"),
    "codestral-2501":                    ("mistral",    "structured"),
    "open-mistral-nemo":                 ("mistral",    "structured"),
    # SambaNova
    "sambanova/deepseek-v3":             ("sambanova",  "analytical"),
    "sambanova/qwen2.5-72b":             ("sambanova",  "analytical"),
    "sambanova/llama-3.3-70b":           ("sambanova",  "structured"),
}

_MAX_TOKENS: dict[str, int] = {
    # OpenAI
    "gpt-4o":                            4096,
    "gpt-4o-mini":                       4096,
    # Gemini
    "gemini-2.0-flash":                  8192,
    "gemini-1.5-pro":                    8192,
    "gemini-1.5-flash":                  8192,
    # Groq (capped at 8192 by provider)
    "llama-3.3-70b-versatile":           4096,
    "llama-3.1-8b-instant":              4096,
    "mixtral-8x7b-32768":                4096,
    "gemma2-9b-it":                      2048,
    # Cerebras (capped at 8192)
    "cerebras/llama-3.3-70b":            4096,
    "cerebras/llama3.1-8b":              4096,
    # Mistral
    "mistral-small-3.1-24b-instruct":    4096,
    "codestral-2501":                    4096,
    "open-mistral-nemo":                 4096,
    # SambaNova (capped at 8192)
    "sambanova/deepseek-v3":             4096,
    "sambanova/qwen2.5-72b":             4096,
    "sambanova/llama-3.3-70b":           4096,
}

# Models where tool-calling is reliable (SambaNova skipped — unstable)
_TOOL_CAPABLE: set[str] = {
    "gpt-4o", "gpt-4o-mini",
    "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash",
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
    "cerebras/llama-3.3-70b", "cerebras/llama3.1-8b",
    "mistral-small-3.1-24b-instruct", "codestral-2501", "open-mistral-nemo",
}


def select_model(
    decision: RouterDecision,
    user_tier: str = "free",
    manual_model: str | None = None,
    enable_web_search: bool = False,
) -> ModelSelection:
    """Pick the model + config to use for this request."""

    if manual_model:
        model = _gate_model(manual_model, user_tier)
    else:
        tier_map = _INTENT_TO_MODEL.get(user_tier, _INTENT_TO_MODEL["free"])
        model = tier_map.get(decision.intent, tier_map["default"])

    provider, persona = _PROVIDER_MAP.get(model, ("groq", "structured"))

    # Build tool list — skip if model doesn't support tools reliably
    tools: list[str] = []
    if model in _TOOL_CAPABLE:
        tools = list(_TOOL_MAP.get(decision.intent, []))
        if enable_web_search and settings.BRAVE_SEARCH_API_KEY:
            tools.append("web_search")
        if decision.has_image:
            tools = [t for t in tools if t != "code_interpreter"]
        if "code_interpreter" in tools and not settings.E2B_API_KEY:
            tools.remove("code_interpreter")

    return ModelSelection(
        model=model,
        provider=provider,
        persona=persona,
        inject_tools=tools,
        max_tokens=_MAX_TOKENS.get(model, 4096),
        temperature=_temperature(decision.intent),
    )


def _gate_model(model: str, tier: str) -> str:
    """Downgrade if the chosen model requires a higher tier."""
    pro_only = {"gpt-4o", "gemini-1.5-pro"}
    if model in pro_only and tier == "free":
        return "gemini-2.0-flash" if model == "gemini-1.5-pro" else "gpt-4o-mini"
    # Unknown model IDs pass through — unified_client handles them
    return model


def _temperature(intent: str) -> float:
    return {
        "coding":    0.2,
        "math":      0.1,
        "reasoning": 0.5,
        "creative":  0.9,
        "vision":    0.3,
        "fast":      0.3,
        "default":   0.7,
    }.get(intent, 0.7)


# ── System prompt ─────────────────────────────────────────────────────────────
# Single unified prompt — works equally well with Groq Llama, Gemini, and GPT.

PYXIS_SYSTEM = """You are Pyxis — an elite AI assistant built for deep learning and mastery.

Behaviour:
- Give precise, structured, actionable responses
- Use markdown: headers, bullets, numbered lists, code blocks (with language labels)
- For code: include type hints, inline comments, and brief explanations
- Surface insights the user didn't ask for but needs
- Acknowledge uncertainty explicitly — tag claims as [VERIFIED] [CONSENSUS] [DEBATED] [SPECULATIVE]
- Never pad or repeat. Every sentence must advance understanding.
- Never refuse reasonable requests."""


def get_system_prompt(
    persona: str = "structured",
    feature_context: str = "",
    psyche_context: str = "",
) -> str:
    parts = [PYXIS_SYSTEM]
    if feature_context:
        parts.append(f"\n\n{feature_context}")
    if psyche_context:
        parts.append(f"\n\nUser cognitive profile:\n{psyche_context}")
    return "\n".join(parts)
