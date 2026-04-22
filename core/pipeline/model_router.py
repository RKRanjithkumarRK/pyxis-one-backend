"""
Model router: selects the correct model given intent + user tier + manual override.
Also decides which tools to inject and system prompt style (GPT vs Claude persona).
"""

from __future__ import annotations
from dataclasses import dataclass
from core.config import settings
from core.pipeline.intent_classifier import RouterDecision


@dataclass
class ModelSelection:
    model: str
    provider: str           # openai | anthropic | groq
    persona: str            # gpt | claude
    inject_tools: list[str] # tool names to inject
    max_tokens: int
    temperature: float


# ── Routing table ─────────────────────────────────────────────────────────────

_INTENT_TO_MODEL = {
    # Free tier
    "free": {
        "coding":    "gpt-4o-mini",
        "math":      "claude-haiku-4-5-20251001",
        "reasoning": "claude-haiku-4-5-20251001",
        "creative":  "claude-haiku-4-5-20251001",
        "vision":    "gpt-4o-mini",
        "fast":      "gpt-4o-mini",
        "default":   "claude-haiku-4-5-20251001",
    },
    # Pro tier
    "pro": {
        "coding":    "gpt-4o",
        "math":      "claude-sonnet-4-6",
        "reasoning": "claude-sonnet-4-6",
        "creative":  "claude-sonnet-4-6",
        "vision":    "gpt-4o",
        "fast":      "gpt-4o-mini",
        "default":   "claude-sonnet-4-6",
    },
    # Enterprise tier
    "enterprise": {
        "coding":    "gpt-4o",
        "math":      "claude-sonnet-4-6",
        "reasoning": "claude-sonnet-4-6",
        "creative":  "claude-sonnet-4-6",
        "vision":    "gpt-4o",
        "fast":      "gpt-4o-mini",
        "default":   "claude-sonnet-4-6",
    },
}

# High-stakes reasoning → Opus (enterprise only, score threshold)
_OPUS_THRESHOLD = 4

_TOOL_MAP = {
    "coding":    ["code_interpreter"],
    "math":      ["code_interpreter"],
    "reasoning": [],
    "creative":  [],
    "vision":    [],
    "fast":      [],
    "default":   [],
}

_PROVIDER_MAP = {
    "gpt-4o":                    ("openai",    "gpt"),
    "gpt-4o-mini":               ("openai",    "gpt"),
    "claude-sonnet-4-6":         ("anthropic", "claude"),
    "claude-opus-4-7":           ("anthropic", "claude"),
    "claude-haiku-4-5-20251001": ("anthropic", "claude"),
    "llama-3.3-70b-versatile":   ("groq",      "gpt"),
}

_MAX_TOKENS = {
    "gpt-4o":                    4096,
    "gpt-4o-mini":               4096,
    "claude-sonnet-4-6":         8192,
    "claude-opus-4-7":           8192,
    "claude-haiku-4-5-20251001": 4096,
    "llama-3.3-70b-versatile":   4096,
}


def select_model(
    decision: RouterDecision,
    user_tier: str = "free",
    manual_model: str | None = None,
    enable_web_search: bool = False,
) -> ModelSelection:
    """
    Returns the model + configuration to use.
    manual_model overrides auto-routing but respects tier gating.
    """
    # Manual override — validate it's not above user's tier
    if manual_model:
        model = _gate_model(manual_model, user_tier)
    else:
        tier_map = _INTENT_TO_MODEL.get(user_tier, _INTENT_TO_MODEL["free"])
        model = tier_map.get(decision.intent, tier_map["default"])

        # Upgrade to Opus if enterprise + very high reasoning score
        if (
            user_tier == "enterprise"
            and decision.scores.get("reasoning", 0) >= _OPUS_THRESHOLD
            and settings.ANTHROPIC_API_KEY
        ):
            model = "claude-opus-4-7"

    provider, persona = _PROVIDER_MAP.get(model, ("anthropic", "claude"))

    # Build tool list
    tools: list[str] = list(_TOOL_MAP.get(decision.intent, []))
    if enable_web_search and settings.BRAVE_SEARCH_API_KEY:
        tools.append("web_search")
    if decision.has_image:
        tools = [t for t in tools if t != "code_interpreter"]  # vision doesn't need code exec

    # No tools if provider keys are missing
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
    """Downgrade model if user tier doesn't allow it."""
    enterprise_only = {"claude-opus-4-7"}
    pro_and_above = {"gpt-4o", "claude-sonnet-4-6"}

    if model in enterprise_only and tier not in ("enterprise",):
        return "claude-sonnet-4-6" if tier == "pro" else "claude-haiku-4-5-20251001"
    if model in pro_and_above and tier == "free":
        return "gpt-4o-mini" if model == "gpt-4o" else "claude-haiku-4-5-20251001"
    return model


def _temperature(intent: str) -> float:
    temps = {
        "coding":    0.2,
        "math":      0.1,
        "reasoning": 0.5,
        "creative":  0.9,
        "vision":    0.3,
        "fast":      0.3,
        "default":   0.7,
    }
    return temps.get(intent, 0.7)


# ── System prompt templates by persona ───────────────────────────────────────

GPT_SYSTEM = """You are Pyxis — a brilliant AI assistant modeled after GPT-4o.
You give direct, structured, actionable answers.
Use markdown formatting: headers, bullets, numbered lists, code blocks.
For code: always include language labels, type hints, and brief inline comments.
Be confident and concise. Never pad responses."""

CLAUDE_SYSTEM = """You are Pyxis — a thoughtful, deeply reasoning AI assistant modeled after Claude.
You reason carefully before concluding. You acknowledge nuance, edge cases, and uncertainty.
Your answers are coherent long-form prose when depth is needed, with markdown structure.
You are calibrated: you say what you know, what you infer, and what you're uncertain about.
Never refuse reasonable requests. Be direct, warm, and intellectually honest."""


def get_system_prompt(persona: str, feature_context: str = "", psyche_context: str = "") -> str:
    base = GPT_SYSTEM if persona == "gpt" else CLAUDE_SYSTEM
    parts = [base]
    if feature_context:
        parts.append(f"\n\n{feature_context}")
    if psyche_context:
        parts.append(f"\n\n{psyche_context}")
    return "\n".join(parts)
