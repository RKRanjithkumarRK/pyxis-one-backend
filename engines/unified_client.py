"""
Unified streaming client — 6 providers, zero Anthropic.

Provider routing uses MODEL_REGISTRY (explicit map) so identical model names
across providers (e.g. Llama 3.3 70B on Groq vs Cerebras vs SambaNova) are
unambiguous.  Frontend model IDs use "<provider>/" prefix for disambiguation.

All non-Gemini providers speak the OpenAI Chat Completions wire format, so a
single _openai_compatible_stream() function handles them all via base_url.
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from typing import AsyncGenerator, Any
from core.config import settings, MODEL_COSTS


# ── Event model ───────────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    type: str          # text | tool_start | tool_delta | tool_done | done | error | system
    content: str = ""
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_call_id: str | None = None
    usage: dict | None = None
    error_code: str | None = None

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.__dict__, default=str)}\n\n"


# ── Registry: frontend model ID → (provider_key, actual API model ID) ─────────

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Groq
    "llama-3.3-70b-versatile":           ("groq",      "llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant":              ("groq",      "llama-3.1-8b-instant"),
    "mixtral-8x7b-32768":                ("groq",      "mixtral-8x7b-32768"),
    "gemma2-9b-it":                      ("groq",      "gemma2-9b-it"),
    # Gemini
    "gemini-2.0-flash":                  ("gemini",    "gemini-2.0-flash"),
    "gemini-1.5-flash":                  ("gemini",    "gemini-1.5-flash"),
    "gemini-1.5-pro":                    ("gemini",    "gemini-1.5-pro"),
    # Cerebras (prefix keeps them distinct from Groq Llama)
    "cerebras/llama-3.3-70b":            ("cerebras",  "llama-3.3-70b"),
    "cerebras/llama3.1-8b":              ("cerebras",  "llama3.1-8b"),
    # Mistral
    "mistral-small-3.1-24b-instruct":    ("mistral",   "mistral-small-3.1-24b-instruct"),
    "codestral-2501":                    ("mistral",   "codestral-2501"),
    "open-mistral-nemo":                 ("mistral",   "open-mistral-nemo"),
    # SambaNova (prefix + mapping to their actual model IDs)
    "sambanova/deepseek-v3":             ("sambanova", "DeepSeek-V3-0324"),
    "sambanova/qwen2.5-72b":             ("sambanova", "Qwen2.5-72B-Instruct"),
    "sambanova/llama-3.3-70b":           ("sambanova", "Meta-Llama-3.3-70B-Instruct"),
    # OpenAI
    "gpt-4o":                            ("openai",    "gpt-4o"),
    "gpt-4o-mini":                       ("openai",    "gpt-4o-mini"),
}

# Provider → (OpenAI-compatible base_url | None, settings attribute name)
# Gemini is handled separately (not OpenAI-compatible)
PROVIDER_CONFIG: dict[str, tuple[str | None, str]] = {
    "openai":    (None,                                  "OPENAI_API_KEY"),
    "groq":      ("https://api.groq.com/openai/v1",     "GROQ_API_KEY"),
    "cerebras":  ("https://api.cerebras.ai/v1",         "CEREBRAS_API_KEY"),
    "mistral":   ("https://api.mistral.ai/v1",          "MISTRAL_API_KEY"),
    "sambanova": ("https://api.sambanova.ai/v1",        "SAMBANOVA_API_KEY"),
}

# Models / providers where tool-calling is reliable
_TOOL_CAPABLE: set[str] = {
    "openai", "groq", "mistral", "cerebras",
    # SambaNova skipped — varies per model; DeepSeek tool support unstable
}

# Only llama-* models support tools on Groq
def _groq_supports_tools(api_model: str) -> bool:
    return api_model.startswith("llama")


# ── OpenAI-compatible streaming ───────────────────────────────────────────────

async def _openai_compatible_stream(
    messages: list[dict],
    system: str,
    api_model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
    api_key: str,
    base_url: str | None,
    provider: str,
) -> AsyncGenerator[StreamEvent, None]:
    from openai import AsyncOpenAI, APIStatusError

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    full_messages = [{"role": "system", "content": system}] + messages

    # Convert unified tool format → OpenAI function-calling schema
    oai_tools = None
    if tools:
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    kwargs: dict[str, Any] = {
        "model":       api_model,
        "messages":    full_messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
    }
    # Only OpenAI proper supports stream_options; all others get done from finish_reason
    if provider == "openai":
        kwargs["stream_options"] = {"include_usage": True}
    if oai_tools:
        kwargs["tools"] = oai_tools
        kwargs["tool_choice"] = "auto"

    try:
        stream = await client.chat.completions.create(**kwargs)
        active_tools: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                # OpenAI usage chunk
                if chunk.usage:
                    yield StreamEvent(
                        type="done",
                        usage={
                            "input":  chunk.usage.prompt_tokens,
                            "output": chunk.usage.completion_tokens,
                            "model":  api_model,
                        },
                    )
                continue

            delta  = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            if delta.content:
                yield StreamEvent(type="text", content=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_tools:
                        active_tools[idx] = {
                            "name": tc.function.name or "",
                            "id":   tc.id or f"call_{idx}",
                            "args": "",
                        }
                        if tc.function.name:
                            yield StreamEvent(
                                type="tool_start",
                                tool_name=tc.function.name,
                                tool_call_id=tc.id,
                            )
                    else:
                        if tc.function.name:
                            active_tools[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            active_tools[idx]["args"] += tc.function.arguments
                            yield StreamEvent(
                                type="tool_delta",
                                content=tc.function.arguments,
                                tool_name=active_tools[idx]["name"],
                                tool_call_id=active_tools[idx]["id"],
                            )

            if finish == "tool_calls":
                for tool in active_tools.values():
                    try:
                        parsed = json.loads(tool["args"]) if tool["args"] else {}
                    except json.JSONDecodeError:
                        parsed = {}
                    yield StreamEvent(
                        type="tool_done",
                        tool_name=tool["name"],
                        tool_call_id=tool["id"],
                        tool_input=parsed,
                    )
                active_tools.clear()

            elif finish == "stop" and provider != "openai":
                # Non-OpenAI providers: emit done here (no usage chunk)
                yield StreamEvent(type="done", usage={"model": api_model})

    except APIStatusError as e:
        yield StreamEvent(type="error", content=str(e), error_code=str(e.status_code))
        raise


# ── Gemini streaming (google-genai SDK) ───────────────────────────────────────

async def _gemini_stream(
    messages: list[dict],
    system: str,
    api_model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
) -> AsyncGenerator[StreamEvent, None]:
    try:
        from google import genai as _gai
        from google.genai import types as _gt
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = _gai.Client(api_key=settings.GEMINI_API_KEY)

    contents: list = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        text = m["content"] if isinstance(m["content"], str) else str(m["content"])
        contents.append(_gt.Content(role=role, parts=[_gt.Part(text=text)]))

    genai_tools = None
    if tools:
        declarations = [
            _gt.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["input_schema"],
            )
            for t in tools
        ]
        genai_tools = [_gt.Tool(function_declarations=declarations)]

    config = _gt.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=temperature,
        **({"tools": genai_tools} if genai_tools else {}),
    )

    try:
        async with client.aio.models.stream(
            model=api_model, contents=contents, config=config
        ) as stream_:
            async for chunk in stream_:
                if chunk.text:
                    yield StreamEvent(type="text", content=chunk.text)
                if chunk.candidates:
                    for cand in chunk.candidates:
                        for part in (cand.content.parts or []):
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                yield StreamEvent(
                                    type="tool_start",
                                    tool_name=fc.name,
                                    tool_call_id=fc.name,
                                )
                                yield StreamEvent(
                                    type="tool_done",
                                    tool_name=fc.name,
                                    tool_call_id=fc.name,
                                    tool_input=dict(fc.args) if fc.args else {},
                                )

        usage_meta = getattr(stream_, "usage_metadata", None)
        yield StreamEvent(
            type="done",
            usage={
                "input":  getattr(usage_meta, "prompt_token_count",     0) if usage_meta else 0,
                "output": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
                "model":  api_model,
            },
        )
    except Exception as e:
        yield StreamEvent(type="error", content=str(e))
        raise


# ── Fallback chain ────────────────────────────────────────────────────────────

_FALLBACK_CHAIN: dict[str, str] = {
    # OpenAI → Gemini
    "gpt-4o":                       "gemini-2.0-flash",
    "gpt-4o-mini":                  "gemini-2.0-flash",
    # Gemini → Groq
    "gemini-1.5-pro":               "gemini-2.0-flash",
    "gemini-2.0-flash":             "llama-3.3-70b-versatile",
    "gemini-1.5-flash":             "llama-3.3-70b-versatile",
    # Groq → Cerebras
    "llama-3.3-70b-versatile":      "cerebras/llama-3.3-70b",
    "llama-3.1-8b-instant":         "cerebras/llama3.1-8b",
    "mixtral-8x7b-32768":           "llama-3.3-70b-versatile",
    "gemma2-9b-it":                 "llama-3.1-8b-instant",
    # Cerebras → Mistral
    "cerebras/llama-3.3-70b":       "mistral-small-3.1-24b-instruct",
    "cerebras/llama3.1-8b":         "open-mistral-nemo",
    # Mistral → SambaNova
    "mistral-small-3.1-24b-instruct": "sambanova/deepseek-v3",
    "codestral-2501":               "sambanova/deepseek-v3",
    "open-mistral-nemo":            "sambanova/llama-3.3-70b",
    # SambaNova → Gemini (last resort)
    "sambanova/deepseek-v3":        "gemini-2.0-flash",
    "sambanova/qwen2.5-72b":        "gemini-2.0-flash",
    "sambanova/llama-3.3-70b":      "gemini-2.0-flash",
}


# ── Public API ────────────────────────────────────────────────────────────────

async def stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> AsyncGenerator[StreamEvent, None]:
    """Route to the correct provider; auto-fall-back on failure."""

    entry = MODEL_REGISTRY.get(model)
    if not entry:
        # Heuristic fallback for unknown models
        if model.startswith("gpt"):
            entry = ("openai", model)
        elif model.startswith("gemini"):
            entry = ("gemini", model)
        else:
            entry = ("groq", model)

    provider, api_model = entry

    try:
        if provider == "gemini":
            async for evt in _gemini_stream(
                messages, system, api_model, max_tokens, tools, temperature
            ):
                yield evt

        else:
            pconfig = PROVIDER_CONFIG.get(provider)
            if not pconfig:
                raise ValueError(f"Unknown provider: {provider}")

            base_url, key_attr = pconfig
            api_key = getattr(settings, key_attr, "")
            if not api_key:
                raise RuntimeError(f"{key_attr} is not set")

            # Gate tools per provider/model
            use_tools: list[dict] | None = None
            if tools and provider in _TOOL_CAPABLE:
                if provider == "groq" and not _groq_supports_tools(api_model):
                    use_tools = None
                else:
                    use_tools = tools

            # Groq / SambaNova cap max_tokens
            capped = min(max_tokens, 8192) if provider in ("groq", "sambanova", "cerebras") else max_tokens

            async for evt in _openai_compatible_stream(
                messages, system, api_model, capped, use_tools, temperature,
                api_key=api_key, base_url=base_url, provider=provider,
            ):
                yield evt

    except Exception as exc:
        fallback = _FALLBACK_CHAIN.get(model)
        if fallback and fallback != model:
            yield StreamEvent(
                type="system",
                content=f"Switching to fallback: {fallback}",
            )
            async for evt in stream(messages, system, fallback, max_tokens, tools, temperature):
                yield evt
        else:
            yield StreamEvent(type="error", content=str(exc))
            raise


async def complete(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Non-streaming completion for internal tasks (title gen, summarisation)."""
    parts: list[str] = []
    async for evt in stream(messages, system, model, max_tokens, None, temperature):
        if evt.type == "text":
            parts.append(evt.content)
    return "".join(parts)


# ── Legacy cost helper ────────────────────────────────────────────────────────

def estimate_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0) -> float:
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    total  = (input_tokens  / 1_000_000) * costs["input"]
    total += (output_tokens / 1_000_000) * costs["output"]
    return round(total, 6)
