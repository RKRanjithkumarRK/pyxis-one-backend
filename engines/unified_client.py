"""
Unified streaming client — Groq · Gemini · OpenAI.

No Anthropic. All providers produce identical StreamEvent types so the
pipeline above this layer is fully provider-agnostic.

Provider detection is by model-name prefix:
  gpt-*            → OpenAI
  o1-* / o3-*      → OpenAI
  llama-* / mixtral-* / gemma-*  → Groq (OpenAI-compatible)
  gemini-*         → Google Gemini (google-genai SDK)
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


# ── OpenAI-compatible streaming (GPT + Groq) ─────────────────────────────────

async def _openai_compatible_stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
    api_key: str,
    base_url: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    from openai import AsyncOpenAI, APIStatusError

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    full_messages = [{"role": "system", "content": system}] + messages

    # Convert unified tool schema → OpenAI function-calling format
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
        "model": model,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    # Groq supports usage in stream; OpenAI requires stream_options
    if base_url is None:                  # OpenAI proper
        kwargs["stream_options"] = {"include_usage": True}
    if oai_tools:
        kwargs["tools"] = oai_tools
        kwargs["tool_choice"] = "auto"

    try:
        stream = await client.chat.completions.create(**kwargs)

        active_tools: dict[int, dict] = {}   # index → {name, id, args}

        async for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk (OpenAI stream_options)
                if chunk.usage:
                    yield StreamEvent(
                        type="done",
                        usage={
                            "input": chunk.usage.prompt_tokens,
                            "output": chunk.usage.completion_tokens,
                            "model": model,
                        },
                    )
                continue

            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            if delta.content:
                yield StreamEvent(type="text", content=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_tools:
                        active_tools[idx] = {
                            "name": tc.function.name or "",
                            "id": tc.id or f"call_{idx}",
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

            elif finish == "stop" and base_url is not None:
                # Groq: no usage chunk, emit done here
                yield StreamEvent(type="done", usage={"model": model})

    except APIStatusError as e:
        yield StreamEvent(type="error", content=str(e), error_code=str(e.status_code))
        raise


# ── Gemini streaming (google-genai SDK) ───────────────────────────────────────

async def _gemini_stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
) -> AsyncGenerator[StreamEvent, None]:
    try:
        from google import genai as _gai
        from google.genai import types as _gt
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        )

    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = _gai.Client(api_key=settings.GEMINI_API_KEY)

    # Build contents list in Gemini format
    contents: list = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        text = m["content"] if isinstance(m["content"], str) else str(m["content"])
        contents.append(_gt.Content(role=role, parts=[_gt.Part(text=text)]))

    # Build tool declarations if any
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
        # google-genai async streaming via context manager
        async with client.aio.models.stream(
            model=model,
            contents=contents,
            config=config,
        ) as stream_:
            async for chunk in stream_:
                # Text token
                if chunk.text:
                    yield StreamEvent(type="text", content=chunk.text)

                # Function call (tool use)
                if chunk.candidates:
                    for cand in chunk.candidates:
                        for part in (cand.content.parts or []):
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                call_id = fc.name  # Gemini has no UUID per call
                                yield StreamEvent(
                                    type="tool_start",
                                    tool_name=fc.name,
                                    tool_call_id=call_id,
                                )
                                yield StreamEvent(
                                    type="tool_done",
                                    tool_name=fc.name,
                                    tool_call_id=call_id,
                                    tool_input=dict(fc.args) if fc.args else {},
                                )

        # Emit usage (available after stream closes)
        usage_meta = getattr(stream_, "usage_metadata", None)
        if usage_meta:
            yield StreamEvent(
                type="done",
                usage={
                    "input":  getattr(usage_meta, "prompt_token_count",     0),
                    "output": getattr(usage_meta, "candidates_token_count", 0),
                    "model":  model,
                },
            )
        else:
            yield StreamEvent(type="done", usage={"model": model})

    except Exception as e:
        yield StreamEvent(type="error", content=str(e))
        raise


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
    provider = _get_provider(model)

    try:
        if provider == "openai":
            async for evt in _openai_compatible_stream(
                messages, system, model, max_tokens, tools, temperature,
                api_key=settings.OPENAI_API_KEY,
            ):
                yield evt

        elif provider == "groq":
            # Groq: tool support only on llama models
            groq_tools = tools if _groq_supports_tools(model) else None
            async for evt in _openai_compatible_stream(
                messages, system, model,
                min(max_tokens, 8192),  # Groq cap
                groq_tools, temperature,
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            ):
                yield evt

        elif provider == "gemini":
            async for evt in _gemini_stream(
                messages, system, model, max_tokens, tools, temperature
            ):
                yield evt

        else:
            raise ValueError(f"No provider found for model: {model}")

    except Exception as exc:
        fallback = _FALLBACK_CHAIN.get(model)
        if fallback and fallback != model:
            yield StreamEvent(
                type="system",
                content=f"Switching to fallback model: {fallback}",
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_provider(model: str) -> str:
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    if model.startswith(("llama", "mixtral", "gemma")):
        return "groq"
    if model.startswith("gemini"):
        return "gemini"
    # Default to Groq (free) if unknown
    return "groq"


def _groq_supports_tools(model: str) -> bool:
    """Groq tool-calling works on llama-3.x models only."""
    return model.startswith("llama")


# Priority: free/fast first, then fallback up quality ladder
_FALLBACK_CHAIN: dict[str, str] = {
    # OpenAI
    "gpt-4o":                   "gemini-2.0-flash",
    "gpt-4o-mini":              "gemini-2.0-flash",
    # Gemini
    "gemini-1.5-pro":           "gemini-2.0-flash",
    "gemini-2.0-flash":         "llama-3.3-70b-versatile",
    "gemini-1.5-flash":         "llama-3.3-70b-versatile",
    # Groq
    "llama-3.3-70b-versatile":  "llama-3.1-8b-instant",
    "mixtral-8x7b-32768":       "llama-3.3-70b-versatile",
    "gemma2-9b-it":             "llama-3.1-8b-instant",
    "llama-3.1-8b-instant":     "gemini-2.0-flash",
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0) -> float:
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    total = (input_tokens  / 1_000_000) * costs["input"]
    total += (output_tokens / 1_000_000) * costs["output"]
    return round(total, 6)
