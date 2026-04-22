"""
Unified streaming client for OpenAI (GPT-4o) and Anthropic (Claude).

Both providers produce the same StreamEvent types so the pipeline is
provider-agnostic above this layer.
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, Any
from core.config import settings, MODEL_COSTS


# ── Event model ──────────────────────────────────────────────────────────────

@dataclass
class StreamEvent:
    type: str          # text | thinking | tool_start | tool_delta | tool_done | done | error | system
    content: str = ""
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_call_id: str | None = None
    usage: dict | None = None
    error_code: str | None = None

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.__dict__, default=str)}\n\n"


# ── Anthropic streaming ───────────────────────────────────────────────────────

async def _anthropic_stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
) -> AsyncGenerator[StreamEvent, None]:
    from anthropic import AsyncAnthropic, APIStatusError

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Inject cache_control on system prompt — saves ~70% cost on repeated calls
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]

    # Also cache the first large user turn if it's a long document
    cached_messages = list(messages)
    if cached_messages and len(cached_messages[0].get("content", "")) > 2000:
        first = cached_messages[0]
        cached_messages[0] = {
            "role": first["role"],
            "content": [
                {
                    "type": "text",
                    "text": first["content"] if isinstance(first["content"], str) else str(first["content"]),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks,
        "messages": cached_messages,
        "temperature": temperature,
        "betas": ["prompt-caching-2024-07-31"],
    }
    if tools:
        # Convert unified tool format to Anthropic format
        kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    try:
        async with client.messages.stream(**kwargs) as stream:
            current_tool_name: str | None = None
            current_tool_id: str | None = None

            async for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_name = block.name
                        current_tool_id = block.id
                        yield StreamEvent(
                            type="tool_start",
                            tool_name=block.name,
                            tool_call_id=block.id,
                        )
                    elif block.type == "thinking":
                        pass  # thinking block opened

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta" and delta.text:
                        yield StreamEvent(type="text", content=delta.text)
                    elif delta.type == "thinking_delta" and delta.thinking:
                        yield StreamEvent(type="thinking", content=delta.thinking)
                    elif delta.type == "input_json_delta":
                        yield StreamEvent(
                            type="tool_delta",
                            content=delta.partial_json,
                            tool_name=current_tool_name,
                            tool_call_id=current_tool_id,
                        )

                elif etype == "content_block_stop":
                    if current_tool_name:
                        yield StreamEvent(
                            type="tool_done",
                            tool_name=current_tool_name,
                            tool_call_id=current_tool_id,
                        )
                        current_tool_name = None
                        current_tool_id = None

                elif etype == "message_stop":
                    final = stream.get_final_message()
                    usage_data = {
                        "input": final.usage.input_tokens,
                        "output": final.usage.output_tokens,
                        "cache_read": getattr(final.usage, "cache_read_input_tokens", 0),
                        "cache_created": getattr(final.usage, "cache_creation_input_tokens", 0),
                    }
                    # Attach stop_reason for tool_use detection
                    usage_data["stop_reason"] = final.stop_reason
                    usage_data["model"] = model
                    yield StreamEvent(type="done", usage=usage_data)

    except APIStatusError as e:
        yield StreamEvent(type="error", content=str(e), error_code=str(e.status_code))
        raise


# ── OpenAI streaming ──────────────────────────────────────────────────────────

async def _openai_stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict] | None,
    temperature: float,
) -> AsyncGenerator[StreamEvent, None]:
    from openai import AsyncOpenAI, APIStatusError

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    full_messages = [{"role": "system", "content": system}] + messages

    # Convert unified tool format to OpenAI format
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
        "stream_options": {"include_usage": True},
    }
    if oai_tools:
        kwargs["tools"] = oai_tools
        kwargs["tool_choice"] = "auto"

    try:
        stream = await client.chat.completions.create(**kwargs)

        # Per-tool accumulation: index → {name, args_buffer, id}
        active_tools: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                # usage-only chunk
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
                            "id": tc.id or "",
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
                for idx, tool in active_tools.items():
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

            elif finish == "stop":
                # usage comes in the next chunk (stream_options)
                pass

    except APIStatusError as e:
        yield StreamEvent(type="error", content=str(e), error_code=str(e.status_code))
        raise


# ── Groq fallback (OpenAI-compatible, no tools) ───────────────────────────────

async def _groq_stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> AsyncGenerator[StreamEvent, None]:
    from openai import AsyncOpenAI, APIStatusError

    client = AsyncOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    full_messages = [{"role": "system", "content": system}] + messages

    try:
        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            max_tokens=min(max_tokens, 8192),
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamEvent(type="text", content=chunk.choices[0].delta.content)
        yield StreamEvent(type="done", usage={"model": "llama-3.3-70b-versatile"})
    except APIStatusError as e:
        yield StreamEvent(type="error", content=str(e), error_code=str(e.status_code))
        raise


# ── Public streaming API ───────────────────────────────────────────────────────

async def stream(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Route to the correct provider based on model name.
    Falls back through providers if primary fails.
    """
    provider = _get_provider(model)

    try:
        if provider == "anthropic":
            async for evt in _anthropic_stream(messages, system, model, max_tokens, tools, temperature):
                yield evt
        elif provider == "openai":
            async for evt in _openai_stream(messages, system, model, max_tokens, tools, temperature):
                yield evt
        elif provider == "groq":
            async for evt in _groq_stream(messages, system, model, max_tokens, temperature):
                yield evt
        else:
            raise ValueError(f"Unknown model: {model}")

    except Exception as exc:
        # Try fallback
        fallback = _get_fallback(model)
        if fallback and fallback != model:
            yield StreamEvent(
                type="system",
                content=f"Primary model unavailable, switching to {fallback}...",
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
    """Non-streaming completion for internal tasks (title gen, summarization)."""
    result = []
    async for evt in stream(messages, system, model, max_tokens, None, temperature):
        if evt.type == "text":
            result.append(evt.content)
    return "".join(result)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if model.startswith("llama") or model.startswith("mixtral"):
        return "groq"
    if model.startswith("gemini"):
        return "gemini"
    return "anthropic"


_FALLBACK_CHAIN: dict[str, str] = {
    "claude-opus-4-7":           "claude-sonnet-4-6",
    "claude-sonnet-4-6":         "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "gpt-4o-mini",
    "gpt-4o":                    "gpt-4o-mini",
    "gpt-4o-mini":               "claude-haiku-4-5-20251001",
}


def _get_fallback(model: str) -> str | None:
    return _FALLBACK_CHAIN.get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0) -> float:
    costs = MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
    total = (input_tokens / 1_000_000) * costs["input"]
    total += (output_tokens / 1_000_000) * costs["output"]
    if cache_read and "cache_read" in costs:
        total += (cache_read / 1_000_000) * costs["cache_read"]
    return round(total, 6)
