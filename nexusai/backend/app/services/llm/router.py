from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Optional
from dataclasses import dataclass

from litellm import acompletion, aembedding
from litellm.exceptions import RateLimitError, APIError, Timeout

from app.core.config import settings
from app.core.redis import redis_client
from app.core.telemetry import tracer


@dataclass
class ModelRoute:
    provider: str
    model_id: str
    litellm_id: str
    vision: bool
    tool_use: bool
    max_input: int
    cost_in_per_1k: float
    cost_out_per_1k: float


ROUTES: dict[str, ModelRoute] = {
    "claude-sonnet-4": ModelRoute("anthropic", "claude-sonnet-4", "claude-sonnet-4-20250514", True,  True,  200_000, 0.003,    0.015),
    "claude-opus-4":   ModelRoute("anthropic", "claude-opus-4",   "claude-opus-4-20250514",   True,  True,  200_000, 0.015,    0.075),
    "gpt-4o":          ModelRoute("openai",    "gpt-4o",          "gpt-4o",                    True,  True,  128_000, 0.0025,   0.010),
    "gpt-4o-mini":     ModelRoute("openai",    "gpt-4o-mini",     "gpt-4o-mini",              True,  True,  128_000, 0.00015,  0.0006),
    "gemini-2-pro":    ModelRoute("google",    "gemini-2-pro",    "gemini/gemini-2.0-pro-exp", True,  True,  2_000_000, 0.00125, 0.005),
    "gemini-2-flash":  ModelRoute("google",    "gemini-2-flash",  "gemini/gemini-2.0-flash",  True,  True,  1_000_000, 0.000075, 0.0003),
    "groq-llama-70b":  ModelRoute("groq",      "groq-llama-70b",  "groq/llama-3.3-70b-versatile", False, True, 128_000, 0.00059, 0.00079),
    "mistral-large":   ModelRoute("mistral",   "mistral-large",   "mistral/mistral-large-latest", False, True, 128_000, 0.002,  0.006),
    "cerebras-llama":  ModelRoute("cerebras",  "cerebras-llama",  "cerebras/llama3.1-70b",    False, False, 128_000, 0.0006,   0.0006),
    "sambanova-llama": ModelRoute("sambanova", "sambanova-llama", "sambanova/Meta-Llama-3.1-70B-Instruct", False, False, 8_000, 0.0006, 0.0012),
}

FALLBACK_CHAIN: dict[str, list[str]] = {
    "claude-sonnet-4": ["claude-opus-4", "gpt-4o"],
    "claude-opus-4":   ["claude-sonnet-4", "gpt-4o"],
    "gpt-4o":          ["claude-sonnet-4", "gemini-2-pro"],
    "gpt-4o-mini":     ["gpt-4o", "gemini-2-flash"],
    "gemini-2-pro":    ["claude-sonnet-4", "gpt-4o"],
    "gemini-2-flash":  ["gemini-2-pro", "gpt-4o-mini"],
    "groq-llama-70b":  ["gemini-2-flash", "gpt-4o-mini"],
    "mistral-large":   ["claude-sonnet-4", "gpt-4o"],
    "cerebras-llama":  ["groq-llama-70b", "gpt-4o-mini"],
    "sambanova-llama": ["groq-llama-70b", "gpt-4o-mini"],
}

_PROVIDER_ENV: dict[str, str] = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "google":     "GOOGLE_API_KEY",
    "groq":       "GROQ_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "cerebras":   "CEREBRAS_API_KEY",
    "sambanova":  "SAMBANOVA_API_KEY",
}


def _configure_litellm() -> None:
    import litellm
    import os
    key_map = {
        "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY,
        "OPENAI_API_KEY":    settings.OPENAI_API_KEY,
        "GOOGLE_API_KEY":    settings.GOOGLE_API_KEY,
        "GROQ_API_KEY":      settings.GROQ_API_KEY,
        "MISTRAL_API_KEY":   settings.MISTRAL_API_KEY,
        "CEREBRAS_API_KEY":  settings.CEREBRAS_API_KEY,
        "SAMBANOVA_API_KEY": settings.SAMBANOVA_API_KEY,
    }
    for env_key, val in key_map.items():
        if val:
            os.environ[env_key] = val
    litellm.drop_params = True  # ignore unsupported params per model


_configure_litellm()


def available_models() -> list[dict]:
    return [
        {
            "id": r.model_id,
            "provider": r.provider,
            "vision": r.vision,
            "tool_use": r.tool_use,
            "max_input": r.max_input,
            "cost_in_per_1k": r.cost_in_per_1k,
            "cost_out_per_1k": r.cost_out_per_1k,
        }
        for r in ROUTES.values()
        if settings.available_providers and r.provider in settings.available_providers
    ] or list(
        # Return all routes if no providers configured (dev/test mode)
        {
            "id": r.model_id,
            "provider": r.provider,
            "vision": r.vision,
            "tool_use": r.tool_use,
            "max_input": r.max_input,
            "cost_in_per_1k": r.cost_in_per_1k,
            "cost_out_per_1k": r.cost_out_per_1k,
        }
        for r in ROUTES.values()
    )


async def get_model_latency(model_id: str) -> int | None:
    """Returns P50 latency in ms from rolling window of 50 calls."""
    key = f"latency:{model_id}"
    raw = await redis_client.lrange(key, 0, -1)
    if not raw:
        return None
    values = sorted(int(v) for v in raw)
    return values[len(values) // 2]


async def stream_chat(
    model_id: str,
    messages: list[dict],
    *,
    tools: Optional[list[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    user_id: str,
    conversation_id: str,
) -> AsyncIterator[dict]:
    if model_id not in ROUTES:
        # Fallback to first available model
        model_id = next(iter(ROUTES))

    route = ROUTES[model_id]

    with tracer.start_as_current_span("llm.stream_chat") as span:
        span.set_attribute("llm.provider", route.provider)
        span.set_attribute("llm.model", route.model_id)
        span.set_attribute("user.id", user_id)
        span.set_attribute("conversation.id", conversation_id)

        attempt = 0
        tried = [model_id]
        current = route

        while True:
            try:
                t0 = time.perf_counter()
                stream = await acompletion(
                    model=current.litellm_id,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata={"user_id": user_id, "conversation_id": conversation_id},
                )
                first_token_time: float | None = None
                usage = {"prompt_tokens": 0, "completion_tokens": 0}

                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                        await _record_latency(current.model_id, first_token_time)
                        span.set_attribute("llm.first_token_ms", int(first_token_time * 1000))

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if delta.content:
                        yield {"type": "token", "content": delta.content}
                    if getattr(delta, "tool_calls", None):
                        yield {"type": "tool_call", "tool_calls": [tc.model_dump() for tc in delta.tool_calls]}
                    if getattr(chunk, "usage", None):
                        usage["prompt_tokens"] = chunk.usage.prompt_tokens or 0
                        usage["completion_tokens"] = chunk.usage.completion_tokens or 0

                yield {"type": "done", "usage": usage, "model": current.model_id}
                await _record_cost(user_id, current, usage)
                return

            except (RateLimitError, APIError, Timeout, asyncio.TimeoutError) as exc:
                chain = FALLBACK_CHAIN.get(model_id, [])
                next_model = next((m for m in chain if m not in tried), None)
                if next_model is None or attempt >= 3:
                    yield {"type": "error", "message": str(exc), "model": current.model_id}
                    return
                tried.append(next_model)
                current = ROUTES[next_model]
                attempt += 1
                yield {"type": "fallback", "from": tried[-2], "to": next_model}
                await asyncio.sleep(min(2 ** attempt, 8))

            except Exception as exc:
                yield {"type": "error", "message": str(exc), "model": current.model_id}
                return


async def _record_latency(model_id: str, seconds: float) -> None:
    key = f"latency:{model_id}"
    await redis_client.lpush(key, int(seconds * 1000))
    await redis_client.ltrim(key, 0, 49)
    await redis_client.expire(key, 86400 * 7)


async def _record_cost(user_id: str, route: ModelRoute, usage: dict) -> None:
    cost = (
        (usage.get("prompt_tokens", 0) / 1000) * route.cost_in_per_1k
        + (usage.get("completion_tokens", 0) / 1000) * route.cost_out_per_1k
    )
    daily_key = f"cost:user:{user_id}:daily"
    monthly_key = f"cost:user:{user_id}:monthly"
    await redis_client.incrbyfloat(daily_key, cost)
    await redis_client.expire(daily_key, 86400 * 2)   # rolls over after 2 days
    await redis_client.incrbyfloat(monthly_key, cost)
    await redis_client.expire(monthly_key, 86400 * 40)  # covers ~1 billing cycle


async def embed(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    resp = await aembedding(model=model, input=texts)
    return [d["embedding"] for d in resp.data]


async def litellm_complete(model: str, messages: list[dict], **kwargs) -> str:
    """Non-streaming single-turn completion with Redis cache. Returns assistant message content."""
    import hashlib, json as _json
    cache_key = "llm:cache:" + hashlib.sha256(
        _json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True).encode()
    ).hexdigest()

    cached = await redis_client.get(cache_key)
    if cached:
        return cached.decode()

    with tracer.start_as_current_span("llm.complete") as span:
        span.set_attribute("llm.model", model)
        resp = await acompletion(model=model, messages=messages, stream=False, **kwargs)
        content = resp.choices[0].message.content or ""

    await redis_client.setex(cache_key, 3600, content)  # cache for 1 hour
    return content
