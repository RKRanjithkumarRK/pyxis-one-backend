"""
Multi-provider AI client with automatic fallback.

Priority order: Groq → Google Gemini → Anthropic Claude
Set whichever keys you have — unused providers are skipped.

Free keys (no credit card required):
  GROQ_API_KEY    → https://console.groq.com          (Llama 3.3 70B, 30 RPM free)
  GEMINI_API_KEY  → https://aistudio.google.com/app/apikey  (Gemini 1.5 Flash, 15 RPM free)
  ANTHROPIC_API_KEY → https://console.anthropic.com   (paid, used as last resort)
"""

import asyncio
from typing import AsyncGenerator
from core.config import settings, DEFAULT_MODEL


# ── Groq (OpenAI-compatible, free tier) ──────────────────────────────────

async def _groq_stream(messages: list[dict], system: str, max_tokens: int) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    stream = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        max_tokens=min(max_tokens, 8192),
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _groq_complete(messages: list[dict], system: str, max_tokens: int) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        max_tokens=min(max_tokens, 8192),
    )
    return response.choices[0].message.content or ""


# ── Google Gemini (OpenAI-compatible endpoint, free tier) ─────────────────

async def _gemini_stream(messages: list[dict], system: str, max_tokens: int) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    stream = await client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=msgs,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _gemini_complete(messages: list[dict], system: str, max_tokens: int) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    response = await client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=msgs,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ── Anthropic Claude (paid, last resort) ──────────────────────────────────

async def _anthropic_stream(messages: list[dict], system: str, model: str, max_tokens: int) -> AsyncGenerator[str, None]:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _anthropic_complete(messages: list[dict], system: str, model: str, max_tokens: int) -> str:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── Provider list builder ─────────────────────────────────────────────────

def _stream_providers(messages: list[dict], system: str, model: str, max_tokens: int):
    """Returns list of (name, async_generator_factory) in fallback order."""
    providers = []
    if settings.GROQ_API_KEY:
        providers.append(("Groq", lambda: _groq_stream(messages, system, max_tokens)))
    if settings.GEMINI_API_KEY:
        providers.append(("Gemini", lambda: _gemini_stream(messages, system, max_tokens)))
    if settings.ANTHROPIC_API_KEY:
        providers.append(("Anthropic", lambda: _anthropic_stream(messages, system, model, max_tokens)))
    return providers


def _complete_providers(messages: list[dict], system: str, model: str, max_tokens: int):
    providers = []
    if settings.GROQ_API_KEY:
        providers.append(("Groq", lambda: _groq_complete(messages, system, max_tokens)))
    if settings.GEMINI_API_KEY:
        providers.append(("Gemini", lambda: _gemini_complete(messages, system, max_tokens)))
    if settings.ANTHROPIC_API_KEY:
        providers.append(("Anthropic", lambda: _anthropic_complete(messages, system, model, max_tokens)))
    return providers


# ── Public API (same signatures as before — all callers unchanged) ─────────

async def stream_response(
    messages: list[dict],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    providers = _stream_providers(messages, system, model, max_tokens)
    if not providers:
        raise RuntimeError(
            "No AI provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY."
        )

    last_error: Exception | None = None
    for name, make_gen in providers:
        yielded_any = False
        try:
            async for chunk in make_gen():
                yielded_any = True
                yield chunk
            return  # success
        except Exception as exc:
            last_error = exc
            if yielded_any:
                return  # can't retry after partial stream
            print(f"[ai-fallback] {name} failed: {exc!r} — trying next provider")

    raise last_error or RuntimeError("All AI providers failed")


async def complete_response(
    messages: list[dict],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    providers = _complete_providers(messages, system, model, max_tokens)
    if not providers:
        raise RuntimeError(
            "No AI provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY."
        )

    last_error: Exception | None = None
    for name, call in providers:
        try:
            return await call()
        except Exception as exc:
            last_error = exc
            print(f"[ai-fallback] {name} failed: {exc!r} — trying next provider")

    raise last_error or RuntimeError("All AI providers failed")


async def parallel_complete(requests: list[dict]) -> list[str]:
    tasks = [
        complete_response(
            messages=req.get("messages", []),
            system=req.get("system", ""),
            model=req.get("model", DEFAULT_MODEL),
            max_tokens=req.get("max_tokens", 4096),
        )
        for req in requests
    ]
    return list(await asyncio.gather(*tasks))
