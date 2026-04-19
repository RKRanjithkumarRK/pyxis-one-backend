import asyncio
from typing import AsyncGenerator
from anthropic import AsyncAnthropic
from core.config import settings, DEFAULT_MODEL

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


async def stream_response(
    messages: list[dict],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    client = get_client()
    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def complete_response(
    messages: list[dict],
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    client = get_client()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return response.content[0].text


async def parallel_complete(
    requests: list[dict],
) -> list[str]:
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
