"""
Token-bucket rate limiter per user.
Works with Redis (Upstash) when available; degrades to in-memory when not.
"""

from __future__ import annotations
import time
import asyncio
from collections import defaultdict
from core.config import settings, TIER_LIMITS


class RateLimitError(Exception):
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s.")


# ── In-memory fallback (single instance) ─────────────────────────────────────

_in_memory_rpm: dict[str, list[float]] = defaultdict(list)
_in_memory_lock = asyncio.Lock()


async def _check_in_memory(user_id: str, tier: str) -> None:
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    now = time.time()
    window_start = now - 60.0

    async with _in_memory_lock:
        # Remove timestamps outside the 1-minute window
        _in_memory_rpm[user_id] = [
            t for t in _in_memory_rpm[user_id] if t > window_start
        ]
        current_count = len(_in_memory_rpm[user_id])

        if current_count >= limits["rpm"]:
            oldest = _in_memory_rpm[user_id][0]
            retry_after = int(60 - (now - oldest)) + 1
            raise RateLimitError(retry_after=max(1, retry_after))

        _in_memory_rpm[user_id].append(now)


# ── Redis-backed limiter (production) ─────────────────────────────────────────

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None and settings.REDIS_URL:
        try:
            import redis.asyncio as aioredis
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
        except Exception:
            _redis_client = None
    return _redis_client


async def _check_redis(user_id: str, tier: str) -> None:
    client = _get_redis()
    if client is None:
        return

    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    key = f"rl:{user_id}:rpm:{int(time.time() // 60)}"

    try:
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 120)
        results = await pipe.execute()
        current = results[0]

        if current > limits["rpm"]:
            retry_after = 60 - (int(time.time()) % 60)
            raise RateLimitError(retry_after=max(1, retry_after))
    except RateLimitError:
        raise
    except Exception:
        pass  # Redis failure → silently allow (don't block users)


# ── Public API ────────────────────────────────────────────────────────────────

async def check(user_id: str, tier: str = "free") -> None:
    """Raise RateLimitError if user exceeds their tier's rate limit."""
    if settings.REDIS_URL:
        await _check_redis(user_id, tier)
    else:
        await _check_in_memory(user_id, tier)
