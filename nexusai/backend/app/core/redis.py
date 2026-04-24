from __future__ import annotations
import redis.asyncio as aioredis
from app.core.config import settings

_pool: aioredis.ConnectionPool | None = None


def get_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=50,
            decode_responses=True,
        )
    return _pool


redis_client: aioredis.Redis = aioredis.Redis(connection_pool=get_pool())  # type: ignore[assignment]


async def ping() -> bool:
    try:
        return await redis_client.ping()
    except Exception:
        return False
