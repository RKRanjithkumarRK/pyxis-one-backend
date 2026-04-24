"""Usage and cost tracking endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import require_bearer, decode_token
from app.core.redis import redis_client

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/me")
async def my_usage(credentials: HTTPAuthorizationCredentials = Depends(require_bearer)):
    """Return the authenticated user's accumulated AI spending."""
    token_payload = decode_token(credentials.credentials)
    user_id = str(token_payload["sub"])

    daily = await redis_client.get(f"cost:user:{user_id}:daily")
    monthly = await redis_client.get(f"cost:user:{user_id}:monthly")

    return {
        "daily_cost_usd": float(daily) if daily else 0.0,
        "monthly_cost_usd": float(monthly) if monthly else 0.0,
    }
