from __future__ import annotations
import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_db
from app.core.redis import ping as redis_ping
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health():
    return {
        "status": "ok",
        "service": "nexusai-backend",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
    }


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    t0 = time.perf_counter()

    db_ok = False
    db_latency_ms = None
    try:
        await db.execute(text("SELECT 1"))
        db_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        db_ok = True
    except Exception as exc:
        db_err = str(exc)
    else:
        db_err = None

    t1 = time.perf_counter()
    redis_ok = await redis_ping()
    redis_latency_ms = round((time.perf_counter() - t1) * 1000, 2)

    all_ok = db_ok and redis_ok
    return {
        "status": "ready" if all_ok else "degraded",
        "checks": {
            "postgres": {
                "ok": db_ok,
                "latency_ms": db_latency_ms,
                "error": db_err if not db_ok else None,
            },
            "redis": {
                "ok": redis_ok,
                "latency_ms": redis_latency_ms,
            },
        },
        "providers": settings.available_providers,
    }


@router.get("/live")
async def liveness():
    return {"alive": True}
