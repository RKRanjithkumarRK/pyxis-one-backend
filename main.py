from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.config import settings
from core.database import create_all_tables, AsyncSessionLocal
from core.models import Session as PyxisSession
from core.middleware.rate_limiter import RateLimitError
from sqlalchemy import select
import uvicorn

from routers import chat, trident, assessment, parliament, features, voice
from routers import vault as vault_router
from routers import analytics
from routers import conversations as conversations_router
from routers import files as files_router

scheduler = AsyncIOScheduler()


async def _scheduled_curriculum_rewrite() -> None:
    from engines.curriculum import curriculum_engine
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(PyxisSession).limit(50))
            sessions = result.scalars().all()
        for session in sessions:
            try:
                await curriculum_engine.rewrite_curriculum(session.id)
            except Exception:
                pass
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    scheduler.add_job(
        _scheduled_curriculum_rewrite,
        "interval",
        seconds=60,
        id="curriculum_rewrite",
        replace_existing=True,
    )
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(
    title="Pyxis One Backend",
    description="Advanced AI product backend — ChatGPT + Claude parity",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit", "retry_after": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat.router, prefix="/api")
app.include_router(conversations_router.router, prefix="/api")
app.include_router(files_router.router, prefix="/api")
app.include_router(trident.router, prefix="/api")
app.include_router(assessment.router, prefix="/api")
app.include_router(parliament.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(vault_router.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from core.middleware.circuit_breaker import status as cb_status
    key_anthropic = bool(settings.ANTHROPIC_API_KEY)
    key_openai = bool(settings.OPENAI_API_KEY)
    key_groq = bool(settings.GROQ_API_KEY)
    return {
        "status": "ok",
        "service": "pyxis-one-backend",
        "version": "2.0.0",
        "providers": {
            "anthropic": key_anthropic,
            "openai": key_openai,
            "groq": key_groq,
            "brave_search": bool(settings.BRAVE_SEARCH_API_KEY),
            "e2b": bool(settings.E2B_API_KEY),
        },
        "circuit_breakers": cb_status(),
    }


@app.get("/")
async def root():
    return {
        "message": "Pyxis One Backend — v2.0 Production",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=False,
    )
