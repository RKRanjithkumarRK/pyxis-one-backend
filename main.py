from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.config import settings
from core.database import create_all_tables, AsyncSessionLocal
from core.models import Session as PyxisSession
from sqlalchemy import select
import uvicorn

from routers import chat, trident, assessment, parliament, features, voice
from routers import vault as vault_router
from routers import analytics

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
    description="Advanced AI learning companion backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(trident.router, prefix="/api")
app.include_router(assessment.router, prefix="/api")
app.include_router(parliament.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(voice.router, prefix="/api")
app.include_router(vault_router.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "pyxis-one-backend"}


@app.get("/")
async def root():
    return {
        "message": "Pyxis One Backend — Active and Online",
        "version": "1.0.0",
        "engines": [
            "psyche", "forge", "curriculum", "oracle", "nemesis",
            "helix", "tides", "gravity", "dark_knowledge", "mirror",
            "civilization", "symphony", "vault", "blind_spots",
            "precognition", "shadow_self",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=False,
    )
