from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.config import settings
from core.database import create_all_tables
import uvicorn

from routers import chat
from routers import conversations as conversations_router
from routers import files as files_router
from routers import features


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    yield


app = FastAPI(
    title="Pyxis Backend",
    description="Multi-provider AI backend — 6 providers, 18 models",
    version="3.0.0",
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

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat.router,                  prefix="/api")
app.include_router(conversations_router.router,  prefix="/api")
app.include_router(files_router.router,          prefix="/api")
app.include_router(features.router,              prefix="/api")

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "pyxis-backend",
        "version": "3.0.0",
        "providers": {
            "groq":      bool(settings.GROQ_API_KEY),
            "gemini":    bool(settings.GEMINI_API_KEY),
            "cerebras":  bool(settings.CEREBRAS_API_KEY),
            "mistral":   bool(settings.MISTRAL_API_KEY),
            "sambanova": bool(settings.SAMBANOVA_API_KEY),
            "openai":    bool(settings.OPENAI_API_KEY),
        },
    }


@app.get("/")
async def root():
    return {
        "message": "Pyxis Backend v3.0 — running",
        "docs":    "/docs",
        "health":  "/health",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(settings.PORT), reload=False)
