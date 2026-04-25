from __future__ import annotations
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.core.database import engine, AsyncSessionLocal
from app.core.telemetry import setup_telemetry
from app.api.v1 import api_router

logger = logging.getLogger("nexusai")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_UNAUTHENTICATED],
    storage_uri=settings.REDIS_URL,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry()
    logger.info("NexusAI backend starting — environment=%s", settings.ENVIRONMENT)
    logger.info("Available AI providers: %s", settings.available_providers)

    from app.services.agents.loader import load_builtin_agents
    async with AsyncSessionLocal() as db:
        count = await load_builtin_agents(db)
        logger.info("Built-in agents ready: %d", count)

    yield
    logger.info("NexusAI backend shutting down")
    await engine.dispose()


app = FastAPI(
    title="NexusAI API",
    version="1.0.0",
    description="NexusAI — all-in-one AI platform",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(api_router)


@app.websocket("/ws/canvas/{doc_id}")
async def canvas_websocket(websocket: WebSocket, doc_id: str):
    from app.websocket.canvas import handle_canvas_ws
    await handle_canvas_ws(websocket, doc_id)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "nexusai-backend", "version": "1.0.0"}
