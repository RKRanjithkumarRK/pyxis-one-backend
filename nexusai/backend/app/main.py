from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine, Base
from app.core.telemetry import setup_telemetry
from app.api.v1 import api_router

logger = logging.getLogger("nexusai")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry()
    logger.info("NexusAI backend starting — environment=%s", settings.ENVIRONMENT)
    logger.info("Available AI providers: %s", settings.available_providers)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "nexusai-backend", "version": "1.0.0"}
