"""Shared pytest fixtures for NexusAI backend tests."""
from __future__ import annotations
import os
import pytest
import pytest_asyncio

# Force test environment before any app imports
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://nexusai:nexusai_dev@localhost:5432/nexusai_test",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault(
    "SECRET_KEY", "test_secret_key_minimum_32_characters_long_for_tests"
)

from httpx import AsyncClient, ASGITransport  # noqa: E402
from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
