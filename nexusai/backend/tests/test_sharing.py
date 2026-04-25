"""Phase 16 — Sharing, search, export endpoint tests."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


def _user():
    u = MagicMock()
    u.id = "00000000-0000-0000-0000-000000000030"
    u.email = "user@nexusai.dev"
    return u


@pytest.mark.asyncio
async def test_share_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/share/conversations/some-id")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_nonexistent_share(client: AsyncClient):
    with patch("app.api.v1.sharing.redis_client") as mock_redis:
        mock_redis.get = AsyncMock(return_value=None)
        resp = await client.get("/api/v1/share/nonexistent-token-xyz")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_existing_share(client: AsyncClient):
    payload = json.dumps({
        "title": "Test Chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "shared_at": "2026-01-01T00:00:00Z",
    })
    with patch("app.api.v1.sharing.redis_client") as mock_redis:
        mock_redis.get = AsyncMock(return_value=payload.encode())
        resp = await client.get("/api/v1/share/valid-token-abc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Chat"


@pytest.mark.asyncio
async def test_search_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/search?q=hello")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_export_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/export/request")
    assert resp.status_code in (401, 403)
