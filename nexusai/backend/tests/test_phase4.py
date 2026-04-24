"""Phase 4 — Multi-provider + Compare endpoint tests."""
from __future__ import annotations
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_usage_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/usage/me")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_usage_returns_zeroes_for_new_user(client: AsyncClient):
    from app.core.security import create_access_token
    from unittest.mock import AsyncMock, patch

    token = create_access_token("00000000-0000-0000-0000-000000000002")

    with patch("app.core.redis.redis_client.get", new_callable=AsyncMock, return_value=None):
        resp = await client.get(
            "/api/v1/usage/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["daily_cost_usd"] == 0.0
    assert data["monthly_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_compare_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/chat/compare", json={
        "models": ["claude-sonnet-4", "gpt-4o"],
        "message": "hello",
    })
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_compare_rejects_single_model(client: AsyncClient):
    from app.core.security import create_access_token
    token = create_access_token("00000000-0000-0000-0000-000000000002")
    resp = await client.post(
        "/api/v1/chat/compare",
        json={"models": ["claude-sonnet-4"], "message": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_compare_rejects_four_models(client: AsyncClient):
    from app.core.security import create_access_token
    token = create_access_token("00000000-0000-0000-0000-000000000002")
    resp = await client.post(
        "/api/v1/chat/compare",
        json={
            "models": ["claude-sonnet-4", "gpt-4o", "gemini-2-flash", "mistral-large"],
            "message": "hi",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_compare_rejects_empty_message(client: AsyncClient):
    from app.core.security import create_access_token
    token = create_access_token("00000000-0000-0000-0000-000000000002")
    resp = await client.post(
        "/api/v1/chat/compare",
        json={"models": ["claude-sonnet-4", "gpt-4o"], "message": "   "},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422
