"""Phase 2 auth endpoint tests.
These tests verify all auth routes without hitting real external providers.
The DB/Redis/real Postgres integration tests require docker-compose up.
For CI without infra, mark the DB-dependent tests with @pytest.mark.integration.
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_register_missing_fields(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_weak_password(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "test@nexusai.dev",
        "password": "weakpass",
        "name": "Test User",
    })
    assert resp.status_code == 422  # missing uppercase + digit


@pytest.mark.asyncio
async def test_login_missing_fields(client: AsyncClient):
    resp = await client.post("/api/v1/auth/login", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_magic_link_no_sendgrid(client: AsyncClient):
    """Magic link endpoint should accept any email and return 202 (email sending is best-effort)."""
    with patch("app.services.auth.service.UserRepository.get_by_email", new_callable=AsyncMock) as mock_get, \
         patch("app.services.auth.service.UserRepository.create", new_callable=AsyncMock) as mock_create, \
         patch("app.services.auth.service.send_magic_link", new_callable=AsyncMock, return_value=True):

        mock_user = MagicMock()
        mock_user.id = "00000000-0000-0000-0000-000000000001"
        mock_user.name = "Test"
        mock_get.return_value = mock_user
        mock_create.return_value = mock_user

        resp = await client.post("/api/v1/auth/magic-link", json={"email": "user@example.com"})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_guest_session(client: AsyncClient):
    with patch("app.services.auth.service.UserRepository.create_guest", new_callable=AsyncMock) as mock_create:
        mock_user = MagicMock()
        mock_user.id = "00000000-0000-0000-0000-000000000002"
        mock_create.return_value = mock_user

        with patch("app.core.redis.redis_client.setex", new_callable=AsyncMock):
            resp = await client.post("/api/v1/auth/guest")

    assert resp.status_code == 201
    data = resp.json()
    assert "guest_id" in data
    assert "token" in data
    assert data["messages_remaining"] == 10


@pytest.mark.asyncio
async def test_logout_requires_token(client: AsyncClient):
    resp = await client.post("/api/v1/auth/logout")
    assert resp.status_code == 403  # SlowAPI/bearer returns 403 without credentials


@pytest.mark.asyncio
async def test_me_requires_token(client: AsyncClient):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_oauth_unsupported_provider(client: AsyncClient):
    resp = await client.post("/api/v1/auth/oauth/callback", json={
        "provider": "twitter",
        "code": "abc",
        "redirect_uri": "http://localhost:3000",
    })
    assert resp.status_code == 422  # pydantic Literal check rejects twitter
