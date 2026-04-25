"""Phase 17 — Settings & BYOK endpoint tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


def _user():
    u = MagicMock()
    u.id = "00000000-0000-0000-0000-000000000020"
    u.email = "user@nexusai.dev"
    u.name = "Test User"
    u.plan = MagicMock(value="plus")
    u.avatar_url = None
    u.created_at = MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z")
    return u


@pytest.mark.asyncio
async def test_get_profile_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/settings/profile")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_profile(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _user()

    resp = await client.get("/api/v1/settings/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == "user@nexusai.dev"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_byok_set_and_get(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _user()

    with patch("app.api.v1.settings.redis_client") as mock_redis:
        mock_redis.set = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"encrypted_blob")

        # Store a key
        resp = await client.post(
            "/api/v1/settings/byok",
            json={"provider": "openai", "api_key": "sk-test-1234567890abcdef"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "stored"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_totp_setup(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _user()

    with patch("app.api.v1.settings.redis_client") as mock_redis:
        mock_redis.setex = AsyncMock()
        resp = await client.post("/api/v1/settings/2fa/setup")

    assert resp.status_code == 200
    data = resp.json()
    assert "secret" in data
    assert "otpauth_url" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sessions_list(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _user()

    with patch("app.api.v1.settings.redis_client") as mock_redis:
        mock_redis.keys = AsyncMock(return_value=[b"session:user:00000000-0000-0000-0000-000000000020:abc123"])
        mock_redis.ttl = AsyncMock(return_value=86400)
        resp = await client.get("/api/v1/settings/sessions")

    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    app.dependency_overrides.clear()
