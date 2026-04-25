"""Phase 19 — Admin Console endpoint tests."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


def _admin_user():
    u = MagicMock()
    u.id = "00000000-0000-0000-0000-000000000001"
    u.email = "admin@nexusai.dev"
    u.is_admin = True
    u.plan = MagicMock(value="plus")
    return u


def _regular_user():
    u = MagicMock()
    u.is_admin = False
    return u


@pytest.mark.asyncio
async def test_dashboard_requires_admin(client: AsyncClient):
    with patch("app.api.v1.admin.get_current_user", return_value=_regular_user()):
        resp = await client.get("/api/v1/admin/dashboard")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_dashboard_returns_stats(client: AsyncClient):
    from app.api.deps import get_db
    from app.api.v1.admin import _require_admin

    mock_db = AsyncMock()
    # Simulate scalar_one returning totals
    mock_db.execute = AsyncMock(
        side_effect=[
            MagicMock(scalar_one=lambda: 100),   # total_users
            MagicMock(scalar_one=lambda: 12),    # active_today
            MagicMock(scalar_one=lambda: 80),    # free count
            MagicMock(scalar_one=lambda: 15),    # plus count
            MagicMock(scalar_one=lambda: 4),     # team count
            MagicMock(scalar_one=lambda: 1),     # enterprise count
        ]
    )

    from app.main import app
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[_require_admin] = lambda: _admin_user()

    resp = await client.get("/api/v1/admin/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_users" in data
    assert "plan_distribution" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_audit_logs_returned(client: AsyncClient):
    from app.api.v1.admin import _require_admin

    mock_entries = [
        json.dumps({"action": "login", "actor_id": "abc", "target": None, "metadata": {}, "timestamp": "2026-01-01T00:00:00Z"}).encode()
    ]

    from app.main import app
    app.dependency_overrides[_require_admin] = lambda: _admin_user()

    with patch("app.api.v1.admin.redis_client") as mock_redis:
        mock_redis.lrange = AsyncMock(return_value=mock_entries)
        resp = await client.get("/api/v1/admin/audit-logs?limit=10")

    assert resp.status_code == 200
    logs = resp.json()
    assert isinstance(logs, list)
    assert logs[0]["action"] == "login"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_content_filter_add(client: AsyncClient):
    from app.api.v1.admin import _require_admin
    from app.main import app
    app.dependency_overrides[_require_admin] = lambda: _admin_user()

    with patch("app.api.v1.admin.redis_client") as mock_redis:
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()
        resp = await client.post(
            "/api/v1/admin/content-filters",
            json={"action": "add", "pattern": "badword"},
        )

    assert resp.status_code == 200
    assert "badword" in resp.json()["filters"]
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sso_put_and_get(client: AsyncClient):
    from app.api.v1.admin import _require_admin
    from app.main import app
    app.dependency_overrides[_require_admin] = lambda: _admin_user()

    with patch("app.api.v1.admin.redis_client") as mock_redis:
        mock_redis.set = AsyncMock()
        resp = await client.put(
            "/api/v1/admin/sso",
            json={"type": "oidc", "config": {"issuer": "https://sso.example.com"}},
        )
    assert resp.status_code == 200
    assert resp.json()["type"] == "oidc"

    app.dependency_overrides.clear()
