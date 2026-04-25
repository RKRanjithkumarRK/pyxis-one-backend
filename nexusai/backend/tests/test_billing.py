"""Phase 18 — Billing endpoint tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


def _free_user():
    u = MagicMock()
    u.id = "00000000-0000-0000-0000-000000000010"
    u.email = "user@nexusai.dev"
    u.plan = MagicMock(value="free")
    return u


@pytest.mark.asyncio
async def test_list_plans(client: AsyncClient):
    resp = await client.get("/api/v1/billing/plans")
    assert resp.status_code == 200
    data = resp.json()
    assert "free" in data
    assert "plus" in data
    assert "team" in data
    assert "enterprise" in data


@pytest.mark.asyncio
async def test_subscription_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/billing/subscription")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_subscription_returns_plan(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _free_user()

    resp = await client.get("/api/v1/billing/subscription")
    assert resp.status_code == 200
    data = resp.json()
    assert data["plan"] == "free"
    assert "features" in data

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_checkout_no_stripe_key(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _free_user()

    with patch("app.api.v1.billing.settings") as mock_settings:
        mock_settings.STRIPE_SECRET_KEY = None
        resp = await client.post(
            "/api/v1/billing/checkout",
            json={"plan": "plus", "success_url": "http://localhost:3000", "cancel_url": "http://localhost:3000"},
        )
    assert resp.status_code == 503

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_checkout_unknown_plan(client: AsyncClient):
    from app.api.deps import get_current_user
    from app.main import app
    app.dependency_overrides[get_current_user] = lambda: _free_user()

    with patch("app.api.v1.billing.settings") as mock_settings:
        mock_settings.STRIPE_SECRET_KEY = "sk_test_xxx"
        resp = await client.post(
            "/api/v1/billing/checkout",
            json={"plan": "nonexistent", "success_url": "http://localhost:3000", "cancel_url": "http://localhost:3000"},
        )
    assert resp.status_code == 400

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_check_feature_gate():
    from app.api.v1.billing import check_feature_gate
    user = MagicMock()
    user.plan = MagicMock(value="plus")
    assert await check_feature_gate(user, "voice") is True
    assert await check_feature_gate(user, "sso") is False

    user.plan = MagicMock(value="free")
    assert await check_feature_gate(user, "voice") is False
