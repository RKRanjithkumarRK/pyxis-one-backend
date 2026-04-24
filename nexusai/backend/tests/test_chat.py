"""Phase 3 chat and conversation endpoint tests."""
from __future__ import annotations
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_models_endpoint(client: AsyncClient):
    """Models list requires no auth; returns model definitions."""
    from unittest.mock import AsyncMock, patch
    with patch("app.api.v1.chat.get_model_latency", new_callable=AsyncMock, return_value=None):
        resp = await client.get("/api/v1/chat/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    # Verify expected model fields
    for m in data["models"]:
        assert "id" in m
        assert "provider" in m
        assert "cost_in_per_1k" in m


@pytest.mark.asyncio
async def test_chat_stream_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/chat/stream", json={
        "message": "hello",
        "model": "claude-sonnet-4",
    })
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_compare_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/chat/compare", json={
        "models": ["claude-sonnet-4", "gpt-4o"],
        "message": "hello",
    })
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_compare_model_count_validation(client: AsyncClient):
    from app.core.security import create_access_token
    token = create_access_token("00000000-0000-0000-0000-000000000001")
    resp = await client.post(
        "/api/v1/chat/compare",
        json={"models": ["claude-sonnet-4"], "message": "hi"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_conversations_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/conversations")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_conversation_search_requires_q(client: AsyncClient):
    from app.core.security import create_access_token
    token = create_access_token("00000000-0000-0000-0000-000000000001")
    resp = await client.get(
        "/api/v1/conversations/search",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 422  # missing required 'q' param


@pytest.mark.asyncio
async def test_empty_message_rejected(client: AsyncClient):
    from app.core.security import create_access_token
    from unittest.mock import AsyncMock, patch, MagicMock
    token = create_access_token("00000000-0000-0000-0000-000000000001")

    with patch("app.api.v1.chat.ConversationService.get_or_create", new_callable=AsyncMock) as mock_conv:
        mock_conv.return_value = MagicMock(id="conv-1", active_branch_id="branch-1")
        resp = await client.post(
            "/api/v1/chat/stream",
            json={"message": "   ", "model": "claude-sonnet-4"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 422
