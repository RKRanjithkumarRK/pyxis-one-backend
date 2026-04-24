"""Phase 5 — Agent Store tests."""
from __future__ import annotations
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_agents_public(client: AsyncClient):
    resp = await client.get("/api/v1/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "total" in data
    assert "pages" in data


@pytest.mark.asyncio
async def test_list_agents_by_category(client: AsyncClient):
    resp = await client.get("/api/v1/agents?category=code")
    assert resp.status_code == 200
    data = resp.json()
    for agent in data["agents"]:
        assert agent["category"] == "code"


@pytest.mark.asyncio
async def test_list_agents_search(client: AsyncClient):
    resp = await client.get("/api/v1/agents?search=code+reviewer")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["agents"], list)


@pytest.mark.asyncio
async def test_get_builtin_agent(client: AsyncClient):
    resp = await client.get("/api/v1/agents/code-reviewer")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "code-reviewer"
    assert data["is_builtin"] is True
    assert data["visibility"] == "public"


@pytest.mark.asyncio
async def test_get_nonexistent_agent(client: AsyncClient):
    resp = await client.get("/api/v1/agents/does-not-exist-agent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_agent_requires_auth(client: AsyncClient):
    resp = await client.post(
        "/api/v1/agents",
        json={"name": "My Agent", "category": "general"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_create_list_delete_agent(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/agents",
        json={
            "name": "Test Agent Phase5",
            "description": "A test agent",
            "category": "code",
            "instructions": "You are a test agent.",
            "starters": ["Hello", "What can you do?"],
            "visibility": "private",
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    agent = create_resp.json()
    assert agent["name"] == "Test Agent Phase5"
    assert agent["slug"] == "test-agent-phase5"
    agent_id = agent["id"]

    list_resp = await client.get("/api/v1/agents/mine", headers=auth_headers)
    assert list_resp.status_code == 200
    ids = [a["id"] for a in list_resp.json()]
    assert agent_id in ids

    del_resp = await client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_update_agent(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/agents",
        json={"name": "Update Me Agent", "category": "writing"},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"description": "Updated description", "name": "Updated Name"},
        headers=auth_headers,
    )
    assert patch_resp.status_code == 200
    updated = patch_resp.json()
    assert updated["description"] == "Updated description"
    assert updated["name"] == "Updated Name"
    assert updated["version"] == 2

    await client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_publish_agent(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/agents",
        json={"name": "Publish Me", "category": "general", "visibility": "private"},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]

    pub_resp = await client.post(
        f"/api/v1/agents/{agent_id}/publish?public=true", headers=auth_headers
    )
    assert pub_resp.status_code == 200
    assert pub_resp.json()["visibility"] == "public"

    unpub_resp = await client.post(
        f"/api/v1/agents/{agent_id}/publish?public=false", headers=auth_headers
    )
    assert unpub_resp.status_code == 200
    assert unpub_resp.json()["visibility"] == "private"

    await client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_cannot_delete_builtin(client: AsyncClient, auth_headers: dict):
    agent_resp = await client.get("/api/v1/agents/code-reviewer")
    assert agent_resp.status_code == 200
    agent_id = agent_resp.json()["id"]

    del_resp = await client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)
    assert del_resp.status_code == 403


@pytest.mark.asyncio
async def test_version_history(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/agents",
        json={"name": "Versioned Agent", "category": "data"},
        headers=auth_headers,
    )
    agent_id = create_resp.json()["id"]

    await client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"description": "v2"},
        headers=auth_headers,
    )
    await client.patch(
        f"/api/v1/agents/{agent_id}",
        json={"description": "v3"},
        headers=auth_headers,
    )

    versions_resp = await client.get(
        f"/api/v1/agents/{agent_id}/versions", headers=auth_headers
    )
    assert versions_resp.status_code == 200
    versions = versions_resp.json()
    assert len(versions) == 2

    restore_resp = await client.post(
        f"/api/v1/agents/{agent_id}/restore/1", headers=auth_headers
    )
    assert restore_resp.status_code == 200

    await client.delete(f"/api/v1/agents/{agent_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_rate_agent(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/agents/code-reviewer/rate",
        json={"rating": 5.0},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rating"] is not None
    assert data["rating_count"] >= 1
