"""Phase 9 — Projects tests."""
from __future__ import annotations
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_and_list_project(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/projects",
        json={"name": "My Project", "description": "A test project", "system_prompt": "You are a helpful assistant."},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My Project"
    assert data["role"] == "owner"
    project_id = data["id"]

    list_resp = await client.get("/api/v1/projects", headers=auth_headers)
    assert list_resp.status_code == 200
    ids = [p["id"] for p in list_resp.json()]
    assert project_id in ids

    # Cleanup
    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_get_project(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "Get Test"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    get_resp = await client.get(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == project_id

    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_update_project(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "Old Name"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    patch_resp = await client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": "New Name", "system_prompt": "Be concise."},
        headers=auth_headers,
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["name"] == "New Name"
    assert data["system_prompt"] == "Be concise."

    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_delete_project(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "To Delete"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_list_members(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "Members Test"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    members_resp = await client.get(
        f"/api/v1/projects/{project_id}/members", headers=auth_headers
    )
    assert members_resp.status_code == 200
    members = members_resp.json()
    assert len(members) >= 1
    assert members[0]["role"] == "owner"

    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_invite_nonexistent_user(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "Invite Test"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    invite_resp = await client.post(
        f"/api/v1/projects/{project_id}/members",
        json={"email": "nonexistent@example.com", "role": "viewer"},
        headers=auth_headers,
    )
    assert invite_resp.status_code == 404

    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_project_conversations(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/projects", json={"name": "Conv Test"}, headers=auth_headers
    )
    project_id = create_resp.json()["id"]

    convs_resp = await client.get(
        f"/api/v1/projects/{project_id}/conversations", headers=auth_headers
    )
    assert convs_resp.status_code == 200
    assert isinstance(convs_resp.json(), list)

    await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_project_access_denied(client: AsyncClient):
    import uuid
    fake_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/projects/{fake_id}")
    assert resp.status_code in (401, 403)
