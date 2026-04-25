"""Tests for Knowledge Base API."""
from __future__ import annotations
import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_kb(auth_client: AsyncClient):
    resp = await auth_client.post("/api/v1/kb", json={"name": "My KB"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "My KB"
    assert data["files"] == []


@pytest.mark.asyncio
async def test_list_kbs(auth_client: AsyncClient):
    await auth_client.post("/api/v1/kb", json={"name": "KB 1"})
    await auth_client.post("/api/v1/kb", json={"name": "KB 2"})
    resp = await auth_client.get("/api/v1/kb")
    assert resp.status_code == 200
    names = [k["name"] for k in resp.json()]
    assert "KB 1" in names
    assert "KB 2" in names


@pytest.mark.asyncio
async def test_get_kb(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "Test KB"})
    kb_id = create.json()["id"]
    resp = await auth_client.get(f"/api/v1/kb/{kb_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == kb_id


@pytest.mark.asyncio
async def test_update_kb(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "Old Name"})
    kb_id = create.json()["id"]
    resp = await auth_client.patch(f"/api/v1/kb/{kb_id}", json={"name": "New Name"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_delete_kb(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "Delete Me"})
    kb_id = create.json()["id"]
    with patch("app.api.v1.knowledge_base.delete_kb_chunks"):
        resp = await auth_client.delete(f"/api/v1/kb/{kb_id}")
    assert resp.status_code == 204
    resp = await auth_client.get(f"/api/v1/kb/{kb_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_file(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "File KB"})
    kb_id = create.json()["id"]

    with (
        patch("app.api.v1.knowledge_base.gcs.upload"),
        patch("app.api.v1.knowledge_base.ingest_file") as mock_task,
    ):
        mock_task.delay = MagicMock()
        content = b"Hello world. This is a test document."
        resp = await auth_client.post(
            f"/api/v1/kb/{kb_id}/files",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test.txt"
    assert data["file_type"] == "txt"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_unsupported_type(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "Bad File KB"})
    kb_id = create.json()["id"]
    with patch("app.api.v1.knowledge_base.gcs.upload"):
        resp = await auth_client.post(
            f"/api/v1/kb/{kb_id}/files",
            files={"file": ("script.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_delete_file(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "File Delete KB"})
    kb_id = create.json()["id"]

    with (
        patch("app.api.v1.knowledge_base.gcs.upload"),
        patch("app.api.v1.knowledge_base.ingest_file") as mock_task,
    ):
        mock_task.delay = MagicMock()
        upload_resp = await auth_client.post(
            f"/api/v1/kb/{kb_id}/files",
            files={"file": ("sample.txt", io.BytesIO(b"content"), "text/plain")},
        )
    file_id = upload_resp.json()["id"]

    with (
        patch("app.api.v1.knowledge_base.delete_file_chunks"),
        patch("app.api.v1.knowledge_base.gcs.delete"),
    ):
        del_resp = await auth_client.delete(f"/api/v1/kb/{kb_id}/files/{file_id}")
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_get_file_status(auth_client: AsyncClient):
    create = await auth_client.post("/api/v1/kb", json={"name": "Status KB"})
    kb_id = create.json()["id"]

    with (
        patch("app.api.v1.knowledge_base.gcs.upload"),
        patch("app.api.v1.knowledge_base.ingest_file") as mock_task,
    ):
        mock_task.delay = MagicMock()
        upload_resp = await auth_client.post(
            f"/api/v1/kb/{kb_id}/files",
            files={"file": ("data.txt", io.BytesIO(b"data"), "text/plain")},
        )
    file_id = upload_resp.json()["id"]

    resp = await auth_client.get(f"/api/v1/kb/{kb_id}/files/{file_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == file_id
