"""Phase 7 — Canvas tests."""
from __future__ import annotations
import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_list_delete_doc(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post(
        "/api/v1/canvas",
        json={"title": "My Research Notes"},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    doc = create_resp.json()
    assert doc["title"] == "My Research Notes"
    assert doc["version"] == 1
    doc_id = doc["id"]

    list_resp = await client.get("/api/v1/canvas", headers=auth_headers)
    assert list_resp.status_code == 200
    ids = [d["id"] for d in list_resp.json()]
    assert doc_id in ids

    del_resp = await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_get_doc(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "Test Doc"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    get_resp = await client.get(f"/api/v1/canvas/{doc_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == doc_id

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_update_doc(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "Draft"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    update_resp = await client.patch(
        f"/api/v1/canvas/{doc_id}",
        json={"title": "Final Report", "content": {"type": "doc", "content": []}, "save_version": False},
        headers=auth_headers,
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["title"] == "Final Report"

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_version_history(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "Versioned"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    await client.patch(
        f"/api/v1/canvas/{doc_id}",
        json={"title": "Version 2", "content": {"type": "doc"}, "save_version": True},
        headers=auth_headers,
    )
    await client.patch(
        f"/api/v1/canvas/{doc_id}",
        json={"title": "Version 3", "content": {"type": "doc"}, "save_version": True},
        headers=auth_headers,
    )

    versions_resp = await client.get(f"/api/v1/canvas/{doc_id}/versions", headers=auth_headers)
    assert versions_resp.status_code == 200
    versions = versions_resp.json()
    assert len(versions) >= 2

    restore_resp = await client.post(
        f"/api/v1/canvas/{doc_id}/versions/{versions[-1]['version']}/restore",
        headers=auth_headers,
    )
    assert restore_resp.status_code == 200

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_ai_edit(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "AI Edit Test"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    with patch("app.api.v1.canvas.ai_edit", new_callable=AsyncMock) as mock_edit:
        mock_edit.return_value = "This is concise."
        resp = await client.post(
            f"/api/v1/canvas/{doc_id}/ai-edit",
            json={
                "selected_text": "This is a very verbose and unnecessarily long sentence that could be shorter.",
                "instruction": "Make this more concise",
            },
            headers=auth_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["original"].startswith("This is a very")
    assert data["suggested"] == "This is concise."

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_export_requires_auth_for_private(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "Private Doc"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    export_resp = await client.get(f"/api/v1/canvas/{doc_id}/export?format=md")
    assert export_resp.status_code == 403

    export_auth_resp = await client.get(
        f"/api/v1/canvas/{doc_id}/export?format=md", headers=auth_headers
    )
    assert export_auth_resp.status_code == 200

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_export_html_format(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "HTML Export"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    resp = await client.get(f"/api/v1/canvas/{doc_id}/export?format=html", headers=auth_headers)
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "")

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


@pytest.mark.asyncio
async def test_export_docx_format(client: AsyncClient, auth_headers: dict):
    create_resp = await client.post("/api/v1/canvas", json={"title": "DOCX Export"}, headers=auth_headers)
    doc_id = create_resp.json()["id"]

    resp = await client.get(f"/api/v1/canvas/{doc_id}/export?format=docx", headers=auth_headers)
    assert resp.status_code == 200
    assert "wordprocessingml" in resp.headers.get("content-type", "")

    await client.delete(f"/api/v1/canvas/{doc_id}", headers=auth_headers)


def test_export_markdown_converts_html():
    from app.services.canvas.service import export_markdown
    html = "<h1>Title</h1><p>Hello <strong>world</strong></p>"
    md = export_markdown(html)
    assert "Title" in md


def test_export_html_wraps_content():
    from app.services.canvas.service import export_html
    html = export_html("My Doc", "<p>Content</p>")
    assert "<!DOCTYPE html>" in html
    assert "My Doc" in html
    assert "<p>Content</p>" in html
