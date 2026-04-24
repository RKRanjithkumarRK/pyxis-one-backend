"""Phase 6 — Deep Research tests."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_start_research_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/research", json={"query": "What is AI?"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_research_empty(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/research", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_start_and_get_research(client: AsyncClient, auth_headers: dict):
    with patch("app.services.research.tasks.deep_research.apply_async") as mock_task:
        mock_task.return_value = MagicMock(id="fake-celery-task-id")
        resp = await client.post(
            "/api/v1/research",
            json={"query": "What is quantum entanglement?", "depth": "quick"},
            headers=auth_headers,
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "pending"
    assert data["query"] == "What is quantum entanglement?"
    assert data["depth"] == "quick"
    report_id = data["id"]

    get_resp = await client.get(f"/api/v1/research/{report_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == report_id

    list_resp = await client.get("/api/v1/research", headers=auth_headers)
    assert list_resp.status_code == 200
    ids = [r["id"] for r in list_resp.json()]
    assert report_id in ids

    del_resp = await client.delete(f"/api/v1/research/{report_id}", headers=auth_headers)
    assert del_resp.status_code == 204


@pytest.mark.asyncio
async def test_get_nonexistent_report(client: AsyncClient, auth_headers: dict):
    import uuid
    fake_id = str(uuid.uuid4())
    resp = await client.get(f"/api/v1/research/{fake_id}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_query_too_short(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/research",
        json={"query": "AI"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_depth(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/research",
        json={"query": "What is machine learning?", "depth": "ultra"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_plan_research_returns_questions():
    """Unit test for pipeline planning (no network calls, uses LLM mock)."""
    with patch("app.services.research.pipeline._llm") as mock_llm:
        mock_llm.return_value = '["What is AI?", "How does ML work?", "What are neural nets?"]'
        from app.services.research.pipeline import plan_research
        questions = plan_research("Tell me about AI")
    assert isinstance(questions, list)
    assert len(questions) >= 1
    assert all(isinstance(q, str) for q in questions)


def test_verify_citations_removes_invalid_refs():
    from app.services.research.pipeline import verify_citations
    report = {
        "executive_summary": "AI is great [1]. See also [5] and [2].",
        "sections": [{"heading": "Intro", "content": "Section with [3] reference."}],
        "citations": [
            {"id": 1, "title": "A", "url": "http://a.com", "snippet": ""},
            {"id": 2, "title": "B", "url": "http://b.com", "snippet": ""},
        ],
    }
    result = verify_citations(report)
    assert "[5]" not in result["executive_summary"]
    assert "[1]" in result["executive_summary"]
    assert "[2]" in result["executive_summary"]
    assert "[3]" not in result["sections"][0]["content"]


def test_serper_client_handles_failure():
    from app.services.research.serper import search
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = Exception("timeout")
        results = search("test query", "fake-key")
    assert results == []


def test_fetch_skips_binary_files():
    from app.services.research.fetcher import fetch_text
    result = fetch_text("https://example.com/file.pdf")
    assert result is None
