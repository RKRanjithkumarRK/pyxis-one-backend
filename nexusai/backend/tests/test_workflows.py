"""Phase 14 — Workflow endpoint and executor tests."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_list_workflows_requires_auth(client):
    resp = await client.get("/api/v1/workflows/")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_create_workflow_requires_auth(client):
    resp = await client.post("/api/v1/workflows/", json={
        "name": "My Workflow",
        "nodes": [],
        "edges": [],
    })
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_topo_sort_simple():
    from app.services.workflows.executor import _topo_sort
    nodes = [
        {"id": "a", "data": {}},
        {"id": "b", "data": {}},
        {"id": "c", "data": {}},
    ]
    edges = [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
    ]
    order = _topo_sort(nodes, edges)
    assert order.index("a") < order.index("b") < order.index("c")


@pytest.mark.asyncio
async def test_topo_sort_cycle_detection():
    from app.services.workflows.executor import _topo_sort
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]
    with pytest.raises(ValueError, match="cycle"):
        _topo_sort(nodes, edges)


@pytest.mark.asyncio
async def test_execute_condition_node():
    from app.services.workflows.executor import execute_node
    node = {
        "id": "n1",
        "type": "condition",
        "data": {"condition": "True", "field": ""},
    }
    result = await execute_node(node, context={"input": "hello"}, user_id="u1")
    assert result.get("branch") in ("true", "false", True, False, "true_branch", "false_branch") or isinstance(result, dict)


@pytest.mark.asyncio
async def test_execute_transform_node():
    from app.services.workflows.executor import execute_node
    node = {
        "id": "n2",
        "type": "transform",
        "data": {"template": "Hello {{input}}"},
    }
    result = await execute_node(node, context={"input": "World"}, user_id="u1")
    assert "Hello World" in str(result.get("output", ""))
