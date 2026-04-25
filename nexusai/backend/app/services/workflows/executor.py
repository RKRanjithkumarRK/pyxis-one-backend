"""Workflow Celery DAG executor — runs nodes in topological order."""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger("nexusai.workflows.executor")


def _topo_sort(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """Topological sort of workflow nodes."""
    adj: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    in_deg: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])
        in_deg[e["target"]] = in_deg.get(e["target"], 0) + 1

    queue = [nid for nid, deg in in_deg.items() if deg == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for child in adj[nid]:
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    node_map = {n["id"]: n for n in nodes}
    return [node_map[nid] for nid in order if nid in node_map]


async def execute_node(node: dict, context: dict, inputs: dict) -> dict:
    """Execute a single workflow node, return its output."""
    ntype = node.get("type", "")
    data = node.get("data", {})

    if ntype == "agent":
        return await _run_agent_node(data, context, inputs)
    elif ntype == "http":
        return await _run_http_node(data, context, inputs)
    elif ntype == "condition":
        return _run_condition_node(data, context, inputs)
    elif ntype == "loop":
        return _run_loop_node(data, context, inputs)
    elif ntype == "transform":
        return _run_transform_node(data, context, inputs)
    elif ntype == "delay":
        await asyncio.sleep(data.get("seconds", 1))
        return {"status": "delayed"}
    elif ntype == "trigger":
        return {"trigger": "fired", "inputs": inputs}
    else:
        logger.warning("Unknown node type: %s", ntype)
        return {"skipped": True, "type": ntype}


async def _run_agent_node(data: dict, context: dict, inputs: dict) -> dict:
    from app.services.llm.router import litellm_complete
    model = data.get("model", "groq/llama-3.3-70b-versatile")
    prompt = _template(data.get("prompt", ""), context, inputs)
    response = await litellm_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=data.get("temperature", 0.7),
    )
    return {"output": response}


async def _run_http_node(data: dict, context: dict, inputs: dict) -> dict:
    url = _template(data.get("url", ""), context, inputs)
    method = data.get("method", "GET").upper()
    headers = data.get("headers", {})
    body = data.get("body")
    if isinstance(body, str):
        body = _template(body, context, inputs)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.request(method, url, headers=headers, content=body)
        try:
            return {"status": r.status_code, "body": r.json()}
        except Exception:
            return {"status": r.status_code, "body": r.text}


def _run_condition_node(data: dict, context: dict, inputs: dict) -> dict:
    expression = data.get("expression", "True")
    try:
        result = bool(eval(expression, {"__builtins__": {}}, {**context, **inputs}))  # noqa: S307
    except Exception:
        result = False
    return {"result": result, "branch": "true" if result else "false"}


def _run_loop_node(data: dict, context: dict, inputs: dict) -> dict:
    items = inputs.get("items", data.get("items", []))
    return {"items": items, "count": len(items) if isinstance(items, list) else 0}


def _run_transform_node(data: dict, context: dict, inputs: dict) -> dict:
    code = data.get("code", "output = inputs")
    local: dict[str, Any] = {"inputs": inputs, "context": context}
    try:
        exec(code, {"__builtins__": {}}, local)  # noqa: S102
        return {"output": local.get("output", {})}
    except Exception as exc:
        return {"error": str(exc)}


def _template(text: str, context: dict, inputs: dict) -> str:
    """Simple {{variable}} template substitution."""
    combined = {**context, **inputs}
    for k, v in combined.items():
        text = text.replace(f"{{{{{k}}}}}", str(v))
    return text


async def run_workflow(workflow_id: str, inputs: dict, db_url: str) -> dict:
    """Execute a complete workflow. Called from Celery task."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import select
    from app.models.workflow import Workflow, WorkflowRun

    engine = create_async_engine(db_url, pool_size=2, max_overflow=2)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        result = await db.execute(select(Workflow).where(Workflow.id == uuid.UUID(workflow_id)))
        workflow = result.scalar_one_or_none()
        if not workflow:
            return {"error": "Workflow not found"}

        run = WorkflowRun(
            workflow_id=workflow.id,
            status="running",
            trigger="manual",
            inputs=inputs,
            outputs={},
            node_results={},
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        t0 = time.perf_counter()
        context: dict[str, Any] = {"workflow_id": workflow_id, "run_id": str(run.id)}
        node_results: dict[str, Any] = {}
        error: str | None = None

        try:
            ordered_nodes = _topo_sort(workflow.nodes, workflow.edges)
            for node in ordered_nodes:
                node_id = node["id"]
                # Feed outputs of connected nodes as inputs
                node_inputs = dict(inputs)
                for edge in workflow.edges:
                    if edge["target"] == node_id and edge["source"] in node_results:
                        node_inputs.update(node_results[edge["source"]])

                node_out = await execute_node(node, context, node_inputs)
                node_results[node_id] = node_out
                logger.info("Node %s (%s) done: %s", node_id, node.get("type"), node_out)

            run.status = "completed"
            run.outputs = node_results.get(ordered_nodes[-1]["id"] if ordered_nodes else "", {})

        except Exception as exc:
            error = str(exc)
            run.status = "failed"
            run.error = error
            logger.error("Workflow %s run %s failed: %s", workflow_id, run.id, exc)

        run.node_results = node_results
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        await db.commit()
        await engine.dispose()

        return {
            "run_id": str(run.id),
            "status": run.status,
            "duration_ms": run.duration_ms,
            "outputs": run.outputs,
            "error": error,
        }
