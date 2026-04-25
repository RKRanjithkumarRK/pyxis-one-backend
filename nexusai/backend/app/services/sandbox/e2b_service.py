"""E2B sandbox service — real Linux containers via Firecracker VMs."""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

from app.core.config import settings
from app.core.telemetry import tracer

logger = logging.getLogger("nexusai.sandbox.e2b")

# Template ID: build with `e2b template build` from nexusai/e2b/Dockerfile
NEXUSAI_TEMPLATE = "nexusai-dev"

# In-memory sandbox registry: project_id → sandbox_id
_registry: dict[str, str] = {}

# Warm pool: pre-started sandbox IDs
_warm_pool: list[str] = []
_pool_lock = asyncio.Lock()


def _client():
    from e2b import Sandbox
    return Sandbox


async def get_or_create_sandbox(project_id: str, timeout: int = 3600) -> str:
    """Return existing sandbox ID for project or create a new one."""
    if project_id in _registry:
        sandbox_id = _registry[project_id]
        if await _is_alive(sandbox_id):
            return sandbox_id
        del _registry[project_id]

    sandbox_id = await _create_sandbox(timeout)
    _registry[project_id] = sandbox_id
    return sandbox_id


async def _is_alive(sandbox_id: str) -> bool:
    try:
        from e2b import Sandbox
        loop = asyncio.get_event_loop()
        sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
        await loop.run_in_executor(None, sandbox.commands.run, "echo alive")
        return True
    except Exception:
        return False


async def _create_sandbox(timeout: int = 3600) -> str:
    async with _pool_lock:
        if _warm_pool:
            sandbox_id = _warm_pool.pop(0)
            logger.info("Warm sandbox claimed: %s", sandbox_id)
            return sandbox_id

    return await _spawn_sandbox(timeout)


async def _spawn_sandbox(timeout: int = 3600) -> str:
    if not settings.E2B_API_KEY:
        fake_id = f"dev-{uuid.uuid4().hex[:8]}"
        logger.warning("E2B_API_KEY not set — returning fake sandbox %s", fake_id)
        return fake_id

    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    with tracer.start_as_current_span("e2b.spawn_sandbox") as span:
        span.set_attribute("e2b.template", NEXUSAI_TEMPLATE)
        t0 = time.perf_counter()
        sandbox = await loop.run_in_executor(
            None,
            lambda: Sandbox(
                template=NEXUSAI_TEMPLATE,
                api_key=settings.E2B_API_KEY,
                timeout=timeout,
            ),
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        span.set_attribute("e2b.cold_start_ms", elapsed_ms)
        logger.info("Spawned E2B sandbox %s (template=%s, cold_start=%dms)", sandbox.sandbox_id, NEXUSAI_TEMPLATE, elapsed_ms)
    return sandbox.sandbox_id


async def prefill_warm_pool(count: int = 3) -> None:
    """Pre-start `count` sandboxes into the warm pool."""
    async with _pool_lock:
        need = max(0, count - len(_warm_pool))

    for _ in range(need):
        try:
            sid = await _spawn_sandbox()
            async with _pool_lock:
                _warm_pool.append(sid)
            logger.info("Warm pool now has %d sandboxes", len(_warm_pool))
        except Exception as exc:
            logger.error("Failed to prefill sandbox: %s", exc)


async def execute_command(
    sandbox_id: str,
    command: str,
    workdir: str = "/workspace",
) -> dict:
    """Run command synchronously; return {stdout, stderr, exit_code}."""
    if sandbox_id.startswith("dev-"):
        return {"stdout": f"[dev mode] would run: {command}", "stderr": "", "exit_code": 0}

    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    result = await loop.run_in_executor(
        None,
        lambda: sandbox.commands.run(command, workdir=workdir),
    )
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "exit_code": result.exit_code,
    }


async def stream_command(
    sandbox_id: str,
    command: str,
    workdir: str = "/workspace",
) -> AsyncIterator[str]:
    """Stream command output line by line."""
    if sandbox_id.startswith("dev-"):
        yield f"[dev mode] $ {command}\n"
        yield "[dev mode] Command executed successfully\n"
        return

    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_stdout(data: str):
        asyncio.run_coroutine_threadsafe(queue.put(data), loop)

    def on_stderr(data: str):
        asyncio.run_coroutine_threadsafe(queue.put(f"[stderr] {data}"), loop)

    def run_blocking():
        sandbox.commands.run(
            command,
            workdir=workdir,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    asyncio.get_event_loop().run_in_executor(None, run_blocking)

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


async def write_file(sandbox_id: str, path: str, content: str) -> None:
    if sandbox_id.startswith("dev-"):
        return
    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    await loop.run_in_executor(None, lambda: sandbox.files.write(path, content))


async def read_file(sandbox_id: str, path: str) -> str:
    if sandbox_id.startswith("dev-"):
        return ""
    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    return await loop.run_in_executor(None, lambda: sandbox.files.read(path))


async def list_files(sandbox_id: str, path: str = "/workspace") -> list[dict]:
    if sandbox_id.startswith("dev-"):
        return []
    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    entries = await loop.run_in_executor(None, lambda: sandbox.files.list(path))
    return [{"name": e.name, "type": "dir" if e.is_dir else "file", "path": f"{path}/{e.name}"} for e in entries]


async def expose_port(sandbox_id: str, port: int) -> str:
    """Return preview URL for exposed port."""
    if sandbox_id.startswith("dev-"):
        return f"http://localhost:{port}"
    from e2b import Sandbox
    loop = asyncio.get_event_loop()
    sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    host = await loop.run_in_executor(None, lambda: sandbox.get_host(port))
    return f"https://{host}"


async def terminate_sandbox(project_id: str) -> None:
    sandbox_id = _registry.pop(project_id, None)
    if not sandbox_id or sandbox_id.startswith("dev-"):
        return
    try:
        from e2b import Sandbox
        loop = asyncio.get_event_loop()
        sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
        await loop.run_in_executor(None, sandbox.close)
        logger.info("Terminated sandbox %s for project %s", sandbox_id, project_id)
    except Exception as exc:
        logger.warning("Failed to terminate sandbox %s: %s", sandbox_id, exc)
