"""Terminal WebSocket — bidirectional stdin/stdout between xterm.js and E2B sandbox PTY."""
from __future__ import annotations
import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.models.user import User
from app.services.sandbox import e2b_service as sb

logger = logging.getLogger("nexusai.ws.terminal")

HEARTBEAT_INTERVAL = 20  # seconds


async def terminal_ws(websocket: WebSocket, project_id: str, user: User, db) -> None:
    await websocket.accept()
    logger.info("Terminal WS connected: project=%s user=%s", project_id, user.id)

    sandbox_id = sb._registry.get(project_id)
    if not sandbox_id:
        await websocket.send_json({"type": "error", "message": "Sandbox not running. Start it first."})
        await websocket.close(1003)
        return

    # Queue for output from the sandbox process back to xterm
    output_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # For dev-mode sandboxes we simulate a shell
    if sandbox_id.startswith("dev-"):
        await _dev_mode_terminal(websocket, project_id)
        return

    # Start a persistent shell process in the sandbox
    try:
        from e2b import Sandbox
        loop = asyncio.get_event_loop()
        sandbox = await loop.run_in_executor(None, lambda: Sandbox.connect(sandbox_id))
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"Cannot connect to sandbox: {exc}"})
        await websocket.close(1011)
        return

    def _on_stdout(data: str):
        asyncio.run_coroutine_threadsafe(output_queue.put(data), asyncio.get_event_loop())

    def _on_stderr(data: str):
        asyncio.run_coroutine_threadsafe(output_queue.put(data), asyncio.get_event_loop())

    # Start a long-running bash process
    loop = asyncio.get_event_loop()
    process = None
    try:
        process = await loop.run_in_executor(
            None,
            lambda: sandbox.commands.run(
                "bash",
                background=True,
                on_stdout=_on_stdout,
                on_stderr=_on_stderr,
                workdir="/workspace",
            ),
        )
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(1011)
        return

    async def read_output():
        while True:
            chunk = await output_queue.get()
            if chunk is None:
                break
            try:
                await websocket.send_json({"type": "output", "data": chunk})
            except Exception:
                break

    async def read_input():
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = {"type": "input", "data": raw}

                if msg.get("type") == "input":
                    data = msg.get("data", "")
                    await loop.run_in_executor(None, lambda: process.send_stdin(data))
                elif msg.get("type") == "resize":
                    cols = msg.get("cols", 80)
                    rows = msg.get("rows", 24)
                    try:
                        await loop.run_in_executor(None, lambda: process.resize_pty(cols=cols, rows=rows))
                    except Exception:
                        pass
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await output_queue.put(None)

    await asyncio.gather(read_output(), read_input(), return_exceptions=True)

    try:
        await loop.run_in_executor(None, process.kill)
    except Exception:
        pass
    logger.info("Terminal WS closed: project=%s", project_id)


async def _dev_mode_terminal(websocket: WebSocket, project_id: str) -> None:
    """Simple echo shell for local dev (no E2B_API_KEY)."""
    await websocket.send_json({"type": "output", "data": f"NexusCode dev shell — project {project_id}\r\n$ "})
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "input", "data": raw}
            if msg.get("type") == "input":
                cmd = msg.get("data", "")
                await websocket.send_json({"type": "output", "data": f"\r\n[dev] {cmd.strip()}: command executed\r\n$ "})
            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
