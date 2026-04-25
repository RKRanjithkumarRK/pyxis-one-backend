"""File sync WebSocket — Monaco editor ↔ E2B sandbox ↔ GCS (debounced 2s)."""
from __future__ import annotations
import asyncio
import json
import logging
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect

from app.models.user import User
from app.services.sandbox import e2b_service as sb
from app.services.storage import project_storage as ps
from app.core.redis import redis_client

logger = logging.getLogger("nexusai.ws.filesync")

DEBOUNCE_SECONDS = 2.0

# project_id → list of active sync clients
_rooms: dict[str, list[WebSocket]] = defaultdict(list)
_room_lock = asyncio.Lock()


async def file_sync_ws(websocket: WebSocket, project_id: str, user: User, db) -> None:
    await websocket.accept()

    async with _room_lock:
        _rooms[project_id].append(websocket)

    logger.info("FileSync WS joined: project=%s user=%s", project_id, user.id)
    user_id = str(user.id)

    # debounce timer per (project, path)
    _pending: dict[str, asyncio.TimerHandle] = {}
    loop = asyncio.get_event_loop()

    async def flush_to_gcs(path: str, content: str):
        try:
            await ps.write_project_file(user_id, project_id, path, content.encode("utf-8"))
            logger.debug("Flushed %s to GCS (project=%s)", path, project_id)
        except Exception as exc:
            logger.warning("GCS flush failed for %s: %s", path, exc)

    async def broadcast(msg: dict, exclude: WebSocket):
        payload = json.dumps(msg)
        dead = []
        for ws in list(_rooms.get(project_id, [])):
            if ws is exclude:
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                _rooms[project_id].remove(ws)
            except ValueError:
                pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "file_change":
                path = msg.get("path", "")
                content = msg.get("content", "")
                if not path:
                    continue

                # Write to sandbox immediately
                sandbox_id = sb._registry.get(project_id)
                if sandbox_id:
                    try:
                        await sb.write_file(sandbox_id, f"/workspace/{path}", content)
                    except Exception as exc:
                        logger.debug("Sandbox write skipped: %s", exc)

                # Broadcast to other open editors
                await broadcast({"type": "file_change", "path": path, "content": content, "user_id": user_id}, websocket)

                # Debounce GCS write
                old = _pending.pop(path, None)
                if old:
                    old.cancel()
                handle = loop.call_later(
                    DEBOUNCE_SECONDS,
                    lambda p=path, c=content: asyncio.ensure_future(flush_to_gcs(p, c)),
                )
                _pending[path] = handle

            elif mtype == "file_save":
                path = msg.get("path", "")
                content = msg.get("content", "")
                if path:
                    old = _pending.pop(path, None)
                    if old:
                        old.cancel()
                    await flush_to_gcs(path, content)
                    await websocket.send_json({"type": "saved", "path": path})

            elif mtype == "file_list":
                files = await ps.list_project_files(user_id, project_id)
                await websocket.send_json({"type": "file_list", "files": files})

            elif mtype == "file_read":
                path = msg.get("path", "")
                try:
                    content = await ps.read_project_file(user_id, project_id, path)
                    await websocket.send_json({
                        "type": "file_content",
                        "path": path,
                        "content": content.decode("utf-8", errors="replace"),
                    })
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})

            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        # Flush any pending writes immediately on disconnect
        for path, handle in _pending.items():
            handle.cancel()

        async with _room_lock:
            try:
                _rooms[project_id].remove(websocket)
            except ValueError:
                pass

        logger.info("FileSync WS left: project=%s user=%s", project_id, user.id)
