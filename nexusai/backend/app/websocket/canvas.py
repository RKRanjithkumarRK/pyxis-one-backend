"""Canvas WebSocket — real-time document collaboration."""
from __future__ import annotations
import asyncio
import json
import logging
import uuid
from collections import defaultdict
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.redis import redis_client
from app.core.security import decode_token

logger = logging.getLogger("nexusai.ws.canvas")

# In-process room registry: doc_id → list[WebSocket]
_rooms: dict[str, list[WebSocket]] = defaultdict(list)
_room_lock = asyncio.Lock()

REDIS_CONTENT_TTL = 3600 * 24  # 24 hours


def _redis_key(doc_id: str) -> str:
    return f"canvas:content:{doc_id}"


async def _broadcast(doc_id: str, message: dict, exclude: WebSocket | None = None) -> None:
    payload = json.dumps(message)
    dead: list[WebSocket] = []
    for ws in list(_rooms.get(doc_id, [])):
        if ws is exclude:
            continue
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            _rooms[doc_id].remove(ws)
        except ValueError:
            pass


async def handle_canvas_ws(websocket: WebSocket, doc_id: str) -> None:
    """WebSocket handler for a single Canvas document room."""
    await websocket.accept()
    token: str | None = None
    user_id: str | None = None

    try:
        # ── Auth handshake ────────────────────────────────────
        auth_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        auth_msg = json.loads(auth_raw)
        if auth_msg.get("type") != "auth":
            await websocket.close(code=4001, reason="Missing auth")
            return

        token = auth_msg.get("token", "")
        try:
            payload = decode_token(token)
            user_id = payload["sub"]
        except Exception:
            await websocket.close(code=4003, reason="Invalid token")
            return

        # ── Join room ─────────────────────────────────────────
        async with _room_lock:
            _rooms[doc_id].append(websocket)

        # ── Send current document state ───────────────────────
        cached = await redis_client.get(_redis_key(doc_id))
        init_content: Any = None
        if cached:
            try:
                init_content = json.loads(cached)
            except Exception:
                init_content = None

        peers = len(_rooms.get(doc_id, [])) - 1
        await websocket.send_text(json.dumps({
            "type": "init",
            "content": init_content,
            "peers": peers,
            "user_id": user_id,
        }))

        # Notify others that a new peer joined
        await _broadcast(doc_id, {"type": "peer_joined", "peers": peers + 1}, exclude=websocket)

        # ── Message loop ──────────────────────────────────────
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "update":
                content = msg.get("content")
                if content is not None:
                    # Cache in Redis
                    await redis_client.setex(
                        _redis_key(doc_id), REDIS_CONTENT_TTL, json.dumps(content)
                    )
                    # Broadcast to room
                    await _broadcast(doc_id, {
                        "type": "update",
                        "content": content,
                        "user_id": user_id,
                    }, exclude=websocket)

            elif msg_type == "cursor":
                await _broadcast(doc_id, {
                    "type": "cursor",
                    "user_id": user_id,
                    "pos": msg.get("pos"),
                    "anchor": msg.get("anchor"),
                    "head": msg.get("head"),
                }, exclude=websocket)

            elif msg_type == "title":
                await _broadcast(doc_id, {
                    "type": "title",
                    "title": msg.get("title", ""),
                    "user_id": user_id,
                }, exclude=websocket)

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        logger.warning("Canvas WS auth timeout for doc %s", doc_id)
    except Exception as exc:
        logger.exception("Canvas WS error for doc %s: %s", doc_id, exc)
    finally:
        async with _room_lock:
            try:
                _rooms[doc_id].remove(websocket)
            except ValueError:
                pass
        peers = len(_rooms.get(doc_id, []))
        if peers > 0:
            await _broadcast(doc_id, {"type": "peer_left", "peers": peers})
        logger.debug("Canvas WS closed for doc %s, %d peers remaining", doc_id, peers)
