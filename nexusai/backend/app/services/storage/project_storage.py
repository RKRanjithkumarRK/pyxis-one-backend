"""Project file storage: GCS (prod) or local disk (dev)."""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path

from app.services.storage.gcs import _use_gcs, _bucket, _LOCAL_ROOT

logger = logging.getLogger("nexusai.storage.projects")


def _project_prefix(user_id: str, project_id: str) -> str:
    return f"projects/{user_id}/{project_id}"


async def list_project_files(user_id: str, project_id: str) -> list[dict]:
    prefix = _project_prefix(user_id, project_id)
    loop = asyncio.get_event_loop()
    if _use_gcs():
        blobs = await loop.run_in_executor(None, lambda: list(_bucket().list_blobs(prefix=prefix + "/")))
        return [
            {
                "path": b.name[len(prefix) + 1:],
                "size": b.size,
                "updated": b.updated.isoformat() if b.updated else None,
            }
            for b in blobs
        ]
    else:
        root = _LOCAL_ROOT / prefix
        if not root.exists():
            return []
        results = []
        for p in root.rglob("*"):
            if p.is_file():
                relative = str(p.relative_to(root)).replace("\\", "/")
                results.append({"path": relative, "size": p.stat().st_size, "updated": None})
        return results


async def read_project_file(user_id: str, project_id: str, path: str) -> bytes:
    key = f"{_project_prefix(user_id, project_id)}/{path}"
    loop = asyncio.get_event_loop()
    if _use_gcs():
        return await loop.run_in_executor(None, lambda: _bucket().blob(key).download_as_bytes())
    local = _LOCAL_ROOT / key
    return local.read_bytes()


async def write_project_file(user_id: str, project_id: str, path: str, content: bytes) -> None:
    key = f"{_project_prefix(user_id, project_id)}/{path}"
    loop = asyncio.get_event_loop()
    if _use_gcs():
        await loop.run_in_executor(None, lambda: _bucket().blob(key).upload_from_string(content))
    else:
        local = _LOCAL_ROOT / key
        await loop.run_in_executor(None, lambda: (local.parent.mkdir(parents=True, exist_ok=True), local.write_bytes(content)))


async def delete_project_file(user_id: str, project_id: str, path: str) -> None:
    key = f"{_project_prefix(user_id, project_id)}/{path}"
    loop = asyncio.get_event_loop()
    if _use_gcs():
        try:
            await loop.run_in_executor(None, lambda: _bucket().blob(key).delete())
        except Exception as exc:
            logger.warning("GCS delete failed %s: %s", key, exc)
    else:
        local = _LOCAL_ROOT / key
        if local.exists():
            await loop.run_in_executor(None, local.unlink)


async def sync_from_gcs_to_sandbox(user_id: str, project_id: str, sandbox_id: str) -> int:
    """Download all project files from GCS into the E2B sandbox. Returns file count."""
    from app.services.sandbox.e2b_service import write_file as sb_write
    files = await list_project_files(user_id, project_id)
    count = 0
    for f in files:
        content = await read_project_file(user_id, project_id, f["path"])
        await sb_write(sandbox_id, f"/workspace/{f['path']}", content.decode("utf-8", errors="replace"))
        count += 1
    logger.info("Synced %d files from GCS → sandbox %s", count, sandbox_id)
    return count


async def sync_from_sandbox_to_gcs(user_id: str, project_id: str, sandbox_id: str, paths: list[str]) -> None:
    """Upload given sandbox file paths back to GCS."""
    from app.services.sandbox.e2b_service import read_file as sb_read
    for path in paths:
        try:
            content = await sb_read(sandbox_id, f"/workspace/{path}")
            await write_project_file(user_id, project_id, path, content.encode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to sync %s to GCS: %s", path, exc)
