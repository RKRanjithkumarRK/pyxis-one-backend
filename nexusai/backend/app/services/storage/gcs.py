"""GCS file storage with local-disk fallback for dev."""
from __future__ import annotations
import logging
import os
from pathlib import Path

logger = logging.getLogger("nexusai.storage.gcs")

_LOCAL_ROOT = Path("/tmp/nexusai_storage")


def _bucket():
    from google.cloud import storage as gcs
    from app.core.config import settings
    client = gcs.Client()
    return client.bucket(settings.GCS_BUCKET_NAME)


def _use_gcs() -> bool:
    from app.core.config import settings
    return bool(getattr(settings, "GCS_BUCKET_NAME", None))


def upload(path: str, content: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to GCS (or local). Returns the storage path."""
    if _use_gcs():
        blob = _bucket().blob(path)
        blob.upload_from_string(content, content_type=content_type)
        logger.debug("Uploaded gs://%s/%s (%d bytes)", blob.bucket.name, path, len(content))
    else:
        dest = _LOCAL_ROOT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        logger.debug("Saved locally %s (%d bytes)", dest, len(content))
    return path


def download(path: str) -> bytes:
    """Download bytes from GCS (or local)."""
    if _use_gcs():
        return _bucket().blob(path).download_as_bytes()
    local = _LOCAL_ROOT / path
    return local.read_bytes()


def delete(path: str) -> None:
    if _use_gcs():
        try:
            _bucket().blob(path).delete()
        except Exception as exc:
            logger.warning("GCS delete failed for %s: %s", path, exc)
    else:
        local = _LOCAL_ROOT / path
        if local.exists():
            local.unlink()


def kb_file_path(kb_id: str, file_id: str, filename: str) -> str:
    return f"kb/{kb_id}/{file_id}/{filename}"
