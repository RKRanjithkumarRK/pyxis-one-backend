"""Celery tasks for RAG ingestion — implemented in Phase 10."""
from app.core.celery_app import celery_app


@celery_app.task(name="rag.ingest_file")
def ingest_file(file_id: str) -> None:
    """Stub — full implementation in Phase 10 (Knowledge Base)."""
