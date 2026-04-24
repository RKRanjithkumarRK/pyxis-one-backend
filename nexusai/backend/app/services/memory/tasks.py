"""Celery tasks for memory extraction — implemented in Phase 8."""
from app.core.celery_app import celery_app


@celery_app.task(name="memory.extract")
def extract_memory(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Stub — full implementation in Phase 8 (Memory)."""
