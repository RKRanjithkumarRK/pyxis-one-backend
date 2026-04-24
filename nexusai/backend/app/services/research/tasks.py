"""Celery tasks for Deep Research — implemented in Phase 6."""
from app.core.celery_app import celery_app


@celery_app.task(name="research.deep_research")
def deep_research(query: str, user_id: str, conversation_id: str, depth: str = "standard") -> None:
    """Stub — full implementation in Phase 6 (Deep Research)."""
