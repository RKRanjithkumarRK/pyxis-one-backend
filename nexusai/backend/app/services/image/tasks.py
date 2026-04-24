"""Celery tasks for image generation — implemented in Phase 11."""
from app.core.celery_app import celery_app


@celery_app.task(name="image.generate")
def generate_images(request_id: str, user_id: str) -> None:
    """Stub — full implementation in Phase 11 (Image Studio)."""
