"""Celery tasks for image generation."""
from __future__ import annotations
import asyncio
import logging
import uuid

from celery import shared_task

logger = logging.getLogger("nexusai.image.tasks")


@shared_task(name="image.generate", bind=True, max_retries=1, acks_late=True)
def generate_images_task(self, request_id: str, user_id: str) -> dict:
    return asyncio.run(_run(request_id, user_id))


async def _run(request_id: str, user_id: str) -> dict:
    from app.core.database import AsyncSessionLocal
    from app.models.image import ImageRequest
    from app.services.image.service import generate_image
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        req = (
            await db.execute(select(ImageRequest).where(ImageRequest.id == uuid.UUID(request_id)))
        ).scalar_one_or_none()
        if not req:
            return {"error": "not found"}

        req.status = "processing"
        await db.commit()

        try:
            urls = await generate_image(
                prompt=req.prompt,
                model=req.model,
                width=req.width,
                height=req.height,
                num_images=req.num_images,
                negative_prompt=req.negative_prompt or "",
            )
            req.status = "done"
            req.result_urls = urls
            await db.commit()
            return {"request_id": request_id, "urls": urls}
        except Exception as exc:
            req.status = "error"
            req.error_msg = str(exc)[:500]
            await db.commit()
            raise
