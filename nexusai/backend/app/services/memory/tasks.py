"""Celery tasks for memory extraction."""
from __future__ import annotations
import asyncio
import logging
import uuid

from app.core.celery_app import celery_app

logger = logging.getLogger("nexusai.memory.tasks")


@celery_app.task(
    name="memory.extract",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def extract_memory(
    self,
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    message_id: str | None = None,
) -> dict:
    """Extract and store user memories from a single conversation exchange."""
    try:
        return asyncio.run(_async_extract(user_id, user_msg, assistant_msg, message_id))
    except Exception as exc:
        logger.exception("Memory extraction task failed for user %s: %s", user_id, exc)
        raise self.retry(exc=exc)


async def _async_extract(
    user_id: str,
    user_msg: str,
    assistant_msg: str,
    message_id: str | None,
) -> dict:
    from app.core.database import AsyncSessionLocal
    from app.services.memory.service import extract_facts, store_memories

    uid = uuid.UUID(user_id)
    mid = uuid.UUID(message_id) if message_id else None

    facts = await extract_facts(user_msg, assistant_msg)
    if not facts:
        logger.debug("No facts extracted for user %s", user_id)
        return {"stored": 0, "facts": []}

    async with AsyncSessionLocal() as db:
        stored = await store_memories(db, uid, facts, source_message_id=mid)

    logger.info("Stored %d memories for user %s", stored, user_id)
    return {"stored": stored, "facts": facts}
