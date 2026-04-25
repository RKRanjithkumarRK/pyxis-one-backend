"""Celery task: ingest a KB file — parse → chunk → embed → upsert Qdrant → store chunks."""
from __future__ import annotations
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from celery import shared_task

logger = logging.getLogger("nexusai.rag.tasks")


@shared_task(
    name="rag.ingest_file",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_file(self, file_id: str) -> dict:
    try:
        return asyncio.run(_async_ingest(file_id))
    except Exception as exc:
        logger.exception("ingest_file failed for file_id=%s", file_id)
        asyncio.run(_mark_error(file_id, str(exc)))
        raise self.retry(exc=exc)


async def _async_ingest(file_id: str) -> dict:
    from app.core.database import AsyncSessionLocal
    from app.models.knowledge_base import KBFile, KBChunk
    from app.services.rag.parser import extract_text
    from app.services.rag.chunker import chunk_text
    from app.services.rag import qdrant_store
    from app.services.storage import gcs
    import litellm
    from sqlalchemy import select, delete as sa_delete

    async with AsyncSessionLocal() as db:
        kb_file = (await db.execute(select(KBFile).where(KBFile.id == uuid.UUID(file_id)))).scalar_one_or_none()
        if not kb_file:
            raise ValueError(f"KBFile {file_id} not found")

        kb_file.status = "processing"
        await db.commit()

        try:
            raw = gcs.download(kb_file.storage_path)
            text = extract_text(raw, kb_file.file_type)
            raw_chunks = chunk_text(text)

            if not raw_chunks:
                kb_file.status = "done"
                kb_file.chunk_count = 0
                await db.commit()
                return {"file_id": file_id, "chunks": 0}

            resp = await litellm.aembedding(
                model="text-embedding-3-small",
                input=raw_chunks,
            )
            embeddings = [item["embedding"] for item in resp.data]

            qdrant_store.ensure_collection()
            qdrant_store.delete_file_chunks(str(kb_file.kb_id), file_id)

            await db.execute(sa_delete(KBChunk).where(KBChunk.file_id == uuid.UUID(file_id)))

            qdrant_store.upsert_chunks(
                kb_id=str(kb_file.kb_id),
                file_id=file_id,
                chunks=[
                    {"chunk_index": i, "content": c, "embedding": e}
                    for i, (c, e) in enumerate(zip(raw_chunks, embeddings))
                ],
            )

            now = datetime.now(timezone.utc)
            db.add_all([
                KBChunk(
                    kb_id=kb_file.kb_id,
                    file_id=uuid.UUID(file_id),
                    chunk_index=i,
                    content=c,
                    created_at=now,
                )
                for i, c in enumerate(raw_chunks)
            ])

            kb_file.status = "done"
            kb_file.chunk_count = len(raw_chunks)
            await db.commit()
            logger.info("Ingested file_id=%s → %d chunks", file_id, len(raw_chunks))
            return {"file_id": file_id, "chunks": len(raw_chunks)}

        except Exception as exc:
            kb_file.status = "error"
            kb_file.error_msg = str(exc)[:500]
            await db.commit()
            raise


async def _mark_error(file_id: str, msg: str) -> None:
    from app.core.database import AsyncSessionLocal
    from app.models.knowledge_base import KBFile
    from sqlalchemy import select
    async with AsyncSessionLocal() as db:
        kb_file = (await db.execute(select(KBFile).where(KBFile.id == uuid.UUID(file_id)))).scalar_one_or_none()
        if kb_file:
            kb_file.status = "error"
            kb_file.error_msg = msg[:500]
            await db.commit()
