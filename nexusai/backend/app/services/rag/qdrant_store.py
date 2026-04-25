"""Qdrant vector store wrapper for KB chunks."""
from __future__ import annotations
import logging
import uuid
from typing import Any

logger = logging.getLogger("nexusai.rag.qdrant")

COLLECTION = "nexusai_kb"
VECTOR_SIZE = 1536  # text-embedding-3-small


def _client():
    from qdrant_client import QdrantClient
    from app.core.config import settings
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection() -> None:
    from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
        )
        logger.info("Created Qdrant collection %s", COLLECTION)


def upsert_chunks(
    kb_id: str,
    file_id: str,
    chunks: list[dict],  # [{"chunk_index": int, "content": str, "embedding": list[float]}]
) -> None:
    from qdrant_client.models import PointStruct
    client = _client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=c["embedding"],
            payload={
                "kb_id": kb_id,
                "file_id": file_id,
                "chunk_index": c["chunk_index"],
                "content": c["content"],
            },
        )
        for c in chunks
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    logger.debug("Upserted %d points for kb=%s file=%s", len(points), kb_id, file_id)


def delete_file_chunks(kb_id: str, file_id: str) -> None:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(key="kb_id", match=MatchValue(value=kb_id)),
                FieldCondition(key="file_id", match=MatchValue(value=file_id)),
            ]
        ),
    )


def delete_kb_chunks(kb_id: str) -> None:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
        ),
    )


def search(
    kb_ids: list[str],
    query_embedding: list[float],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Cosine search restricted to the given kb_ids. Returns payload dicts with score."""
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    client = _client()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="kb_id", match=MatchAny(any=kb_ids))]
        ),
        limit=limit,
        with_payload=True,
    )
    return [
        {**r.payload, "score": r.score}
        for r in results
    ]
