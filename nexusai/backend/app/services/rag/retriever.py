"""Hybrid retrieval: BM25 (Postgres FTS) + Qdrant vector + Cohere rerank."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("nexusai.rag.retriever")

MAX_BM25 = 20
MAX_VECTOR = 20
MAX_RERANK = 8


async def _get_embedding(text: str) -> list[float] | None:
    try:
        import litellm
        resp = await litellm.aembedding(model="text-embedding-3-small", input=[text])
        return resp.data[0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None


async def _bm25_search(db: "AsyncSession", kb_ids: list[str], query: str, limit: int) -> list[dict]:
    from sqlalchemy import text as sql_text
    import uuid
    kb_uuids = [uuid.UUID(k) for k in kb_ids]
    sql = sql_text("""
        SELECT c.content, c.chunk_index, c.file_id::text, c.kb_id::text,
               ts_rank_cd(to_tsvector('english', c.content),
                          plainto_tsquery('english', :q)) AS score
        FROM kb_chunks c
        WHERE c.kb_id = ANY(:kb_ids)
          AND to_tsvector('english', c.content) @@ plainto_tsquery('english', :q)
        ORDER BY score DESC
        LIMIT :limit
    """)
    rows = (await db.execute(sql, {"q": query, "kb_ids": kb_uuids, "limit": limit})).fetchall()
    return [
        {"content": r.content, "chunk_index": r.chunk_index,
         "file_id": r.file_id, "kb_id": r.kb_id, "score": float(r.score), "source": "bm25"}
        for r in rows
    ]


async def _vector_search(kb_ids: list[str], query: str, limit: int) -> list[dict]:
    from app.services.rag.qdrant_store import search as qdrant_search
    emb = await _get_embedding(query)
    if not emb:
        return []
    results = qdrant_search(kb_ids, emb, limit=limit)
    return [{**r, "source": "vector"} for r in results]


def _dedup_merge(bm25: list[dict], vector: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    merged = []
    for item in bm25 + vector:
        key = (item["file_id"], item["chunk_index"])
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


async def _cohere_rerank(query: str, docs: list[dict], top_n: int) -> list[dict]:
    try:
        import cohere
        from app.core.config import settings
        if not getattr(settings, "COHERE_API_KEY", None):
            return docs[:top_n]
        co = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)
        texts = [d["content"] for d in docs]
        resp = await co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_n,
        )
        reranked = [docs[r.index] for r in resp.results]
        return reranked
    except Exception as exc:
        logger.warning("Cohere rerank failed (%s), using original order", exc)
        return docs[:top_n]


async def retrieve(
    db: "AsyncSession",
    kb_ids: list[str],
    query: str,
    top_n: int = MAX_RERANK,
) -> list[dict]:
    """Return top_n deduplicated, reranked chunks from the given KBs."""
    bm25_task = _bm25_search(db, kb_ids, query, MAX_BM25)
    vector_task = _vector_search(kb_ids, query, MAX_VECTOR)

    import asyncio
    bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

    merged = _dedup_merge(bm25_results, vector_results)
    if not merged:
        return []

    reranked = await _cohere_rerank(query, merged, top_n)
    return reranked


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the system prompt."""
    if not chunks:
        return ""
    parts = ["<knowledge_base_context>"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source {i}]\n{c['content']}")
    parts.append("</knowledge_base_context>")
    return "\n\n".join(parts)
