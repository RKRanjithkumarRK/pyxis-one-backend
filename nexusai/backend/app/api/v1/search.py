"""Full-text search across conversations, messages, canvas docs, research reports."""
from __future__ import annotations
import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.api.deps import get_current_user, get_db
from app.models.user import User

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search(
    q: str = Query(min_length=1, max_length=500),
    types: str = Query(default="messages,canvas,research", description="comma-separated"),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Full-text search using Postgres GIN index (tsvector)."""
    type_list = [t.strip() for t in types.split(",")]
    results = []

    if "messages" in type_list:
        rows = await db.execute(
            text("""
                SELECT
                    m.id::text AS id,
                    'message' AS type,
                    m.content AS excerpt,
                    c.id::text AS parent_id,
                    c.title AS parent_title,
                    m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = :uid
                  AND to_tsvector('english', m.content) @@ plainto_tsquery('english', :q)
                ORDER BY m.created_at DESC
                LIMIT :lim
            """),
            {"uid": str(current_user.id), "q": q, "lim": limit},
        )
        for row in rows.mappings():
            excerpt = _snippet(row["excerpt"], q)
            results.append({**dict(row), "excerpt": excerpt})

    if "canvas" in type_list:
        try:
            rows = await db.execute(
                text("""
                    SELECT
                        cd.id::text AS id,
                        'canvas' AS type,
                        cd.content AS excerpt,
                        cd.id::text AS parent_id,
                        cd.title AS parent_title,
                        cd.updated_at AS created_at
                    FROM canvas_documents cd
                    WHERE cd.owner_id = :uid
                      AND to_tsvector('english', COALESCE(cd.content, '')) @@ plainto_tsquery('english', :q)
                    ORDER BY cd.updated_at DESC
                    LIMIT :lim
                """),
                {"uid": str(current_user.id), "q": q, "lim": limit},
            )
            for row in rows.mappings():
                results.append({**dict(row), "excerpt": _snippet(row["excerpt"] or "", q)})
        except Exception:
            pass  # Table may not exist in dev

    if "research" in type_list:
        try:
            rows = await db.execute(
                text("""
                    SELECT
                        rr.id::text AS id,
                        'research' AS type,
                        rr.summary AS excerpt,
                        rr.id::text AS parent_id,
                        rr.query AS parent_title,
                        rr.created_at
                    FROM research_reports rr
                    WHERE rr.user_id = :uid
                      AND to_tsvector('english', COALESCE(rr.summary, '')) @@ plainto_tsquery('english', :q)
                    ORDER BY rr.created_at DESC
                    LIMIT :lim
                """),
                {"uid": str(current_user.id), "q": q, "lim": limit},
            )
            for row in rows.mappings():
                results.append({**dict(row), "excerpt": _snippet(row["excerpt"] or "", q)})
        except Exception:
            pass

    results.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"query": q, "total": len(results), "results": results[:limit]}


def _snippet(text: str, query: str, window: int = 150) -> str:
    """Extract a short snippet around the first query word match."""
    lower = text.lower()
    word = query.split()[0].lower()
    idx = lower.find(word)
    if idx == -1:
        return text[:window] + ("..." if len(text) > window else "")
    start = max(0, idx - 60)
    end = min(len(text), idx + window)
    snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
    return snippet
