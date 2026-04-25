"""Memory service — fact extraction, embedding, and retrieval."""
from __future__ import annotations
import json
import logging
import uuid
from typing import Any

import litellm
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.memory import MemoryRepository

logger = logging.getLogger("nexusai.memory")

EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_FACTS_PER_EXCHANGE = 5
MAX_SYSTEM_MEMORIES = 6


_EXTRACTION_SYSTEM = """\
You extract short personal facts about the USER from a conversation exchange.
Return a valid JSON array of strings — zero to five concise facts.

Rules:
- Only extract facts explicitly stated or clearly implied about the user.
- Do NOT extract general knowledge, opinions about third-party topics, or AI responses.
- Facts must be relevant to personalizing future conversations.
- Each fact is one short sentence (< 20 words).
- Return [] if nothing worth remembering is present.

Example output: ["User prefers Python over JavaScript", "User is building a SaaS product"]"""


async def extract_facts(user_msg: str, assistant_msg: str) -> list[str]:
    """Use LiteLLM to extract personalisable facts from one exchange."""
    content = (
        f"User said: {user_msg[:1500]}\n\n"
        f"Assistant said: {assistant_msg[:1500]}"
    )
    try:
        resp = await litellm.acompletion(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        facts: Any = json.loads(raw)
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if f and isinstance(f, str)][:MAX_FACTS_PER_EXCHANGE]
        return []
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)
        return []


async def get_embedding(text: str) -> list[float] | None:
    """Embed text with text-embedding-3-small (1536 dims)."""
    try:
        resp = await litellm.aembedding(
            model=EMBEDDING_MODEL,
            input=[text],
        )
        return resp.data[0].embedding
    except Exception as exc:
        logger.warning("Embedding failed for memory: %s", exc)
        return None


async def store_memories(
    db: AsyncSession,
    user_id: uuid.UUID,
    facts: list[str],
    source_message_id: uuid.UUID | None = None,
) -> int:
    """Embed facts, deduplicate, and persist to DB. Returns count stored."""
    stored = 0
    for fact in facts:
        embedding = await get_embedding(fact)
        if embedding is None:
            # Store without embedding (no dedup possible)
            await MemoryRepository.create(
                db, user_id, fact, embedding=None, source_message_id=source_message_id
            )
            stored += 1
            continue

        is_dup = await MemoryRepository.deduplicate_check(db, user_id, embedding)
        if not is_dup:
            await MemoryRepository.create(
                db, user_id, fact,
                embedding=embedding,
                source_message_id=source_message_id,
            )
            stored += 1
        else:
            logger.debug("Skipping duplicate memory: %s", fact[:60])
    await db.commit()
    return stored


async def retrieve_for_message(
    db: AsyncSession,
    user_id: uuid.UUID,
    user_message: str,
    *,
    limit: int = MAX_SYSTEM_MEMORIES,
) -> list[str]:
    """Return a list of memory strings relevant to the user's message."""
    embedding = await get_embedding(user_message)
    if embedding is None:
        # Fall back to most recently used
        mems = await MemoryRepository.list_for_user(db, user_id, limit=limit)
        return [m.fact for m in mems[:limit]]

    mems = await MemoryRepository.search_similar(db, user_id, embedding, limit=limit)
    facts = []
    for m in mems:
        await MemoryRepository.mark_used(db, m)
        facts.append(m.fact)
    await db.flush()
    return facts


def build_memory_block(facts: list[str]) -> str:
    """Format memory facts for injection into the system prompt."""
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts)
    return f"[User memories — use these to personalise your response]\n{lines}"
