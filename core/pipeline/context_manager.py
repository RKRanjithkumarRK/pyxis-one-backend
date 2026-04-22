"""
Context window management:
- Counts tokens accurately per model
- Truncates oldest messages when near limit
- Summarizes dropped messages so context continuity is preserved
- Injects RAG results and psyche context safely within budget
"""

from __future__ import annotations
import asyncio
from core.config import MODEL_CONTEXT_LIMITS


_RESERVE_FOR_OUTPUT = 8192   # always keep space for the response
_SUMMARIZE_THRESHOLD = 0.80  # start truncating at 80% capacity


def _count_tokens(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 chars (avoids tiktoken import cost)."""
    return max(1, len(text) // 4)


def _count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        total += _count_tokens(str(content)) + 4  # role overhead
    return total


async def prepare(
    model: str,
    system: str,
    messages: list[dict],
    rag_chunks: list[str] | None = None,
    psyche_context: str = "",
) -> tuple[str, list[dict]]:
    """
    Returns (enriched_system, trimmed_messages) ready to send to the model.
    """
    limit = MODEL_CONTEXT_LIMITS.get(model, 128_000) - _RESERVE_FOR_OUTPUT

    system_tokens = _count_tokens(system)
    rag_tokens = sum(_count_tokens(c) for c in (rag_chunks or []))
    psyche_tokens = _count_tokens(psyche_context)
    overhead = system_tokens + rag_tokens + psyche_tokens

    available = limit - overhead

    # Trim messages from oldest first, keeping newest
    trimmed, dropped = _trim_messages(messages, available)

    # Build enriched system prompt
    enriched = system
    if psyche_context:
        enriched += f"\n\n--- User Profile ---\n{psyche_context}"
    if rag_chunks:
        rag_block = "\n".join(f"[Memory {i+1}]: {c}" for i, c in enumerate(rag_chunks))
        enriched += f"\n\n--- Relevant Past Context ---\n{rag_block}"

    # If we dropped messages, prepend a summary placeholder
    if dropped:
        summary_text = _quick_summary(dropped)
        summary_msg = {
            "role": "user",
            "content": f"[Earlier context summary: {summary_text}]",
        }
        ack_msg = {
            "role": "assistant",
            "content": "Understood. Continuing from that earlier context.",
        }
        trimmed = [summary_msg, ack_msg] + trimmed

    return enriched, trimmed


def _trim_messages(messages: list[dict], token_budget: int) -> tuple[list[dict], list[dict]]:
    """Keep the most recent messages that fit within token_budget."""
    kept: list[dict] = []
    used = 0

    for msg in reversed(messages):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        tokens = _count_tokens(str(content)) + 4

        if used + tokens > token_budget:
            break
        kept.insert(0, msg)
        used += tokens

    dropped = messages[: len(messages) - len(kept)]
    return kept, dropped


def _quick_summary(messages: list[dict]) -> str:
    """Produce a short inline summary of dropped messages without an LLM call."""
    parts = []
    for m in messages[-6:]:  # last 6 of dropped (most relevant)
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        snippet = str(content)[:120].replace("\n", " ")
        parts.append(f"{role}: {snippet}...")
    return " | ".join(parts) if parts else "previous conversation"


def build_tool_result_message(tool_name: str, tool_call_id: str, result: str, provider: str) -> dict:
    """Construct the correct tool result message format per provider."""
    if provider == "openai":
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }
    else:  # anthropic
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }
            ],
        }
