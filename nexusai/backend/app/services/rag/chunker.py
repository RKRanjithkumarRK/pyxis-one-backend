"""Text chunking with character-level sliding window."""
from __future__ import annotations

CHUNK_SIZE = 1200   # characters per chunk
CHUNK_OVERLAP = 150  # overlap between adjacent chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into overlapping chunks at paragraph/sentence boundaries where possible.
    Returns a list of non-empty strings.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break at a paragraph boundary
        boundary = _find_boundary(text, end, lookahead=200)
        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, boundary - overlap)

    return chunks


def _find_boundary(text: str, pos: int, lookahead: int = 200) -> int:
    """Find the nearest good boundary (double-newline, sentence end) near `pos`."""
    # Prefer double newline (paragraph break)
    idx = text.find("\n\n", pos, pos + lookahead)
    if idx != -1:
        return idx + 2

    # Settle for sentence-ending punctuation followed by space
    for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.find(punct, pos, pos + lookahead)
        if idx != -1:
            return idx + len(punct)

    # Last resort: single newline
    idx = text.find("\n", pos, pos + lookahead)
    if idx != -1:
        return idx + 1

    # Hard cut
    return pos
