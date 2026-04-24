"""Fetch and extract text content from URLs."""
from __future__ import annotations
import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger("nexusai.research.fetcher")

FETCH_TIMEOUT = 8
MAX_CHARS = 4000
MAX_REDIRECTS = 3

SKIP_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3", ".zip"}


class _TextExtractor(HTMLParser):
    """Minimal HTML → text extractor. Strips tags, scripts, and styles."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        self._current_tag = ""

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        raw = re.sub(r"\s{2,}", " ", raw)
        return raw.strip()


def _should_skip(url: str) -> bool:
    lower = url.lower()
    return any(lower.endswith(ext) for ext in SKIP_EXTENSIONS)


def fetch_text(url: str) -> str | None:
    """Fetch URL and return extracted plain text, or None on failure."""
    if _should_skip(url):
        return None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; NexusAI-Research/1.0; "
            "+https://nexusai.app/research)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            max_redirects=MAX_REDIRECTS,
            follow_redirects=True,
        ) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                return None
            parser = _TextExtractor()
            parser.feed(resp.text)
            text = parser.get_text()
            return text[:MAX_CHARS] if text else None
    except Exception as exc:
        logger.debug("Fetch failed for %s: %s", url, exc)
        return None
