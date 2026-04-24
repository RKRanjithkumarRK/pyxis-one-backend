"""Serper.dev API client for web search."""
from __future__ import annotations
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("nexusai.research.serper")

SERPER_URL = "https://google.serper.dev/search"
TIMEOUT_SECONDS = 10


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    date: str | None = None
    position: int = 0


def search(query: str, api_key: str, *, num: int = 10) -> list[SearchResult]:
    """Call Serper API and return organic search results."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num, "gl": "us", "hl": "en"}
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            resp = client.post(SERPER_URL, json=payload, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Serper request failed for '%s': %s", query, exc)
        return []

    data = resp.json()
    results: list[SearchResult] = []
    for i, item in enumerate(data.get("organic", []), start=1):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                date=item.get("date"),
                position=i,
            )
        )
    return results


def knowledge_graph(query: str, api_key: str) -> dict | None:
    """Return knowledge graph entry if available."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            resp = client.post(SERPER_URL, json={"q": query}, headers=headers)
            resp.raise_for_status()
        return resp.json().get("knowledgeGraph")
    except Exception:
        return None
