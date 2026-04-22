"""
Web search tool using Brave Search API.
Falls back to a basic DuckDuckGo instant-answer scrape when key is missing.
"""

from __future__ import annotations
import httpx
from core.config import settings


async def search(query: str, num_results: int = 5) -> str:
    """Returns formatted search results as a string for model consumption."""
    if settings.BRAVE_SEARCH_API_KEY:
        return await _brave_search(query, num_results)
    return await _ddg_fallback(query)


async def _brave_search(query: str, num_results: int) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": settings.BRAVE_SEARCH_API_KEY,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": min(num_results, 10),
        "text_decorations": "false",
        "search_lang": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()

        results = data.get("web", {}).get("results", [])
        if not results:
            return "No results found."

        parts = [f"Search results for: {query}\n"]
        for i, result in enumerate(results[:num_results], 1):
            title = result.get("title", "No title")
            url_ = result.get("url", "")
            desc = result.get("description", result.get("snippet", "No description"))
            parts.append(f"[{i}] {title}\n    URL: {url_}\n    {desc}\n")

        return "\n".join(parts)

    except httpx.HTTPStatusError as e:
        return f"Search failed (HTTP {e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Search error: {str(e)[:200]}"


async def _ddg_fallback(query: str) -> str:
    """DuckDuckGo instant answer — no API key required, limited results."""
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params=params)
            data = r.json()

        abstract = data.get("AbstractText", "")
        if abstract:
            source = data.get("AbstractSource", "DuckDuckGo")
            return f"[{source}]: {abstract}"

        related = data.get("RelatedTopics", [])[:3]
        parts = [f"Search: {query}"]
        for item in related:
            if isinstance(item, dict) and "Text" in item:
                parts.append(f"- {item['Text'][:200]}")
        return "\n".join(parts) if len(parts) > 1 else f"No results found for: {query}"

    except Exception as e:
        return f"Search unavailable: {str(e)[:100]}"
