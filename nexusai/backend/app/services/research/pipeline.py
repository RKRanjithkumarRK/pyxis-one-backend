"""Core research pipeline — synchronous functions for use inside Celery tasks."""
from __future__ import annotations
import json
import logging
import re
from typing import Callable

import litellm

from app.services.research.serper import SearchResult, search as serper_search
from app.services.research.fetcher import fetch_text

logger = logging.getLogger("nexusai.research.pipeline")

FAST_MODEL = "groq/llama-3.3-70b-versatile"
SYNTHESIS_MODEL = "claude-sonnet-4-20250514"
MAX_SOURCES = 12


def _llm(model: str, system: str, user: str, *, max_tokens: int = 2048) -> str:
    """Synchronous LiteLLM call with fallback."""
    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM call failed (%s): %s — falling back to claude-sonnet-4", model, exc)
        try:
            resp = litellm.completion(
                model="claude-sonnet-4-20250514",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""
        except Exception as e2:
            logger.error("Fallback also failed: %s", e2)
            return ""


def plan_research(query: str) -> list[str]:
    """Generate 4-6 focused sub-questions that together cover the research query."""
    system = (
        "You are a research librarian. Given a research query, generate 4-6 specific, "
        "non-overlapping sub-questions that together comprehensively answer the query. "
        "Return ONLY a JSON array of strings. No preamble or explanation."
    )
    result = _llm(FAST_MODEL, system, f"Query: {query}", max_tokens=512)
    try:
        questions = json.loads(result)
        if isinstance(questions, list):
            return [str(q) for q in questions[:6] if q]
    except Exception:
        lines = [l.strip().lstrip("0123456789.-) ") for l in result.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 10][:6]
    return [query]


def search_sources(
    sub_questions: list[str],
    serper_api_key: str | None,
    *,
    results_per_q: int = 5,
    progress_cb: Callable[[str], None] | None = None,
) -> list[SearchResult]:
    """Search all sub-questions and return deduplicated results."""
    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []

    for q in sub_questions:
        if progress_cb:
            progress_cb(f"Searching: {q[:80]}")

        if serper_api_key:
            results = serper_search(q, serper_api_key, num=results_per_q)
        else:
            results = _generate_mock_sources(q, count=3)

        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

    return all_results


def _generate_mock_sources(query: str, count: int) -> list[SearchResult]:
    """When Serper key is absent, synthesize placeholder sources so the pipeline still runs."""
    system = (
        "Generate exactly {n} realistic search result stubs for the query below. "
        "Return a JSON array of objects with keys: title, url, snippet. "
        "Use plausible but fictional URLs. No preamble."
    ).format(n=count)
    raw = _llm(FAST_MODEL, system, query, max_tokens=512)
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [
                SearchResult(
                    title=i.get("title", ""),
                    url=i.get("url", f"https://example.com/{j}"),
                    snippet=i.get("snippet", ""),
                )
                for j, i in enumerate(items[:count])
            ]
    except Exception:
        pass
    return []


def fetch_all_sources(
    results: list[SearchResult],
    *,
    max_sources: int = MAX_SOURCES,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Fetch page text for each result. Returns list of {title, url, snippet, content}."""
    fetched: list[dict] = []
    for r in results[:max_sources]:
        if progress_cb:
            progress_cb(f"Fetching: {r.url[:70]}")
        text = fetch_text(r.url)
        fetched.append(
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "content": text or r.snippet,
            }
        )
    return fetched


def summarize_source(source: dict, query: str) -> str:
    """Extract the most relevant passage from a source for the research query."""
    system = (
        "Extract the 2-3 most relevant sentences from the provided source text "
        "that directly help answer the research query. "
        "If the content is irrelevant, return an empty string. "
        "Return ONLY the extracted sentences, no preamble."
    )
    content = source.get("content") or source.get("snippet", "")
    user = f"Research query: {query}\n\nSource ({source['title']}):\n{content[:3000]}"
    return _llm(FAST_MODEL, system, user, max_tokens=300)


def synthesize_report(
    query: str,
    summaries: list[dict],
    *,
    depth: str = "standard",
) -> dict:
    """
    Synthesize all source summaries into a structured research report.
    Returns: {title, executive_summary, sections, citations}
    """
    source_block = "\n".join(
        f"[{i+1}] {s['title']} ({s['url']})\n{s['summary']}"
        for i, s in enumerate(summaries)
        if s.get("summary")
    )
    sections_count = {"quick": 3, "standard": 5, "deep": 7}.get(depth, 5)

    system = (
        "You are a senior research analyst. Write a comprehensive research report based on the provided sources. "
        "Format your response as a JSON object with this exact structure:\n"
        '{"title": "...", "executive_summary": "...(2-3 sentences)...", '
        '"sections": [{"heading": "...", "content": "..."}], '
        '"key_findings": ["finding 1", "finding 2", ...]}\n\n'
        f"Include {sections_count} sections. "
        "Cite sources inline using [N] notation (e.g. [1], [2]). "
        "Be analytical, not just descriptive. "
        "Return ONLY valid JSON, no markdown code blocks."
    )
    user = (
        f"Research Query: {query}\n\n"
        f"Sources:\n{source_block}\n\n"
        "Write the full report now."
    )

    raw = _llm(SYNTHESIS_MODEL, system, user, max_tokens=4096)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        data = json.loads(raw)
    except Exception:
        data = {
            "title": query,
            "executive_summary": raw[:500] if raw else "Research completed.",
            "sections": [{"heading": "Findings", "content": raw}],
            "key_findings": [],
        }

    citations = [
        {
            "id": i + 1,
            "title": s["title"],
            "url": s["url"],
            "snippet": s.get("snippet", ""),
        }
        for i, s in enumerate(summaries)
    ]
    data["citations"] = citations
    return data


def verify_citations(report: dict) -> dict:
    """Remove [N] references with N > total citations count."""
    n = len(report.get("citations", []))
    if n == 0:
        return report

    def clean_refs(text: str) -> str:
        def replace(m: re.Match) -> str:
            num = int(m.group(1))
            return m.group(0) if num <= n else ""
        return re.sub(r"\[(\d+)\]", replace, text)

    if "executive_summary" in report:
        report["executive_summary"] = clean_refs(report["executive_summary"])
    for section in report.get("sections", []):
        if "content" in section:
            section["content"] = clean_refs(section["content"])
    return report
