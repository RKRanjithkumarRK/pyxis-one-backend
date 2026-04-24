"""Celery tasks for Deep Research — Phase 6."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.research import pipeline

logger = logging.getLogger("nexusai.research.tasks")

CHANNEL_PREFIX = "research:"


def _get_sync_db():
    url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(url, pool_pre_ping=True)
    Session = sessionmaker(engine, autoflush=True)
    return Session()


def _publish(r: redis.Redis, report_id: str, event: dict) -> None:
    r.publish(f"{CHANNEL_PREFIX}{report_id}:progress", json.dumps(event))


@celery_app.task(name="research.deep_research", bind=True, max_retries=0)
def deep_research(
    self,
    report_id: str,
    query: str,
    user_id: str,
    depth: str = "standard",
) -> None:
    """
    Full deep research pipeline:
    plan → search fan-out → fetch → dedup → summarize → synthesize → verify citations
    """
    import uuid as _uuid

    sync_redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
    db = _get_sync_db()

    def pub(stage: str, progress: int, message: str) -> None:
        event = {"stage": stage, "progress": progress, "message": message}
        _publish(sync_redis, report_id, event)
        logger.info("[research:%s] %s %d%% — %s", report_id[:8], stage, progress, message)

    try:
        from app.models.research import ResearchReport
        report = db.query(ResearchReport).filter(ResearchReport.id == _uuid.UUID(report_id)).first()
        if not report:
            logger.error("Report %s not found in DB", report_id)
            return

        report.status = "running"
        report.task_id = self.request.id
        db.commit()

        # ── Stage 1: Plan ────────────────────────────────────────
        pub("planning", 5, "Generating research plan…")
        sub_questions = pipeline.plan_research(query)
        pub("planning", 12, f"Plan ready: {len(sub_questions)} sub-questions")

        # ── Stage 2: Search fan-out ───────────────────────────────
        results_per_q = {"quick": 3, "standard": 5, "deep": 8}.get(depth, 5)

        def search_progress_cb(msg: str) -> None:
            pub("searching", 20, msg)

        raw_results = pipeline.search_sources(
            sub_questions,
            settings.SERPER_API_KEY,
            results_per_q=results_per_q,
            progress_cb=search_progress_cb,
        )
        pub("searching", 38, f"Found {len(raw_results)} unique sources")

        # ── Stage 3: Fetch ────────────────────────────────────────
        max_sources = {"quick": 6, "standard": 10, "deep": 16}.get(depth, 10)
        fetched_idx = [0]

        def fetch_progress_cb(msg: str) -> None:
            fetched_idx[0] += 1
            pct = 38 + int(20 * fetched_idx[0] / max_sources)
            pub("fetching", min(pct, 58), msg)

        sources = pipeline.fetch_all_sources(
            raw_results,
            max_sources=max_sources,
            progress_cb=fetch_progress_cb,
        )
        pub("fetching", 58, f"Fetched {len(sources)} sources")

        # ── Stage 4: Summarize ────────────────────────────────────
        summaries: list[dict] = []
        for i, src in enumerate(sources):
            pct = 58 + int(17 * (i + 1) / len(sources))
            pub("summarizing", pct, f"Summarizing source {i + 1}/{len(sources)}: {src['title'][:50]}")
            summary_text = pipeline.summarize_source(src, query)
            if summary_text:
                summaries.append({**src, "summary": summary_text})

        if not summaries:
            summaries = [{**s, "summary": s.get("snippet", "")} for s in sources[:6]]

        pub("summarizing", 75, f"Summarized {len(summaries)} relevant sources")

        # ── Stage 5: Synthesize ────────────────────────────────────
        pub("synthesizing", 78, "Synthesizing findings into report…")
        report_data = pipeline.synthesize_report(query, summaries, depth=depth)
        pub("synthesizing", 90, "Synthesis complete")

        # ── Stage 6: Verify citations ──────────────────────────────
        pub("verifying", 93, "Verifying citations…")
        report_data = pipeline.verify_citations(report_data)
        report_data["sub_questions"] = sub_questions
        report_data["depth"] = depth
        report_data["generated_at"] = datetime.now(timezone.utc).isoformat()

        # ── Save to DB ────────────────────────────────────────────
        pub("saving", 97, "Saving report…")
        report.status = "complete"
        report.title = report_data.get("title", query)
        report.report = report_data
        report.sources_count = len(report_data.get("citations", []))
        report.completed_at = datetime.now(timezone.utc)
        db.commit()

        # ── Final event ───────────────────────────────────────────
        final_event = {
            "stage": "complete",
            "progress": 100,
            "message": "Research complete!",
            "report_id": report_id,
            "title": report.title,
            "sources_count": report.sources_count,
        }
        _publish(sync_redis, report_id, final_event)
        logger.info("Research report %s complete (%d sources)", report_id[:8], report.sources_count)

    except Exception as exc:
        logger.exception("Research task failed for %s: %s", report_id, exc)
        try:
            from app.models.research import ResearchReport
            report = db.query(ResearchReport).filter(
                ResearchReport.id == _uuid.UUID(report_id)
            ).first()
            if report:
                report.status = "error"
                report.error = str(exc)
                report.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass

        _publish(
            sync_redis,
            report_id,
            {"stage": "error", "progress": 0, "message": f"Research failed: {exc}"},
        )
    finally:
        db.close()
        sync_redis.close()
