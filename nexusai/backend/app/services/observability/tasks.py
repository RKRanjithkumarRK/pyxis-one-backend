"""SLO burn-rate monitoring — runs every 5 minutes via Celery Beat."""
from __future__ import annotations
import asyncio
import logging
from app.core.celery_app import celery_app

logger = logging.getLogger("nexusai.observability")

# SLO thresholds
SLO_THRESHOLDS = {
    "p99_latency_ms": 2000.0,    # 99th pct latency must be < 2s
    "error_rate_1m": 0.05,        # error rate < 5% over 1m window
    "llm_error_rate": 0.10,       # LLM call failures < 10%
}


@celery_app.task(name="observability.check_slo_burn", bind=True, max_retries=1)
def check_slo_burn(self):  # noqa: ANN001
    """Reads Prometheus-style counters from Redis and fires PagerDuty if SLOs are burning."""
    asyncio.run(_async_check())


async def _async_check() -> None:
    from app.core.redis import redis_client
    from app.core.telemetry import record_slo_burn

    for metric, threshold in SLO_THRESHOLDS.items():
        raw = await redis_client.get(f"slo:{metric}")
        if raw is None:
            continue
        try:
            value = float(raw)
            await record_slo_burn(metric, value, threshold)
        except ValueError:
            logger.warning("Invalid SLO metric value for %s: %s", metric, raw)
