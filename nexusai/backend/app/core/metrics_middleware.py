"""Request metrics middleware — writes p99 latency and error rate counters to Redis."""
from __future__ import annotations
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("nexusai.metrics")

# Rolling window length for error rate (samples)
_WINDOW = 1000


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            status = 500
            raise exc from exc
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            await self._record(elapsed_ms, status)
        return response

    async def _record(self, latency_ms: float, status: int) -> None:
        try:
            from app.core.redis import redis_client

            is_error = 1 if status >= 500 else 0

            # Append latency to sorted set keyed by timestamp — prune to last 1000
            await redis_client.lpush("metrics:latency", latency_ms)
            await redis_client.ltrim("metrics:latency", 0, _WINDOW - 1)

            # Maintain error window
            await redis_client.lpush("metrics:errors", is_error)
            await redis_client.ltrim("metrics:errors", 0, _WINDOW - 1)

            # Compute and store p99 + error rate for SLO task to read
            raw_latencies = await redis_client.lrange("metrics:latency", 0, -1)
            if raw_latencies:
                latencies = sorted(float(v) for v in raw_latencies)
                idx = int(len(latencies) * 0.99)
                p99 = latencies[min(idx, len(latencies) - 1)]
                await redis_client.set("slo:p99_latency_ms", str(p99))

            raw_errors = await redis_client.lrange("metrics:errors", 0, -1)
            if raw_errors:
                error_rate = sum(int(v) for v in raw_errors) / len(raw_errors)
                await redis_client.set("slo:error_rate_1m", str(error_rate))
        except Exception:
            pass  # metrics are best-effort; never break request handling
