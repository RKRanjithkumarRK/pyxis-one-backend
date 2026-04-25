from __future__ import annotations
import time
import hashlib
import logging
from typing import Any
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode
from app.core.config import settings

logger = logging.getLogger("nexusai.telemetry")
_provider: TracerProvider | None = None


def setup_telemetry() -> None:
    global _provider
    resource = Resource.create({SERVICE_NAME: "nexusai-backend", "deployment.environment": settings.ENVIRONMENT})
    _provider = TracerProvider(resource=resource)

    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    else:
        exporter = ConsoleSpanExporter() if not settings.is_production else None  # type: ignore[assignment]

    if exporter:
        _provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(_provider)

    # FastAPI auto-instrumentation
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument()
    except Exception:
        pass

    # Sentry — scrub PII from events
    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        def _scrub_pii(event: dict, hint: dict) -> dict | None:  # noqa: ARG001
            for key in ("email", "username", "name", "ip_address", "phone"):
                if key in event.get("user", {}):
                    event["user"][key] = "[Filtered]"
            return event

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            traces_sample_rate=0.05,
            profiles_sample_rate=0.01,
            send_default_pii=False,
            before_send=_scrub_pii,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        )

    # PostHog
    if settings.POSTHOG_API_KEY:
        try:
            import posthog
            posthog.api_key = settings.POSTHOG_API_KEY
            posthog.host = "https://app.posthog.com"
        except ImportError:
            logger.warning("posthog package not installed; skipping analytics setup")


tracer = trace.get_tracer("nexusai")


def track_event(distinct_id: str, event: str, properties: dict | None = None) -> None:
    """Fire-and-forget PostHog event. Silently no-ops if PostHog is unconfigured."""
    if not settings.POSTHOG_API_KEY:
        return
    try:
        import posthog
        posthog.capture(distinct_id, event, properties or {})
    except Exception:
        pass


class LLMSpan:
    """Context manager that wraps an LLM call in an OTel span with standard attributes."""

    def __init__(self, model: str, user_id: str | None = None, operation: str = "chat") -> None:
        self._model = model
        self._user_id = user_id
        self._operation = operation
        self._span: Any = None
        self._t0: float = 0.0

    def __enter__(self) -> "LLMSpan":
        self._t0 = time.perf_counter()
        self._span = tracer.start_span(f"llm.{self._operation}")
        self._span.set_attribute("llm.model", self._model)
        if self._user_id:
            # hash to avoid PII in traces
            self._span.set_attribute("user.id_hash", hashlib.sha256(self._user_id.encode()).hexdigest()[:16])
        return self

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        if self._span:
            self._span.set_attribute("llm.prompt_tokens", prompt_tokens)
            self._span.set_attribute("llm.completion_tokens", completion_tokens)
            self._span.set_attribute("llm.total_tokens", prompt_tokens + completion_tokens)

    def record_error(self, exc: Exception) -> None:
        if self._span:
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
            self._span.record_exception(exc)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span:
            elapsed_ms = (time.perf_counter() - self._t0) * 1000
            self._span.set_attribute("llm.latency_ms", round(elapsed_ms, 1))
            if exc_type:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self._span.end()


async def record_slo_burn(metric: str, value: float, threshold: float) -> None:
    """Emit a SLO burn-rate alert to the configured alerting sink."""
    if value <= threshold:
        return
    logger.error("SLO BURN ALERT — %s=%.2f exceeds threshold %.2f", metric, value, threshold)
    # PagerDuty integration
    if not getattr(settings, "PAGERDUTY_ROUTING_KEY", None):
        return
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json={
                    "routing_key": settings.PAGERDUTY_ROUTING_KEY,  # type: ignore[attr-defined]
                    "event_action": "trigger",
                    "payload": {
                        "summary": f"NexusAI SLO burn: {metric}={value:.2f} (threshold {threshold:.2f})",
                        "severity": "critical",
                        "source": "nexusai-backend",
                        "custom_details": {"metric": metric, "value": value, "threshold": threshold},
                    },
                },
                timeout=5,
            )
    except Exception as exc:
        logger.warning("PagerDuty alert failed: %s", exc)
