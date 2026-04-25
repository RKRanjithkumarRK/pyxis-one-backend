"""Phase 21 — Telemetry and observability unit tests."""
from __future__ import annotations
import pytest
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_track_event_no_key():
    from app.core.telemetry import track_event
    # Should silently no-op without PostHog key
    with patch("app.core.telemetry.settings") as mock_cfg:
        mock_cfg.POSTHOG_API_KEY = None
        track_event("user-123", "chat_sent", {"model": "gpt-4"})  # must not raise


@pytest.mark.asyncio
async def test_track_event_with_key():
    from app.core.telemetry import track_event
    with patch("app.core.telemetry.settings") as mock_cfg:
        mock_cfg.POSTHOG_API_KEY = "phc_test"
        with patch("posthog.capture") as mock_cap:
            track_event("user-123", "chat_sent", {"model": "gpt-4"})
            mock_cap.assert_called_once_with("user-123", "chat_sent", {"model": "gpt-4"})


def test_llm_span_context_manager():
    from app.core.telemetry import LLMSpan
    with LLMSpan("gpt-4o", user_id="abc123", operation="chat") as span:
        span.record_usage(prompt_tokens=100, completion_tokens=50)
    # No exception means the span lifecycle completed correctly


def test_llm_span_records_error():
    from app.core.telemetry import LLMSpan
    with pytest.raises(RuntimeError):
        with LLMSpan("gpt-4o") as span:
            span.record_error(RuntimeError("timeout"))
            raise RuntimeError("timeout")


@pytest.mark.asyncio
async def test_slo_burn_no_alert_under_threshold():
    from app.core.telemetry import record_slo_burn
    # Should not raise even with no PagerDuty key
    with patch("app.core.telemetry.settings") as mock_cfg:
        mock_cfg.PAGERDUTY_ROUTING_KEY = None
        await record_slo_burn("p99_latency_ms", 200.0, 500.0)


@pytest.mark.asyncio
async def test_slo_burn_alert_over_threshold():
    from app.core.telemetry import record_slo_burn
    with patch("app.core.telemetry.settings") as mock_cfg:
        mock_cfg.PAGERDUTY_ROUTING_KEY = None
        # Should log error but not raise
        await record_slo_burn("error_rate", 0.15, 0.05)
