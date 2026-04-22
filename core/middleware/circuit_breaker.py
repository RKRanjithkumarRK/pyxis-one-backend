"""
Circuit breaker per AI provider.
Prevents cascading failures when a provider is down.

States:
  CLOSED   → normal operation
  OPEN     → blocking all calls, returning cached error
  HALF_OPEN → testing recovery with one probe request
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5      # failures before opening
    recovery_timeout: float = 30.0  # seconds before trying recovery
    success_threshold: int = 2       # successes in HALF_OPEN before closing

    _failures: int = field(default=0, init=False, repr=False)
    _successes: int = field(default=0, init=False, repr=False)
    _state: str = field(default="closed", init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)

    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half_open"
                self._successes = 0
                return False
            return True
        return False

    def record_success(self) -> None:
        if self._state == "half_open":
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state = "closed"
                self._failures = 0
        elif self._state == "closed":
            self._failures = max(0, self._failures - 1)

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "open"

    @property
    def state(self) -> str:
        return self._state


# ── Global registry ───────────────────────────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {
    "openai":    CircuitBreaker(name="openai",    failure_threshold=5, recovery_timeout=30),
    "anthropic": CircuitBreaker(name="anthropic", failure_threshold=5, recovery_timeout=30),
    "groq":      CircuitBreaker(name="groq",      failure_threshold=3, recovery_timeout=20),
    "gemini":    CircuitBreaker(name="gemini",    failure_threshold=3, recovery_timeout=20),
}


def get(provider: str) -> CircuitBreaker:
    return _breakers.get(provider, CircuitBreaker(name=provider))


def check_open(provider: str) -> bool:
    return get(provider).is_open()


def success(provider: str) -> None:
    get(provider).record_success()


def failure(provider: str) -> None:
    get(provider).record_failure()


def status() -> dict[str, str]:
    return {name: cb.state for name, cb in _breakers.items()}
