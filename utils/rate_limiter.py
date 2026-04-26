import asyncio
import threading
import time

from providers.errors import CircuitOpenError

PROVIDER_DEFAULTS: dict[str, int] = {
    "anthropic": 5,
    "openai":    5,
    "groq":      5,
    "ollama":    1,
}

_semaphore: asyncio.Semaphore | None = None


def get_semaphore(provider: str, user_override: int | None = None) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        limit = user_override if user_override is not None \
                else PROVIDER_DEFAULTS.get(provider, 5)
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore


def reset_semaphore() -> None:
    global _semaphore
    _semaphore = None


class CircuitBreaker:
    """
    Globally shared circuit breaker for all provider calls.

    Trips after `threshold` consecutive provider failures (RateLimitError /
    ServerError) and holds OPEN for `cooldown` seconds.  After the cooldown
    elapsed one probe is allowed through; success closes the circuit, failure
    restarts the cooldown.

    Uses a threading.Lock (not asyncio.Lock) so state persists correctly
    across multiple asyncio.run() invocations (e.g., between test runs).
    The lock is never held across an `await`, so it never blocks the event loop.
    """

    def __init__(self) -> None:
        self._failures: int = 0
        self._open_since: float | None = None
        self._lock = threading.Lock()

    def check(self, threshold: int, cooldown: float) -> None:
        """Raise CircuitOpenError if the circuit is open and the cooldown has not elapsed."""
        with self._lock:
            if self._open_since is None:
                return
            elapsed = time.monotonic() - self._open_since
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                raise CircuitOpenError(
                    f"API appears unavailable — circuit open, retry in {remaining}s."
                )
            # Cooldown elapsed: allow one probe (half-open — close tentatively)
            self._open_since = None

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self, threshold: int) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= threshold:
                self._open_since = time.monotonic()

    def reset(self) -> None:
        """Reset all state. Call between test runs to prevent state leakage."""
        with self._lock:
            self._failures = 0
            self._open_since = None


_circuit_breaker: CircuitBreaker = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    return _circuit_breaker


def reset_circuit_breaker() -> None:
    _circuit_breaker.reset()
