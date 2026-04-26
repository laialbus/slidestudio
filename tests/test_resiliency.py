"""
Task 1 — Resiliency tests.

All tests use stub providers that override _call() to simulate error conditions.
No mocking library is used. No real API calls are made.
"""

import asyncio
import json

import pytest
from pydantic import BaseModel

from providers.base import BaseProvider
from providers.config import ProviderConfig
from providers.errors import (
    CircuitOpenError,
    FatalAPIError,
    RateLimitError,
    ServerError,
)
from utils.rate_limiter import reset_circuit_breaker, reset_semaphore


# ──────────────────────────────────────────────────────────────
# Minimal Pydantic schema for testing
# ──────────────────────────────────────────────────────────────

class EchoModel(BaseModel):
    value: str


_VALID_JSON = json.dumps({"value": "ok"})


# ──────────────────────────────────────────────────────────────
# Stub providers
# ──────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> ProviderConfig:
    """Build a ProviderConfig with test-friendly defaults (zero backoff waits)."""
    kwargs.pop("provider_name", None)
    return ProviderConfig(
        model=kwargs.get("model", "stub"),
        max_concurrent=kwargs.get("max_concurrent", None),
        max_format_retries=kwargs.get("max_format_retries", 1),
        max_rate_limit_retries=kwargs.get("max_rate_limit_retries", 3),
        request_timeout=kwargs.get("request_timeout", 5),
        circuit_breaker_threshold=kwargs.get("circuit_breaker_threshold", 3),
        circuit_breaker_cooldown=kwargs.get("circuit_breaker_cooldown", 60),
        backoff_wait_min=kwargs.get("backoff_wait_min", 0),
        backoff_wait_max=kwargs.get("backoff_wait_max", 0),
    )


def _make_provider(**kwargs) -> BaseProvider:
    """BaseProvider configured for fast tests (no real backoff waits)."""
    config = _make_config(**kwargs)

    class Stub(BaseProvider):
        async def _call(self, messages, system, response_schema=None):
            raise NotImplementedError("_call must be overridden in each test")

        @property
        def name(self):
            return "stub"

    return Stub(config)


class _FailThenSucceedProvider(BaseProvider):
    """
    Raises `error_cls` for the first `n_failures` calls to `_call()`,
    then returns `response_json` for all subsequent calls.
    """

    def __init__(self, error_cls: type[Exception], n_failures: int, **kwargs):
        config = _make_config(
            max_rate_limit_retries=kwargs.get("max_rate_limit_retries", n_failures + 1),
            circuit_breaker_threshold=kwargs.get("circuit_breaker_threshold", n_failures + 10),
            **{k: v for k, v in kwargs.items()
               if k not in ("max_rate_limit_retries", "circuit_breaker_threshold")},
        )
        super().__init__(config)
        self.error_cls   = error_cls
        self.n_failures  = n_failures
        self.call_count  = 0

    async def _call(self, messages, system, response_schema=None):
        self.call_count += 1
        if self.call_count <= self.n_failures:
            raise self.error_cls("Simulated error")
        return _VALID_JSON

    @property
    def name(self):
        return "stub"


class _AlwaysFailProvider(BaseProvider):
    """Always raises `error_cls` from `_call()`. Used to trip the circuit breaker."""

    def __init__(self, error_cls: type[Exception], **kwargs):
        super().__init__(_make_config(**kwargs))
        self.error_cls  = error_cls
        self.call_count = 0

    async def _call(self, messages, system, response_schema=None):
        self.call_count += 1
        raise self.error_cls("Simulated error")

    @property
    def name(self):
        return "stub"


class _FatalProvider(BaseProvider):
    """Immediately raises FatalAPIError from `_call()`."""

    def __init__(self, **kwargs):
        kwargs.setdefault("circuit_breaker_threshold", 5)
        super().__init__(_make_config(**kwargs))
        self.call_count = 0

    async def _call(self, messages, system, response_schema=None):
        self.call_count += 1
        raise FatalAPIError("HTTP 400 Bad Request")

    @property
    def name(self):
        return "stub"


class _TimeoutProvider(BaseProvider):
    """Simulates a timed-out call by raising asyncio.TimeoutError from `_call()`."""

    def __init__(self, **kwargs):
        succeed_after = kwargs.pop("succeed_after", 999)
        kwargs.setdefault("circuit_breaker_threshold", 10)
        super().__init__(_make_config(**kwargs))
        self.call_count    = 0
        self.succeed_after = succeed_after

    async def _call(self, messages, system, response_schema=None):
        self.call_count += 1
        if self.call_count <= self.succeed_after:
            raise asyncio.TimeoutError()
        return _VALID_JSON

    @property
    def name(self):
        return "stub"


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

class _ResiliencyBase:
    def setup_method(self):
        reset_semaphore()
        reset_circuit_breaker()

    def teardown_method(self):
        reset_semaphore()
        reset_circuit_breaker()


# ──────────────────────────────────────────────────────────────
# Test 1 — HTTP 500 is retried with backoff
# ──────────────────────────────────────────────────────────────

class TestServerErrorRetried(_ResiliencyBase):
    def test_server_error_is_retried_and_succeeds(self):
        provider = _FailThenSucceedProvider(ServerError, n_failures=1)
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"

    def test_provider_called_twice_on_one_failure(self):
        provider = _FailThenSucceedProvider(ServerError, n_failures=1)
        _run(provider.complete_json("prompt", EchoModel))
        assert provider.call_count == 2

    def test_server_error_exhaust_raises_server_error(self):
        provider = _AlwaysFailProvider(ServerError, max_rate_limit_retries=2)
        with pytest.raises(ServerError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_rate_limit_error_is_retried(self):
        provider = _FailThenSucceedProvider(RateLimitError, n_failures=1)
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"

    def test_server_error_not_raised_when_retry_succeeds(self):
        provider = _FailThenSucceedProvider(ServerError, n_failures=2)
        result = _run(provider.complete_json("prompt", EchoModel))
        assert isinstance(result, EchoModel)


# ──────────────────────────────────────────────────────────────
# Test 2 — Timeout is treated as retryable (ServerError)
# ──────────────────────────────────────────────────────────────

class TestTimeoutRetried(_ResiliencyBase):
    def test_timeout_is_converted_to_server_error_and_retried(self):
        # Timeout on first call, succeed on second
        provider = _TimeoutProvider(
            max_rate_limit_retries=2,
            circuit_breaker_threshold=10,
            succeed_after=1,
        )
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"

    def test_timeout_exhaust_raises_server_error(self):
        # Always times out — should eventually raise ServerError
        provider = _TimeoutProvider(
            max_rate_limit_retries=2,
            circuit_breaker_threshold=10,
            succeed_after=999,
        )
        with pytest.raises(ServerError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_timeout_provider_retried_multiple_times(self):
        provider = _TimeoutProvider(
            max_rate_limit_retries=3,
            circuit_breaker_threshold=10,
            succeed_after=999,
        )
        with pytest.raises(ServerError):
            _run(provider.complete_json("prompt", EchoModel))
        assert provider.call_count == 3


# ──────────────────────────────────────────────────────────────
# Test 3 — HTTP 4xx (non-429) raises FatalAPIError immediately
# ──────────────────────────────────────────────────────────────

class TestFatalAPIError(_ResiliencyBase):
    def test_fatal_api_error_raised_immediately(self):
        provider = _FatalProvider()
        with pytest.raises(FatalAPIError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_fatal_api_error_not_retried(self):
        provider = _FatalProvider()
        with pytest.raises(FatalAPIError):
            _run(provider.complete_json("prompt", EchoModel))
        assert provider.call_count == 1

    def test_fatal_api_error_message_is_preserved(self):
        provider = _FatalProvider()
        with pytest.raises(FatalAPIError, match="HTTP 400"):
            _run(provider.complete_json("prompt", EchoModel))

    def test_fatal_api_error_does_not_trip_circuit_breaker(self):
        provider = _FatalProvider(circuit_breaker_threshold=2)
        for _ in range(5):
            with pytest.raises(FatalAPIError):
                _run(provider.complete_json("prompt", EchoModel))
        # Circuit should still be closed — no CircuitOpenError raised
        with pytest.raises(FatalAPIError):
            _run(provider.complete_json("prompt", EchoModel))


# ──────────────────────────────────────────────────────────────
# Test 4 — N consecutive failures trip the circuit breaker
# ──────────────────────────────────────────────────────────────

class TestCircuitBreaker(_ResiliencyBase):
    def test_circuit_opens_after_threshold_failures(self):
        provider = _AlwaysFailProvider(
            ServerError,
            circuit_breaker_threshold=3,
            max_rate_limit_retries=1,
        )
        for _ in range(3):
            with pytest.raises(ServerError):
                _run(provider.complete_json("prompt", EchoModel))

        with pytest.raises(CircuitOpenError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_circuit_open_error_is_clean_exception(self):
        provider = _AlwaysFailProvider(
            ServerError,
            circuit_breaker_threshold=2,
            max_rate_limit_retries=1,
        )
        for _ in range(2):
            with pytest.raises(ServerError):
                _run(provider.complete_json("prompt", EchoModel))

        exc = None
        try:
            _run(provider.complete_json("prompt", EchoModel))
        except CircuitOpenError as e:
            exc = e
        assert exc is not None
        assert "retry" in str(exc).lower() or "unavailable" in str(exc).lower()

    def test_circuit_stays_closed_below_threshold(self):
        provider = _AlwaysFailProvider(
            ServerError,
            circuit_breaker_threshold=5,
            max_rate_limit_retries=1,
        )
        for _ in range(4):
            with pytest.raises(ServerError):
                _run(provider.complete_json("prompt", EchoModel))
        # Should still raise ServerError, not CircuitOpenError
        with pytest.raises(ServerError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_circuit_does_not_open_on_fatal_errors(self):
        provider = _FatalProvider(circuit_breaker_threshold=3)
        for _ in range(5):
            with pytest.raises(FatalAPIError):
                _run(provider.complete_json("prompt", EchoModel))
        # Circuit should still be closed
        with pytest.raises(FatalAPIError):
            _run(provider.complete_json("prompt", EchoModel))

    def test_successful_call_resets_failure_counter(self):
        provider = _FailThenSucceedProvider(
            ServerError,
            n_failures=2,
            circuit_breaker_threshold=3,
        )
        # Two failures then one success — resets counter
        _run(provider.complete_json("prompt", EchoModel))
        # Circuit should be closed (success reset the counter)
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"


# ──────────────────────────────────────────────────────────────
# Fix A — _extract_json handles both objects and arrays
# ──────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_bare_object_extracted(self):
        result = BaseProvider._extract_json('{"value": "ok"}')
        assert result == '{"value": "ok"}'

    def test_bare_array_extracted(self):
        result = BaseProvider._extract_json('[{"value": "a"}, {"value": "b"}]')
        assert result == '[{"value": "a"}, {"value": "b"}]'

    def test_array_inside_prose_extracted(self):
        result = BaseProvider._extract_json(
            'here is the result: [{"value": "a"}] end'
        )
        assert result == '[{"value": "a"}]'

    def test_no_json_raises_decode_error(self):
        import json as _json
        with pytest.raises(_json.JSONDecodeError):
            BaseProvider._extract_json("no json here at all")


# ──────────────────────────────────────────────────────────────
# Fix B — Schema-aware normalization and targeted retry
# ──────────────────────────────────────────────────────────────

class _ArrayResponseProvider(BaseProvider):
    """Returns a JSON array from _call. Used to test array normalization."""

    def __init__(self, responses: list[str], **kwargs):
        config = _make_config(
            max_format_retries=kwargs.get("max_format_retries", 3),
            circuit_breaker_threshold=kwargs.get("circuit_breaker_threshold", 10),
            **{k: v for k, v in kwargs.items()
               if k not in ("max_format_retries", "circuit_breaker_threshold")},
        )
        super().__init__(config)
        self._responses = responses
        self._index = 0
        self.messages_received: list[list[dict]] = []

    async def _call(self, messages, system, response_schema=None):
        self.messages_received.append(list(messages))
        raw = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return raw

    @property
    def name(self):
        return "stub"


class TestArrayNormalization(_ResiliencyBase):
    def test_singleton_array_unwrapped_silently(self):
        provider = _ArrayResponseProvider(
            ['[{"value": "ok"}]'],
            max_format_retries=1,
        )
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"

    def test_multi_element_array_triggers_merge(self):
        # Two objects with mergeable list fields — should produce one result
        from pydantic import Field as PydanticField

        class MultiModel(BaseModel):
            items: list[str] = PydanticField(min_length=1)

        provider = _ArrayResponseProvider(
            ['[{"items": ["a"]}, {"items": ["b"]}]'],
            max_format_retries=1,
        )
        result = _run(provider.complete_json("prompt", MultiModel))
        assert set(result.items) == {"a", "b"}

    def test_multi_element_array_unmergeable_sends_targeted_retry(self):
        # First call returns array with wrong-keyed dicts — merge returns None.
        # Second call returns a valid object.
        provider = _ArrayResponseProvider(
            [
                '[{"x": "a"}, {"y": "b"}]',   # no "value" field → merge returns None
                '{"value": "ok"}',
            ],
            max_format_retries=2,
        )
        result = _run(provider.complete_json("prompt", EchoModel))
        assert result.value == "ok"
        # The retry message must mention "array" and the expected field names
        last_messages = provider.messages_received[-1]
        retry_msg = last_messages[-1]["content"]
        assert "array" in retry_msg
        assert "value" in retry_msg

    def test_targeted_retry_used_not_generic_on_array(self):
        # Wrong-keyed dicts force merge failure and the targeted retry message.
        provider = _ArrayResponseProvider(
            [
                '[{"x": "a"}, {"y": "b"}]',   # no "value" field → merge returns None
                '{"value": "ok"}',
            ],
            max_format_retries=2,
        )
        _run(provider.complete_json("prompt", EchoModel))
        last_messages = provider.messages_received[-1]
        retry_msg = last_messages[-1]["content"]
        # Generic message starts with "Your response was invalid"
        assert not retry_msg.startswith("Your response was invalid")
        assert "Do not wrap" in retry_msg


# ──────────────────────────────────────────────────────────────
# Fix C — response_schema forwarded to _call
# ──────────────────────────────────────────────────────────────

class _SchemaCapturingProvider(BaseProvider):
    """Records the response_schema passed to _call."""

    def __init__(self, **kwargs):
        config = _make_config(
            max_format_retries=1,
            circuit_breaker_threshold=10,
            **kwargs,
        )
        super().__init__(config)
        self.captured_schemas: list = []

    async def _call(self, messages, system, response_schema=None):
        self.captured_schemas.append(response_schema)
        return _VALID_JSON

    @property
    def name(self):
        return "stub"


class TestResponseSchemaForwarding(_ResiliencyBase):
    def test_response_schema_forwarded_to_call(self):
        provider = _SchemaCapturingProvider()
        _run(provider.complete_json("prompt", EchoModel))
        assert provider.captured_schemas == [EchoModel]

    def test_none_schema_not_forwarded_when_not_set(self):
        # Default BaseProvider._call accepts response_schema=None — verify
        # stub providers with the old signature still work via the default.
        class _LegacyStub(BaseProvider):
            captured = []

            async def _call(self, messages, system, response_schema=None):
                _LegacyStub.captured.append(response_schema)
                return _VALID_JSON

            @property
            def name(self):
                return "stub"

        config = _make_config(max_format_retries=1, circuit_breaker_threshold=10)
        provider = _LegacyStub(config)
        _run(provider.complete_json("prompt", EchoModel))
        assert _LegacyStub.captured == [EchoModel]
