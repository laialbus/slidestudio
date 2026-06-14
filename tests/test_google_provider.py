"""
Milestone 9 — GoogleProvider tests.

Uses unittest.mock.patch to mock google.genai.Client directly.
No real API calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from providers.base import RateLimitError
from providers.config import ProviderConfig
from providers.google import GoogleProvider


def _make_provider() -> GoogleProvider:
    return GoogleProvider(
        config=ProviderConfig(
            model="gemini-2.5-pro",
            max_concurrent=5,
            max_format_retries=1,
            max_rate_limit_retries=1,
            request_timeout=5,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            backoff_wait_min=0,
            backoff_wait_max=0,
        ),
        api_key="test-key",
    )


def _setup_mock_client(mock_client_cls, text: str = '{"result": "ok"}'):
    """Wire up a mock Client whose aio.models.generate_content returns text."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = text
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return mock_client


# ──────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────

class TestGoogleProviderConstruction:
    def test_model_stored_on_instance(self):
        provider = GoogleProvider(
            config=ProviderConfig(
                model="gemini-3.1-flash-lite-preview",
                max_concurrent=5,
                max_format_retries=3,
                max_rate_limit_retries=6,
                request_timeout=5,
                circuit_breaker_threshold=3,
                circuit_breaker_cooldown=60,
                backoff_wait_min=0,
                backoff_wait_max=0,
            ),
            api_key="key",
        )
        assert provider.model == "gemini-3.1-flash-lite-preview"

    def test_name_property_returns_google(self):
        assert _make_provider().name == "google"

    def test_api_key_stored(self):
        assert _make_provider().api_key == "test-key"


# ──────────────────────────────────────────────────────────────
# Rate-limit error handling
# ──────────────────────────────────────────────────────────────

class TestRateLimitHandling:
    def test_raises_rate_limit_error_on_resource_exhausted(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.aio.models.generate_content = AsyncMock(
                    side_effect=google_exceptions.ResourceExhausted("429 quota")
                )
                with pytest.raises(RateLimitError):
                    await _make_provider()._call(
                        [{"role": "user", "content": "test"}], ""
                    )

        asyncio.run(run())

    def test_raises_rate_limit_error_on_too_many_requests(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.aio.models.generate_content = AsyncMock(
                    side_effect=google_exceptions.TooManyRequests("429")
                )
                with pytest.raises(RateLimitError):
                    await _make_provider()._call(
                        [{"role": "user", "content": "test"}], ""
                    )

        asyncio.run(run())

    def test_rate_limit_error_is_base_rate_limit_error(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.aio.models.generate_content = AsyncMock(
                    side_effect=google_exceptions.ResourceExhausted("429 quota")
                )
                try:
                    await _make_provider()._call(
                        [{"role": "user", "content": "test"}], ""
                    )
                    assert False, "Expected RateLimitError"
                except RateLimitError:
                    pass

        asyncio.run(run())


# ──────────────────────────────────────────────────────────────
# Successful call + system instruction + role mapping + client isolation
# ──────────────────────────────────────────────────────────────

class TestSuccessfulCall:
    def test_successful_call_returns_text(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls, '{"result": "ok"}')
                result = await _make_provider()._call(
                    [{"role": "user", "content": "test"}], ""
                )
                assert result == '{"result": "ok"}'

        asyncio.run(run())

    def test_system_string_passed_as_system_instruction(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                await _make_provider()._call(
                    [{"role": "user", "content": "hi"}],
                    "You are strict.",
                )
                call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
                assert call_kwargs["config"].system_instruction == "You are strict."

        asyncio.run(run())

    def test_model_name_passed_to_generate_content(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                provider = GoogleProvider(
                    config=ProviderConfig(
                        model="gemini-3.1-flash-lite-preview",
                        max_concurrent=5,
                        max_format_retries=1,
                        max_rate_limit_retries=1,
                        request_timeout=5,
                        circuit_breaker_threshold=3,
                        circuit_breaker_cooldown=60,
                        backoff_wait_min=0,
                        backoff_wait_max=0,
                    ),
                    api_key="test-key",
                )
                await provider._call([{"role": "user", "content": "test"}], "")
                call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
                assert call_kwargs["model"] == "gemini-3.1-flash-lite-preview"

        asyncio.run(run())

    def test_assistant_role_mapped_to_model(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                messages = [
                    {"role": "user",      "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user",      "content": "again"},
                ]
                await _make_provider()._call(messages, "")
                contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
                assert [c.role for c in contents] == ["user", "model", "user"]

        asyncio.run(run())

    def test_user_role_preserved(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                await _make_provider()._call([{"role": "user", "content": "solo"}], "")
                contents = mock_client.aio.models.generate_content.call_args.kwargs["contents"]
                assert contents[0].role == "user"
                assert contents[0].parts[0].text == "solo"

        asyncio.run(run())

    def test_client_created_with_api_key_inside_call(self):
        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                _setup_mock_client(mock_cls)
                provider = GoogleProvider(
                    config=ProviderConfig(
                        model="gemini-2.5-pro",
                        max_concurrent=5,
                        max_format_retries=1,
                        max_rate_limit_retries=1,
                        request_timeout=5,
                        circuit_breaker_threshold=3,
                        circuit_breaker_cooldown=60,
                        backoff_wait_min=0,
                        backoff_wait_max=0,
                    ),
                    api_key="live-key-123",
                )
                # Client must NOT be instantiated at construction time
                mock_cls.assert_not_called()

                await provider._call([{"role": "user", "content": "test"}], "")
                mock_cls.assert_called_once_with(api_key="live-key-123")

        asyncio.run(run())

    def test_response_schema_not_forwarded(self):
        # Native structured output is intentionally disabled: dict-bearing
        # schemas compile to additionalProperties, which Gemini rejects. The
        # provider must not forward response_schema even when one is supplied.
        from pydantic import BaseModel as _BM

        class _Schema(_BM):
            result: str

        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                result = await _make_provider()._call(
                    [{"role": "user", "content": "test"}], "",
                    response_schema=_Schema,
                )
                cfg = mock_client.aio.models.generate_content.call_args.kwargs["config"]
                assert cfg.response_schema is None
                assert result == '{"result": "ok"}'

        asyncio.run(run())

    def test_thinking_level_enabled(self):
        from providers.google import _THINKING_LEVEL

        async def run():
            with patch("providers.google.genai.Client") as mock_cls:
                mock_client = _setup_mock_client(mock_cls)
                await _make_provider()._call(
                    [{"role": "user", "content": "test"}], ""
                )
                cfg = mock_client.aio.models.generate_content.call_args.kwargs["config"]
                assert cfg.thinking_config is not None
                assert cfg.thinking_config.thinking_level == _THINKING_LEVEL

        asyncio.run(run())

    def test_client_created_before_generate_content(self):
        """genai.Client must be instantiated before generate_content is awaited."""
        call_order: list[str] = []

        async def run():
            def _client_side_effect(**_kwargs):
                call_order.append("client")
                mock_client = MagicMock()
                async def _generate(**_kw):
                    call_order.append("generate")
                    r = MagicMock()
                    r.text = "{}"
                    return r
                mock_client.aio.models.generate_content = _generate
                return mock_client

            with patch("providers.google.genai.Client", side_effect=_client_side_effect):
                await _make_provider()._call([{"role": "user", "content": "test"}], "")

        asyncio.run(run())
        assert call_order == ["client", "generate"]
