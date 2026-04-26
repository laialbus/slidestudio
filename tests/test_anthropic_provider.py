"""
Milestone 6 — AnthropicProvider tests.

Uses unittest.mock.patch to mock anthropic.AsyncAnthropic.
No real API calls.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from providers.anthropic import AnthropicProvider
from providers.base import RateLimitError
from providers.config import ProviderConfig


def _make_provider() -> AnthropicProvider:
    with patch("anthropic.AsyncAnthropic"):
        return AnthropicProvider(
            config=ProviderConfig(
                model="claude-sonnet-4-20250514",
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


def _make_rate_limit_exc() -> anthropic.RateLimitError:
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.request = MagicMock()
    return anthropic.RateLimitError(
        message="Too many requests",
        response=mock_response,
        body=None,
    )


# ──────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────

class TestAnthropicProviderConstruction:
    def test_model_stored_on_instance(self):
        with patch("anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(
                config=ProviderConfig(
                    model="claude-opus-4-20250514",
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
        assert provider.model == "claude-opus-4-20250514"

    def test_name_property_returns_anthropic(self):
        provider = _make_provider()
        assert provider.name == "anthropic"


# ──────────────────────────────────────────────────────────────
# Rate-limit error handling
# ──────────────────────────────────────────────────────────────

class TestRateLimitHandling:
    def test_raises_rate_limit_error_on_429(self):
        async def run():
            with patch("anthropic.AsyncAnthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.messages.create = AsyncMock(
                    side_effect=_make_rate_limit_exc()
                )
                provider = AnthropicProvider(
                    config=ProviderConfig(
                        model="claude-sonnet-4-20250514",
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
                with pytest.raises(RateLimitError):
                    await provider._call([{"role": "user", "content": "test"}], "")

        asyncio.run(run())

    def test_rate_limit_error_is_base_rate_limit_error(self):
        async def run():
            with patch("anthropic.AsyncAnthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.messages.create = AsyncMock(
                    side_effect=_make_rate_limit_exc()
                )
                provider = AnthropicProvider(
                    config=ProviderConfig(
                        model="claude-sonnet-4-20250514",
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
                try:
                    await provider._call([{"role": "user", "content": "test"}], "")
                    assert False, "Expected RateLimitError"
                except RateLimitError:
                    pass  # correct

        asyncio.run(run())

    def test_successful_call_returns_text(self):
        async def run():
            with patch("anthropic.AsyncAnthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client

                mock_content = MagicMock()
                mock_content.text = '{"result": "ok"}'
                mock_response = MagicMock()
                mock_response.content = [mock_content]
                mock_client.messages.create = AsyncMock(return_value=mock_response)

                provider = AnthropicProvider(
                    config=ProviderConfig(
                        model="claude-sonnet-4-20250514",
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
                result = await provider._call([{"role": "user", "content": "test"}], "")
                assert result == '{"result": "ok"}'

        asyncio.run(run())

    def test_anthropic_sdk_called_with_correct_model(self):
        async def run():
            with patch("anthropic.AsyncAnthropic") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client

                mock_content = MagicMock()
                mock_content.text = "response"
                mock_response = MagicMock()
                mock_response.content = [mock_content]
                mock_client.messages.create = AsyncMock(return_value=mock_response)

                provider = AnthropicProvider(
                    config=ProviderConfig(
                        model="claude-opus-4-20250514",
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
                await provider._call([{"role": "user", "content": "test"}], "system prompt")

                call_kwargs = mock_client.messages.create.call_args
                assert call_kwargs.kwargs["model"] == "claude-opus-4-20250514"

        asyncio.run(run())
