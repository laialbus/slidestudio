import anthropic as sdk

from providers.base import BaseProvider, RateLimitError


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        max_concurrent: int | None,
        max_format_retries: int,
        max_rate_limit_retries: int,
    ):
        super().__init__(provider_name, max_concurrent, max_format_retries, max_rate_limit_retries)
        self.model   = model
        self._client = sdk.AsyncAnthropic(api_key=api_key)

    async def _call(self, messages: list, system: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "You are a helpful assistant. Return only valid JSON.",
                messages=messages,
            )
            return response.content[0].text
        except sdk.RateLimitError as exc:
            raise RateLimitError(f"HTTP 429 from Anthropic: {exc}") from exc

    @property
    def name(self) -> str:
        return "anthropic"
