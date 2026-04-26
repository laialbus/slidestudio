import anthropic as sdk
from pydantic import BaseModel

from providers.base import BaseProvider
from providers.config import ProviderConfig
from providers.errors import FatalAPIError, RateLimitError, ServerError


class AnthropicProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, api_key: str):
        super().__init__(config)
        self._client = sdk.AsyncAnthropic(api_key=api_key)

    async def _call(
        self,
        messages: list,
        system: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        try:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                system=system or "You are a helpful assistant. Return only valid JSON.",
                messages=messages,
            )
            return response.content[0].text
        except sdk.RateLimitError as exc:
            raise RateLimitError(f"HTTP 429 from Anthropic: {exc}") from exc
        except sdk.APIStatusError as exc:
            # RateLimitError is a subclass of APIStatusError, so it is
            # already handled above; what remains is 5xx and other 4xx.
            if exc.status_code >= 500:
                raise ServerError(
                    f"HTTP {exc.status_code} from Anthropic: {exc}"
                ) from exc
            raise FatalAPIError(
                f"HTTP {exc.status_code} from Anthropic: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return "anthropic"
