from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types

from providers.base import BaseProvider, RateLimitError


class GoogleProvider(BaseProvider):
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
        self.api_key = api_key

    async def _call(self, messages: list[dict], system: str) -> str:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else m["role"],
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=system or "You are a helpful assistant. Return only valid JSON.",
            response_mime_type="application/json", # Enforce JSON output deterministically
        )

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text
        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as exc:
            raise RateLimitError(f"HTTP 429 from Google: {exc}") from exc

    @property
    def name(self) -> str:
        return "google"
