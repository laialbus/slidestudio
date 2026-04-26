from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types
from pydantic import BaseModel

from providers.base import BaseProvider, RateLimitError
from providers.config import ProviderConfig


class GoogleProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key

    async def _call(
        self,
        messages: list[dict],
        system: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else m["role"],
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]

        client = genai.Client(api_key=self.api_key)
        cfg = types.GenerateContentConfig(
            system_instruction=system or "You are a helpful assistant. Return only valid JSON.",
            response_mime_type="application/json",
            # response_schema=(
            #     response_schema.model_json_schema() if response_schema else None
            # ),
        )

        try:
            response = await client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=cfg,
            )
            return response.text
        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as exc:
            raise RateLimitError(f"HTTP 429 from Google: {exc}") from exc

    @property
    def name(self) -> str:
        return "google"
