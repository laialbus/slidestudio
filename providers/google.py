import httpx
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types
from pydantic import BaseModel

from providers.base import BaseProvider, RateLimitError, ServerError
from providers.config import ProviderConfig

# Reasoning depth for Gemini flash calls. Flash defaults to little or no
# thinking, which leaves the Writer/Critic/Refiner reasoning shallow. Gemini 3
# controls this with a thinking *level* (an enum), not a token budget — passing
# a token count to thinking_level is invalid. MEDIUM buys deeper reasoning at a
# moderate cost. Internal provider tuning, so it stays local rather than in
# config.py (per the coding standards).
_THINKING_LEVEL = types.ThinkingLevel.MEDIUM


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
        # response_schema (native structured output) is deliberately NOT set.
        # Some agent schemas contain dict fields — e.g. ChunkMap.figure_purposes
        # (dict[str, ...]) — which compile to JSON Schema `additionalProperties`,
        # and the Gemini API rejects that construct ("additionalProperties is not
        # supported"). response_mime_type + the BaseProvider format-retry loop
        # already guarantee valid JSON for every schema, so we rely on that.
        cfg = types.GenerateContentConfig(
            system_instruction=system or "You are a helpful assistant. Return only valid JSON.",
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=_THINKING_LEVEL),
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
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            raise ServerError(f"Network error from Google API: {exc}") from exc

    @property
    def name(self) -> str:
        return "google"
