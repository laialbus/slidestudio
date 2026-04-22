import json

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.rate_limiter import get_semaphore


class RateLimitError(Exception):
    """Raised by provider subclasses when the API returns HTTP 429."""


class BaseProvider:
    def __init__(
        self,
        provider_name: str,
        max_concurrent: int | None,
        max_format_retries: int,
        max_rate_limit_retries: int,
    ):
        self.provider_name     = provider_name
        self.max_concurrent    = max_concurrent
        self.max_format_retries = max_format_retries

        # Bind tenacity retry logic at construction time so max_rate_limit_retries
        # comes from DI rather than a module-level constant.
        self._call_with_backoff = retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(max_rate_limit_retries),
        )(self._raw_call_with_backoff)

    async def complete_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        semaphore = get_semaphore(self.provider_name, self.max_concurrent)
        async with semaphore:
            return await self._call_with_backoff(prompt, schema, system)

    async def _raw_call_with_backoff(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str,
    ) -> BaseModel:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(1, self.max_format_retries + 1):
            raw = await self._call(messages, system)
            try:
                data = json.loads(self._extract_json(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == self.max_format_retries:
                    raise RuntimeError(
                        f"Format retries exhausted after {self.max_format_retries} attempts.\n"
                        f"Last error: {e}\nLast response: {raw[:300]}"
                    )
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was invalid. Error: {e}\n\n"
                        "Please return ONLY a valid JSON object that matches "
                        "the required schema. No explanation, no markdown."
                    ),
                })

    @staticmethod
    def _extract_json(text: str) -> str:
        clean = text.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", clean, 0)
        return clean[start:end]

    async def _call(self, messages: list, system: str) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
