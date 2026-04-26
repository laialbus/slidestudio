import asyncio
import json
from typing import get_origin

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from providers.config import ProviderConfig
from providers.errors import (
    CircuitOpenError,
    FatalAPIError,
    RateLimitError,
    ServerError,
)
from utils.rate_limiter import get_circuit_breaker, get_semaphore

# Re-export for backward compatibility — importers that do
# `from providers.base import RateLimitError` continue to work.
__all__ = [
    "BaseProvider",
    "RateLimitError",
    "ServerError",
    "FatalAPIError",
    "CircuitOpenError",
]


class BaseProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config

        # Bind tenacity at construction time so parameters come from DI,
        # not from module-level constants. reraise=True ensures the original
        # RateLimitError/ServerError propagates when retries are exhausted,
        # not a tenacity.RetryError wrapper.
        self._call_with_backoff = retry(
            retry=retry_if_exception_type((RateLimitError, ServerError)),
            wait=wait_exponential(
                multiplier=1,
                min=config.backoff_wait_min,
                max=config.backoff_wait_max,
            ),
            stop=stop_after_attempt(config.max_rate_limit_retries),
            reraise=True,
        )(self._raw_call_with_backoff)

    @property
    def model(self) -> str:
        return self.config.model

    async def complete_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        cb = get_circuit_breaker()
        cb.check(self.config.circuit_breaker_threshold, self.config.circuit_breaker_cooldown)

        semaphore = get_semaphore(self.name, self.config.max_concurrent)
        async with semaphore:
            try:
                result = await self._call_with_backoff(prompt, schema, system)
                cb.record_success()
                return result
            except (RateLimitError, ServerError):
                cb.record_failure(self.config.circuit_breaker_threshold)
                raise

    async def _raw_call_with_backoff(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str,
    ) -> BaseModel:
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, self.config.max_format_retries + 1):
            try:
                raw = await asyncio.wait_for(
                    self._call(messages, system, response_schema=schema),
                    timeout=self.config.request_timeout,
                )
            except asyncio.TimeoutError:
                raise ServerError(
                    f"Request timed out after {self.config.request_timeout}s"
                )

            try:
                data = json.loads(self._extract_json(raw))
            except json.JSONDecodeError as e:
                if attempt == self.config.max_format_retries:
                    raise RuntimeError(
                        f"Format retries exhausted after {self.config.max_format_retries} attempts.\n"
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
                continue

            # Normalize list responses before schema validation
            if isinstance(data, list):
                if len(data) == 1:
                    data = data[0]
                elif len(data) > 1:
                    merged = self._merge_array_response(data, schema)
                    if merged is not None:
                        data = merged
                    else:
                        if attempt == self.config.max_format_retries:
                            raise RuntimeError(
                                f"Format retries exhausted after {self.config.max_format_retries} attempts.\n"
                                f"Last error: returned array of {len(data)} objects that could not be merged.\n"
                                f"Last response: {raw[:300]}"
                            )
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"You returned an array of {len(data)} objects. "
                                f"Return exactly ONE JSON object with these fields: "
                                f"{list(schema.model_fields.keys())}. "
                                f"Do not wrap the result in an array."
                            ),
                        })
                        continue

            try:
                return schema.model_validate(data)
            except ValidationError as e:
                if attempt == self.config.max_format_retries:
                    raise RuntimeError(
                        f"Format retries exhausted after {self.config.max_format_retries} attempts.\n"
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
        obj_start = clean.find("{")
        arr_start = clean.find("[")

        if obj_start == -1 and arr_start == -1:
            raise json.JSONDecodeError("No JSON object found", clean, 0)

        if arr_start == -1 or (obj_start != -1 and obj_start < arr_start):
            start = obj_start
            end   = clean.rfind("}") + 1
        else:
            start = arr_start
            end   = clean.rfind("]") + 1

        return clean[start:end]

    @staticmethod
    def _merge_array_response(data: list, schema: type[BaseModel]) -> dict | None:
        if not data or not all(isinstance(elem, dict) for elem in data):
            return None

        merged: dict = {}
        for field_name, field_info in schema.model_fields.items():
            values = [
                elem[field_name]
                for elem in data
                if field_name in elem and elem[field_name] is not None
            ]
            if not values:
                continue
            if get_origin(field_info.annotation) is list:
                combined: list = []
                seen: set = set()
                for v in values:
                    if isinstance(v, list):
                        for item in v:
                            key = (
                                json.dumps(item, sort_keys=True)
                                if isinstance(item, (dict, list))
                                else item
                            )
                            if key not in seen:
                                seen.add(key)
                                combined.append(item)
                merged[field_name] = combined
            else:
                merged[field_name] = values[0]

        return merged if merged else None

    async def _call(
        self,
        messages: list,
        system: str,
        response_schema: type[BaseModel] | None = None,
    ) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError
