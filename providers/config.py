# Internal initialization contract for provider instances.
# Do not edit defaults here — set values in the top-level config.py instead.
# ProviderConfig carries values from config.py into provider __init__ methods.

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    model: str
    max_concurrent: int | None
    max_format_retries: int
    max_rate_limit_retries: int
    request_timeout: float
    circuit_breaker_threshold: int
    circuit_breaker_cooldown: float
    backoff_wait_min: float
    backoff_wait_max: float
