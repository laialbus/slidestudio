import asyncio

PROVIDER_DEFAULTS: dict[str, int] = {
    "anthropic": 5,
    "openai":    5,
    "groq":      5,
    "ollama":    1,
}

_semaphore: asyncio.Semaphore | None = None


def get_semaphore(provider: str, user_override: int | None = None) -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        limit = user_override if user_override is not None \
                else PROVIDER_DEFAULTS.get(provider, 5)
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore


def reset_semaphore() -> None:
    global _semaphore
    _semaphore = None
