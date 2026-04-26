class RateLimitError(Exception):
    """Raised by provider subclasses when the API returns HTTP 429."""


class ServerError(Exception):
    """Raised by provider subclasses when the API returns HTTP 5xx or times out."""


class FatalAPIError(Exception):
    """
    Raised for non-retryable 4xx responses (400, 401, 403, 413, etc.).
    These will never succeed on retry — fail immediately with a clear message.
    """


class CircuitOpenError(Exception):
    """
    Raised when the circuit breaker is open — the API appears completely unavailable.
    The caller should surface this as a clean error rather than retrying.
    """
