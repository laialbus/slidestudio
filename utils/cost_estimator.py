import warnings

CHARS_PER_TOKEN = 4

# Per-1M-token USD prices. Sources: provider pricing pages. Google rates are the
# standard-tier text prices from ai.google.dev/gemini-api/docs/pricing.
PRICING: dict[str, dict[str, dict[str, float]]] = {
    "anthropic": {
        "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
        "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
    },
    "openai": {
        "gpt-4o":      {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    },
    "ollama": {
        "*": {"input": 0.00, "output": 0.00},
    },
    "google": {
        "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    },
    "google-fast": {
        # Gemma 4 is free of charge on the Gemini API.
        "gemma-4-31b-it": {"input": 0.00, "output": 0.00},
    },
}


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    provider: str,
    model: str,
) -> float:
    prices = (
        PRICING.get(provider, {}).get(model)
        or PRICING.get(provider, {}).get("*")
    )
    if prices is None:
        # No pricing entry: warn loudly rather than present $0 as a real
        # figure, since the point of an estimate is a trustworthy number.
        warnings.warn(
            f"No pricing table for {provider}/{model}; "
            f"cost estimate unavailable (reporting $0).",
            RuntimeWarning,
            stacklevel=2,
        )
        prices = {"input": 0.0, "output": 0.0}
    return (
        input_tokens  / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )


def analyze_pdf_cost(extraction, provider: str, model: str) -> dict:
    """
    Accepts either an ExtractionResult (Pydantic model with a .chunks attribute)
    or a legacy dict with a "chunks" key.
    """
    chunks = extraction.chunks if hasattr(extraction, "chunks") else extraction["chunks"]
    input_tokens   = sum(estimate_tokens(chunk) for chunk in chunks)
    output_tokens  = 250 * 15  # 250 output tokens per slide × 15 slides
    estimated_cost = calculate_cost(input_tokens, output_tokens, provider, model)
    return {
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "estimated_cost":  estimated_cost,
    }
