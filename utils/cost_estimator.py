CHARS_PER_TOKEN = 4

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
        or {"input": 0.0, "output": 0.0}
    )
    return (
        input_tokens  / 1_000_000 * prices["input"]
        + output_tokens / 1_000_000 * prices["output"]
    )


def analyze_pdf_cost(extraction: dict, provider: str, model: str) -> dict:
    chunks = extraction["chunks"]
    input_tokens  = sum(estimate_tokens(chunk) for chunk in chunks)
    output_tokens = 250 * 15  # 250 output tokens per slide × 15 slides
    estimated_cost = calculate_cost(input_tokens, output_tokens, provider, model)
    return {
        "input_tokens":    input_tokens,
        "output_tokens":   output_tokens,
        "estimated_cost":  estimated_cost,
    }
