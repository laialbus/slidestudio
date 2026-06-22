"""
Milestone 6 — cost_estimator tests.

No real API calls, no external tokenizer libraries.
"""

import pytest

from utils.cost_estimator import (
    CHARS_PER_TOKEN,
    PRICING,
    analyze_pdf_cost,
    calculate_cost,
    estimate_tokens,
)


# ──────────────────────────────────────────────────────────────
# estimate_tokens
# ──────────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_four_chars_returns_one_token(self):
        assert estimate_tokens("abcd") == 1

    def test_eight_chars_returns_two_tokens(self):
        assert estimate_tokens("abcdefgh") == 2

    def test_chars_per_token_constant_is_4(self):
        assert CHARS_PER_TOKEN == 4

    def test_result_equals_len_divided_by_chars_per_token(self):
        text = "x" * 1000
        assert estimate_tokens(text) == 1000 // CHARS_PER_TOKEN

    def test_truncates_remainder(self):
        # 9 chars → 9 // 4 = 2
        assert estimate_tokens("abcdefghi") == 2

    def test_large_text(self):
        text = "a" * 500_000
        assert estimate_tokens(text) == 125_000


# ──────────────────────────────────────────────────────────────
# calculate_cost
# ──────────────────────────────────────────────────────────────

class TestCalculateCost:
    def test_zero_tokens_returns_zero(self):
        cost = calculate_cost(0, 0, "anthropic", "claude-sonnet-4-20250514")
        assert cost == 0.0

    def test_anthropic_sonnet_input_cost(self):
        # 1M input tokens at $3.00/M = $3.00
        cost = calculate_cost(1_000_000, 0, "anthropic", "claude-sonnet-4-20250514")
        assert abs(cost - 3.00) < 1e-9

    def test_anthropic_sonnet_output_cost(self):
        # 1M output tokens at $15.00/M = $15.00
        cost = calculate_cost(0, 1_000_000, "anthropic", "claude-sonnet-4-20250514")
        assert abs(cost - 15.00) < 1e-9

    def test_anthropic_opus_input_cost(self):
        # 1M input tokens at $15.00/M
        cost = calculate_cost(1_000_000, 0, "anthropic", "claude-opus-4-20250514")
        assert abs(cost - 15.00) < 1e-9

    def test_openai_gpt4o_input_cost(self):
        # 1M input tokens at $2.50/M
        cost = calculate_cost(1_000_000, 0, "openai", "gpt-4o")
        assert abs(cost - 2.50) < 1e-9

    def test_ollama_is_free(self):
        cost = calculate_cost(1_000_000, 1_000_000, "ollama", "llama3.1")
        assert cost == 0.0

    def test_unknown_provider_returns_zero(self):
        with pytest.warns(RuntimeWarning):
            cost = calculate_cost(
                1_000_000, 1_000_000, "unknown_provider", "some-model"
            )
        assert cost == 0.0

    def test_unknown_model_returns_zero(self):
        with pytest.warns(RuntimeWarning):
            cost = calculate_cost(1_000_000, 0, "anthropic", "nonexistent-model")
        assert cost == 0.0

    def test_unknown_pricing_warns_loudly(self):
        # C5: an unpriced provider/model must warn, not silently report $0.
        with pytest.warns(RuntimeWarning, match="No pricing table"):
            calculate_cost(1_000, 1_000, "unknown_provider", "some-model")

    def test_google_gemini_flash_input_cost(self):
        # 1M input tokens at $0.50/M
        cost = calculate_cost(1_000_000, 0, "google", "gemini-3-flash-preview")
        assert abs(cost - 0.50) < 1e-9

    def test_google_gemini_flash_output_cost(self):
        # 1M output tokens at $3.00/M
        cost = calculate_cost(0, 1_000_000, "google", "gemini-3-flash-preview")
        assert abs(cost - 3.00) < 1e-9

    def test_google_fast_gemma_is_free(self):
        # Gemma 4 is free; a known $0 must NOT warn (unlike unknown pricing).
        cost = calculate_cost(1_000_000, 1_000_000, "google-fast", "gemma-4-31b-it")
        assert cost == 0.0

    def test_combined_input_and_output(self):
        # 100k input at $3/M + 100k output at $15/M
        cost = calculate_cost(100_000, 100_000, "anthropic", "claude-sonnet-4-20250514")
        expected = 100_000 / 1_000_000 * 3.00 + 100_000 / 1_000_000 * 15.00
        assert abs(cost - expected) < 1e-9

    def test_groq_model_has_nonzero_cost(self):
        cost = calculate_cost(1_000_000, 0, "groq", "llama-3.1-70b-versatile")
        assert cost > 0.0


# ──────────────────────────────────────────────────────────────
# analyze_pdf_cost
# ──────────────────────────────────────────────────────────────

class TestAnalyzePdfCost:
    def _extraction(self, chunks: list[str]) -> dict:
        return {"headers": [], "chunks": chunks}

    def test_returns_dict_with_required_keys(self):
        result = analyze_pdf_cost(
            self._extraction(["chunk text"]),
            "anthropic",
            "claude-sonnet-4-20250514",
        )
        assert "input_tokens"   in result
        assert "output_tokens"  in result
        assert "estimated_cost" in result

    def test_output_tokens_is_250_times_15(self):
        result = analyze_pdf_cost(
            self._extraction(["x" * 400]),
            "anthropic",
            "claude-sonnet-4-20250514",
        )
        assert result["output_tokens"] == 250 * 15

    def test_input_tokens_sum_of_chunk_estimates(self):
        chunks = ["a" * 400, "b" * 800]
        result = analyze_pdf_cost(
            self._extraction(chunks),
            "anthropic",
            "claude-sonnet-4-20250514",
        )
        expected = estimate_tokens("a" * 400) + estimate_tokens("b" * 800)
        assert result["input_tokens"] == expected

    def test_zero_chunks_returns_zero_input(self):
        result = analyze_pdf_cost(self._extraction([]), "anthropic", "claude-sonnet-4-20250514")
        assert result["input_tokens"] == 0

    def test_cost_is_float(self):
        result = analyze_pdf_cost(
            self._extraction(["chunk"]),
            "anthropic",
            "claude-sonnet-4-20250514",
        )
        assert isinstance(result["estimated_cost"], float)

    def test_ollama_cost_is_zero(self):
        result = analyze_pdf_cost(
            self._extraction(["chunk text here"]),
            "ollama",
            "llama3.1",
        )
        assert result["estimated_cost"] == 0.0

    def test_pricing_dict_has_expected_providers(self):
        assert "anthropic"   in PRICING
        assert "openai"      in PRICING
        assert "groq"        in PRICING
        assert "ollama"      in PRICING
        assert "google"      in PRICING
        assert "google-fast" in PRICING
