"""
Tests for BaseProvider._build_retry_message.

No API calls — ValidationError instances are produced directly from
minimal Pydantic schemas so each error type can be exercised in isolation.
"""

import pytest
from pydantic import BaseModel, Field, ValidationError

from providers.base import BaseProvider
from providers.config import ProviderConfig


# ──────────────────────────────────────────────────────────────
# Minimal concrete provider (satisfies the abstract interface)
# ──────────────────────────────────────────────────────────────

class _Provider(BaseProvider):
    @property
    def name(self) -> str:
        return "test"

    async def _call(self, messages, system, response_schema=None) -> str:
        raise NotImplementedError


def _make_provider() -> _Provider:
    return _Provider(
        config=ProviderConfig(
            model="test-model",
            max_concurrent=1,
            max_format_retries=3,
            max_rate_limit_retries=1,
            request_timeout=5,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            backoff_wait_min=0,
            backoff_wait_max=0,
        )
    )


# ──────────────────────────────────────────────────────────────
# Inline schemas used only in these tests
# ──────────────────────────────────────────────────────────────

class _StringCapped(BaseModel):
    text: str = Field(max_length=5)


class _StringFloored(BaseModel):
    text: str = Field(min_length=10)


class _ListCapped(BaseModel):
    items: list[str] = Field(max_length=2)


class _ListFloored(BaseModel):
    items: list[str] = Field(min_length=2)


class _NumericBounded(BaseModel):
    count: int = Field(ge=1, le=10)


class _NumericStrict(BaseModel):
    value: int = Field(gt=0, lt=100)


class _TwoConstraints(BaseModel):
    first:  str = Field(max_length=3)
    second: str = Field(max_length=3)


class _Required(BaseModel):
    name: str
    age:  int


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _provoke(schema: type[BaseModel], data: dict) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        schema.model_validate(data)
    return exc_info.value


# ──────────────────────────────────────────────────────────────
# Constraint violations
# ──────────────────────────────────────────────────────────────

class TestConstraintErrors:
    def setup_method(self):
        self.provider = _make_provider()

    def test_string_too_long_names_field_and_instructs_rewrite(self):
        error = _provoke(_StringCapped, {"text": "x" * 6})
        msg = self.provider._build_retry_message(error, _StringCapped)

        assert "Field 'text'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_string_too_short_names_field_and_instructs_rewrite(self):
        error = _provoke(_StringFloored, {"text": "hi"})
        msg = self.provider._build_retry_message(error, _StringFloored)

        assert "Field 'text'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_list_too_long_names_field_and_instructs_rewrite(self):
        error = _provoke(_ListCapped, {"items": ["a", "b", "c"]})
        msg = self.provider._build_retry_message(error, _ListCapped)

        assert "Field 'items'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_list_too_short_names_field_and_instructs_rewrite(self):
        error = _provoke(_ListFloored, {"items": ["only_one"]})
        msg = self.provider._build_retry_message(error, _ListFloored)

        assert "Field 'items'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_ge_violation_names_field_and_instructs_rewrite(self):
        error = _provoke(_NumericBounded, {"count": 0})
        msg = self.provider._build_retry_message(error, _NumericBounded)

        assert "Field 'count'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_le_violation_names_field_and_instructs_rewrite(self):
        error = _provoke(_NumericBounded, {"count": 11})
        msg = self.provider._build_retry_message(error, _NumericBounded)

        assert "Field 'count'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_gt_violation_names_field_and_instructs_rewrite(self):
        error = _provoke(_NumericStrict, {"value": 0})
        msg = self.provider._build_retry_message(error, _NumericStrict)

        assert "Field 'value'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_lt_violation_names_field_and_instructs_rewrite(self):
        error = _provoke(_NumericStrict, {"value": 100})
        msg = self.provider._build_retry_message(error, _NumericStrict)

        assert "Field 'value'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg

    def test_constraint_errors_do_not_contain_structural_phrase(self):
        error = _provoke(_StringCapped, {"text": "toolong"})
        msg = self.provider._build_retry_message(error, _StringCapped)

        assert "Structural errors detected" not in msg

    def test_multiple_constraint_violations_each_get_own_line(self):
        error = _provoke(_TwoConstraints, {"first": "aaaa", "second": "bbbb"})
        msg = self.provider._build_retry_message(error, _TwoConstraints)

        assert "Field 'first'" in msg
        assert "Field 'second'" in msg
        # Each field entry is on a separate line
        assert msg.count("Rewrite this field to satisfy the constraint.") == 2


# ──────────────────────────────────────────────────────────────
# Structural violations
# ──────────────────────────────────────────────────────────────

class TestStructuralErrors:
    def setup_method(self):
        self.provider = _make_provider()

    def test_missing_field_gives_structural_message(self):
        error = _provoke(_Required, {"name": "Alice"})  # age missing
        msg = self.provider._build_retry_message(error, _Required)

        assert "Structural errors detected" in msg

    def test_structural_message_lists_schema_fields(self):
        error = _provoke(_Required, {})
        msg = self.provider._build_retry_message(error, _Required)

        assert "name" in msg
        assert "age" in msg

    def test_structural_errors_do_not_contain_constraint_phrase(self):
        error = _provoke(_Required, {})
        msg = self.provider._build_retry_message(error, _Required)

        assert "Rewrite this field to satisfy the constraint." not in msg


# ──────────────────────────────────────────────────────────────
# Mixed violations (constraint + structural in one response)
# ──────────────────────────────────────────────────────────────

class _Mixed(BaseModel):
    label: str = Field(max_length=3)
    count: int


class TestMixedErrors:
    def setup_method(self):
        self.provider = _make_provider()

    def test_mixed_errors_include_both_constraint_and_structural_sections(self):
        # label is too long (constraint) and count is missing (structural)
        error = _provoke(_Mixed, {"label": "toolong"})
        msg = self.provider._build_retry_message(error, _Mixed)

        assert "Field 'label'" in msg
        assert "Rewrite this field to satisfy the constraint." in msg
        assert "Structural errors detected" in msg


# ──────────────────────────────────────────────────────────────
# Message envelope (applies to all error types)
# ──────────────────────────────────────────────────────────────

class TestMessageEnvelope:
    def setup_method(self):
        self.provider = _make_provider()

    def test_message_starts_with_header(self):
        error = _provoke(_StringCapped, {"text": "toolong"})
        msg = self.provider._build_retry_message(error, _StringCapped)

        assert msg.startswith("Your response was invalid:\n")

    def test_message_ends_with_return_instruction(self):
        error = _provoke(_Required, {})
        msg = self.provider._build_retry_message(error, _Required)

        assert msg.endswith(
            "Return ONLY the corrected JSON object. No explanation, no markdown."
        )

    def test_envelope_present_for_constraint_error(self):
        error = _provoke(_StringCapped, {"text": "toolong"})
        msg = self.provider._build_retry_message(error, _StringCapped)

        assert msg.startswith("Your response was invalid:\n")
        assert msg.endswith(
            "Return ONLY the corrected JSON object. No explanation, no markdown."
        )

    def test_envelope_present_for_structural_error(self):
        error = _provoke(_Required, {})
        msg = self.provider._build_retry_message(error, _Required)

        assert msg.startswith("Your response was invalid:\n")
        assert msg.endswith(
            "Return ONLY the corrected JSON object. No explanation, no markdown."
        )
