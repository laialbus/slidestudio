"""
Validates the multi-deck routing decision.

The router requires BOTH conditions to activate multi-deck mode:
  1. chapter_count > multi_deck_chapter_threshold
  2. total_chars   > multi_deck_length_threshold
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pipeline import route
from schemas.deck_index import DeckIndex
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slides_final import FinalSlide, SlidesFinal


def _run(coro):
    return asyncio.run(coro)


def _skeleton(n_chapters: int) -> GlobalSkeleton:
    return GlobalSkeleton(
        title="Test",
        document_type="research_paper",
        core_thesis="A thesis.",
        sections=[
            SectionEntry(heading=f"Chapter {i + 1}", level=1, position=i)
            for i in range(n_chapters)
        ],
    )


def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Test",
        document_type="research_paper",
        technical_level="intermediate",
        core_thesis="A thesis.",
        key_concepts=["concept"],
        sections=[Section(heading="Intro", importance="high", summary="Summary.")],
    )


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test",
        slides=[FinalSlide(index=1, title="S", bullets=["B."], tag="Key Concept")],
    )


def _agents():
    import types
    provider = types.SimpleNamespace(name="stub", model="stub-model")
    planner  = types.SimpleNamespace(provider=provider)
    return {"planner": planner}


def _route(skeleton, total_chars, chapter_threshold=3, length_threshold=40_000, **kw):
    """Helper that patches run_single_deck and runs route()."""
    with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
        mock_rsd.return_value = (_slides_final(), [], Path("outputs/test.json"))
        return _run(route(
            "Test", skeleton, _doc_map(), ["chunk"],
            _agents(),
            multi_deck_chapter_threshold=chapter_threshold,
            multi_deck_length_threshold=length_threshold,
            total_chars=total_chars,
            max_review_cycles=1,
            debug=False,
            **kw,
        ))


# ──────────────────────────────────────────────────────────────
# Both conditions must be true for multi-deck
# ──────────────────────────────────────────────────────────────

class TestDualConditionRouting:
    def test_multi_deck_when_both_conditions_met(self, tmp_path):
        # chapters: 4 > 3, chars: 50_000 > 40_000
        result, _, _ = _route(_skeleton(4), total_chars=50_000, output_dir=tmp_path)
        assert isinstance(result, DeckIndex)

    def test_single_deck_when_only_chapters_exceeded(self, tmp_path):
        # chapters: 4 > 3, but chars: 1_000 < 40_000
        result, _, _ = _route(_skeleton(4), total_chars=1_000, output_dir=tmp_path)
        assert isinstance(result, SlidesFinal)

    def test_single_deck_when_only_length_exceeded(self, tmp_path):
        # chars: 50_000 > 40_000, but chapters: 2 <= 3
        result, _, _ = _route(_skeleton(2), total_chars=50_000, output_dir=tmp_path)
        assert isinstance(result, SlidesFinal)

    def test_single_deck_when_neither_condition_met(self, tmp_path):
        # chapters: 2 <= 3, chars: 1_000 < 40_000
        result, _, _ = _route(_skeleton(2), total_chars=1_000, output_dir=tmp_path)
        assert isinstance(result, SlidesFinal)


# ──────────────────────────────────────────────────────────────
# Boundary: exactly at threshold is NOT multi-deck (strictly greater)
# ──────────────────────────────────────────────────────────────

class TestThresholdBoundary:
    def test_chapters_exactly_at_threshold_is_single_deck(self, tmp_path):
        # 3 chapters, threshold=3 → 3 > 3 is False
        result, _, _ = _route(_skeleton(3), total_chars=50_000, output_dir=tmp_path)
        assert isinstance(result, SlidesFinal)

    def test_chars_exactly_at_length_threshold_is_single_deck(self, tmp_path):
        # 4 chapters but chars == threshold → 40_000 > 40_000 is False
        result, _, _ = _route(_skeleton(4), total_chars=40_000, output_dir=tmp_path)
        assert isinstance(result, SlidesFinal)

    def test_chapters_one_above_threshold_with_sufficient_length(self, tmp_path):
        result, _, _ = _route(_skeleton(4), total_chars=40_001, output_dir=tmp_path)
        assert isinstance(result, DeckIndex)


# ──────────────────────────────────────────────────────────────
# Real-world scenarios described in IMPROVE_MULTI_DECK_ROUTING.md
# ──────────────────────────────────────────────────────────────

class TestRealWorldScenarios:
    def test_short_paper_with_many_sections_stays_single_deck(self, tmp_path):
        # 8 sections, ~15 pages ≈ 30_000 chars — should NOT trigger multi-deck
        result, _, _ = _route(
            _skeleton(8), total_chars=30_000,
            chapter_threshold=3, length_threshold=40_000,
            output_dir=tmp_path,
        )
        assert isinstance(result, SlidesFinal)

    def test_long_survey_with_few_chapters_stays_single_deck(self, tmp_path):
        # 3 sections, ~80 pages ≈ 160_000 chars — chapter count not exceeded
        result, _, _ = _route(
            _skeleton(3), total_chars=160_000,
            chapter_threshold=3, length_threshold=40_000,
            output_dir=tmp_path,
        )
        assert isinstance(result, SlidesFinal)

    def test_large_textbook_with_many_chapters_triggers_multi_deck(self, tmp_path):
        # 18 chapters, ~600 pages ≈ 1_200_000 chars — both conditions met
        result, _, _ = _route(
            _skeleton(18), total_chars=1_200_000,
            chapter_threshold=3, length_threshold=40_000,
            output_dir=tmp_path,
        )
        assert isinstance(result, DeckIndex)
