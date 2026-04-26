"""
Milestone 7 — multi-deck routing and orchestration tests.

Uses unittest.mock.patch to mock pipeline.run_single_deck for testing run_multi_deck.
"""

import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pipeline import route, run_multi_deck
from schemas.deck_index import DeckIndex
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slides_final import FinalSlide, SlidesFinal


def _run(coro):
    return asyncio.run(coro)


def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Test Paper",
        document_type="research_paper",
        technical_level="intermediate",
        core_thesis="A thesis.",
        key_concepts=["concept"],
        sections=[Section(heading="Intro", importance="high", summary="Summary.")],
    )


def _skeleton_with_chapters(n: int) -> GlobalSkeleton:
    return GlobalSkeleton(
        title="Test Paper",
        document_type="research_paper",
        core_thesis="A thesis.",
        sections=[
            SectionEntry(heading=f"Chapter {i + 1}", level=1, position=i)
            for i in range(n)
        ],
    )


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test",
        slides=[FinalSlide(index=1, title="Slide 1", bullets=["Bullet."], tag="Key Concept")],
    )


def _make_agents() -> dict:
    provider = types.SimpleNamespace(name="stub", model="stub-model")
    planner  = types.SimpleNamespace(provider=provider)
    return {"planner": planner}


# ──────────────────────────────────────────────────────────────
# route() delegation
# ──────────────────────────────────────────────────────────────

class TestRouteMultiDeckDelegation:
    def test_route_returns_deck_index_when_threshold_exceeded(self, tmp_path):
        skeleton = _skeleton_with_chapters(4)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.return_value = (_slides_final(), [], Path("outputs/test.json"))
            result, _, _ = _run(route(
                "Test Paper", skeleton, _doc_map(), ["chunk"],
                _make_agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
                total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            ))
        assert isinstance(result, DeckIndex)

    def test_route_returns_slides_final_at_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.return_value = (_slides_final(), [], tmp_path / "test.json")
            result, _, _ = _run(route(
                "Test Paper", skeleton, _doc_map(), ["chunk"],
                _make_agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
                total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            ))
        assert isinstance(result, SlidesFinal)


# ──────────────────────────────────────────────────────────────
# run_multi_deck() orchestration
# ──────────────────────────────────────────────────────────────

class TestRunMultiDeckOrchestration:
    def test_spawns_one_task_per_chapter(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.return_value = (_slides_final(), [], Path("outputs/test.json"))
            _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        assert mock_rsd.call_count == 3

    def test_each_call_receives_non_none_scope(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.return_value = (_slides_final(), [], Path("outputs/test.json"))
            _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        for call in mock_rsd.call_args_list:
            assert call.kwargs["scope"] is not None

    def test_each_call_receives_correct_chapter_scope(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        chapters = [s for s in skeleton.sections if s.level == 1]
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.return_value = (_slides_final(), [], Path("outputs/test.json"))
            _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        for i, call in enumerate(mock_rsd.call_args_list):
            assert call.kwargs["scope"] == chapters[i]


# ──────────────────────────────────────────────────────────────
# Blast radius — single chapter failure must not crash the run
# ──────────────────────────────────────────────────────────────

class TestBlastRadius:
    def _make_side_effect(self):
        async def side_effect(*args, **kwargs):
            if kwargs["scope"].heading == "Chapter 2":
                raise Exception("Chapter 2 failed")
            return (_slides_final(), [], Path("outputs/test.json"))
        return side_effect

    def test_survives_chapter_exception(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.side_effect = self._make_side_effect()
            result, _, _ = _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        assert isinstance(result, DeckIndex)

    def test_index_contains_only_successful_chapters(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.side_effect = self._make_side_effect()
            result, _, _ = _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        assert len(result.decks) == 2

    def test_failed_chapter_not_in_index(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        with patch("pipeline.run_single_deck", new_callable=AsyncMock) as mock_rsd:
            mock_rsd.side_effect = self._make_side_effect()
            result, _, _ = _run(run_multi_deck(
                "Test", _doc_map(), skeleton, ["chunk"],
                _make_agents(), max_review_cycles=1,
                debug=False, output_dir=tmp_path,
            ))
        titles = [d.chapter_title for d in result.decks]
        assert "Chapter 2" not in titles
