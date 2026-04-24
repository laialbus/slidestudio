"""
Milestone 5 — pipeline.py tests (route, run_single_deck, write_output).

Uses lightweight stub agents — no mocking library, no real API calls.
All file I/O uses the pytest tmp_path fixture; the real outputs/ dir is never touched.
"""

import asyncio
import json
import types

import pytest

from pipeline import route, run_single_deck, write_output
from schemas.deck_index import DeckIndex
from schemas.critique import Critique, Issue, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import DraftSlide, SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal
from utils.slugify import slugify


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

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


def _slide_plan() -> SlidePlan:
    slides = [
        PlannedSlide(
            index=i + 1,
            tag="Key Concept",
            source_section="Intro",
            intention="Explain.",
            emphasis="Note.",
            chunk_indices=[0],
        )
        for i in range(4)
    ]
    return SlidePlan(title="Test Deck", total_slides=4, slides=slides)


def _slides_draft() -> SlidesDraft:
    return SlidesDraft(
        title="Test Deck",
        slides=[
            DraftSlide(index=i + 1, title=f"Slide {i + 1}", bullets=["Bullet."], tag="Key Concept")
            for i in range(4)
        ],
    )


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test Deck",
        slides=[FinalSlide(index=1, title="Slide 1", bullets=["Bullet."], tag="Key Concept")],
    )


def _all_passing_critique(draft: SlidesDraft) -> Critique:
    return Critique(slides=[SlideReview(index=s.index, passed=True) for s in draft.slides])


# ──────────────────────────────────────────────────────────────
# Stub agents
# ──────────────────────────────────────────────────────────────

class StubPlanner:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0
        self.provider  = types.SimpleNamespace(name="stub", model="stub-model")

    async def run(self, doc_map, skeleton, scope=None):
        if self.call_log is not None:
            self.call_log.append("planner")
        self.call_count += 1
        return _slide_plan()


class StubWriter:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0

    async def run(self, slide_plan, doc_map, chunks):
        if self.call_log is not None:
            self.call_log.append("writer")
        self.call_count += 1
        return _slides_draft()


class StubCritic:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0

    async def run(self, doc_map, slides):
        if self.call_log is not None:
            self.call_log.append("critic")
        self.call_count += 1
        return _all_passing_critique(slides)


class StubRefiner:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0

    async def run(self, doc_map, slides, critique):
        if self.call_log is not None:
            self.call_log.append("refiner")
        self.call_count += 1
        return slides


def _agents(call_log: list | None = None) -> dict:
    return {
        "planner": StubPlanner(call_log),
        "writer":  StubWriter(call_log),
        "critic":  StubCritic(call_log),
        "refiner": StubRefiner(call_log),
    }


def _intermediates() -> dict:
    draft = _slides_draft()
    return {
        "slide_plan":   _slide_plan(),
        "slides_draft": draft,
        "critique":     _all_passing_critique(draft),
    }


# ──────────────────────────────────────────────────────────────
# Tests — route()
# ──────────────────────────────────────────────────────────────

class TestRoute:
    def test_returns_slides_final_when_chapters_at_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(3)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"],
            _agents(), multi_deck_threshold=3, max_review_cycles=1,
            debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_slides_final_when_chapters_below_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(2)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"],
            _agents(), multi_deck_threshold=3, max_review_cycles=1,
            debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_deck_index_when_chapters_exceed_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(4)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"],
            _agents(), multi_deck_threshold=3, max_review_cycles=1,
            debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, DeckIndex)

    def test_threshold_is_strictly_greater(self, tmp_path):
        # exactly threshold → single-deck (not raised)
        skeleton = _skeleton_with_chapters(3)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"],
            _agents(), multi_deck_threshold=3, max_review_cycles=1,
            debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, SlidesFinal)

    def test_threshold_plus_one_returns_deck_index(self, tmp_path):
        skeleton = _skeleton_with_chapters(4)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"],
            _agents(), multi_deck_threshold=3, max_review_cycles=1,
            debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, DeckIndex)


# ──────────────────────────────────────────────────────────────
# Tests — write_output()
# ──────────────────────────────────────────────────────────────

class TestWriteOutput:
    def test_writes_json_at_slugified_path(self, tmp_path):
        write_output(_slides_final(), "My Test Paper", False, tmp_path, _intermediates())
        assert (tmp_path / "my_test_paper.json").exists()

    def test_output_is_valid_json(self, tmp_path):
        write_output(_slides_final(), "My Test Paper", False, tmp_path, _intermediates())
        data = json.loads((tmp_path / "my_test_paper.json").read_text())
        assert "slides" in data

    def test_hostile_title_slugified_correctly(self, tmp_path):
        hostile = "Chapter 1: The /\\ File * System?"
        write_output(_slides_final(), hostile, False, tmp_path, _intermediates())
        slug = slugify(hostile)
        assert (tmp_path / f"{slug}.json").exists()

    def test_hostile_title_output_is_valid_json(self, tmp_path):
        hostile = "Chapter 1: The /\\ File * System?"
        write_output(_slides_final(), hostile, False, tmp_path, _intermediates())
        slug = slugify(hostile)
        data = json.loads((tmp_path / f"{slug}.json").read_text())
        assert "slides" in data

    def test_no_debug_dir_when_debug_false(self, tmp_path):
        write_output(_slides_final(), "Test", False, tmp_path, _intermediates())
        assert not (tmp_path / "debug").exists()

    def test_debug_creates_slide_plan_file(self, tmp_path):
        write_output(_slides_final(), "Test Paper", True, tmp_path, _intermediates())
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "01_slide_plan.json").exists()

    def test_debug_creates_slides_draft_file(self, tmp_path):
        write_output(_slides_final(), "Test Paper", True, tmp_path, _intermediates())
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "02_slides_draft.json").exists()

    def test_debug_creates_critique_file(self, tmp_path):
        write_output(_slides_final(), "Test Paper", True, tmp_path, _intermediates())
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "03_critique.json").exists()

    def test_debug_files_are_valid_json(self, tmp_path):
        write_output(_slides_final(), "Test Paper", True, tmp_path, _intermediates())
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        for fname in ["01_slide_plan.json", "02_slides_draft.json", "03_critique.json"]:
            data = json.loads((debug_dir / fname).read_text())
            assert isinstance(data, dict)

    def test_debug_dir_uses_slugified_title(self, tmp_path):
        write_output(_slides_final(), "My Paper Title", True, tmp_path, _intermediates())
        assert (tmp_path / "debug" / "my_paper_title").is_dir()


# ──────────────────────────────────────────────────────────────
# Tests — run_single_deck() agent call sequence
# ──────────────────────────────────────────────────────────────

class TestRunSingleDeckSequence:
    def test_returns_slides_final(self, tmp_path):
        result, _, _ = _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_unresolved_list(self, tmp_path):
        _, unresolved, _ = _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert isinstance(unresolved, list)

    def test_planner_called_before_writer(self, tmp_path):
        call_log = []
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(call_log), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert call_log.index("planner") < call_log.index("writer")

    def test_writer_called_before_critic(self, tmp_path):
        call_log = []
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(call_log), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert call_log.index("writer") < call_log.index("critic")

    def test_planner_called_exactly_once(self, tmp_path):
        agents = _agents()
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            agents, max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert agents["planner"].call_count == 1

    def test_writer_called_exactly_once(self, tmp_path):
        agents = _agents()
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            agents, max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert agents["writer"].call_count == 1

    def test_writes_output_file(self, tmp_path):
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert (tmp_path / f"{slugify('Test Paper')}.json").exists()

    def test_debug_writes_all_intermediate_files(self, tmp_path):
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"],
            _agents(), max_review_cycles=1, debug=True, output_dir=tmp_path,
        ))
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "01_slide_plan.json").exists()
        assert (debug_dir / "02_slides_draft.json").exists()
        assert (debug_dir / "03_critique.json").exists()
