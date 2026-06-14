"""
Milestone 5 — pipeline.py tests (route, run_single_deck, write_output).

Uses lightweight stub agents — no mocking library, no real API calls.
All file I/O uses the pytest tmp_path fixture; the real outputs/ dir is never touched.
"""

import asyncio
import json
import re
import types

import pytest

from pipeline import (
    _cleanup_stale_output,
    build_figure_catalog,
    route,
    run_single_deck,
    write_output,
)
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
            DraftSlide(index=i + 1, heading=f"Slide {i + 1}", body="Bullet.", tag="Key Concept")
            for i in range(4)
        ],
    )


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test Deck",
        slides=[FinalSlide(index=1, heading="Slide 1", body="Bullet.", tag="Key Concept")],
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

    async def run(self, doc_map, skeleton, figure_catalog=None, scope=None):
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

    async def write_summary(self, completed_slides, summary_index):
        return _slides_draft()


class StubCritic:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0

    async def run(self, doc_map, slides, **kwargs):
        if self.call_log is not None:
            self.call_log.append("critic")
        self.call_count += 1
        return _all_passing_critique(slides)


class StubRefiner:
    def __init__(self, call_log: list | None = None):
        self.call_log  = call_log
        self.call_count = 0

    async def run(self, doc_map, slides, critique, deck_feedback=None, **kwargs):
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
            "Test Paper", skeleton, _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite",
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_slides_final_when_chapters_below_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(2)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite",
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_deck_index_when_chapters_exceed_threshold(self, tmp_path):
        skeleton = _skeleton_with_chapters(4)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite",
        ))
        assert isinstance(result, DeckIndex)

    def test_threshold_is_strictly_greater(self, tmp_path):
        # exactly threshold → single-deck (not raised)
        skeleton = _skeleton_with_chapters(3)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite",
        ))
        assert isinstance(result, SlidesFinal)

    def test_threshold_plus_one_returns_deck_index(self, tmp_path):
        skeleton = _skeleton_with_chapters(4)
        result, _, _ = _run(route(
            "Test Paper", skeleton, _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=50_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite",
        ))
        assert isinstance(result, DeckIndex)


# ──────────────────────────────────────────────────────────────
# Tests — write_output()
# ──────────────────────────────────────────────────────────────

_PROVIDER = "stub"
_MODEL    = "stub-model"


class TestWriteOutput:
    def test_writes_json_at_slugified_timestamped_path(self, tmp_path):
        path = write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert path.exists()
        assert path.parent == tmp_path
        assert re.fullmatch(r"my_test_paper_\d{8}T\d{6}(_\d+)?\.json", path.name)

    def test_output_is_valid_json(self, tmp_path):
        path = write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        data = json.loads(path.read_text())
        assert "slides" in data

    def test_output_contains_provider_and_model(self, tmp_path):
        path = write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        data = json.loads(path.read_text())
        assert data["provider"] == _PROVIDER
        assert data["model"]    == _MODEL

    def test_hostile_title_slugified_correctly(self, tmp_path):
        hostile = "Chapter 1: The /\\ File * System?"
        path = write_output(_slides_final(), [], hostile, False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert path.name.startswith(slugify(hostile) + "_")
        assert path.exists()

    def test_hostile_title_output_is_valid_json(self, tmp_path):
        hostile = "Chapter 1: The /\\ File * System?"
        path = write_output(_slides_final(), [], hostile, False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        data = json.loads(path.read_text())
        assert "slides" in data

    def test_empty_title_falls_back_to_untitled_document(self, tmp_path):
        path = write_output(_slides_final(), [], "???", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert path.name.startswith("untitled_document_")
        assert path.exists()

    def test_two_runs_same_title_keep_distinct_files(self, tmp_path):
        p1 = write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        p2 = write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_no_debug_dir_when_debug_false(self, tmp_path):
        write_output(_slides_final(), [], "Test", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert not (tmp_path / "debug").exists()

    def test_debug_creates_slide_plan_file(self, tmp_path):
        write_output(_slides_final(), [], "Test Paper", True, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "01_slide_plan.json").exists()

    def test_debug_creates_slides_draft_file(self, tmp_path):
        write_output(_slides_final(), [], "Test Paper", True, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "02_slides_draft.json").exists()

    def test_debug_creates_critique_file(self, tmp_path):
        write_output(_slides_final(), [], "Test Paper", True, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "03_critique.json").exists()

    def test_debug_files_are_valid_json(self, tmp_path):
        write_output(_slides_final(), [], "Test Paper", True, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        for fname in ["01_slide_plan.json", "02_slides_draft.json", "03_critique.json"]:
            data = json.loads((debug_dir / fname).read_text())
            assert isinstance(data, dict)

    def test_debug_dir_uses_slugified_title(self, tmp_path):
        write_output(_slides_final(), [], "My Paper Title", True, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert (tmp_path / "debug" / "my_paper_title").is_dir()

    def test_writes_library_manifest(self, tmp_path):
        write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        assert (tmp_path / "library.json").exists()

    def test_library_manifest_contains_entry(self, tmp_path):
        write_output(_slides_final(), [], "My Test Paper", False, tmp_path, _intermediates(), _PROVIDER, _MODEL)
        data = json.loads((tmp_path / "library.json").read_text())
        assert len(data) == 1
        # manifest title comes from SlidesFinal.title, not the slug title param
        assert data[0]["title"] == _slides_final().title
        assert data[0]["type"]  == "single_deck"


# ──────────────────────────────────────────────────────────────
# Tests — run_single_deck() agent call sequence
# ──────────────────────────────────────────────────────────────

class TestRunSingleDeckSequence:
    def test_returns_slides_final(self, tmp_path):
        result, _, _ = _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert isinstance(result, SlidesFinal)

    def test_returns_unresolved_list(self, tmp_path):
        _, unresolved, _ = _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert isinstance(unresolved, list)

    def test_planner_called_before_writer(self, tmp_path):
        call_log = []
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(call_log), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert call_log.index("planner") < call_log.index("writer")

    def test_writer_called_before_critic(self, tmp_path):
        call_log = []
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(call_log), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert call_log.index("writer") < call_log.index("critic")

    def test_planner_called_exactly_once(self, tmp_path):
        agents = _agents()
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            agents, max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert agents["planner"].call_count == 1

    def test_writer_called_exactly_once(self, tmp_path):
        agents = _agents()
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            agents, max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert agents["writer"].call_count == 1

    def test_writes_output_file(self, tmp_path):
        _, _, output_path = _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(), max_review_cycles=1, debug=False, output_dir=tmp_path,
        ))
        assert output_path is not None and output_path.exists()
        assert output_path.name.startswith(f"{slugify('Test Paper')}_")

    def test_debug_writes_all_intermediate_files(self, tmp_path):
        _run(run_single_deck(
            "Test Paper", _doc_map(), _skeleton_with_chapters(1), ["chunk"], [],
            _agents(), max_review_cycles=1, debug=True, output_dir=tmp_path,
        ))
        debug_dir = tmp_path / "debug" / slugify("Test Paper")
        assert (debug_dir / "01_slide_plan.json").exists()
        assert (debug_dir / "02_slides_draft.json").exists()
        assert (debug_dir / "03_critique.json").exists()


# ──────────────────────────────────────────────────────────────
# Tests — duplicate_policy and stale-output cleanup
# ──────────────────────────────────────────────────────────────

class TestDuplicatePolicy:
    def _route(self, tmp_path, policy):
        return _run(route(
            "Test Paper", _skeleton_with_chapters(1), _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=10_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy=policy,
        ))

    def _outputs(self, tmp_path):
        return sorted(p.name for p in tmp_path.glob("test_paper*.json"))

    def test_overwrite_leaves_single_output_after_two_runs(self, tmp_path):
        self._route(tmp_path, "overwrite")
        self._route(tmp_path, "overwrite")
        assert len(self._outputs(tmp_path)) == 1

    def test_keep_both_leaves_two_outputs_after_two_runs(self, tmp_path):
        self._route(tmp_path, "keep_both")
        self._route(tmp_path, "keep_both")
        assert len(self._outputs(tmp_path)) == 2

    def test_overwrite_removes_legacy_untimestamped_file(self, tmp_path):
        (tmp_path / "test_paper.json").write_text("{}")
        self._route(tmp_path, "overwrite")
        names = self._outputs(tmp_path)
        assert "test_paper.json" not in names
        assert len(names) == 1

    def test_overwrite_does_not_touch_archived_copy(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "test_paper.json").write_text("{}")
        self._route(tmp_path, "overwrite")
        assert (archive / "test_paper.json").exists()


class TestCleanupStaleOutput:
    def test_removes_legacy_file_and_directory(self, tmp_path):
        (tmp_path / "my_paper.json").write_text("{}")
        (tmp_path / "my_paper").mkdir()
        _cleanup_stale_output(tmp_path, "My Paper")
        assert not (tmp_path / "my_paper.json").exists()
        assert not (tmp_path / "my_paper").exists()

    def test_removes_timestamped_variants(self, tmp_path):
        (tmp_path / "my_paper_20260101T000000.json").write_text("{}")
        (tmp_path / "my_paper_20260101T000000_2.json").write_text("{}")
        (tmp_path / "my_paper_20260101T000000").mkdir()
        _cleanup_stale_output(tmp_path, "My Paper")
        assert list(tmp_path.iterdir()) == []

    def test_does_not_remove_longer_slug_sharing_prefix(self, tmp_path):
        (tmp_path / "my_paper_extended_20260101T000000.json").write_text("{}")
        _cleanup_stale_output(tmp_path, "My Paper")
        assert (tmp_path / "my_paper_extended_20260101T000000.json").exists()

    def test_never_removes_reserved_names(self, tmp_path):
        (tmp_path / "archive").mkdir()
        (tmp_path / "debug").mkdir()
        (tmp_path / "library.json").write_text("[]")
        _cleanup_stale_output(tmp_path, "Archive")
        _cleanup_stale_output(tmp_path, "Debug")
        _cleanup_stale_output(tmp_path, "Library")
        assert (tmp_path / "archive").is_dir()
        assert (tmp_path / "debug").is_dir()
        assert (tmp_path / "library.json").exists()

    def test_empty_title_cleans_untitled_document(self, tmp_path):
        (tmp_path / "untitled_document_20260101T000000.json").write_text("{}")
        _cleanup_stale_output(tmp_path, "???")
        assert list(tmp_path.iterdir()) == []

    def test_missing_output_dir_is_a_no_op(self, tmp_path):
        _cleanup_stale_output(tmp_path / "does_not_exist", "My Paper")


# ──────────────────────────────────────────────────────────────
# build_figure_catalog
# ──────────────────────────────────────────────────────────────

class TestBuildFigureCatalog:
    def _images(self, *captions: str) -> list[dict]:
        return [
            {"index": i, "caption": c, "data_uri": "data:,", "page": 1}
            for i, c in enumerate(captions)
        ]

    def test_referenceable_figure_in_catalog(self):
        images = self._images("Figure 0 caption", "Figure 1 caption")
        chunk_images = [[0], [1]]
        figure_purposes = [{0: "conceptual"}, {1: "evidential"}]
        catalog = build_figure_catalog(images, chunk_images, figure_purposes)
        by_id = {e["figure_id"]: e for e in catalog}
        assert by_id[0]["caption"] == "Figure 0 caption"
        assert by_id[0]["purpose"] == "conceptual"
        assert by_id[0]["source_chunk"] == 0
        assert by_id[1]["purpose"] == "evidential"
        assert by_id[1]["source_chunk"] == 1

    def test_unreferenced_figure_excluded(self):
        # Figure 1 has a caption but never appears in chunk_images → excluded.
        images = self._images("Figure 0 caption", "Figure 1 caption")
        chunk_images = [[0], []]
        catalog = build_figure_catalog(images, chunk_images, [])
        assert [e["figure_id"] for e in catalog] == [0]

    def test_empty_caption_excluded(self):
        images = self._images("", "Figure 1 caption")
        chunk_images = [[0], [1]]
        catalog = build_figure_catalog(images, chunk_images, [])
        assert [e["figure_id"] for e in catalog] == [1]

    def test_purpose_defaults_to_unspecified(self):
        images = self._images("Figure 0 caption")
        catalog = build_figure_catalog(images, [[0]], [])
        assert catalog[0]["purpose"] == "unspecified"

    def test_first_source_chunk_wins(self):
        # Figure 0's marker appears in chunks 1 and 3; the first wins.
        images = self._images("Figure 0 caption")
        chunk_images = [[], [0], [], [0]]
        catalog = build_figure_catalog(images, chunk_images, [])
        assert catalog[0]["source_chunk"] == 1

    def test_sorted_by_figure_id(self):
        images = self._images("Figure 0", "Figure 1", "Figure 2")
        chunk_images = [[2], [0], [1]]
        catalog = build_figure_catalog(images, chunk_images, [])
        assert [e["figure_id"] for e in catalog] == [0, 1, 2]

    def test_no_figures_yields_empty_catalog(self):
        catalog = build_figure_catalog(self._images("Figure 0"), [[], []], [])
        assert catalog == []
