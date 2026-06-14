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

    def test_output_stem_overrides_title_for_filename(self, tmp_path):
        # In production run() passes a content-keyed stem; the display title is
        # unaffected (it comes from SlidesFinal.title, asserted elsewhere).
        path = write_output(
            _slides_final(), [], "Ignored Title", False, tmp_path,
            _intermediates(), _PROVIDER, _MODEL, output_stem="myfile_deadbeef",
        )
        assert re.fullmatch(r"myfile_deadbeef_\d{8}T\d{6}(_\d+)?\.json", path.name)

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

_HASH = "deadbeef"


class TestDuplicatePolicy:
    def _route(self, tmp_path, policy, *, name_slug="test_paper", doc_hash=_HASH):
        return _run(route(
            "Test Paper", _skeleton_with_chapters(1), _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=10_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy=policy,
            output_stem=f"{name_slug}_{doc_hash}", doc_hash=doc_hash,
        ))

    def _outputs(self, tmp_path):
        return sorted(p.name for p in tmp_path.glob(f"*_{_HASH}_*.json"))

    def _library(self, tmp_path):
        return json.loads((tmp_path / "library.json").read_text())

    def test_overwrite_leaves_single_output_after_two_runs(self, tmp_path):
        self._route(tmp_path, "overwrite")
        self._route(tmp_path, "overwrite")
        assert len(self._outputs(tmp_path)) == 1

    def test_overwrite_leaves_single_library_entry_after_two_runs(self, tmp_path):
        # The manifest entry is keyed on the timestamped path, so without
        # hash-based pruning the prior entry would linger (the original bug).
        self._route(tmp_path, "overwrite")
        self._route(tmp_path, "overwrite")
        assert len(self._library(tmp_path)) == 1

    def test_keep_both_leaves_two_outputs_after_two_runs(self, tmp_path):
        self._route(tmp_path, "keep_both")
        self._route(tmp_path, "keep_both")
        assert len(self._outputs(tmp_path)) == 2

    def test_overwrite_matches_same_content_under_renamed_file(self, tmp_path):
        # Content hash is the identity: a re-run of the same PDF overwrites even
        # when the source file was renamed (different readable stem, same hash).
        self._route(tmp_path, "overwrite", name_slug="old_name")
        self._route(tmp_path, "overwrite", name_slug="new_name")
        assert len(self._outputs(tmp_path)) == 1

    def test_overwrite_does_not_touch_archived_copy(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / f"test_paper_{_HASH}_20260101T000000.json").write_text("{}")
        self._route(tmp_path, "overwrite")
        assert (archive / f"test_paper_{_HASH}_20260101T000000.json").exists()

    def test_overwrite_cleans_prior_deck_even_when_resuming(self, tmp_path):
        # A resume only ever continues a failed run, so a successful resumed run
        # must still overwrite the prior (successful) deck — cleanup is no longer
        # gated on the resume flag.
        from utils.checkpoint import Checkpoint
        old = tmp_path / f"test_paper_{_HASH}_20260101T000000.json"
        old.write_text("{}")
        (tmp_path / "library.json").write_text(json.dumps([{
            "title": "Old", "file": "/" + old.relative_to(tmp_path.parent).as_posix(),
            "type": "single_deck", "doc_hash": _HASH, "archived": False,
        }]))
        ck = Checkpoint(base_dir=tmp_path / ".ckpt", run_key="k", resume=True)
        _run(route(
            "Test Paper", _skeleton_with_chapters(1), _doc_map(), ["chunk"], [],
            _agents(), multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
            total_chars=10_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
            duplicate_policy="overwrite", checkpoint=ck,
            output_stem=f"test_paper_{_HASH}", doc_hash=_HASH,
        ))
        assert not old.exists()
        assert len(self._outputs(tmp_path)) == 1

    def test_overwrite_keeps_old_output_when_generation_fails(self, tmp_path):
        # Crash-safety: cleanup runs only after a successful write, so a failure
        # mid-generation leaves the prior deck (and its manifest entry) intact.
        old = tmp_path / f"test_paper_{_HASH}_20260101T000000.json"
        old.write_text("{}")
        (tmp_path / "library.json").write_text(json.dumps([{
            "title": "Old", "file": f"/outputs/test_paper_{_HASH}_20260101T000000.json",
            "type": "single_deck", "generated_at": "2026-01-01T00:00:00+00:00",
            "doc_hash": _HASH, "archived": False,
        }]))
        agents = _agents()
        async def boom(**_kwargs):
            raise RuntimeError("planner exploded")
        agents["planner"].run = boom
        with pytest.raises(RuntimeError):
            _run(route(
                "Test Paper", _skeleton_with_chapters(1), _doc_map(), ["chunk"], [],
                agents, multi_deck_chapter_threshold=3, multi_deck_length_threshold=40_000,
                total_chars=10_000, max_review_cycles=1, debug=False, output_dir=tmp_path,
                duplicate_policy="overwrite",
                output_stem=f"test_paper_{_HASH}", doc_hash=_HASH,
            ))
        assert old.exists()
        assert len(self._library(tmp_path)) == 1


class TestCleanupStaleOutput:
    """Cleanup is manifest-driven: it deletes the files referenced by library.json
    entries whose stored doc_hash matches — never by parsing the filename."""

    def _entry(self, tmp_path, name, doc_hash, *, multi=False, archived=False):
        if multi:
            d = tmp_path / name
            d.mkdir()
            (d / "index.json").write_text("{}")
            on_disk, file_rel = d, d / "index.json"
            dtype = "multi_deck"
        else:
            f = tmp_path / name
            f.write_text("{}")
            on_disk, file_rel = f, f
            dtype = "single_deck"
        entry = {
            "title": "X", "type": dtype, "archived": archived,
            "doc_hash": doc_hash,
            "file": "/" + file_rel.relative_to(tmp_path.parent).as_posix(),
        }
        return on_disk, entry

    def _write_lib(self, tmp_path, entries):
        (tmp_path / "library.json").write_text(json.dumps(entries), encoding="utf-8")

    def _lib(self, tmp_path):
        return json.loads((tmp_path / "library.json").read_text())

    def test_removes_matching_files_and_entries(self, tmp_path):
        f1, e1 = self._entry(tmp_path, "a.json", _HASH)
        d1, e2 = self._entry(tmp_path, "book", _HASH, multi=True)
        self._write_lib(tmp_path, [e1, e2])
        _cleanup_stale_output(tmp_path, _HASH)
        assert not f1.exists()
        assert not d1.exists()
        assert self._lib(tmp_path) == []

    def test_matches_regardless_of_filename(self, tmp_path):
        # Same stored hash, different names (renamed PDF) → both this document.
        f1, e1 = self._entry(tmp_path, "alpha.json", _HASH)
        f2, e2 = self._entry(tmp_path, "beta.json", _HASH)
        self._write_lib(tmp_path, [e1, e2])
        _cleanup_stale_output(tmp_path, _HASH)
        assert not f1.exists() and not f2.exists()

    def test_does_not_remove_different_hash(self, tmp_path):
        f1, e1 = self._entry(tmp_path, "other.json", "cafebabe")
        self._write_lib(tmp_path, [e1])
        _cleanup_stale_output(tmp_path, _HASH)
        assert f1.exists()
        assert self._lib(tmp_path) == [e1]

    def test_preserves_archived_entry_and_file(self, tmp_path):
        f1, e1 = self._entry(tmp_path, "archived.json", _HASH, archived=True)
        self._write_lib(tmp_path, [e1])
        _cleanup_stale_output(tmp_path, _HASH)
        assert f1.exists()
        assert self._lib(tmp_path) == [e1]

    def test_orphan_file_not_in_manifest_is_left(self, tmp_path):
        # A file on disk but absent from the manifest is invisible to cleanup
        # (the documented tradeoff of manifest-driven deletion; library-refresh
        # is the repair path).
        orphan = tmp_path / f"orphan_{_HASH}_20260101T000000.json"
        orphan.write_text("{}")
        self._write_lib(tmp_path, [])
        _cleanup_stale_output(tmp_path, _HASH)
        assert orphan.exists()

    def test_empty_hash_is_a_no_op(self, tmp_path):
        f1, e1 = self._entry(tmp_path, "a.json", _HASH)
        self._write_lib(tmp_path, [e1])
        _cleanup_stale_output(tmp_path, "")
        assert f1.exists()

    def test_missing_manifest_is_a_no_op(self, tmp_path):
        (tmp_path / "stray.json").write_text("{}")
        _cleanup_stale_output(tmp_path, _HASH)  # no library.json — must not raise

    def test_missing_output_dir_is_a_no_op(self, tmp_path):
        _cleanup_stale_output(tmp_path / "does_not_exist", _HASH)

    def test_keep_excludes_the_just_written_file(self, tmp_path):
        old, e_old = self._entry(tmp_path, "paper_old.json", _HASH)
        new, e_new = self._entry(tmp_path, "paper_new.json", _HASH)
        self._write_lib(tmp_path, [e_old, e_new])
        _cleanup_stale_output(tmp_path, _HASH, keep=new)
        assert not old.exists()
        assert new.exists()
        assert [e["file"] for e in self._lib(tmp_path)] == [e_new["file"]]

    def test_keep_excludes_the_just_written_multi_deck_dir(self, tmp_path):
        old_dir, e_old = self._entry(tmp_path, "book_old", _HASH, multi=True)
        new_dir, e_new = self._entry(tmp_path, "book_new", _HASH, multi=True)
        self._write_lib(tmp_path, [e_old, e_new])
        _cleanup_stale_output(tmp_path, _HASH, keep=new_dir / "index.json")
        assert not old_dir.exists()
        assert new_dir.exists()


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
