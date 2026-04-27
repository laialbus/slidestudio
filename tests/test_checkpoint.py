"""
Task 2 — Checkpoint tests.

Tests verify:
1. Analyst is skipped when a valid checkpoint exists (--resume).
2. Checkpoint files are never written for outputs that fail Pydantic validation.
3. --force ignores existing checkpoints, runs fresh, and overwrites cache.
4. In multi-deck mode, only the failed/missing chapter is regenerated on resume.
"""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

from agents.analyst import AnalystResult
import pipeline as _pipeline_module
from pipeline import run as pipeline_run
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.chapter_map import ChapterMap
from schemas.critique import Critique, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import DraftSlide, SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal
from utils.checkpoint import Checkpoint


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def _skeleton() -> GlobalSkeleton:
    return GlobalSkeleton(
        title="Test Paper",
        document_type="research_paper",
        core_thesis="A thesis.",
        sections=[SectionEntry(heading="Introduction", level=1, position=0)],
    )


def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Test Paper",
        document_type="research_paper",
        technical_level="intermediate",
        core_thesis="A thesis.",
        key_concepts=["concept"],
        sections=[Section(heading="Introduction", importance="high", summary="Summary.")],
    )


def _slide_plan() -> SlidePlan:
    return SlidePlan(
        title="Test Deck",
        total_slides=4,
        slides=[
            PlannedSlide(
                index=i + 1,
                tag="Key Concept",
                source_section="Introduction",
                intention="Teach it.",
                emphasis="Key point.",
                chunk_indices=[0],
            )
            for i in range(4)
        ],
    )


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test Deck",
        slides=[
            FinalSlide(index=i + 1, heading=f"Slide {i+1}", body="A point.", tag="Key Concept")
            for i in range(4)
        ],
    )


def _slides_draft() -> SlidesDraft:
    return SlidesDraft(
        title="Test Deck",
        slides=[
            DraftSlide(index=i + 1, heading=f"Slide {i+1}", body="A point.", tag="Key Concept")
            for i in range(4)
        ],
    )


def _passing_critique(slides: SlidesDraft) -> Critique:
    return Critique(slides=[SlideReview(index=s.index, passed=True) for s in slides.slides])


_FAKE_EXTRACTION = {"headers": ["Introduction"], "chunks": ["chunk text"]}


# ──────────────────────────────────────────────────────────────
# Stub provider and agents
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self, responses: dict[type, list] | None = None):
        super().__init__(ProviderConfig(
            model="stub-model",
            max_concurrent=None,
            max_format_retries=1,
            max_rate_limit_retries=1,
            request_timeout=5,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            backoff_wait_min=0,
            backoff_wait_max=0,
        ))
        self._responses = {k: list(v) for k, v in (responses or {}).items()}
        self._indices: dict[type, int] = {}
        self.call_log: list[type] = []

    async def complete_json(self, prompt, schema, system=""):
        self.call_log.append(schema)
        idx = self._indices.get(schema, 0)
        self._indices[schema] = idx + 1
        return self._responses[schema][idx % len(self._responses[schema])]

    async def _call(self, messages, system, response_schema=None):
        raise NotImplementedError

    @property
    def name(self):
        return "stub"


class _DummyProvider:
    name  = "stub"
    model = "stub-model"


class StubAnalyst:
    def __init__(self, result: AnalystResult):
        self._result    = result
        self.call_count = 0

    async def run(self, extraction: dict) -> AnalystResult:
        self.call_count += 1
        return self._result


class StubPlanner:
    def __init__(self, plan: SlidePlan, provider=None):
        self.provider   = provider or _DummyProvider()
        self._plan      = plan
        self.call_count = 0

    async def run(self, **kwargs) -> SlidePlan:
        self.call_count += 1
        return self._plan


class StubWriter:
    def __init__(self, draft: SlidesDraft):
        self._draft     = draft
        self.call_count = 0

    async def run(self, **kwargs) -> SlidesDraft:
        self.call_count += 1
        return self._draft


class StubCritic:
    def __init__(self, critique_fn=None):
        self._fn        = critique_fn or (lambda slides: _passing_critique(slides))
        self.call_count = 0

    async def run(self, doc_map, slides) -> Critique:
        self.call_count += 1
        return self._fn(slides)


class StubRefiner:
    async def run(self, **kwargs) -> SlidesDraft:
        return kwargs["slides"]


# ──────────────────────────────────────────────────────────────
# Context managers for patching pipeline internals
# ──────────────────────────────────────────────────────────────

@contextmanager
def _patch_extractor(extraction=None):
    """Patch PDFExtractor in the pipeline module to return a fixed extraction."""
    if extraction is None:
        extraction = _FAKE_EXTRACTION

    class _FakeExtractor:
        def __init__(self, **kwargs):
            pass
        def extract(self, path):
            return extraction

    original = _pipeline_module.PDFExtractor
    _pipeline_module.PDFExtractor = _FakeExtractor
    try:
        yield
    finally:
        _pipeline_module.PDFExtractor = original


@contextmanager
def _patch_analyst(analyst: StubAnalyst):
    """Patch AnalystAgent in the pipeline module to use the given stub."""
    class _Patched:
        def __init__(self, provider):
            self._delegate = analyst
        async def run(self, extraction):
            return await self._delegate.run(extraction)

    original = _pipeline_module.AnalystAgent
    _pipeline_module.AnalystAgent = _Patched
    try:
        yield
    finally:
        _pipeline_module.AnalystAgent = original


# ──────────────────────────────────────────────────────────────
# Fixture: temporary checkpoint and output directories
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    return {
        "cache":  tmp_path / ".checkpoints",
        "output": tmp_path / "outputs",
    }


def _checkpoint(tmp_dirs, resume: bool) -> Checkpoint:
    return Checkpoint(
        base_dir=tmp_dirs["cache"],
        run_key="testkey",
        resume=resume,
    )


def _default_agents(planner=None, analyst=None):
    return {
        "planner": planner or StubPlanner(_slide_plan()),
        "writer":  StubWriter(_slides_draft()),
        "critic":  StubCritic(),
        "refiner": StubRefiner(),
    }


async def _run_pipeline_async(tmp_dirs, checkpoint, agents):
    return await pipeline_run(
        file_path=Path("dummy.pdf"),
        provider=StubProvider(),
        agents=agents,
        output_dir=tmp_dirs["output"],
        chunk_size=8000,
        overlap_size=1500,
        multi_deck_chapter_threshold=3,
        multi_deck_length_threshold=0,
        max_review_cycles=1,
        debug=False,
        checkpoint=checkpoint,
    )


def _full_run(tmp_dirs, checkpoint, agents, analyst):
    with _patch_extractor(), _patch_analyst(analyst):
        return _run(_run_pipeline_async(tmp_dirs, checkpoint, agents))


# ──────────────────────────────────────────────────────────────
# Test 1 — Analyst skipped when checkpoint exists (--resume)
# ──────────────────────────────────────────────────────────────

class TestAnalystCheckpoint:
    def test_analyst_not_called_when_checkpoint_exists(self, tmp_dirs):
        sk     = _skeleton()
        dm     = _doc_map()

        # Pre-populate cache
        cp = _checkpoint(tmp_dirs, resume=False)
        cp.save("skeleton", sk)
        cp.save("doc_map",  dm)

        analyst    = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))
        cp_resume  = _checkpoint(tmp_dirs, resume=True)

        _full_run(tmp_dirs, cp_resume, _default_agents(), analyst)

        assert analyst.call_count == 0

    def test_analyst_called_when_no_checkpoint(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        cp = _checkpoint(tmp_dirs, resume=True)
        _full_run(tmp_dirs, cp, _default_agents(), analyst)

        assert analyst.call_count == 1

    def test_checkpoint_written_after_analyst_runs(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        cp = _checkpoint(tmp_dirs, resume=False)
        _full_run(tmp_dirs, cp, _default_agents(), analyst)

        assert (tmp_dirs["cache"] / "testkey" / "skeleton.json").exists()
        assert (tmp_dirs["cache"] / "testkey" / "doc_map.json").exists()

    def test_resume_on_fresh_cache_runs_normally(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        cp = _checkpoint(tmp_dirs, resume=True)
        # Should not raise even though no cache exists
        _full_run(tmp_dirs, cp, _default_agents(), analyst)

        assert analyst.call_count == 1


# ──────────────────────────────────────────────────────────────
# Test 2 — Checkpoint file never written for invalid output
# ──────────────────────────────────────────────────────────────

class TestCheckpointAtomicity:
    def test_save_then_load_roundtrip(self, tmp_dirs):
        cp     = _checkpoint(tmp_dirs, resume=True)
        sk     = _skeleton()
        cp.save("skeleton", sk)
        loaded = cp.load("skeleton", GlobalSkeleton)
        assert loaded is not None
        assert loaded.title == sk.title

    def test_no_file_before_save(self, tmp_dirs):
        cp   = _checkpoint(tmp_dirs, resume=True)
        path = tmp_dirs["cache"] / "testkey" / "skeleton.json"
        assert not path.exists()

    def test_tmp_file_promoted_atomically(self, tmp_dirs):
        cp = _checkpoint(tmp_dirs, resume=False)
        sk = _skeleton()
        cp.save("skeleton", sk)
        assert (tmp_dirs["cache"] / "testkey" / "skeleton.json").exists()
        assert not (tmp_dirs["cache"] / "testkey" / "skeleton.tmp").exists()

    def test_saved_file_is_valid_json(self, tmp_dirs):
        cp = _checkpoint(tmp_dirs, resume=False)
        sk = _skeleton()
        cp.save("skeleton", sk)
        raw    = (tmp_dirs["cache"] / "testkey" / "skeleton.json").read_text()
        parsed = json.loads(raw)
        assert "title" in parsed

    def test_load_returns_none_when_not_resuming(self, tmp_dirs):
        cp = _checkpoint(tmp_dirs, resume=False)
        cp.save("skeleton", _skeleton())
        assert cp.load("skeleton", GlobalSkeleton) is None

    def test_load_returns_none_when_file_absent(self, tmp_dirs):
        cp = _checkpoint(tmp_dirs, resume=True)
        assert cp.load("skeleton", GlobalSkeleton) is None

    def test_no_checkpoint_on_pipeline_failure(self, tmp_dirs):
        """
        When the analyst raises, no checkpoint is written because save() is
        only called after the agent returns a valid model.
        """
        class _FailingAnalyst:
            async def run(self, extraction):
                raise RuntimeError("Simulated failure before save")

        cp = _checkpoint(tmp_dirs, resume=False)

        class _Patched:
            def __init__(self, provider):
                pass
            async def run(self, extraction):
                raise RuntimeError("Simulated failure before save")

        original = _pipeline_module.AnalystAgent
        _pipeline_module.AnalystAgent = _Patched
        try:
            with pytest.raises(RuntimeError):
                with _patch_extractor():
                    _run(_run_pipeline_async(tmp_dirs, cp, _default_agents()))
        finally:
            _pipeline_module.AnalystAgent = original

        assert not (tmp_dirs["cache"] / "testkey" / "skeleton.json").exists()
        assert not (tmp_dirs["cache"] / "testkey" / "doc_map.json").exists()


# ──────────────────────────────────────────────────────────────
# Test 3 — --force ignores existing checkpoints, overwrites cache
# ──────────────────────────────────────────────────────────────

class TestForceFlag:
    def test_force_ignores_existing_cache(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        # Pre-populate cache
        cp_write = _checkpoint(tmp_dirs, resume=False)
        cp_write.save("skeleton", sk)
        cp_write.save("doc_map", dm)

        # Run with resume=False (--force semantics: ignore cache)
        cp_force = _checkpoint(tmp_dirs, resume=False)
        _full_run(tmp_dirs, cp_force, _default_agents(), analyst)

        assert analyst.call_count == 1

    def test_force_overwrites_stale_cache(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        # Write stale cache content
        cp_old    = _checkpoint(tmp_dirs, resume=False)
        stale_sk  = GlobalSkeleton(
            title="Old Title",
            document_type="textbook",
            core_thesis="Old thesis.",
            sections=[SectionEntry(heading="Old Chapter", level=1, position=0)],
        )
        cp_old.save("skeleton", stale_sk)

        # Force run rewrites it
        cp_force = _checkpoint(tmp_dirs, resume=False)
        _full_run(tmp_dirs, cp_force, _default_agents(), analyst)

        cp_check = _checkpoint(tmp_dirs, resume=True)
        reloaded = cp_check.load("skeleton", GlobalSkeleton)
        assert reloaded is not None
        assert reloaded.title == "Test Paper"

    def test_force_leaves_valid_cache_behind(self, tmp_dirs):
        sk      = _skeleton()
        dm      = _doc_map()
        analyst = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))

        cp_force = _checkpoint(tmp_dirs, resume=False)
        _full_run(tmp_dirs, cp_force, _default_agents(), analyst)

        assert (tmp_dirs["cache"] / "testkey" / "skeleton.json").exists()
        assert (tmp_dirs["cache"] / "testkey" / "doc_map.json").exists()


# ──────────────────────────────────────────────────────────────
# Test 4 — Multi-deck: only failed chapter re-runs on resume
# ──────────────────────────────────────────────────────────────

class TestMultiDeckResume:
    def test_completed_chapter_not_rerun(self, tmp_dirs):
        sk = GlobalSkeleton(
            title="Textbook",
            document_type="textbook",
            core_thesis="A thesis.",
            sections=[
                SectionEntry(heading="Chapter 1", level=1, position=0),
                SectionEntry(heading="Chapter 2", level=1, position=2),
                SectionEntry(heading="Chapter 3", level=1, position=4),
                SectionEntry(heading="Chapter 4", level=1, position=6),
            ],
        )
        dm = DocumentMap(
            title="Textbook",
            document_type="textbook",
            technical_level="beginner",
            core_thesis="A thesis.",
            key_concepts=["concept"],
            sections=[
                Section(heading=h, importance="high", summary="Summary.")
                for h in ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"]
            ],
        )
        final = _slides_final()

        # Pre-checkpoint skeleton, doc_map, and two of the four chapters
        cp = _checkpoint(tmp_dirs, resume=False)
        cp.save("skeleton", sk)
        cp.save("doc_map", dm)
        cp.scoped("chapter_1").save("slides_final", final)
        cp.scoped("chapter_3").save("slides_final", final)

        planner_call_log: list[str] = []

        class TrackingPlanner:
            provider = _DummyProvider()

            async def run(self, doc_map, skeleton, scope=None):
                if scope:
                    planner_call_log.append(scope.heading)
                return _slide_plan()

        agents = {
            "planner": TrackingPlanner(),
            "writer":  StubWriter(_slides_draft()),
            "critic":  StubCritic(),
            "refiner": StubRefiner(),
        }

        analyst    = StubAnalyst(AnalystResult(skeleton=sk, doc_map=dm))
        cp_resume  = _checkpoint(tmp_dirs, resume=True)

        _full_run(tmp_dirs, cp_resume, agents, analyst)

        # Chapters 1 and 3 were checkpointed — planner should NOT have run for them
        assert "Chapter 1" not in planner_call_log
        assert "Chapter 3" not in planner_call_log
        # Chapters 2 and 4 were NOT checkpointed — planner MUST have run for them
        assert "Chapter 2" in planner_call_log
        assert "Chapter 4" in planner_call_log


# ──────────────────────────────────────────────────────────────
# Test 5 — Checkpoint.compute_key is stable and discriminating
# ──────────────────────────────────────────────────────────────

class TestComputeKey:
    def test_same_inputs_produce_same_key(self, tmp_path):
        f = tmp_path / "paper.pdf"
        f.write_bytes(b"content")
        k1 = Checkpoint.compute_key(f, "model-a", 8000)
        k2 = Checkpoint.compute_key(f, "model-a", 8000)
        assert k1 == k2

    def test_different_model_produces_different_key(self, tmp_path):
        f = tmp_path / "paper.pdf"
        f.write_bytes(b"content")
        k1 = Checkpoint.compute_key(f, "model-a", 8000)
        k2 = Checkpoint.compute_key(f, "model-b", 8000)
        assert k1 != k2

    def test_different_chunk_size_produces_different_key(self, tmp_path):
        f = tmp_path / "paper.pdf"
        f.write_bytes(b"content")
        k1 = Checkpoint.compute_key(f, "model-a", 8000)
        k2 = Checkpoint.compute_key(f, "model-a", 4000)
        assert k1 != k2

    def test_key_is_16_hex_chars(self, tmp_path):
        f = tmp_path / "paper.pdf"
        f.write_bytes(b"content")
        key = Checkpoint.compute_key(f, "model-a", 8000)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)
