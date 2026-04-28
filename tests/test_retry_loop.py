"""
Milestone 5 — run_review_loop tests (pipeline.py).

Uses lightweight stub agents — no mocking library, no real API calls.
"""

import asyncio

from schemas.critique import Critique, Issue, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.slides_draft import DraftSlide, SlidesDraft

from pipeline import run_review_loop


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Test Paper",
        document_type="research_paper",
        technical_level="intermediate",
        core_thesis="A thesis.",
        key_concepts=["concept"],
        sections=[Section(heading="Intro", importance="high", summary="Summary.")],
    )


def _slides_draft(count: int = 2) -> SlidesDraft:
    return SlidesDraft(
        title="Test Deck",
        slides=[
            DraftSlide(index=i + 1, heading=f"Slide {i + 1}", body="Bullet.", tag="Key Concept")
            for i in range(count)
        ],
    )


def _passing_critique(slides: SlidesDraft) -> Critique:
    return Critique(slides=[SlideReview(index=s.index, passed=True) for s in slides.slides])


def _failing_critique(slides: SlidesDraft) -> Critique:
    return Critique(slides=[
        SlideReview(
            index=s.index,
            passed=False,
            issues=[Issue(type="clarity", detail=f"Slide {s.index} still has issues.")],
        )
        for s in slides.slides
    ])


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# Stub agents
# ──────────────────────────────────────────────────────────────

class AlwaysPassingCritic:
    def __init__(self):
        self.call_count = 0

    async def run(self, doc_map, slides):
        self.call_count += 1
        return _passing_critique(slides)


class AlwaysFailingCritic:
    def __init__(self):
        self.call_count = 0

    async def run(self, doc_map, slides):
        self.call_count += 1
        return _failing_critique(slides)


class PassAfterNCritic:
    """Fails for the first n-1 calls, then passes on call n."""
    def __init__(self, pass_on_call: int):
        self.pass_on_call = pass_on_call
        self.call_count = 0

    async def run(self, doc_map, slides):
        self.call_count += 1
        if self.call_count >= self.pass_on_call:
            return _passing_critique(slides)
        return _failing_critique(slides)


class IdentityRefiner:
    def __init__(self):
        self.call_count = 0

    async def run(self, doc_map, slides, critique, deck_feedback=None):
        self.call_count += 1
        return slides


# ──────────────────────────────────────────────────────────────
# Tests — clean exit (all slides pass)
# ──────────────────────────────────────────────────────────────

class TestLoopCleanExit:
    def test_returns_empty_unresolved_when_all_pass(self):
        draft   = _slides_draft()
        critic  = AlwaysPassingCritic()
        refiner = IdentityRefiner()

        _, unresolved = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert unresolved == []

    def test_returns_slides_draft_on_clean_exit(self):
        draft   = _slides_draft()
        critic  = AlwaysPassingCritic()
        refiner = IdentityRefiner()

        result, _ = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert isinstance(result, SlidesDraft)

    def test_no_refiner_call_when_first_critique_passes(self):
        draft   = _slides_draft()
        critic  = AlwaysPassingCritic()
        refiner = IdentityRefiner()

        _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert refiner.call_count == 0

    def test_critic_called_once_when_first_critique_passes(self):
        draft   = _slides_draft()
        critic  = AlwaysPassingCritic()
        refiner = IdentityRefiner()

        _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert critic.call_count == 1

    def test_exits_early_on_second_pass(self):
        draft   = _slides_draft()
        critic  = PassAfterNCritic(pass_on_call=2)
        refiner = IdentityRefiner()

        _, unresolved = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert unresolved == []
        assert critic.call_count == 2
        assert refiner.call_count == 1


# ──────────────────────────────────────────────────────────────
# Tests — max cycles reached (returns warning)
# ──────────────────────────────────────────────────────────────

class TestLoopMaxCycles:
    def test_non_empty_unresolved_when_max_cycles_reached(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _, unresolved = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert len(unresolved) > 0

    def test_unresolved_is_list_of_strings(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _, unresolved = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert all(isinstance(s, str) for s in unresolved)

    def test_exact_critic_call_count_equals_max_cycles(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert critic.call_count == 3

    def test_refiner_called_max_minus_one_times(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert refiner.call_count == 2

    def test_loop_never_exceeds_max_cycles(self):
        for max_cycles in [1, 2, 5]:
            critic  = AlwaysFailingCritic()
            refiner = IdentityRefiner()
            _run(run_review_loop(_slides_draft(), _doc_map(), critic, refiner, max_review_cycles=max_cycles))
            assert critic.call_count == max_cycles

    def test_best_available_slides_returned_on_max(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        result, _ = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert isinstance(result, SlidesDraft)

    def test_unresolved_contains_slide_index_reference(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _, unresolved = _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=3))

        assert any("Slide" in msg for msg in unresolved)

    def test_max_cycles_1_calls_refiner_zero_times(self):
        draft   = _slides_draft(2)
        critic  = AlwaysFailingCritic()
        refiner = IdentityRefiner()

        _run(run_review_loop(draft, _doc_map(), critic, refiner, max_review_cycles=1))

        assert refiner.call_count == 0
        assert critic.call_count == 1
