"""
Milestone 5 — RefinerAgent tests.

StubProvider is defined locally; no mocking library; no real API calls.
"""

import asyncio

from pydantic import BaseModel

from agents.refiner import RefinerAgent
from providers.base import BaseProvider
from schemas.critique import Critique, Issue, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.slides_draft import DraftSlide, SlidesDraft


# ──────────────────────────────────────────────────────────────
# Local StubProvider
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self, responses: dict[type, list]):
        super().__init__("stub", 5, 3, 1)
        self._responses = {k: list(v) for k, v in responses.items()}
        self._indices: dict[type, int] = {}
        self.received_prompts: list[tuple[type, str]] = []

    async def complete_json(
        self, prompt: str, schema: type[BaseModel], system: str = ""
    ) -> BaseModel:
        self.received_prompts.append((schema, prompt))
        idx = self._indices.get(schema, 0)
        self._indices[schema] = idx + 1
        items = self._responses[schema]
        return items[idx % len(items)]

    async def _call(self, messages: list, system: str) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "stub"


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Test Paper",
        document_type="research_paper",
        technical_level="advanced",
        core_thesis="A thesis.",
        key_concepts=["concept"],
        sections=[Section(heading="Intro", importance="high", summary="Summary.")],
    )


def _draft_slide(index: int, title: str = "") -> DraftSlide:
    return DraftSlide(
        index=index,
        title=title or f"Slide {index}",
        bullets=["Original bullet."],
        tag="Key Concept",
    )


def _all_passing_critique(count: int) -> Critique:
    return Critique(slides=[SlideReview(index=i + 1, passed=True) for i in range(count)])


def _critique_with_failures(failed_indices: list[int], total: int) -> Critique:
    reviews = []
    for i in range(1, total + 1):
        if i in failed_indices:
            reviews.append(SlideReview(
                index=i,
                passed=False,
                issues=[Issue(type="clarity", detail=f"Slide {i} needs improvement.")],
            ))
        else:
            reviews.append(SlideReview(index=i, passed=True))
    return Critique(slides=reviews)


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# Tests — zero flagged slides (no API call, exact object returned)
# ──────────────────────────────────────────────────────────────

class TestRefinerZeroFlagged:
    def test_returns_exact_same_object_when_no_flags(self):
        original = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(i + 1) for i in range(4)],
        )
        stub = StubProvider({SlidesDraft: []})
        agent = RefinerAgent(stub)
        critique = _all_passing_critique(4)

        result = _run(agent.run(_doc_map(), original, critique))

        assert result is original

    def test_no_api_call_when_zero_flagged(self):
        original = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(i + 1) for i in range(3)],
        )
        stub = StubProvider({SlidesDraft: []})
        agent = RefinerAgent(stub)
        critique = _all_passing_critique(3)

        _run(agent.run(_doc_map(), original, critique))

        assert stub._indices.get(SlidesDraft, 0) == 0

    def test_no_prompt_recorded_when_zero_flagged(self):
        original = SlidesDraft(title="Test", slides=[_draft_slide(1)])
        stub = StubProvider({SlidesDraft: []})
        agent = RefinerAgent(stub)
        critique = _all_passing_critique(1)

        _run(agent.run(_doc_map(), original, critique))

        assert stub.received_prompts == []


# ──────────────────────────────────────────────────────────────
# Tests — only flagged slides sent to LLM
# ──────────────────────────────────────────────────────────────

class TestRefinerOnlyFlaggedSent:
    FLAGGED_TITLE   = "UNIQUE_FLAGGED_SLIDE_ZETA"
    UNFLAGGED_TITLE = "UNIQUE_UNFLAGGED_SLIDE_ALPHA"

    def _setup(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[
                _draft_slide(1, title=self.UNFLAGGED_TITLE),
                _draft_slide(2, title=self.FLAGGED_TITLE),
            ],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(2, title="FIXED_SLIDE")],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[2], total=2)
        return agent, stub, slides, critique

    def test_prompt_contains_flagged_slide_title(self):
        agent, stub, slides, critique = self._setup()
        _run(agent.run(_doc_map(), slides, critique))
        _, prompt = stub.received_prompts[0]
        assert self.FLAGGED_TITLE in prompt

    def test_prompt_does_not_contain_unflagged_slide_title(self):
        agent, stub, slides, critique = self._setup()
        _run(agent.run(_doc_map(), slides, critique))
        _, prompt = stub.received_prompts[0]
        assert self.UNFLAGGED_TITLE not in prompt

    def test_prompt_contains_critique_detail(self):
        agent, stub, slides, critique = self._setup()
        _run(agent.run(_doc_map(), slides, critique))
        _, prompt = stub.received_prompts[0]
        assert "needs improvement" in prompt

    def test_exactly_one_api_call_when_one_slide_flagged(self):
        agent, stub, slides, critique = self._setup()
        _run(agent.run(_doc_map(), slides, critique))
        assert stub._indices.get(SlidesDraft, 0) == 1


# ──────────────────────────────────────────────────────────────
# Tests — merge corrected subset back into full list
# ──────────────────────────────────────────────────────────────

class TestRefinerMerge:
    def test_unflagged_slides_unchanged_by_identity(self):
        original_slide_1 = _draft_slide(1, title="ORIGINAL_SLIDE_ONE")
        original_slide_3 = _draft_slide(3, title="ORIGINAL_SLIDE_THREE")
        slides = SlidesDraft(
            title="Test Deck",
            slides=[
                original_slide_1,
                _draft_slide(2, title="FLAGGED_SLIDE"),
                original_slide_3,
            ],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(2, title="FIXED_SLIDE_TWO")],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[2], total=3)

        result = _run(agent.run(_doc_map(), slides, critique))

        slide_1 = next(s for s in result.slides if s.index == 1)
        slide_3 = next(s for s in result.slides if s.index == 3)
        assert slide_1.title == "ORIGINAL_SLIDE_ONE"
        assert slide_3.title == "ORIGINAL_SLIDE_THREE"

    def test_flagged_slide_replaced_by_corrected_version(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(2, title="ORIGINAL_SLIDE_TWO")],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(2, title="FIXED_SLIDE_TWO")],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[2], total=2)

        slides_full = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(1), _draft_slide(2, title="ORIGINAL_SLIDE_TWO")],
        )
        result = _run(agent.run(_doc_map(), slides_full, critique))

        slide_2 = next(s for s in result.slides if s.index == 2)
        assert slide_2.title == "FIXED_SLIDE_TWO"

    def test_merge_preserves_slide_count(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(i + 1) for i in range(5)],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(3, title="FIXED")],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[3], total=5)

        result = _run(agent.run(_doc_map(), slides, critique))

        assert len(result.slides) == 5

    def test_returns_slides_draft_instance(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(1), _draft_slide(2)],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(1, title="FIXED")],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[1], total=2)

        result = _run(agent.run(_doc_map(), slides, critique))

        assert isinstance(result, SlidesDraft)

    def test_merge_multiple_flagged_slides(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(i + 1) for i in range(4)],
        )
        corrected = SlidesDraft(
            title="Test Deck",
            slides=[
                _draft_slide(1, title="FIXED_ONE"),
                _draft_slide(3, title="FIXED_THREE"),
            ],
        )
        stub = StubProvider({SlidesDraft: [corrected]})
        agent = RefinerAgent(stub)
        critique = _critique_with_failures(failed_indices=[1, 3], total=4)

        result = _run(agent.run(_doc_map(), slides, critique))

        slide_1 = next(s for s in result.slides if s.index == 1)
        slide_3 = next(s for s in result.slides if s.index == 3)
        assert slide_1.title == "FIXED_ONE"
        assert slide_3.title == "FIXED_THREE"
        assert len(result.slides) == 4
