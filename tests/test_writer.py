"""
Milestone 4 — WriterAgent tests.

StubProvider is defined locally; no mocking library; no real API calls.
"""

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from agents.writer import WriterAgent
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.document_map import DocumentMap, Section
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import DraftSlide, SlidesDraft


# ──────────────────────────────────────────────────────────────
# Local StubProvider
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self, responses: dict[type, list]):
        super().__init__(ProviderConfig(model="stub", max_concurrent=5, max_format_retries=3, max_rate_limit_retries=1, request_timeout=5, circuit_breaker_threshold=3, circuit_breaker_cooldown=60, backoff_wait_min=0, backoff_wait_max=0))
        self._responses   = {k: list(v) for k, v in responses.items()}
        self._indices:    dict[type, int] = {}
        self.received_prompts: list[tuple[type, str]] = []

    async def complete_json(
        self, prompt: str, schema: type[BaseModel], system: str = ""
    ) -> BaseModel:
        self.received_prompts.append((schema, prompt))
        idx = self._indices.get(schema, 0)
        self._indices[schema] = idx + 1
        items = self._responses[schema]
        return items[idx % len(items)]

    async def _call(self, messages: list, system: str, response_schema=None) -> str:
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


def _planned_slide(
    index: int,
    chunk_indices: list[int],
    source_section: str = "Intro",
) -> PlannedSlide:
    return PlannedSlide(
        index=index,
        tag="Key Concept",
        source_section=source_section,
        intention="Explain the concept.",
        emphasis="Key point.",
        chunk_indices=chunk_indices,
    )


def _draft_slide(index: int) -> DraftSlide:
    return DraftSlide(index=index, heading=f"Slide {index}", body="A bullet.", tag="Key Concept")


def _draft(slides: list[DraftSlide]) -> SlidesDraft:
    return SlidesDraft(title="Test Deck", slides=slides)


def _slide_plan(slides: list[PlannedSlide]) -> SlidePlan:
    n = len(slides)
    assert 4 <= n <= 20, "total_slides must be 4–20 for schema validity"
    return SlidePlan(title="Test Deck", total_slides=n, slides=slides)


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# chunk_indices Pydantic constraint — enforced before Writer runs
# ──────────────────────────────────────────────────────────────

class TestChunkIndicesConstraint:
    def test_four_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _planned_slide(1, chunk_indices=[0, 1, 2, 3])

    def test_five_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _planned_slide(1, chunk_indices=[0, 1, 2, 3, 4])

    def test_three_indices_valid(self):
        slide = _planned_slide(1, chunk_indices=[0, 1, 2])
        assert len(slide.chunk_indices) == 3

    def test_one_index_valid(self):
        slide = _planned_slide(1, chunk_indices=[0])
        assert slide.chunk_indices == [0]

    def test_empty_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            _planned_slide(1, chunk_indices=[])


# ──────────────────────────────────────────────────────────────
# 16-slide deck → exactly 4 API calls with batch_size=5
# ──────────────────────────────────────────────────────────────

class TestBatchCount:
    def test_16_slides_batch5_produces_4_calls(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(16)]
        plan   = _slide_plan(slides)

        # 4 batches: [5,5,5,1] — need 4 SlidesDraft responses
        batch_drafts = [
            _draft([_draft_slide(i + 1) for i in range(j, min(j + 5, 16))])
            for j in range(0, 16, 5)
        ]
        stub  = StubProvider({SlidesDraft: batch_drafts})
        agent = WriterAgent(stub, writer_batch_size=5)

        _run(agent.run(plan, _doc_map(), ["chunk text"] * 3))

        assert stub._indices.get(SlidesDraft, 0) == 4

    def test_4_slides_batch4_produces_1_call(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(4)]
        plan   = _slide_plan(slides)
        stub   = StubProvider({SlidesDraft: [_draft([_draft_slide(i + 1) for i in range(4)])]})
        agent  = WriterAgent(stub, writer_batch_size=4)

        _run(agent.run(plan, _doc_map(), ["chunk text"]))

        assert stub._indices.get(SlidesDraft, 0) == 1

    def test_5_slides_batch2_produces_3_calls(self):
        # batches: [2, 2, 1]
        slides = [_planned_slide(i + 1, [0]) for i in range(5)]
        plan   = SlidePlan(title="Test Deck", total_slides=5, slides=slides)
        drafts = [
            _draft([_draft_slide(1), _draft_slide(2)]),
            _draft([_draft_slide(3), _draft_slide(4)]),
            _draft([_draft_slide(5)]),
        ]
        stub  = StubProvider({SlidesDraft: drafts})
        agent = WriterAgent(stub, writer_batch_size=2)

        _run(agent.run(plan, _doc_map(), ["chunk"]))

        assert stub._indices.get(SlidesDraft, 0) == 3


# ──────────────────────────────────────────────────────────────
# Only referenced chunks injected per batch
# ──────────────────────────────────────────────────────────────

class TestChunkInjection:
    CHUNK_0 = "UNIQUE_ALPHA_CONTENT_CHUNK_ZERO"
    CHUNK_1 = "UNIQUE_BETA_CONTENT_CHUNK_ONE"
    CHUNK_2 = "UNIQUE_GAMMA_CONTENT_CHUNK_TWO"

    def _setup(self):
        # 4 slides, batch_size=2
        # Batch 0: slides 1,2 → chunk 0 only
        # Batch 1: slides 3,4 → chunk 1 only
        slides = [
            _planned_slide(1, [0]),
            _planned_slide(2, [0]),
            _planned_slide(3, [1]),
            _planned_slide(4, [1]),
        ]
        plan   = _slide_plan(slides)
        drafts = [
            _draft([_draft_slide(1), _draft_slide(2)]),
            _draft([_draft_slide(3), _draft_slide(4)]),
        ]
        stub   = StubProvider({SlidesDraft: drafts})
        agent  = WriterAgent(stub, writer_batch_size=2)
        chunks = [self.CHUNK_0, self.CHUNK_1, self.CHUNK_2]
        return agent, stub, plan, chunks

    def test_batch0_prompt_contains_only_chunk0(self):
        agent, stub, plan, chunks = self._setup()
        _run(agent.run(plan, _doc_map(), chunks))
        _, prompt0 = stub.received_prompts[0]
        assert self.CHUNK_0 in prompt0
        assert self.CHUNK_1 not in prompt0

    def test_batch1_prompt_contains_only_chunk1(self):
        agent, stub, plan, chunks = self._setup()
        _run(agent.run(plan, _doc_map(), chunks))
        _, prompt1 = stub.received_prompts[1]
        assert self.CHUNK_1 in prompt1
        assert self.CHUNK_0 not in prompt1

    def test_unreferenced_chunk_never_injected(self):
        # CHUNK_2 is in the list but no slide references index 2
        agent, stub, plan, chunks = self._setup()
        _run(agent.run(plan, _doc_map(), chunks))
        all_prompts = " ".join(p for _, p in stub.received_prompts)
        assert self.CHUNK_2 not in all_prompts

    def test_multi_index_batch_injects_all_referenced_chunks(self):
        # Batch with one slide referencing chunks 0 AND 1
        slides = [
            _planned_slide(1, [0, 1]),
            _planned_slide(2, [0, 1]),
            _planned_slide(3, [0]),
            _planned_slide(4, [0]),
        ]
        plan   = _slide_plan(slides)
        drafts = [
            _draft([_draft_slide(1), _draft_slide(2)]),
            _draft([_draft_slide(3), _draft_slide(4)]),
        ]
        stub   = StubProvider({SlidesDraft: drafts})
        agent  = WriterAgent(stub, writer_batch_size=2)
        chunks = [self.CHUNK_0, self.CHUNK_1, self.CHUNK_2]

        _run(agent.run(plan, _doc_map(), chunks))

        _, prompt0 = stub.received_prompts[0]
        assert self.CHUNK_0 in prompt0
        assert self.CHUNK_1 in prompt0   # both referenced by this batch


# ──────────────────────────────────────────────────────────────
# Batch concatenation → single SlidesDraft
# ──────────────────────────────────────────────────────────────

class TestBatchConcatenation:
    def test_slides_from_all_batches_appear_in_final_draft(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(8)]
        plan   = _slide_plan(slides)
        drafts = [
            _draft([_draft_slide(1), _draft_slide(2), _draft_slide(3), _draft_slide(4)]),
            _draft([_draft_slide(5), _draft_slide(6), _draft_slide(7), _draft_slide(8)]),
        ]
        stub  = StubProvider({SlidesDraft: drafts})
        agent = WriterAgent(stub, writer_batch_size=4)

        result = _run(agent.run(plan, _doc_map(), ["chunk"]))

        assert len(result.slides) == 8

    def test_final_title_comes_from_slide_plan(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(4)]
        plan   = SlidePlan(title="My Custom Title", total_slides=4, slides=slides)
        stub   = StubProvider({SlidesDraft: [_draft([_draft_slide(i + 1) for i in range(4)])]})
        agent  = WriterAgent(stub, writer_batch_size=4)

        result = _run(agent.run(plan, _doc_map(), ["chunk"]))

        assert result.title == "My Custom Title"

    def test_batch_order_preserved_in_slides(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(4)]
        plan   = _slide_plan(slides)
        # Two batches of 2; first batch returns indices 1,2; second returns 3,4
        drafts = [
            _draft([_draft_slide(1), _draft_slide(2)]),
            _draft([_draft_slide(3), _draft_slide(4)]),
        ]
        stub  = StubProvider({SlidesDraft: drafts})
        agent = WriterAgent(stub, writer_batch_size=2)

        result = _run(agent.run(plan, _doc_map(), ["chunk"]))

        assert result.slides[0].index == 1
        assert result.slides[2].index == 3

    def test_returns_slides_draft_instance(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(4)]
        plan   = _slide_plan(slides)
        stub   = StubProvider({SlidesDraft: [_draft([_draft_slide(i + 1) for i in range(4)])]})
        agent  = WriterAgent(stub, writer_batch_size=4)

        result = _run(agent.run(plan, _doc_map(), ["chunk"]))

        assert isinstance(result, SlidesDraft)


# ──────────────────────────────────────────────────────────────
# writer_batch_size is a required init parameter
# ──────────────────────────────────────────────────────────────

class TestWriterBatchSizeRequired:
    def test_writer_batch_size_stored_on_instance(self):
        stub  = StubProvider({})
        agent = WriterAgent(stub, writer_batch_size=5)
        assert agent.writer_batch_size == 5

    def test_different_batch_sizes_produce_different_call_counts(self):
        slides = [_planned_slide(i + 1, [0]) for i in range(8)]
        plan   = _slide_plan(slides)

        # batch_size=4 → 2 calls
        drafts_2 = [_draft([_draft_slide(i + 1) for i in range(4)])] * 2
        stub2  = StubProvider({SlidesDraft: drafts_2})
        agent2 = WriterAgent(stub2, writer_batch_size=4)
        _run(agent2.run(plan, _doc_map(), ["chunk"]))
        assert stub2._indices.get(SlidesDraft, 0) == 2

        # batch_size=8 → 1 call
        drafts_1 = [_draft([_draft_slide(i + 1) for i in range(8)])]
        stub1  = StubProvider({SlidesDraft: drafts_1})
        agent1 = WriterAgent(stub1, writer_batch_size=8)
        _run(agent1.run(plan, _doc_map(), ["chunk"]))
        assert stub1._indices.get(SlidesDraft, 0) == 1


# ──────────────────────────────────────────────────────────────
# image_ref post-processing — PlannedSlide value always wins
# ──────────────────────────────────────────────────────────────

def _planned_slide_with_ref(index: int, image_ref: int | None) -> PlannedSlide:
    return PlannedSlide(
        index=index,
        tag="Key Concept",
        source_section="Intro",
        intention="Explain.",
        emphasis="Note.",
        chunk_indices=[0],
        image_ref=image_ref,
    )


class TestImageRefPostProcessing:
    def test_planned_image_ref_propagates_to_draft(self):
        slides = [
            _planned_slide_with_ref(1, 3),
            _planned_slide_with_ref(2, None),
            _planned_slide_with_ref(3, None),
            _planned_slide_with_ref(4, None),
        ]
        plan = _slide_plan(slides)
        # LLM returns null for all image_refs (as instructed)
        stub  = StubProvider({SlidesDraft: [_draft([_draft_slide(i + 1) for i in range(4)])]})
        agent = WriterAgent(stub, writer_batch_size=4)
        result = _run(agent.run(plan, _doc_map(), ["chunk"]))
        assert result.slides[0].image_ref == 3

    def test_none_planned_ref_yields_null_in_draft(self):
        slides = [_planned_slide_with_ref(i + 1, None) for i in range(4)]
        plan   = _slide_plan(slides)
        stub   = StubProvider({SlidesDraft: [_draft([_draft_slide(i + 1) for i in range(4)])]})
        agent  = WriterAgent(stub, writer_batch_size=4)
        result = _run(agent.run(plan, _doc_map(), ["chunk"]))
        for slide in result.slides:
            assert slide.image_ref is None

    def test_planned_ref_overrides_llm_hallucinated_ref(self):
        # LLM outputs image_ref=99 for slide 1 but the planned value is None
        slides = [_planned_slide_with_ref(i + 1, None) for i in range(4)]
        plan   = _slide_plan(slides)
        # Inject a hallucinated image_ref into the LLM response
        hallucinated = DraftSlide(index=1, heading="S", body="B.", tag="Key Concept", image_ref=99)
        draft_with_hallucination = SlidesDraft(
            title="Test Deck",
            slides=[hallucinated] + [_draft_slide(i + 2) for i in range(3)],
        )
        stub  = StubProvider({SlidesDraft: [draft_with_hallucination]})
        agent = WriterAgent(stub, writer_batch_size=4)
        result = _run(agent.run(plan, _doc_map(), ["chunk"]))
        assert result.slides[0].image_ref is None
