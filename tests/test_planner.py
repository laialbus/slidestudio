"""
Milestone 4 — PlannerAgent tests.

StubProvider is defined locally; no mocking library; no real API calls.
"""

import asyncio

import pytest
from pydantic import BaseModel

from agents.planner import PlannerAgent, _format_catalog, _validate_figure_ids
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.constants import MAX_FIGURES_PER_SLIDE
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan


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
# Shared fixtures
# ──────────────────────────────────────────────────────────────

def _doc_map() -> DocumentMap:
    return DocumentMap(
        title="Attention Is All You Need",
        document_type="research_paper",
        technical_level="advanced",
        core_thesis="Attention mechanisms alone are sufficient for sequence modelling.",
        key_concepts=["self-attention", "transformer", "encoder-decoder"],
        sections=[
            Section(heading="Introduction", importance="high",
                    summary="Motivation for removing recurrence."),
            Section(heading="Model Architecture", importance="high",
                    summary="Encoder and decoder stack with multi-head attention."),
        ],
    )


def _skeleton() -> GlobalSkeleton:
    return GlobalSkeleton(
        title="Attention Is All You Need",
        document_type="research_paper",
        core_thesis="Attention mechanisms alone are sufficient.",
        sections=[
            SectionEntry(heading="Introduction",       level=1, position=0),
            SectionEntry(heading="Model Architecture", level=1, position=1),
        ],
    )


def _slide(index: int, section: str = "Introduction",
           figure_ids: list[int] | None = None) -> PlannedSlide:
    return PlannedSlide(
        index=index,
        tag="Key Concept",
        source_section=section,
        intention="Explain the key idea.",
        emphasis="Remember this point.",
        chunk_indices=[0],
        figure_ids=figure_ids or [],
    )


def _slide_plan(slides: list[PlannedSlide] | None = None) -> SlidePlan:
    if slides is None:
        slides = [_slide(i + 1) for i in range(4)]
    return SlidePlan(title="Test Deck", total_slides=len(slides), slides=slides)


def _catalog(*entries: tuple[int, int, str]) -> list[dict]:
    """Each entry is (figure_id, source_chunk, purpose)."""
    return [
        {"figure_id": fid, "caption": f"Figure {fid} caption",
         "purpose": purpose, "source_chunk": chunk}
        for fid, chunk, purpose in entries
    ]


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# Tests — no scope (full document)
# ──────────────────────────────────────────────────────────────

class TestPlannerNoScope:
    def _make_agent(self, plan: SlidePlan | None = None) -> tuple[PlannerAgent, StubProvider]:
        stub  = StubProvider({SlidePlan: [plan or _slide_plan()]})
        agent = PlannerAgent(stub)
        return agent, stub

    def test_returns_slide_plan_instance(self):
        agent, _ = self._make_agent()
        result = _run(agent.run(_doc_map(), _skeleton(), []))
        assert isinstance(result, SlidePlan)

    def test_title_matches_stub_response(self):
        expected = _slide_plan()
        agent, _ = self._make_agent(expected)
        result = _run(agent.run(_doc_map(), _skeleton(), []))
        assert result.title == expected.title

    def test_no_scope_instruction_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), []))
        _, prompt = stub.received_prompts[0]
        # Empty scope_instruction → the placeholder is substituted with ""
        assert "Generate slides ONLY for" not in prompt

    def test_doc_map_title_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), []))
        _, prompt = stub.received_prompts[0]
        assert "Attention Is All You Need" in prompt

    def test_skeleton_core_thesis_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), []))
        _, prompt = stub.received_prompts[0]
        assert "Attention mechanisms alone are sufficient" in prompt

    def test_figure_catalog_rendered_in_prompt(self):
        agent, stub = self._make_agent()
        catalog = _catalog((5, 0, "conceptual"))
        _run(agent.run(_doc_map(), _skeleton(), catalog))
        _, prompt = stub.received_prompts[0]
        assert "Figure 5 caption" in prompt
        assert "id 5" in prompt

    def test_empty_catalog_renders_placeholder(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), []))
        _, prompt = stub.received_prompts[0]
        assert "no figures available" in prompt

    def test_slide_count_matches_stub(self):
        plan = _slide_plan([_slide(i + 1) for i in range(6)])
        agent, _ = self._make_agent(plan)
        result = _run(agent.run(_doc_map(), _skeleton(), []))
        assert result.total_slides == 6

    def test_exactly_one_provider_call_made(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), []))
        assert stub._indices.get(SlidePlan, 0) == 1


# ──────────────────────────────────────────────────────────────
# Tests — scoped (chapter mode)
# ──────────────────────────────────────────────────────────────

class TestPlannerWithScope:
    def _scope_section(self) -> SectionEntry:
        return SectionEntry(heading="Model Architecture", level=1, position=1)

    def _make_agent(self, slides: list[PlannedSlide] | None = None) -> tuple[PlannerAgent, StubProvider]:
        if slides is None:
            slides = [
                _slide(i + 1, "Model Architecture") for i in range(4)
            ]
        stub  = StubProvider({SlidePlan: [_slide_plan(slides)]})
        agent = PlannerAgent(stub)
        return agent, stub

    def test_returns_slide_plan(self):
        agent, _ = self._make_agent()
        result = _run(agent.run(_doc_map(), _skeleton(), [], scope=self._scope_section()))
        assert isinstance(result, SlidePlan)

    def test_scope_heading_appears_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), [], scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert "Model Architecture" in prompt

    def test_scope_instruction_text_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), [], scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert "Generate slides ONLY for" in prompt

    def test_scope_chapter_name_quoted_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), [], scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert '"Model Architecture"' in prompt

    def test_scoped_slides_reference_only_that_chapter(self):
        slides = [_slide(i + 1, "Model Architecture") for i in range(4)]
        agent, _ = self._make_agent(slides)
        result = _run(agent.run(_doc_map(), _skeleton(), [], scope=self._scope_section()))
        for slide in result.slides:
            assert slide.source_section == "Model Architecture"

    def test_scope_none_same_as_no_scope(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), [], scope=None))
        _, prompt = stub.received_prompts[0]
        assert "Generate slides ONLY for" not in prompt


# ──────────────────────────────────────────────────────────────
# _format_catalog
# ──────────────────────────────────────────────────────────────

class TestFormatCatalog:
    def test_empty_catalog_placeholder(self):
        out = _format_catalog([])
        assert "no figures available" in out

    def test_lists_id_purpose_chunk_and_caption(self):
        out = _format_catalog(_catalog((7, 3, "evidential")))
        assert "id 7" in out
        assert "evidential" in out
        assert "source chunk 3" in out
        assert "Figure 7 caption" in out


# ──────────────────────────────────────────────────────────────
# _validate_figure_ids — deterministic post-processing
# ──────────────────────────────────────────────────────────────

class TestValidateFigureIds:
    def _plan(self, slides: list[PlannedSlide]) -> SlidePlan:
        return SlidePlan(title="Test", total_slides=max(4, len(slides)), slides=slides)

    def _slide(self, index: int, chunk_indices: list[int],
               figure_ids: list[int] | None = None) -> PlannedSlide:
        return PlannedSlide(
            index=index,
            tag="Key Concept",
            source_section="Intro",
            intention="Explain.",
            emphasis="Note.",
            chunk_indices=chunk_indices,
            figure_ids=figure_ids or [],
        )

    def test_requested_valid_figure_kept(self):
        plan = self._plan([self._slide(1, [0], figure_ids=[5])])
        catalog = _catalog((5, 0, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == [5]

    def test_unknown_figure_id_dropped(self):
        plan = self._plan([self._slide(1, [0], figure_ids=[99])])
        catalog = _catalog((5, 0, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == []

    def test_no_request_stays_empty(self):
        plan = self._plan([self._slide(i + 1, [i]) for i in range(4)])
        catalog = _catalog((5, 0, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        for slide in result.slides:
            assert slide.figure_ids == []

    def test_no_reuse_across_slides(self):
        # Both slides request figure 3; it can only land on one of them.
        slides = [
            self._slide(1, [0], figure_ids=[3]),
            self._slide(2, [0], figure_ids=[3]),
        ]
        plan = self._plan(slides)
        catalog = _catalog((3, 0, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        all_refs = [fid for s in result.slides for fid in s.figure_ids]
        assert all_refs.count(3) == 1

    def test_overlap_wins_contested_figure(self):
        # Figure 8 has source_chunk 2. Slide 2 overlaps (chunk 2); slide 1 does not.
        slides = [
            self._slide(1, [0], figure_ids=[8]),
            self._slide(2, [2], figure_ids=[8]),
        ]
        plan = self._plan(slides)
        catalog = _catalog((8, 2, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == []
        assert result.slides[1].figure_ids == [8]

    def test_tie_broken_by_slide_order(self):
        # Neither slide overlaps source_chunk 5 → earliest slide wins.
        slides = [
            self._slide(1, [0], figure_ids=[4]),
            self._slide(2, [1], figure_ids=[4]),
        ]
        plan = self._plan(slides)
        catalog = _catalog((4, 5, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == [4]
        assert result.slides[1].figure_ids == []

    def test_soft_preference_assigns_without_overlap(self):
        # Single requester, no chunk overlap — still assigned (overlap is soft).
        plan = self._plan([self._slide(1, [0], figure_ids=[6])])
        catalog = _catalog((6, 9, "conceptual"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == [6]

    def test_multiple_figures_on_one_slide(self):
        plan = self._plan([self._slide(1, [0], figure_ids=[1, 2, 3])])
        catalog = _catalog((1, 0, "conceptual"), (2, 0, "conceptual"), (3, 0, "evidential"))
        result = _validate_figure_ids(plan, catalog)
        assert sorted(result.slides[0].figure_ids) == [1, 2, 3]
        assert len(result.slides[0].figure_ids) <= MAX_FIGURES_PER_SLIDE

    def test_empty_catalog_clears_all_requests(self):
        plan = self._plan([self._slide(1, [0], figure_ids=[5])])
        result = _validate_figure_ids(plan, [])
        assert result.slides[0].figure_ids == []

    def test_original_plan_not_mutated(self):
        plan = self._plan([self._slide(1, [0], figure_ids=[5])])
        catalog = _catalog((5, 0, "conceptual"))
        _validate_figure_ids(plan, catalog)
        assert plan.slides[0].figure_ids == [5]

    def test_evidential_figure_assignable(self):
        # The old conceptual-only gate is gone — a Data Point slide can carry
        # an evidential chart.
        slide = PlannedSlide(
            index=1, tag="Data Point", source_section="Results",
            intention="Show the numbers.", emphasis="Note the gain.",
            chunk_indices=[0], figure_ids=[5],
        )
        plan = self._plan([slide])
        catalog = _catalog((5, 0, "evidential"))
        result = _validate_figure_ids(plan, catalog)
        assert result.slides[0].figure_ids == [5]
