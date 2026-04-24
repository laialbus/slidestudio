"""
Milestone 4 — PlannerAgent tests.

StubProvider is defined locally; no mocking library; no real API calls.
"""

import asyncio

import pytest
from pydantic import BaseModel

from agents.planner import PlannerAgent
from providers.base import BaseProvider
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan


# ──────────────────────────────────────────────────────────────
# Local StubProvider
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self, responses: dict[type, list]):
        super().__init__("stub", 5, 3, 1)
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

    async def _call(self, messages: list, system: str) -> str:
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


def _slide(index: int, section: str = "Introduction") -> PlannedSlide:
    return PlannedSlide(
        index=index,
        tag="Key Concept",
        source_section=section,
        intention="Explain the key idea.",
        emphasis="Remember this point.",
        chunk_indices=[0],
    )


def _slide_plan(slides: list[PlannedSlide] | None = None) -> SlidePlan:
    if slides is None:
        slides = [_slide(i + 1) for i in range(4)]
    return SlidePlan(title="Test Deck", total_slides=len(slides), slides=slides)


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
        result = _run(agent.run(_doc_map(), _skeleton()))
        assert isinstance(result, SlidePlan)

    def test_title_matches_stub_response(self):
        expected = _slide_plan()
        agent, _ = self._make_agent(expected)
        result = _run(agent.run(_doc_map(), _skeleton()))
        assert result.title == expected.title

    def test_no_scope_instruction_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton()))
        _, prompt = stub.received_prompts[0]
        # Empty scope_instruction → the placeholder is substituted with ""
        assert "Generate slides ONLY for" not in prompt

    def test_doc_map_title_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton()))
        _, prompt = stub.received_prompts[0]
        assert "Attention Is All You Need" in prompt

    def test_skeleton_core_thesis_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton()))
        _, prompt = stub.received_prompts[0]
        assert "Attention mechanisms alone are sufficient" in prompt

    def test_slide_count_matches_stub(self):
        plan = _slide_plan([_slide(i + 1) for i in range(6)])
        agent, _ = self._make_agent(plan)
        result = _run(agent.run(_doc_map(), _skeleton()))
        assert result.total_slides == 6

    def test_exactly_one_provider_call_made(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton()))
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
        result = _run(agent.run(_doc_map(), _skeleton(), scope=self._scope_section()))
        assert isinstance(result, SlidePlan)

    def test_scope_heading_appears_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert "Model Architecture" in prompt

    def test_scope_instruction_text_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert "Generate slides ONLY for" in prompt

    def test_scope_chapter_name_quoted_in_prompt(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), scope=self._scope_section()))
        _, prompt = stub.received_prompts[0]
        assert '"Model Architecture"' in prompt

    def test_scoped_slides_reference_only_that_chapter(self):
        slides = [_slide(i + 1, "Model Architecture") for i in range(4)]
        agent, _ = self._make_agent(slides)
        result = _run(agent.run(_doc_map(), _skeleton(), scope=self._scope_section()))
        for slide in result.slides:
            assert slide.source_section == "Model Architecture"

    def test_scope_none_same_as_no_scope(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _skeleton(), scope=None))
        _, prompt = stub.received_prompts[0]
        assert "Generate slides ONLY for" not in prompt
