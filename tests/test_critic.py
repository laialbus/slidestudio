"""
Milestone 5 — CriticAgent tests.

StubProvider is defined locally; no mocking library; no real API calls.
"""

import asyncio

from pydantic import BaseModel

from agents.critic import CriticAgent
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.critique import Critique, Issue, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.slides_draft import DraftSlide, SlidesDraft


# ──────────────────────────────────────────────────────────────
# Local StubProvider
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self, responses: dict[type, list]):
        super().__init__(ProviderConfig(model="stub", max_concurrent=5, max_format_retries=3, max_rate_limit_retries=1, request_timeout=5, circuit_breaker_threshold=3, circuit_breaker_cooldown=60, backoff_wait_min=0, backoff_wait_max=0))
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
        title="Attention Is All You Need",
        document_type="research_paper",
        technical_level="advanced",
        core_thesis="Attention mechanisms alone are sufficient.",
        key_concepts=["self-attention", "transformer"],
        sections=[Section(heading="Introduction", importance="high", summary="Motivation.")],
    )


def _draft_slide(index: int, title: str = "") -> DraftSlide:
    return DraftSlide(
        index=index,
        title=title or f"Slide {index}",
        bullets=["A bullet point."],
        tag="Key Concept",
    )


def _slides_draft(count: int) -> SlidesDraft:
    return SlidesDraft(
        title="Test Deck",
        slides=[_draft_slide(i + 1) for i in range(count)],
    )


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# Tests — all slides pass
# ──────────────────────────────────────────────────────────────

class TestCriticAllPass:
    def _make_agent(self, count: int = 2) -> tuple[CriticAgent, StubProvider]:
        critique = Critique(
            slides=[SlideReview(index=i + 1, passed=True) for i in range(count)]
        )
        stub = StubProvider({Critique: [critique]})
        return CriticAgent(stub), stub

    def test_returns_critique_instance(self):
        agent, _ = self._make_agent()
        result = _run(agent.run(_doc_map(), _slides_draft(2)))
        assert isinstance(result, Critique)

    def test_all_passed_true_when_all_pass(self):
        agent, _ = self._make_agent(3)
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        assert result.all_passed is True

    def test_failed_slides_empty_when_all_pass(self):
        agent, _ = self._make_agent(2)
        result = _run(agent.run(_doc_map(), _slides_draft(2)))
        assert result.failed_slides == []

    def test_prompt_contains_doc_map_title(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _slides_draft(2)))
        _, prompt = stub.received_prompts[0]
        assert "Attention Is All You Need" in prompt

    def test_prompt_contains_slides_content(self):
        slides = SlidesDraft(
            title="Test Deck",
            slides=[_draft_slide(1, title="UNIQUE_SLIDE_TITLE_X")],
        )
        critique = Critique(slides=[SlideReview(index=1, passed=True)])
        stub = StubProvider({Critique: [critique]})
        agent = CriticAgent(stub)
        _run(agent.run(_doc_map(), slides))
        _, prompt = stub.received_prompts[0]
        assert "UNIQUE_SLIDE_TITLE_X" in prompt

    def test_exactly_one_provider_call(self):
        agent, stub = self._make_agent()
        _run(agent.run(_doc_map(), _slides_draft(2)))
        assert stub._indices.get(Critique, 0) == 1


# ──────────────────────────────────────────────────────────────
# Tests — some slides fail
# ──────────────────────────────────────────────────────────────

class TestCriticWithFailures:
    def _make_agent_with_failure(self) -> tuple[CriticAgent, StubProvider, Critique]:
        critique = Critique(slides=[
            SlideReview(
                index=1,
                passed=False,
                issues=[Issue(type="clarity", detail="Too vague to understand.")],
            ),
            SlideReview(index=2, passed=True, issues=[]),
            SlideReview(
                index=3,
                passed=False,
                issues=[Issue(type="inaccuracy", detail="Formula is wrong.")],
            ),
        ])
        stub = StubProvider({Critique: [critique]})
        return CriticAgent(stub), stub, critique

    def test_all_passed_false_when_any_fail(self):
        agent, _, _ = self._make_agent_with_failure()
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        assert result.all_passed is False

    def test_failed_slides_count_correct(self):
        agent, _, _ = self._make_agent_with_failure()
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        assert len(result.failed_slides) == 2

    def test_failed_slides_indices_correct(self):
        agent, _, _ = self._make_agent_with_failure()
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        failed_indices = {s.index for s in result.failed_slides}
        assert failed_indices == {1, 3}

    def test_failed_slide_issues_populated(self):
        agent, _, _ = self._make_agent_with_failure()
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        slide1 = next(s for s in result.failed_slides if s.index == 1)
        assert len(slide1.issues) == 1
        assert slide1.issues[0].type == "clarity"
        assert "vague" in slide1.issues[0].detail

    def test_passing_slides_have_empty_issues(self):
        agent, _, _ = self._make_agent_with_failure()
        result = _run(agent.run(_doc_map(), _slides_draft(3)))
        slide2 = next(s for s in result.slides if s.index == 2)
        assert slide2.passed is True
        assert slide2.issues == []
