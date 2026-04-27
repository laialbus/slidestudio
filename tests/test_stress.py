"""
Milestone 6 — Stress tests.

No mocking library. StubProvider pattern for kill-switch test.
"""

import asyncio

import pytest
from pydantic import BaseModel

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.refiner import RefinerAgent
from agents.writer import WriterAgent
from extractors.pdf import PDFExtractor
from pipeline import run_single_deck
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.critique import Critique, Issue, SlideReview
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import DraftSlide, SlidesDraft


# ──────────────────────────────────────────────────────────────
# Local StubProvider — returns failing Critique unconditionally
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    def __init__(self):
        super().__init__(ProviderConfig(model="stub", max_concurrent=5, max_format_retries=3, max_rate_limit_retries=1, request_timeout=5, circuit_breaker_threshold=3, circuit_breaker_cooldown=60, backoff_wait_min=0, backoff_wait_max=0))
        self.call_count = 0

    async def complete_json(
        self, prompt: str, schema: type[BaseModel], system: str = ""
    ) -> BaseModel:
        self.call_count += 1

        if schema is Critique:
            return Critique(slides=[
                SlideReview(
                    index=1,
                    passed=False,
                    issues=[Issue(type="clarity", detail="This slide still has issues.")],
                )
            ])

        if schema is SlidesDraft:
            return SlidesDraft(
                title="Test",
                slides=[DraftSlide(
                    index=1, heading="Slide 1", body="Bullet.", tag="Key Concept"
                )],
            )

        if schema is SlidePlan:
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

        raise ValueError(f"StubProvider: unexpected schema {schema}")

    async def _call(self, messages: list, system: str, response_schema=None) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "stub"


# ──────────────────────────────────────────────────────────────
# Helpers
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


def _skeleton() -> GlobalSkeleton:
    return GlobalSkeleton(
        title="Test Paper",
        document_type="research_paper",
        core_thesis="A thesis.",
        sections=[SectionEntry(heading="Intro", level=1, position=0)],
    )


# ──────────────────────────────────────────────────────────────
# Scale test — chunking 500 000 chars
# ──────────────────────────────────────────────────────────────

class TestChunkScale:
    def test_chunks_500k_string_without_hanging(self):
        extractor = PDFExtractor(chunk_size=8_000, overlap_size=1_500)
        big_text  = "word " * 100_000   # 500 000 chars
        chunks    = extractor._chunk(big_text)
        assert len(chunks) > 0

    def test_chunks_500k_string_produces_reasonable_count(self):
        extractor = PDFExtractor(chunk_size=8_000, overlap_size=1_500)
        big_text  = "word " * 100_000
        chunks    = extractor._chunk(big_text)
        # With 8000-char chunks and 1500 overlap, ~500000/6500 ≈ 77 chunks
        assert 50 <= len(chunks) <= 200

    def test_each_chunk_respects_size_limit(self):
        extractor = PDFExtractor(chunk_size=8_000, overlap_size=1_500)
        big_text  = "word " * 100_000
        chunks    = extractor._chunk(big_text)
        for chunk in chunks:
            assert len(chunk) <= 8_000


# ──────────────────────────────────────────────────────────────
# Kill-switch test — pipeline exits at max_review_cycles
# ──────────────────────────────────────────────────────────────

class TestKillSwitch:
    def test_exits_with_non_empty_unresolved(self, tmp_path):
        stub = StubProvider()
        agents = {
            "planner": PlannerAgent(stub),
            "writer":  WriterAgent(stub, writer_batch_size=4),
            "critic":  CriticAgent(stub),
            "refiner": RefinerAgent(stub),
        }

        _, unresolved, _ = asyncio.run(run_single_deck(
            title="Test Paper",
            doc_map=_doc_map(),
            skeleton=_skeleton(),
            chunks=["chunk text here"],
            agents=agents,
            max_review_cycles=2,
            debug=False,
            output_dir=tmp_path,
        ))

        assert isinstance(unresolved, list)
        assert len(unresolved) > 0

    def test_unresolved_contains_strings(self, tmp_path):
        stub = StubProvider()
        agents = {
            "planner": PlannerAgent(stub),
            "writer":  WriterAgent(stub, writer_batch_size=4),
            "critic":  CriticAgent(stub),
            "refiner": RefinerAgent(stub),
        }

        _, unresolved, _ = asyncio.run(run_single_deck(
            title="Test Paper",
            doc_map=_doc_map(),
            skeleton=_skeleton(),
            chunks=["chunk"],
            agents=agents,
            max_review_cycles=2,
            debug=False,
            output_dir=tmp_path,
        ))

        assert all(isinstance(s, str) for s in unresolved)
