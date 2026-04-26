"""
Milestone 3 — AnalystAgent tests.

All tests use a hand-written StubProvider that inherits from BaseProvider.
No mocking library is used. No real API calls are made.
"""

import asyncio

import pytest
from pydantic import BaseModel

from agents.analyst import AnalystAgent, AnalystResult
from providers.base import BaseProvider
from providers.config import ProviderConfig
from schemas.chapter_map import ChapterMap
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry


# ──────────────────────────────────────────────────────────────
# Stub provider — returns pre-built schema instances in order
# ──────────────────────────────────────────────────────────────

class StubProvider(BaseProvider):
    """
    Inherits from BaseProvider. Overrides complete_json to return
    pre-built Pydantic instances keyed by schema type.
    No API calls are made. No mocking library is used.
    """

    def __init__(self, responses: dict[type, list]):
        super().__init__(ProviderConfig(
            model="stub",
            max_concurrent=5,
            max_format_retries=3,
            max_rate_limit_retries=1,
            request_timeout=5,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            backoff_wait_min=0,
            backoff_wait_max=0,
        ))
        self._responses = {k: list(v) for k, v in responses.items()}
        self._indices:  dict[type, int] = {}
        self.call_log:  list[type] = []

    async def complete_json(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        self.call_log.append(schema)
        idx = self._indices.get(schema, 0)
        self._indices[schema] = idx + 1
        items = self._responses[schema]
        return items[idx % len(items)]

    async def _call(self, messages: list, system: str, response_schema=None) -> str:
        raise NotImplementedError("StubProvider._call should never be reached")

    @property
    def name(self) -> str:
        return "stub"


# ──────────────────────────────────────────────────────────────
# Shared fixture helpers
# ──────────────────────────────────────────────────────────────

def _skeleton(sections: list[SectionEntry] | None = None) -> GlobalSkeleton:
    if sections is None:
        sections = [SectionEntry(heading="Introduction", level=1, position=0)]
    return GlobalSkeleton(
        title="Test Paper",
        document_type="research_paper",
        core_thesis="A thesis statement.",
        sections=sections,
    )


def _doc_map(summary: str = "A section summary.") -> DocumentMap:
    return DocumentMap(
        title="Test Paper",
        document_type="research_paper",
        technical_level="advanced",
        core_thesis="A thesis statement.",
        key_concepts=["concept"],
        sections=[Section(heading="Introduction", importance="high", summary=summary)],
    )


def _chapter_map(heading: str = "Introduction", chunk_range: tuple = (0, 0)) -> ChapterMap:
    return ChapterMap(
        chapter_heading=heading,
        key_concepts=["concept"],
        summary="Chapter overview.",
        chunk_range=chunk_range,
    )


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────
# _group_by_chapter — chunk-to-chapter assignment
# ──────────────────────────────────────────────────────────────

class TestGroupByChapter:
    def setup_method(self):
        self.agent = AnalystAgent(StubProvider({}))

    def _three_chapter_sections(self) -> list[SectionEntry]:
        return [
            SectionEntry(heading="Introduction", level=1, position=0),
            SectionEntry(heading="Methods",      level=1, position=3),
            SectionEntry(heading="Results",      level=1, position=6),
        ]

    def test_early_chunks_assigned_to_first_chapter(self):
        maps = [_doc_map() for _ in range(3)]
        groups = self.agent._group_by_chapter(maps, self._three_chapter_sections())
        assert "Introduction" in groups
        assert len(groups["Introduction"]) == 3

    def test_chunk_at_boundary_goes_to_new_chapter(self):
        maps = [_doc_map() for _ in range(4)]
        groups = self.agent._group_by_chapter(maps, self._three_chapter_sections())
        # chunk index 3 has Methods at position 3 → should land in Methods
        assert "Methods" in groups
        assert len(groups["Methods"]) >= 1

    def test_all_chunks_are_assigned(self):
        maps = [_doc_map() for _ in range(9)]
        groups = self.agent._group_by_chapter(maps, self._three_chapter_sections())
        total = sum(len(v) for v in groups.values())
        assert total == 9

    def test_no_level1_sections_all_go_to_main(self):
        maps = [_doc_map() for _ in range(3)]
        sections = [SectionEntry(heading="Background", level=2, position=0)]
        groups = self.agent._group_by_chapter(maps, sections)
        assert list(groups.keys()) == ["main"]
        assert len(groups["main"]) == 3

    def test_chunks_in_correct_chapters(self):
        maps = [_doc_map() for _ in range(9)]
        groups = self.agent._group_by_chapter(maps, self._three_chapter_sections())
        # Introduction: chunks 0,1,2; Methods: 3,4,5; Results: 6,7,8
        assert len(groups["Introduction"]) == 3
        assert len(groups["Methods"])      == 3
        assert len(groups["Results"])      == 3

    def test_empty_partial_maps_returns_empty_groups(self):
        groups = self.agent._group_by_chapter([], self._three_chapter_sections())
        assert groups == {}


# ──────────────────────────────────────────────────────────────
# _trim_skeleton — level-1 headings always retained
# ──────────────────────────────────────────────────────────────

class TestTrimSkeleton:
    def setup_method(self):
        self.agent = AnalystAgent(StubProvider({}))

    def _rich_skeleton(self) -> GlobalSkeleton:
        return GlobalSkeleton(
            title="Big Textbook",
            document_type="textbook",
            core_thesis="Comprehensive coverage.",
            sections=[
                SectionEntry(heading="Chapter 1",    level=1, position=0),
                SectionEntry(heading="1.1 Intro",    level=2, position=0),
                SectionEntry(heading="1.2 Detail",   level=3, position=1),
                SectionEntry(heading="Chapter 2",    level=1, position=2),
                SectionEntry(heading="2.1 Methods",  level=2, position=2),
                SectionEntry(heading="2.2 Analysis", level=3, position=3),
                SectionEntry(heading="Chapter 3",    level=1, position=5),
                SectionEntry(heading="3.1 Results",  level=2, position=5),
            ],
        )

    def test_all_level1_headings_retained_regardless_of_chunk(self):
        sk = self._rich_skeleton()
        for chunk_idx in (0, 1, 3, 5):
            trimmed = self.agent._trim_skeleton(sk, chunk_idx)
            level1  = [s for s in trimmed.sections if s.level == 1]
            assert len(level1) == 3, f"Expected 3 level-1 headings at chunk {chunk_idx}"

    def test_level1_headings_names_unchanged(self):
        sk = self._rich_skeleton()
        trimmed = self.agent._trim_skeleton(sk, current_chunk_index=3)
        level1_headings = {s.heading for s in trimmed.sections if s.level == 1}
        assert level1_headings == {"Chapter 1", "Chapter 2", "Chapter 3"}

    def test_level2_from_other_chapters_dropped(self):
        sk = self._rich_skeleton()
        # chunk 2 is inside Chapter 2
        trimmed = self.agent._trim_skeleton(sk, current_chunk_index=2)
        lower_headings = {s.heading for s in trimmed.sections if s.level > 1}
        assert "1.1 Intro" not in lower_headings
        assert "2.1 Methods" in lower_headings

    def test_trim_does_not_mutate_original(self):
        sk = self._rich_skeleton()
        original_count = len(sk.sections)
        self.agent._trim_skeleton(sk, current_chunk_index=3)
        assert len(sk.sections) == original_count

    def test_trim_at_chunk_zero_keeps_first_chapter_detail(self):
        sk = self._rich_skeleton()
        trimmed = self.agent._trim_skeleton(sk, current_chunk_index=0)
        lower_headings = {s.heading for s in trimmed.sections if s.level > 1}
        assert "1.1 Intro" in lower_headings


# ──────────────────────────────────────────────────────────────
# Figure placeholder flow-through
# ──────────────────────────────────────────────────────────────

class TestFigurePlaceholderFlowThrough:
    FIGURE_TEXT = "[FIGURE EXCLUDED: Figure 1 shows the transformer architecture]"

    def _make_stub(self):
        sk = _skeleton(sections=[
            SectionEntry(heading="Architecture", level=1, position=0),
        ])
        doc_with_figure = DocumentMap(
            title="Test Paper",
            document_type="research_paper",
            technical_level="advanced",
            core_thesis="A thesis.",
            key_concepts=["transformer", "attention"],
            sections=[Section(
                heading="Architecture",
                importance="high",
                summary=f"The model uses self-attention. {self.FIGURE_TEXT}",
            )],
        )
        return StubProvider({
            GlobalSkeleton: [sk],
            DocumentMap:    [doc_with_figure],
        })

    def test_figure_text_in_single_chunk_result(self):
        stub  = self._make_stub()
        agent = AnalystAgent(stub)
        result = _run(agent.run({
            "headers": ["Architecture"],
            "chunks":  [f"Self-attention mechanism. {self.FIGURE_TEXT} More text."],
        }))
        assert isinstance(result, AnalystResult)
        all_summaries = " ".join(s.summary for s in result.doc_map.sections)
        assert "[FIGURE EXCLUDED:" in all_summaries

    def test_single_chunk_skips_merge_returns_directly(self):
        stub  = self._make_stub()
        agent = AnalystAgent(stub)
        expected = stub._responses[DocumentMap][0]
        result   = _run(agent.run({
            "headers": ["Architecture"],
            "chunks":  ["chunk text with figure placeholder"],
        }))
        assert result.doc_map is expected

    def test_figure_text_preserved_through_merge(self):
        """With two chunks the merge path runs; figure text must reach final DocumentMap."""
        sk = _skeleton(sections=[
            SectionEntry(heading="Intro", level=1, position=0),
        ])
        doc0 = _doc_map(f"Chunk 0 text. {self.FIGURE_TEXT}")
        doc1 = _doc_map("Chunk 1 text.")
        chap = _chapter_map()
        final = DocumentMap(
            title="Test Paper",
            document_type="research_paper",
            technical_level="advanced",
            core_thesis="A thesis.",
            key_concepts=["attention"],
            sections=[Section(
                heading="Intro",
                importance="high",
                summary=f"Final merged summary. {self.FIGURE_TEXT}",
            )],
        )
        stub  = StubProvider({
            GlobalSkeleton: [sk],
            DocumentMap:    [doc0, doc1, final],
            ChapterMap:     [chap],
        })
        agent = AnalystAgent(stub)
        result = _run(agent.run({
            "headers": ["Intro"],
            "chunks":  ["chunk 0", "chunk 1"],
        }))
        assert isinstance(result, AnalystResult)
        all_summaries = " ".join(s.summary for s in result.doc_map.sections)
        assert "[FIGURE EXCLUDED:" in all_summaries


# ──────────────────────────────────────────────────────────────
# Concurrent chunk analysis via asyncio.gather
# ──────────────────────────────────────────────────────────────

class TestConcurrentChunkAnalysis:
    def _make_multi_chunk_stub(self, n: int):
        sk       = _skeleton(sections=[SectionEntry(heading="Ch1", level=1, position=0)])
        doc_maps = [_doc_map(f"Summary chunk {i}") for i in range(n)]
        chap     = _chapter_map()
        final    = _doc_map("Final merged.")
        return StubProvider({
            GlobalSkeleton: [sk],
            DocumentMap:    doc_maps + [final],
            ChapterMap:     [chap],
        })

    def test_all_chunks_produce_partial_maps(self):
        n     = 4
        stub  = self._make_multi_chunk_stub(n)
        agent = AnalystAgent(stub)
        result = _run(agent.run({
            "headers": ["Ch1"],
            "chunks":  [f"text {i}" for i in range(n)],
        }))
        assert isinstance(result, AnalystResult)
        assert isinstance(result.doc_map, DocumentMap)
        # n DocumentMap calls for chunks + 1 for the final document merge
        assert stub._indices.get(DocumentMap, 0) == n + 1

    def test_chunk_analysis_calls_provider_per_chunk(self):
        n     = 5
        stub  = self._make_multi_chunk_stub(n)
        agent = AnalystAgent(stub)
        _run(agent.run({
            "headers": ["Ch1"],
            "chunks":  ["chunk"] * n,
        }))
        doc_calls = stub.call_log.count(DocumentMap)
        # n chunk calls + 1 document-merge call
        assert doc_calls == n + 1

    def test_single_chunk_no_merge_called(self):
        sk   = _skeleton()
        doc  = _doc_map()
        stub = StubProvider({
            GlobalSkeleton: [sk],
            DocumentMap:    [doc],
        })
        agent = AnalystAgent(stub)
        result = _run(agent.run({
            "headers": ["Introduction"],
            "chunks":  ["single chunk"],
        }))
        assert result.doc_map is doc
        assert ChapterMap not in stub._indices

    def test_skeleton_call_happens_before_chunks(self):
        n    = 3
        stub = self._make_multi_chunk_stub(n)
        agent = AnalystAgent(stub)
        _run(agent.run({
            "headers": ["Ch1"],
            "chunks":  ["text"] * n,
        }))
        # GlobalSkeleton must be the first call
        assert stub.call_log[0] is GlobalSkeleton

    def test_chapter_merge_called_once_per_chapter(self):
        sk   = _skeleton(sections=[
            SectionEntry(heading="Ch1", level=1, position=0),
            SectionEntry(heading="Ch2", level=1, position=3),
        ])
        n    = 6
        doc_maps = [_doc_map(f"Summary {i}") for i in range(n)]
        # Two chapters → two ChapterMap responses
        chap_maps = [_chapter_map("Ch1", (0, 2)), _chapter_map("Ch2", (3, 5))]
        final     = _doc_map("Final.")
        stub = StubProvider({
            GlobalSkeleton: [sk],
            DocumentMap:    doc_maps + [final],
            ChapterMap:     chap_maps,
        })
        agent = AnalystAgent(stub)
        _run(agent.run({
            "headers": ["Ch1", "Ch2"],
            "chunks":  ["text"] * n,
        }))
        assert stub._indices.get(ChapterMap, 0) == 2
