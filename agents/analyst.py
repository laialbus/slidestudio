import asyncio
from dataclasses import dataclass, field
from string import Template
from typing import Literal

from schemas.chapter_map import ChapterMap
from schemas.chunk_map import ChunkMap
from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry

from agents.base import BaseAgent


@dataclass
class AnalystResult:
    skeleton: GlobalSkeleton
    doc_map: DocumentMap
    figure_purposes: list[dict[int, Literal["conceptual", "evidential"]]] = field(default_factory=list)


class AnalystAgent(BaseAgent):
    name = "analyst"
    output_schema = DocumentMap

    async def run(self, extraction: dict) -> AnalystResult:
        toc_items = extraction.get("toc_items", [])
        headers   = extraction.get("headers", [])
        chunks    = extraction["chunks"]
        pdf_title = extraction.get("pdf_title", "")

        skeleton = await self._build_skeleton(toc_items, headers, pdf_title)

        tasks = [
            self._analyse_chunk(chunks[i], skeleton, index=i)
            for i in range(len(chunks))
        ]
        chunk_results: list[ChunkMap] = list(await asyncio.gather(*tasks))

        # Extract per-chunk figure purposes before stripping the field for merge.
        figure_purposes: list[dict[int, Literal["conceptual", "evidential"]]] = [
            {int(k): v for k, v in cm.figure_purposes.items()}
            for cm in chunk_results
        ]

        partial_maps = [
            DocumentMap(**cm.model_dump(exclude={"figure_purposes"}))
            for cm in chunk_results
        ]

        if len(partial_maps) == 1:
            doc_map = partial_maps[0]
        else:
            doc_map = await self._merge(skeleton, partial_maps)

        return AnalystResult(skeleton=skeleton, doc_map=doc_map, figure_purposes=figure_purposes)

    # ──────────────────────────────────────────────
    # Pass 1 — build global skeleton
    # When toc_items is available, use them directly (zero API cost).
    # Fall back to an LLM call only when toc_items is empty.
    # ──────────────────────────────────────────────

    async def _build_skeleton(
        self, toc_items: list[dict], headers: list[str], pdf_title: str = ""
    ) -> GlobalSkeleton:
        if toc_items:
            return _skeleton_from_toc(toc_items, pdf_title)
        raw = self._load_named_prompt("analyst_skeleton")
        prompt = Template(raw).safe_substitute(headers="\n".join(headers))
        return await self._call(prompt, GlobalSkeleton, "")

    # ──────────────────────────────────────────────
    # Pass 2 — analyse each chunk concurrently
    # ──────────────────────────────────────────────

    async def _analyse_chunk(
        self, chunk: str, skeleton: GlobalSkeleton, index: int
    ) -> ChunkMap:
        trimmed = self._trim_skeleton(skeleton, index)
        raw = self._load_named_prompt("analyst_chunk")
        prompt = Template(raw).safe_substitute(
            global_skeleton=trimmed.as_context(),
            chunk_index=index + 1,
            chunk_text=chunk,
        )
        system = f"You are analysing chunk {index + 1} of an academic document."
        return await self._call(prompt, ChunkMap, system)

    def _trim_skeleton(
        self, skeleton: GlobalSkeleton, current_chunk_index: int
    ) -> GlobalSkeleton:
        current_chapter_pos = max(
            (s.position for s in skeleton.sections
             if s.level == 1 and s.position <= current_chunk_index),
            default=0,
        )
        trimmed = [
            s for s in skeleton.sections
            if s.level == 1 or s.position >= current_chapter_pos
        ]
        return skeleton.model_copy(update={"sections": trimmed})

    # ──────────────────────────────────────────────
    # Hierarchical Map-Reduce merge
    # ──────────────────────────────────────────────

    async def _merge(
        self, skeleton: GlobalSkeleton, partial_maps: list[DocumentMap]
    ) -> DocumentMap:
        chapter_groups = self._group_by_chapter(partial_maps, skeleton.sections)
        chapter_tasks = [
            self._merge_chapter(skeleton, group, chapter)
            for chapter, group in chapter_groups.items()
        ]
        chapter_maps = list(await asyncio.gather(*chapter_tasks))
        return await self._merge_document(skeleton, chapter_maps)

    def _group_by_chapter(
        self,
        partial_maps: list[DocumentMap],
        sections: list[SectionEntry],
    ) -> dict[str, list[DocumentMap]]:
        groups: dict[str, list[DocumentMap]] = {}
        chapter_boundaries = [s for s in sections if s.level == 1]

        for i, pmap in enumerate(partial_maps):
            chapter = next(
                (s.heading for s in reversed(chapter_boundaries)
                 if s.position <= i),
                chapter_boundaries[0].heading if chapter_boundaries else "main",
            )
            groups.setdefault(chapter, []).append(pmap)

        return groups

    async def _merge_chapter(
        self,
        skeleton: GlobalSkeleton,
        group: list[DocumentMap],
        chapter: str,
    ) -> ChapterMap:
        partial_json = "\n---\n".join(m.model_dump_json(indent=2) for m in group)
        raw = self._load_named_prompt("analyst_merge")
        prompt = Template(raw).safe_substitute(
            output_type="ChapterMap",
            global_skeleton=skeleton.as_context(),
            context_label=f"Partial analyses for chapter: {chapter}",
            input_data=partial_json,
            schema_description=(
                'chapter_heading (str), key_concepts (list of str, min 1), '
                'summary (str, max 500 chars), chunk_range ([start_int, end_int])'
            ),
        )
        return await self._call(prompt, ChapterMap, "")

    async def _merge_document(
        self,
        skeleton: GlobalSkeleton,
        chapter_maps: list[ChapterMap],
    ) -> DocumentMap:
        chapter_json = "\n---\n".join(m.model_dump_json(indent=2) for m in chapter_maps)
        raw = self._load_named_prompt("analyst_merge")
        prompt = Template(raw).safe_substitute(
            output_type="DocumentMap",
            global_skeleton=skeleton.as_context(),
            context_label="Chapter analyses",
            input_data=chapter_json,
            schema_description=(
                'title (str), document_type ("research_paper"|"textbook"|"lecture_notes"|"other"), '
                'technical_level ("beginner"|"intermediate"|"advanced"), core_thesis (str), '
                'key_concepts (list of str, min 1), '
                'sections (list of {heading, importance ("high"|"medium"|"low"), summary})'
            ),
        )
        return await self._call(prompt, DocumentMap, "")


def _skeleton_from_toc(toc_items: list[dict], pdf_title: str = "") -> GlobalSkeleton:
    """
    Build a GlobalSkeleton directly from the PDF's embedded TOC.
    Each entry is a dict with keys: level (int), heading (str), page (int).
    No LLM call is made — this is the zero-cost path for structured PDFs.

    pdf_title, when non-empty, takes precedence over the first TOC heading.
    This avoids using section headings like "Abstract" as the document title.
    """
    sections = [
        SectionEntry(
            heading=item.get("heading", ""),
            level=item.get("level", 1),
            position=i,
        )
        for i, item in enumerate(toc_items)
        if item.get("heading", "").strip()
    ]
    if pdf_title:
        title = pdf_title
    else:
        title = next(
            (item["heading"] for item in toc_items if item.get("level") == 1),
            toc_items[0].get("heading", "Document") if toc_items else "Document",
        )
    return GlobalSkeleton(
        title=title[:120],
        document_type="other",
        core_thesis="",
        sections=sections,
    )
