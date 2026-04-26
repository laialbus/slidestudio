import asyncio
from dataclasses import dataclass
from string import Template

from schemas.chapter_map import ChapterMap
from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry

from agents.base import BaseAgent


@dataclass
class AnalystResult:
    skeleton: GlobalSkeleton
    doc_map: DocumentMap


class AnalystAgent(BaseAgent):
    name = "analyst"
    output_schema = DocumentMap

    async def run(self, extraction: dict) -> AnalystResult:
        headers = extraction["headers"]
        chunks  = extraction["chunks"]

        skeleton = await self._build_skeleton(headers)

        tasks = [
            self._analyse_chunk(chunks[i], skeleton, index=i)
            for i in range(len(chunks))
        ]
        partial_maps = list(await asyncio.gather(*tasks))

        if len(partial_maps) == 1:
            doc_map = partial_maps[0]
        else:
            doc_map = await self._merge(skeleton, partial_maps)

        return AnalystResult(skeleton=skeleton, doc_map=doc_map)

    # ──────────────────────────────────────────────
    # Pass 1 — build global skeleton from headers
    # ──────────────────────────────────────────────

    async def _build_skeleton(self, headers: list[str]) -> GlobalSkeleton:
        raw = self._load_named_prompt("analyst_skeleton")
        prompt = Template(raw).safe_substitute(headers="\n".join(headers))
        return await self._call(prompt, GlobalSkeleton, "")

    # ──────────────────────────────────────────────
    # Pass 2 — analyse each chunk concurrently
    # ──────────────────────────────────────────────

    async def _analyse_chunk(
        self, chunk: str, skeleton: GlobalSkeleton, index: int
    ) -> DocumentMap:
        trimmed = self._trim_skeleton(skeleton, index)
        raw = self._load_named_prompt("analyst_chunk")
        prompt = Template(raw).safe_substitute(
            global_skeleton=trimmed.as_context(),
            chunk_index=index + 1,
            chunk_text=chunk,
        )
        system = f"You are analysing chunk {index + 1} of an academic document."
        return await self._call(prompt, DocumentMap, system)

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
