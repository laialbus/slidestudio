import asyncio
from pathlib import Path
from string import Template

from schemas.document_map import DocumentMap
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import SlidesDraft

from agents.base import BaseAgent


class WriterAgent(BaseAgent):
    name = "writer"
    output_schema = SlidesDraft

    def __init__(self, provider, writer_batch_size: int):
        super().__init__(provider)
        self.writer_batch_size = writer_batch_size

    async def run(
        self,
        slide_plan: SlidePlan,
        doc_map: DocumentMap,
        chunks: list[str],
    ) -> SlidesDraft:
        batches = self._batch(slide_plan.slides)
        tasks = [
            self._write_batch(batch, doc_map, chunks)
            for batch in batches
        ]
        batch_drafts = list(await asyncio.gather(*tasks))

        all_slides = []
        for draft in batch_drafts:
            all_slides.extend(draft.slides)

        return SlidesDraft(title=slide_plan.title, slides=all_slides)

    def _batch(self, slides: list[PlannedSlide]) -> list[list[PlannedSlide]]:
        return [
            slides[i : i + self.writer_batch_size]
            for i in range(0, len(slides), self.writer_batch_size)
        ]

    async def write_summary(
        self,
        completed_slides: SlidesDraft,
        summary_index: int,
    ) -> SlidesDraft:
        summary_template = Path("prompts/writer_summary.txt").read_text()
        prompt = Template(summary_template).safe_substitute(
            completed_slides=completed_slides.model_dump_json(indent=2),
            summary_index=summary_index,
        )
        return await self._call(prompt, SlidesDraft)

    async def _write_batch(
        self,
        batch: list[PlannedSlide],
        doc_map: DocumentMap,
        chunks: list[str],
    ) -> SlidesDraft:
        needed_ids = {idx for slide in batch for idx in slide.chunk_indices}
        source_text = "\n\n---\n\n".join(
            f"[Chunk {i}]\n{chunks[i]}"
            for i in sorted(needed_ids)
            if i < len(chunks)
        )
        batch_plan = "\n".join(
            f"Slide {s.index} [{s.tag}] — {s.source_section}: {s.intention}"
            for s in batch
        )
        prompt = Template(self.prompt_template).safe_substitute(
            doc_map=doc_map.model_dump_json(indent=2),
            batch_plan=batch_plan,
            source_text=source_text,
        )
        return await self._call(prompt, SlidesDraft)
