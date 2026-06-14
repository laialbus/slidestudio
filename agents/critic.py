from string import Template

from schemas.critique import Critique
from schemas.document_map import DocumentMap
from schemas.slides_draft import SlidesDraft

from agents.base import BaseAgent, format_source_chunks


class CriticAgent(BaseAgent):
    name = "critic"
    output_schema = Critique

    async def run(
        self,
        doc_map: DocumentMap,
        slides: SlidesDraft,
        chunks: list[str] | None = None,
        slide_chunks: dict[int, list[int]] | None = None,
    ) -> Critique:
        source_chunks = format_source_chunks(
            chunks or [],
            slide_chunks or {},
            slide_indices=[s.index for s in slides.slides],
        )
        prompt = Template(self.prompt_template).safe_substitute(
            doc_map=doc_map.model_dump_json(indent=2),
            slides=slides.model_dump_json(indent=2),
            source_chunks=source_chunks,
        )
        return await self._call(prompt, Critique)
