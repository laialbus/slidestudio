from string import Template

from schemas.critique import Critique
from schemas.document_map import DocumentMap
from schemas.slides_draft import SlidesDraft

from agents.base import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic"
    output_schema = Critique

    async def run(self, doc_map: DocumentMap, slides: SlidesDraft) -> Critique:
        prompt = Template(self.prompt_template).safe_substitute(
            doc_map=doc_map.model_dump_json(indent=2),
            slides=slides.model_dump_json(indent=2),
        )
        return await self._call(prompt, Critique)
