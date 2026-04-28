from string import Template

from schemas.critique import Critique
from schemas.document_map import DocumentMap
from schemas.slides_draft import DraftSlide, SlidesDraft

from agents.base import BaseAgent


class RefinerAgent(BaseAgent):
    name = "refiner"
    output_schema = SlidesDraft

    async def run(
        self,
        doc_map: DocumentMap,
        slides: SlidesDraft,
        critique: Critique,
        deck_feedback: str | None = None,
    ) -> SlidesDraft:
        failed = critique.failed_slides
        if not failed:
            return slides

        failed_indices = {r.index for r in failed}
        flagged_slides = [s for s in slides.slides if s.index in failed_indices]
        critiques_text = "\n".join(
            f"Slide {r.index}: " + "; ".join(issue.detail for issue in r.issues)
            for r in failed
        )
        flagged_json = SlidesDraft(
            title=slides.title,
            slides=flagged_slides,
        ).model_dump_json(indent=2)

        prompt = Template(self.prompt_template).safe_substitute(
            doc_map=doc_map.model_dump_json(indent=2),
            all_slides=slides.model_dump_json(indent=2),
            deck_feedback=deck_feedback or "none",
            flagged_slides=flagged_json,
            critiques=critiques_text,
        )
        corrected: SlidesDraft = await self._call(prompt, SlidesDraft)

        corrected_by_index: dict[int, DraftSlide] = {
            s.index: s for s in corrected.slides
        }
        merged = [corrected_by_index.get(s.index, s) for s in slides.slides]
        return SlidesDraft(title=slides.title, slides=merged)
