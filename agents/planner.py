from string import Template

from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan

from agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"
    output_schema = SlidePlan

    async def run(
        self,
        doc_map: DocumentMap,
        skeleton: GlobalSkeleton,
        chunk_images: list[list[int]],
        scope: SectionEntry | None = None,
    ) -> SlidePlan:
        scope_instruction = ""
        if scope is not None:
            scope_instruction = (
                f'Generate slides ONLY for the chapter: "{scope.heading}".\n'
                f"Do not write slides for other chapters.\n"
                f"Use the Global Skeleton to ensure your definitions and "
                f"terminology align with the rest of the document."
            )

        prompt = Template(self.prompt_template).safe_substitute(
            doc_map=doc_map.model_dump_json(indent=2),
            skeleton=skeleton.as_context(),
            scope_instruction=scope_instruction,
        )
        slide_plan = await self._call(prompt, SlidePlan)
        return _assign_image_refs(slide_plan, chunk_images)


def _assign_image_refs(slide_plan: SlidePlan, chunk_images: list[list[int]]) -> SlidePlan:
    """
    Deterministically assign image_ref to each slide from figures owned by
    its selected chunks. First-come-first-served; no figure is reused.
    """
    assigned: set[int] = set()
    updated: list[PlannedSlide] = []
    for slide in slide_plan.slides:
        candidates = [
            fig_id
            for chunk_idx in slide.chunk_indices
            if chunk_idx < len(chunk_images)
            for fig_id in chunk_images[chunk_idx]
            if fig_id not in assigned
        ]
        image_ref = candidates[0] if candidates else None
        if image_ref is not None:
            assigned.add(image_ref)
        updated.append(slide.model_copy(update={"image_ref": image_ref}))
    return slide_plan.model_copy(update={"slides": updated})
