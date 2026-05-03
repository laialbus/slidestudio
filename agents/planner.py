from string import Template
from typing import Literal

from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan

from agents.base import BaseAgent

FigurePurposes = list[dict[int, Literal["conceptual", "evidential"]]]


class PlannerAgent(BaseAgent):
    name = "planner"
    output_schema = SlidePlan

    async def run(
        self,
        doc_map: DocumentMap,
        skeleton: GlobalSkeleton,
        chunk_images: list[list[int]],
        figure_purposes: FigurePurposes = [],
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
        return _assign_image_refs(slide_plan, chunk_images, figure_purposes)


def _assign_image_refs(
    slide_plan: SlidePlan,
    chunk_images: list[list[int]],
    figure_purposes: FigurePurposes = [],
) -> SlidePlan:
    """
    Assign image_ref to each slide using two gates:
      1. slide.wants_image must be True (LLM signals pedagogical need for a visual)
      2. the figure must have a "conceptual" reference in one of the slide's chunks

    First-come-first-served within those constraints; no figure is reused.
    """
    assigned: set[int] = set()
    updated: list[PlannedSlide] = []
    for slide in slide_plan.slides:
        if not slide.wants_image:
            updated.append(slide.model_copy(update={"image_ref": None}))
            continue

        candidates = []
        for chunk_idx in slide.chunk_indices:
            if chunk_idx >= len(chunk_images):
                continue
            for fig_id in chunk_images[chunk_idx]:
                if fig_id in assigned:
                    continue
                is_conceptual = any(
                    chunk_idx2 < len(figure_purposes)
                    and figure_purposes[chunk_idx2].get(fig_id) == "conceptual"
                    for chunk_idx2 in slide.chunk_indices
                )
                if is_conceptual:
                    candidates.append(fig_id)

        image_ref = candidates[0] if candidates else None
        if image_ref is not None:
            assigned.add(image_ref)
        updated.append(slide.model_copy(update={"image_ref": image_ref}))
    return slide_plan.model_copy(update={"slides": updated})
