from string import Template

from schemas.constants import MAX_FIGURES_PER_SLIDE
from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan

from agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"
    output_schema = SlidePlan

    def __init__(self, provider, max_slides: int):
        super().__init__(provider)
        self._max_slides = max_slides

    async def run(
        self,
        doc_map: DocumentMap,
        skeleton: GlobalSkeleton,
        figure_catalog: list[dict] | None = None,
        scope: SectionEntry | None = None,
    ) -> SlidePlan:
        if figure_catalog is None:
            figure_catalog = []
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
            figure_catalog=_format_catalog(figure_catalog),
            max_slides=self._max_slides,
        )
        slide_plan = await self._call(
            prompt, SlidePlan, context={"max_slides": self._max_slides}
        )
        return _validate_figure_ids(slide_plan, figure_catalog)


def _format_catalog(figure_catalog: list[dict]) -> str:
    """Render the figure catalog for the planner prompt, one line per figure."""
    if not figure_catalog:
        return "(no figures available — leave figure_ids empty for every slide)"
    return "\n".join(
        f"- id {e['figure_id']} [{e['purpose']}, source chunk {e['source_chunk']}]: "
        f"{e['caption']}"
        for e in figure_catalog
    )


def _validate_figure_ids(
    slide_plan: SlidePlan,
    figure_catalog: list[dict],
) -> SlidePlan:
    """
    Resolve each slide's requested figure_ids against the catalog:
      - a requested id must exist in the catalog (unknown ids are dropped);
      - no figure is shown on more than one slide (no reuse);
      - when several slides request the same figure, the one whose chunk_indices
        overlap the figure's source chunk wins (soft ranking), ties broken by
        slide order;
      - each slide carries at most MAX_FIGURES_PER_SLIDE figures.

    Chunk overlap is a *ranking* signal only, never a gate — a figure may be
    assigned to a slide it requested even when their chunks do not overlap.
    """
    source_chunk = {e["figure_id"]: e["source_chunk"] for e in figure_catalog}
    valid_ids = set(source_chunk)

    def overlaps(fig_id: int, slide: PlannedSlide) -> bool:
        return source_chunk[fig_id] in slide.chunk_indices

    # Phase A — assign each requested-and-valid figure to a single best slide.
    owner: dict[int, int] = {}  # figure_id -> winning slide index
    for fig_id in valid_ids:
        requesters = [s for s in slide_plan.slides if fig_id in s.figure_ids]
        if not requesters:
            continue
        best = max(requesters, key=lambda s: (overlaps(fig_id, s), -s.index))
        owner[fig_id] = best.index

    # Phase B — each slide keeps the figures it won, overlap-ranked, then capped.
    updated: list[PlannedSlide] = []
    for slide in slide_plan.slides:
        won = [fid for fid, idx in owner.items() if idx == slide.index]
        won.sort(key=lambda fid: (0 if overlaps(fid, slide) else 1, fid))
        updated.append(
            slide.model_copy(update={"figure_ids": won[:MAX_FIGURES_PER_SLIDE]})
        )
    return slide_plan.model_copy(update={"slides": updated})
