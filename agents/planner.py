from string import Template

from schemas.document_map import DocumentMap
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import SlidePlan

from agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"
    output_schema = SlidePlan

    async def run(
        self,
        doc_map: DocumentMap,
        skeleton: GlobalSkeleton,
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
        return await self._call(prompt, SlidePlan)
