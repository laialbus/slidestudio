from typing import Literal

from pydantic import BaseModel, Field

from schemas.constants import (
    DECK_TITLE_MAX,
    MAX_FIGURES_PER_SLIDE,
    PLANNED_SLIDE_ANNOTATION_MAX,
)

VALID_TAGS = Literal[
    "Introduction", "Key Concept", "Definition",
    "Example", "Insight", "Data Point", "Takeaway", "Summary"
]


class PlannedSlide(BaseModel):
    index:          int
    tag:            VALID_TAGS
    source_section: str
    intention:      str       = Field(max_length=PLANNED_SLIDE_ANNOTATION_MAX)
    emphasis:       str       = Field(max_length=PLANNED_SLIDE_ANNOTATION_MAX)
    chunk_indices:  list[int] = Field(min_length=1, max_length=3)
    # Figures the Planner requests from the catalog by id; the pipeline then
    # validates and prunes this list (exists in catalog, no reuse, capped).
    figure_ids:     list[int] = Field(default_factory=list, max_length=MAX_FIGURES_PER_SLIDE)


class SlidePlan(BaseModel):
    title:        str = Field(max_length=DECK_TITLE_MAX)
    total_slides: int = Field(ge=4, le=20)
    slides:       list[PlannedSlide]
