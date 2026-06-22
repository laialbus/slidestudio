from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

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

    @model_validator(mode="after")
    def _enforce_max_slides(self, info: ValidationInfo) -> "SlidePlan":
        """Cap the deck at the caller's configured max_slides, if supplied.

        The limit originates in config (PIPELINE["max_slides"]) and is passed in
        via validation context so the schema never imports config. With no
        context (e.g. direct construction in tests) only the static ge/le bounds
        apply. Exceeding the cap raises like any constraint, so the provider's
        format-retry loop asks the model to regenerate a shorter plan.
        """
        max_slides = (info.context or {}).get("max_slides")
        if max_slides is not None:
            count = len(self.slides)
            if self.total_slides > max_slides or count > max_slides:
                raise ValueError(
                    f"deck has {count} slides (total_slides="
                    f"{self.total_slides}); the configured maximum is "
                    f"{max_slides}"
                )
        return self
