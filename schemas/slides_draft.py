from pydantic import BaseModel, Field

from schemas.constants import SLIDE_BODY_MAX


class DraftSlide(BaseModel):
    index:      int
    tag:        str
    heading:    str
    body:       str      = Field(min_length=1, max_length=SLIDE_BODY_MAX)
    latex:      str | None = None   # verbatim LaTeX for formula-centric slides
    image_refs: list[int] = Field(default_factory=list)  # indices into DeckOutput.images, stacked


class SlidesDraft(BaseModel):
    title:  str
    slides: list[DraftSlide] = Field(min_length=1)
