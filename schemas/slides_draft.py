from pydantic import BaseModel, Field


class DraftSlide(BaseModel):
    index:   int
    title:   str
    bullets: list[str] = Field(min_length=1)
    tag:     str


class SlidesDraft(BaseModel):
    title:  str
    slides: list[DraftSlide] = Field(min_length=1)
