from pydantic import BaseModel, Field


class DraftSlide(BaseModel):
    index:   int
    tag:     str
    heading: str
    body:    str = Field(min_length=1, max_length=1200)


class SlidesDraft(BaseModel):
    title:  str
    slides: list[DraftSlide] = Field(min_length=1)
