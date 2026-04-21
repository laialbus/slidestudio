from pydantic import BaseModel, Field


class FinalSlide(BaseModel):
    index:   int
    title:   str
    bullets: list[str] = Field(min_length=1)
    tag:     str


class SlidesFinal(BaseModel):
    title:  str
    slides: list[FinalSlide] = Field(min_length=1)
