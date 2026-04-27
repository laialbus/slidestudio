from pydantic import BaseModel, Field


class FinalSlide(BaseModel):
    index:   int
    tag:     str
    heading: str
    body:    str = Field(min_length=1)


class SlidesFinal(BaseModel):
    title:  str
    slides: list[FinalSlide] = Field(min_length=1)
