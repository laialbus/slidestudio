from pydantic import BaseModel, Field


class FinalSlide(BaseModel):
    index:     int
    tag:       str
    heading:   str
    body:      str      = Field(min_length=1)
    latex:     str | None = None   # verbatim LaTeX for formula-centric slides
    image_ref: int | None = None   # index into DeckOutput.images


class SlidesFinal(BaseModel):
    title:  str
    slides: list[FinalSlide] = Field(min_length=1)
