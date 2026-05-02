from pydantic import BaseModel

from schemas.slides_final import FinalSlide


class ImageEntry(BaseModel):
    index:    int
    caption:  str
    data_uri: str   # always a base64 data URI — never a filesystem path
    page:     int


class DeckOutput(BaseModel):
    """
    Top-level output written to disk and loaded by the viewer.
    Wraps SlidesFinal's slide list alongside the images array so that
    SlidesFinal remains a pure slide-only schema.
    """
    title:        str
    type:         str            = "single_deck"
    generated_at: str
    provider:     str
    model:        str
    slides:       list[FinalSlide]
    images:       list[ImageEntry] = []
