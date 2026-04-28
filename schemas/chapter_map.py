from pydantic import BaseModel, Field

from schemas.constants import SECTION_SUMMARY_MAX


class ChapterMap(BaseModel):
    chapter_heading: str
    key_concepts:    list[str] = Field(min_length=1)
    summary:         str = Field(max_length=SECTION_SUMMARY_MAX)
    chunk_range:     tuple[int, int]
