from pydantic import BaseModel, Field


class ChapterMap(BaseModel):
    chapter_heading: str
    key_concepts:    list[str] = Field(min_length=1)
    summary:         str = Field(max_length=1000)
    chunk_range:     tuple[int, int]
