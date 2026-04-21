from typing import Literal

from pydantic import BaseModel, Field


class Section(BaseModel):
    heading:    str
    importance: Literal["high", "medium", "low"]
    summary:    str


class DocumentMap(BaseModel):
    title:           str
    document_type:   Literal["research_paper", "textbook", "lecture_notes", "other"]
    technical_level: Literal["beginner", "intermediate", "advanced"]
    core_thesis:     str
    key_concepts:    list[str] = Field(min_length=1)
    sections:        list[Section] = Field(min_length=1)
