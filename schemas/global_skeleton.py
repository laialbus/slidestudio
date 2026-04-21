from pydantic import BaseModel, Field


class SectionEntry(BaseModel):
    heading:  str
    level:    int
    position: int


class GlobalSkeleton(BaseModel):
    title:         str = Field(max_length=120)
    document_type: str
    core_thesis:   str = Field(max_length=400)
    sections:      list[SectionEntry]

    def as_context(self) -> str:
        return self.model_dump_json(indent=2)
