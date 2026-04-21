from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    type:   Literal["inaccuracy", "clarity", "gap", "density", "heading_mismatch"]
    detail: str


class SlideReview(BaseModel):
    index:  int
    passed: bool
    issues: list[Issue] = []


class Critique(BaseModel):
    slides: list[SlideReview]

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.slides)

    @property
    def failed_slides(self) -> list[SlideReview]:
        return [s for s in self.slides if not s.passed]
