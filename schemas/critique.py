from typing import Literal

from pydantic import BaseModel, Field

from schemas.constants import DECK_FEEDBACK_MAX


class Issue(BaseModel):
    type:   Literal[
        "inaccuracy", "clarity", "gap", "density", "heading_mismatch", "superficial"
    ]
    detail: str


class SlideReview(BaseModel):
    index:  int
    passed: bool
    issues: list[Issue] = []


class Critique(BaseModel):
    slides:        list[SlideReview]
    deck_feedback: str = Field(default="none", max_length=DECK_FEEDBACK_MAX)

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.slides)

    @property
    def failed_slides(self) -> list[SlideReview]:
        return [s for s in self.slides if not s.passed]
