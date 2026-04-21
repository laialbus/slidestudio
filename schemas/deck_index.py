from pydantic import BaseModel


class DeckEntry(BaseModel):
    chapter_title: str
    file:          str


class DeckIndex(BaseModel):
    title:        str
    type:         str = "multi_deck"
    generated_at: str
    provider:     str
    model:        str
    decks:        list[DeckEntry]
