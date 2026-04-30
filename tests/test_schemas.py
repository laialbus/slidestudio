import pytest
from pydantic import ValidationError

from schemas.chapter_map import ChapterMap
from schemas.critique import Critique, Issue, SlideReview
from schemas.deck_index import DeckEntry, DeckIndex
from schemas.deck_output import DeckOutput, ImageEntry
from schemas.document_map import DocumentMap, Section
from schemas.global_skeleton import GlobalSkeleton, SectionEntry
from schemas.slide_plan import PlannedSlide, SlidePlan
from schemas.slides_draft import DraftSlide, SlidesDraft
from schemas.slides_final import FinalSlide, SlidesFinal


# ──────────────────────────────────────────────────────────────
# PlannedSlide.chunk_indices — the key Milestone 2 constraint
# ──────────────────────────────────────────────────────────────

class TestChunkIndicesBound:
    def _slide(self, chunk_indices):
        return PlannedSlide(
            index=1,
            tag="Key Concept",
            source_section="Introduction",
            intention="Explain the main concept clearly",
            emphasis="Focus on the core idea",
            chunk_indices=chunk_indices,
        )

    def test_one_index_is_valid(self):
        slide = self._slide([0])
        assert slide.chunk_indices == [0]

    def test_two_indices_valid(self):
        slide = self._slide([0, 1])
        assert len(slide.chunk_indices) == 2

    def test_three_indices_valid(self):
        slide = self._slide([0, 1, 2])
        assert len(slide.chunk_indices) == 3

    def test_four_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            self._slide([0, 1, 2, 3])

    def test_five_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            self._slide([0, 1, 2, 3, 4])

    def test_empty_chunk_indices_raises_validation_error(self):
        with pytest.raises(ValidationError):
            self._slide([])


# ──────────────────────────────────────────────────────────────
# Critique computed properties
# ──────────────────────────────────────────────────────────────

class TestCritiqueProperties:
    def _critique(self, passed_flags):
        reviews = [SlideReview(index=i, passed=p) for i, p in enumerate(passed_flags)]
        return Critique(slides=reviews)

    def test_all_passed_when_all_true(self):
        assert self._critique([True, True, True]).all_passed is True

    def test_all_passed_false_when_one_fails(self):
        assert self._critique([True, False, True]).all_passed is False

    def test_all_passed_false_when_all_fail(self):
        assert self._critique([False, False]).all_passed is False

    def test_failed_slides_empty_when_all_pass(self):
        assert self._critique([True, True]).failed_slides == []

    def test_failed_slides_returns_only_failures(self):
        c = self._critique([True, False, True, False])
        assert len(c.failed_slides) == 2
        assert all(not s.passed for s in c.failed_slides)

    def test_issue_attaches_to_slide_review(self):
        issue = Issue(type="inaccuracy", detail="Wrong formula cited")
        review = SlideReview(index=0, passed=False, issues=[issue])
        c = Critique(slides=[review])
        assert c.failed_slides[0].issues[0].type == "inaccuracy"


# ──────────────────────────────────────────────────────────────
# GlobalSkeleton field constraints
# ──────────────────────────────────────────────────────────────

class TestGlobalSkeleton:
    def _valid(self):
        return GlobalSkeleton(
            title="Test Document",
            document_type="research_paper",
            core_thesis="A short thesis statement.",
            sections=[SectionEntry(heading="Introduction", level=1, position=0)],
        )

    def test_valid_construction(self):
        sk = self._valid()
        assert sk.title == "Test Document"

    def test_title_max_length_exceeded(self):
        with pytest.raises(ValidationError):
            GlobalSkeleton(
                title="x" * 121,
                document_type="other",
                core_thesis="t",
                sections=[],
            )

    def test_core_thesis_max_length_exceeded(self):
        with pytest.raises(ValidationError):
            GlobalSkeleton(
                title="T",
                document_type="other",
                core_thesis="x" * 401,
                sections=[],
            )

    def test_as_context_returns_json_string(self):
        sk = self._valid()
        ctx = sk.as_context()
        assert isinstance(ctx, str)
        assert "Test Document" in ctx

    def test_section_entry_fields(self):
        entry = SectionEntry(heading="Methods", level=2, position=3)
        assert entry.level == 2
        assert entry.position == 3


# ──────────────────────────────────────────────────────────────
# SlidePlan field constraints
# ──────────────────────────────────────────────────────────────

class TestSlidePlan:
    def test_total_slides_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            SlidePlan(title="T", total_slides=3, slides=[])

    def test_total_slides_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            SlidePlan(title="T", total_slides=21, slides=[])

    def test_title_max_length_exceeded(self):
        with pytest.raises(ValidationError):
            SlidePlan(title="x" * 61, total_slides=5, slides=[])

    def test_valid_construction(self):
        plan = SlidePlan(title="Short title", total_slides=4, slides=[])
        assert plan.total_slides == 4


# ──────────────────────────────────────────────────────────────
# DocumentMap field constraints
# ──────────────────────────────────────────────────────────────

class TestDocumentMap:
    def _section(self):
        return Section(heading="Intro", importance="high", summary="An intro.")

    def test_valid_construction(self):
        dm = DocumentMap(
            title="Paper",
            document_type="research_paper",
            technical_level="advanced",
            core_thesis="The thesis.",
            key_concepts=["concept"],
            sections=[self._section()],
        )
        assert dm.document_type == "research_paper"

    def test_empty_sections_raises(self):
        with pytest.raises(ValidationError):
            DocumentMap(
                title="T",
                document_type="other",
                technical_level="beginner",
                core_thesis="t",
                key_concepts=["a"],
                sections=[],
            )

    def test_empty_key_concepts_raises(self):
        with pytest.raises(ValidationError):
            DocumentMap(
                title="T",
                document_type="other",
                technical_level="beginner",
                core_thesis="t",
                key_concepts=[],
                sections=[self._section()],
            )

    def test_invalid_document_type_raises(self):
        with pytest.raises(ValidationError):
            DocumentMap(
                title="T",
                document_type="unknown_type",
                technical_level="beginner",
                core_thesis="t",
                key_concepts=["a"],
                sections=[self._section()],
            )

    def test_invalid_technical_level_raises(self):
        with pytest.raises(ValidationError):
            DocumentMap(
                title="T",
                document_type="other",
                technical_level="expert",
                core_thesis="t",
                key_concepts=["a"],
                sections=[self._section()],
            )


# ──────────────────────────────────────────────────────────────
# ChapterMap
# ──────────────────────────────────────────────────────────────

class TestChapterMap:
    def test_valid_construction(self):
        cm = ChapterMap(
            chapter_heading="Chapter 1",
            key_concepts=["osmosis", "diffusion"],
            summary="Overview of transport mechanisms.",
            chunk_range=(0, 5),
        )
        assert cm.chunk_range == (0, 5)

    def test_empty_key_concepts_raises(self):
        with pytest.raises(ValidationError):
            ChapterMap(
                chapter_heading="Ch1",
                key_concepts=[],
                summary="s",
                chunk_range=(0, 1),
            )

    def test_summary_max_length(self):
        with pytest.raises(ValidationError):
            ChapterMap(
                chapter_heading="Ch1",
                key_concepts=["a"],
                summary="x" * 1001,
                chunk_range=(0, 1),
            )


# ──────────────────────────────────────────────────────────────
# SlidesDraft and SlidesFinal
# ──────────────────────────────────────────────────────────────

class TestSlidesDraft:
    def test_valid_construction(self):
        slide = DraftSlide(index=1, heading="Slide 1", body="Point A.", tag="Key Concept")
        draft = SlidesDraft(title="Deck", slides=[slide])
        assert len(draft.slides) == 1

    def test_empty_slides_raises(self):
        with pytest.raises(ValidationError):
            SlidesDraft(title="Deck", slides=[])

    def test_empty_body_raises(self):
        with pytest.raises(ValidationError):
            DraftSlide(index=1, heading="S", body="", tag="Summary")


class TestSlidesFinal:
    def test_valid_construction(self):
        slide = FinalSlide(index=1, heading="Slide 1", body="Point A.", tag="Key Concept")
        final = SlidesFinal(title="Deck", slides=[slide])
        assert final.title == "Deck"

    def test_empty_slides_raises(self):
        with pytest.raises(ValidationError):
            SlidesFinal(title="Deck", slides=[])


# ──────────────────────────────────────────────────────────────
# DeckIndex
# ──────────────────────────────────────────────────────────────

class TestDeckIndex:
    def test_valid_construction(self):
        idx = DeckIndex(
            title="Biology 101",
            generated_at="2025-04-18T14:32:00Z",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            decks=[DeckEntry(chapter_title="Introduction", file="01_introduction.json")],
        )
        assert idx.type == "multi_deck"
        assert len(idx.decks) == 1

    def test_type_defaults_to_multi_deck(self):
        idx = DeckIndex(
            title="T",
            generated_at="2025-01-01",
            provider="openai",
            model="gpt-4o",
            decks=[],
        )
        assert idx.type == "multi_deck"


# ──────────────────────────────────────────────────────────────
# DeckOutput
# ──────────────────────────────────────────────────────────────

class TestDeckOutput:
    def _slide(self) -> FinalSlide:
        return FinalSlide(index=1, heading="Slide 1", body="Body text.", tag="Key Concept")

    def test_valid_construction_no_images(self):
        out = DeckOutput(title="My Deck", slides=[self._slide()])
        assert out.title == "My Deck"
        assert out.type == "single_deck"
        assert out.images == []

    def test_type_defaults_to_single_deck(self):
        out = DeckOutput(title="T", slides=[self._slide()])
        assert out.type == "single_deck"

    def test_images_list_populated(self):
        img = ImageEntry(
            index=0, caption="Figure 1", data_uri="data:image/png;base64,abc", page=1
        )
        out = DeckOutput(title="T", slides=[self._slide()], images=[img])
        assert len(out.images) == 1
        assert out.images[0].data_uri.startswith("data:image/")

    def test_slide_latex_and_image_ref_accepted(self):
        slide = FinalSlide(
            index=1, heading="Equations", body="Body.",
            tag="Key Concept", latex=r"\alpha + \beta = \gamma", image_ref=0,
        )
        out = DeckOutput(title="T", slides=[slide])
        assert out.slides[0].latex is not None
        assert out.slides[0].image_ref == 0

    def test_slide_latex_optional_null(self):
        slide = FinalSlide(index=1, heading="H", body="B.", tag="Definition")
        assert slide.latex is None
        assert slide.image_ref is None

    def test_draft_slide_accepts_latex_and_image_ref(self):
        draft = DraftSlide(
            index=1, heading="H", body="B.", tag="Key Concept",
            latex=r"\sum_{i=0}^{n} x_i", image_ref=2,
        )
        assert draft.latex == r"\sum_{i=0}^{n} x_i"
        assert draft.image_ref == 2
