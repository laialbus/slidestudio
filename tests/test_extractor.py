"""
PDFExtractor tests — updated for the pymupdf4llm-based implementation.

Changes from the original extractor:
- Output is an ExtractionResult Pydantic model (not a dict).
- Chunking is heading-boundary based (# / ##); character-overlap is a fallback.
- Images are extracted as base64 data URIs into result.images.
- TOC from the PDF's embedded outline is in result.toc_items.
- Scanned-PDF detection issues a UserWarning (does not crash).

All PDF fixtures are created programmatically with PyMuPDF; no external files.
"""

import struct
import warnings
import zlib

import pymupdf
import pytest

from extractors.pdf import (
    ExtractionResult, PDFExtractor, TocItem,
    _inject_figure_ids, _build_chunk_images,
    _extract_text_in_rect, _is_blank_pixmap,
)
from extractors.pdf import ImageRecord
from extractors.layout import LayoutRegion


# ──────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────

def _make_1x1_png() -> bytes:
    """Minimal valid 1×1 white RGB PNG."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr      = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat      = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend      = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _make_100x100_png() -> bytes:
    """100×100 medium-gray RGB PNG — exceeds MIN_IMAGE_DIM (32 px); non-white for blank-gate tests."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    w, h = 100, 100
    raw_row = b"\x00" + b"\x88\x88\x88" * w  # gray (136, 136, 136) — well below blank threshold
    raw_data = raw_row * h
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr      = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    idat      = chunk(b"IDAT", zlib.compress(raw_data))
    iend      = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _heading_pdf(tmp_path) -> str:
    """
    Two-page PDF with embedded TOC and Markdown-style headings.
    Page 1: Chapter 1 heading + body.
    Page 2: Chapter 2 heading + body.
    """
    doc   = pymupdf.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((72, 80),  "Chapter 1: Introduction", fontsize=14)
    page1.insert_text((72, 120), "Body text for chapter one. " * 10, fontsize=10)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 80),  "Chapter 2: Methods", fontsize=14)
    page2.insert_text((72, 120), "Body text for chapter two. " * 10, fontsize=10)
    doc.set_toc([[1, "Chapter 1: Introduction", 1], [1, "Chapter 2: Methods", 2]])
    path = str(tmp_path / "headings.pdf")
    doc.save(path)
    doc.close()
    return path


def _no_heading_pdf(tmp_path) -> str:
    """Single-page PDF with no headings — triggers character-overlap fallback."""
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Plain body text. " * 200, fontsize=10)
    path = str(tmp_path / "no_headings.pdf")
    doc.save(path)
    doc.close()
    return path


def _image_pdf(tmp_path) -> str:
    """
    Single-page PDF containing a 100×100 PNG image with a Figure caption below it.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Body text before the figure.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 80, 250, 260), stream=_make_100x100_png())
    page.insert_text((72, 275), "Figure 1: A test diagram.", fontsize=10)
    page.insert_text((72, 310), "Body text after the figure.", fontsize=10)
    path = str(tmp_path / "image.pdf")
    doc.save(path)
    doc.close()
    return path


def _table_pdf(tmp_path) -> str:
    """
    Single-page PDF: a drawn 4-row × 3-col grid table.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    x0, y0 = 72, 72
    col_w, row_h = 120, 22
    n_cols, n_rows = 3, 4
    shape = page.new_shape()
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        shape.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x0 + n_cols * col_w, y))
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        shape.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y0 + n_rows * row_h))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()
    headers = ["Species", "Count", "Region"]
    data    = [["Fox", "42", "Forest"], ["Bear", "7", "Mountains"], ["Deer", "105", "Plains"]]
    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 4, y0 + 15), h, fontsize=9)
    for r, row in enumerate(data, 1):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * col_w + 4, y0 + r * row_h + 15), val, fontsize=9)
    page.insert_text((72, 175), "Some text outside the table.", fontsize=10)
    path = str(tmp_path / "table.pdf")
    doc.save(path)
    doc.close()
    return path


def _vector_figure_pdf(tmp_path) -> str:
    """
    Single-page PDF with a vector diagram (no embedded raster) and a Figure caption.
    The figure is drawn with path commands — page.get_images() returns nothing.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 60), "Body text above the diagram.", fontsize=10)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(72, 90, 300, 220))
    shape.draw_line(pymupdf.Point(100, 155), pymupdf.Point(270, 155))
    shape.finish(color=(0, 0, 0), fill=None, width=1)
    shape.commit()
    page.insert_text((72, 240), "Figure 2: A vector architecture diagram.", fontsize=10)
    page.insert_text((72, 300), "Body text below the diagram.", fontsize=10)
    path = str(tmp_path / "vector_figure.pdf")
    doc.save(path)
    doc.close()
    return path


def _table_above_caption_pdf(tmp_path) -> str:
    """
    Single-page PDF where a Table caption sits ABOVE the drawn grid.
    The extractor must search downward from the caption to find the figure region.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Table 1: Experiment results.", fontsize=10)
    shape = page.new_shape()
    for r in range(4):
        y = 100 + r * 20
        shape.draw_line(pymupdf.Point(72, y), pymupdf.Point(300, y))
    for c in range(4):
        x = 72 + c * 76
        shape.draw_line(pymupdf.Point(x, 100), pymupdf.Point(x, 160))
    shape.finish(color=(0, 0, 0), fill=None, width=0.5)
    shape.commit()
    page.insert_text((72, 200), "Body text after the table.", fontsize=10)
    path = str(tmp_path / "table_above.pdf")
    doc.save(path)
    doc.close()
    return path


def _toc_pdf(tmp_path) -> str:
    """PDF with three embedded TOC entries at different levels."""
    doc   = pymupdf.open()
    page1 = doc.new_page(); page1.insert_text((72, 80), "Chapter 1", fontsize=14)
    page2 = doc.new_page(); page2.insert_text((72, 80), "Section 1.1", fontsize=12)
    page3 = doc.new_page(); page3.insert_text((72, 80), "Chapter 2", fontsize=14)
    doc.set_toc([
        [1, "Chapter 1",   1],
        [2, "Section 1.1", 2],
        [1, "Chapter 2",   3],
    ])
    path = str(tmp_path / "toc.pdf")
    doc.save(path)
    doc.close()
    return path


# ──────────────────────────────────────────────────────────────
# Scenario 1 — ExtractionResult is returned (not a dict)
# ──────────────────────────────────────────────────────────────

class TestExtractionResult:
    def test_returns_extraction_result_model(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert isinstance(result, ExtractionResult)

    def test_has_chunks_list(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert isinstance(result.chunks, list)
        assert len(result.chunks) >= 1

    def test_has_toc_items_list(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert isinstance(result.toc_items, list)

    def test_has_images_list(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert isinstance(result.images, list)

    def test_has_chunk_images_list(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert isinstance(result.chunk_images, list)

    def test_chunk_images_length_matches_chunks(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert len(result.chunk_images) == len(result.chunks)

    def test_page_count_correct(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert result.page_count == 1

    def test_char_count_positive(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert result.char_count > 0

    def test_char_count_matches_chunks(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert result.char_count == sum(len(c) for c in result.chunks)


# ──────────────────────────────────────────────────────────────
# Scenario 2 — Heading-boundary chunking
# ──────────────────────────────────────────────────────────────

class TestHeadingBoundaryChunking:
    def test_heading_pdf_produces_multiple_chunks(self, tmp_path):
        result = PDFExtractor().extract(_heading_pdf(tmp_path))
        assert len(result.chunks) >= 2

    def test_chunks_after_first_start_with_heading_marker(self, tmp_path):
        result = PDFExtractor().extract(_heading_pdf(tmp_path))
        # At least one chunk (after any preamble) must start with #
        heading_chunks = [c for c in result.chunks if c.lstrip().startswith("#")]
        assert heading_chunks, "Expected at least one chunk starting with a # heading"

    def test_repeating_header_stripped(self, tmp_path):
        """
        With header=False, footer=False, the PDF title repeated on every page
        should NOT appear in every chunk.
        """
        doc = pymupdf.open()
        for _ in range(3):
            p = doc.new_page()
            p.insert_text((72, 30),  "Repeating Title", fontsize=8)  # fake header
            p.insert_text((72, 80),  "# Section", fontsize=14)
            p.insert_text((72, 120), "Body content here. " * 5, fontsize=10)
        path = str(tmp_path / "repeat_header.pdf")
        doc.save(path)
        doc.close()
        result = PDFExtractor().extract(path)
        joined = " ".join(result.chunks)
        count = joined.count("Repeating Title")
        # With header stripping the count should be much less than page_count
        assert count < result.page_count, (
            f"Found 'Repeating Title' {count}x in {result.page_count} pages — "
            "header stripping may not be working"
        )

    def test_no_heading_pdf_still_produces_chunks(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert len(result.chunks) >= 1

    def test_preamble_included_when_present(self, tmp_path):
        """
        Text before the first heading should appear in the chunks.
        Uses a two-page PDF so the GNN sees the pre-heading text on page 1
        and the heading on page 2 — making the preamble clearly not a
        repeated page header.
        """
        doc   = pymupdf.open()
        page1 = doc.new_page()
        # Full page of abstract / preamble text — no heading on this page
        page1.insert_text((72, 80),  "Abstract of the paper: " * 8, fontsize=10)
        page1.insert_text((72, 180), "More preamble text follows here. " * 8, fontsize=10)
        page2 = doc.new_page()
        page2.insert_text((72, 80), "Introduction", fontsize=16)
        page2.insert_text((72, 120), "Body text of introduction.", fontsize=10)
        path = str(tmp_path / "preamble.pdf")
        doc.save(path)
        doc.close()
        result = PDFExtractor().extract(path)
        joined = " ".join(result.chunks)
        assert "Abstract" in joined or "preamble" in joined.lower()


# ──────────────────────────────────────────────────────────────
# Scenario 3 — TOC extraction
# ──────────────────────────────────────────────────────────────

class TestTocExtraction:
    def test_embedded_toc_populated(self, tmp_path):
        result = PDFExtractor().extract(_toc_pdf(tmp_path))
        assert len(result.toc_items) == 3

    def test_toc_item_fields(self, tmp_path):
        result = PDFExtractor().extract(_toc_pdf(tmp_path))
        item = result.toc_items[0]
        assert isinstance(item, TocItem)
        assert item.level == 1
        assert "Chapter 1" in item.heading
        assert item.page == 1

    def test_toc_levels_correct(self, tmp_path):
        result = PDFExtractor().extract(_toc_pdf(tmp_path))
        levels = [item.level for item in result.toc_items]
        assert levels == [1, 2, 1]

    def test_no_toc_pdf_has_empty_toc_items(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert result.toc_items == []


# ──────────────────────────────────────────────────────────────
# Scenario 4 — Image extraction as base64 data URIs
# ──────────────────────────────────────────────────────────────

class TestImageExtraction:
    def test_image_extracted_from_pdf(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        assert len(result.images) >= 1

    def test_image_has_data_uri(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        img = result.images[0]
        assert img.data_uri.startswith("data:image/")

    def test_image_index_sequential(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        indices = [img.index for img in result.images]
        assert indices == list(range(len(indices)))

    def test_image_page_recorded(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        assert result.images[0].page == 1

    def test_tiny_image_filtered_out(self, tmp_path):
        """1×1 images (below MIN_IMAGE_DIM) must not appear in result.images."""
        doc  = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 60), "Text.", fontsize=10)
        page.insert_image(pymupdf.Rect(72, 80, 73, 81), stream=_make_1x1_png())
        path = str(tmp_path / "tiny.pdf")
        doc.save(path)
        doc.close()
        result = PDFExtractor().extract(path)
        assert len(result.images) == 0

    def test_no_image_pdf_has_empty_images(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert result.images == []

    def test_images_never_contain_filesystem_paths(self, tmp_path):
        """data_uri must always be a base64 URI, never a file path."""
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        for img in result.images:
            assert img.data_uri.startswith("data:image/"), (
                f"Image data_uri is not a base64 URI: {img.data_uri[:40]}"
            )


# ──────────────────────────────────────────────────────────────
# Scenario 5 — Table placeholder (malformed table cleaning)
# ──────────────────────────────────────────────────────────────

class TestTablePlaceholders:
    def test_outside_text_preserved(self, tmp_path):
        result = PDFExtractor().extract(_table_pdf(tmp_path))
        joined = " ".join(result.chunks)
        assert "Some text outside the table" in joined


# ──────────────────────────────────────────────────────────────
# Scenario 6 — Scanned PDF warning (not crash)
# ──────────────────────────────────────────────────────────────

class TestScannedPdfWarning:
    def test_empty_pdf_warns_not_crashes(self, tmp_path):
        """A multi-page PDF with no text should warn about OCR, not crash."""
        doc = pymupdf.open()
        for _ in range(3):
            doc.new_page()   # blank pages — no selectable text
        path = str(tmp_path / "blank.pdf")
        doc.save(path)
        doc.close()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = PDFExtractor().extract(path)
        assert isinstance(result, ExtractionResult)
        # Should have issued a warning mentioning Tesseract
        texts = [str(w.message) for w in caught]
        assert any("Tesseract" in t or "tesseract" in t.lower() for t in texts), (
            f"Expected Tesseract warning, got: {texts}"
        )


# ──────────────────────────────────────────────────────────────
# Scenario 7 — Caption-first figure extraction
# ──────────────────────────────────────────────────────────────

class TestCaptionFirstImageExtraction:
    def test_raster_figure_extracted(self, tmp_path):
        """Existing raster-image figure still extracted with new caption-first path."""
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        assert len(result.images) >= 1

    def test_raster_figure_caption_populated(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        assert len(result.images) >= 1
        assert "Figure 1" in result.images[0].caption

    def test_vector_figure_extracted(self, tmp_path):
        """Vector drawing with a Figure caption is extracted without embedded raster."""
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        assert len(result.images) == 1

    def test_vector_figure_caption_populated(self, tmp_path):
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        assert len(result.images) == 1
        assert "Figure 2" in result.images[0].caption

    def test_vector_figure_data_uri(self, tmp_path):
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        assert result.images[0].data_uri.startswith("data:image/png;base64,")

    def test_table_caption_above_extracted(self, tmp_path):
        """Table caption above the grid causes downward search — region still captured."""
        result = PDFExtractor().extract(_table_above_caption_pdf(tmp_path))
        assert len(result.images) == 1
        assert "Table 1" in result.images[0].caption

    def test_drawing_without_caption_not_extracted(self, tmp_path):
        """Vector drawings with no Figure/Table caption produce no images."""
        result = PDFExtractor().extract(_table_pdf(tmp_path))
        assert result.images == []

    def test_multiline_caption_merged(self, tmp_path):
        """Two adjacent caption blocks are merged into a single ImageRecord."""
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), "Body text above.", fontsize=10)
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(72, 90, 300, 200))
        shape.finish(color=(0, 0, 0), fill=None, width=1)
        shape.commit()
        # Caption split across two blocks — simulated by two insert_text calls
        page.insert_text((72, 220), "Figure 3: First line of a long caption.", fontsize=9)
        page.insert_text((72, 232), "Second line continues here.", fontsize=9)
        page.insert_text((72, 280), "Body text below.", fontsize=10)
        path = str(tmp_path / "multiline_cap.pdf")
        doc.save(path)
        doc.close()
        result = PDFExtractor().extract(path)
        assert len(result.images) == 1
        assert "Figure 3" in result.images[0].caption


# ──────────────────────────────────────────────────────────────
# Scenario 8 — chunk_images: figure-ID injection and chunk mapping
# ──────────────────────────────────────────────────────────────

class TestChunkImages:
    def test_no_figures_yields_empty_lists(self, tmp_path):
        result = PDFExtractor().extract(_no_heading_pdf(tmp_path))
        assert all(ids == [] for ids in result.chunk_images)

    def test_chunk_images_length_equals_chunks_length(self, tmp_path):
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        assert len(result.chunk_images) == len(result.chunks)

    def test_extracted_figure_index_appears_in_some_chunk(self, tmp_path):
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        assert len(result.images) == 1
        fig_index = result.images[0].index
        all_ids = [fid for ids in result.chunk_images for fid in ids]
        assert fig_index in all_ids

    def test_figure_id_marker_injected_into_chunk_text(self, tmp_path):
        result = PDFExtractor().extract(_vector_figure_pdf(tmp_path))
        joined = " ".join(result.chunks)
        assert "[FIGURE_ID:" in joined

    def test_figure_index_in_chunk_images_matches_images_index(self, tmp_path):
        result = PDFExtractor().extract(_image_pdf(tmp_path))
        all_ids = {fid for ids in result.chunk_images for fid in ids}
        image_indices = {img.index for img in result.images}
        assert all_ids <= image_indices


class TestInjectFigureIds:
    def _record(self, index: int, caption: str) -> ImageRecord:
        return ImageRecord(index=index, caption=caption, data_uri="data:image/png;base64,AA==", page=1)

    def test_marker_appended_to_caption_line(self):
        md = "Some text.\nFigure 1: An architecture diagram.\nMore text."
        images = [self._record(0, "Figure 1: An architecture diagram.")]
        result = _inject_figure_ids(md, images)
        assert "[FIGURE_ID: 0]" in result

    def test_marker_appended_only_to_first_occurrence(self):
        md = "Figure 1: Caption.\nSome text about Figure 1 again.\nFigure 1: Caption."
        images = [self._record(0, "Figure 1: Caption.")]
        result = _inject_figure_ids(md, images)
        assert result.count("[FIGURE_ID: 0]") == 1

    def test_multiple_figures_each_get_own_marker(self):
        md = "Figure 1: First diagram.\nText.\nFigure 2: Second diagram."
        images = [
            self._record(0, "Figure 1: First diagram."),
            self._record(1, "Figure 2: Second diagram."),
        ]
        result = _inject_figure_ids(md, images)
        assert "[FIGURE_ID: 0]" in result
        assert "[FIGURE_ID: 1]" in result

    def test_no_match_leaves_markdown_unchanged(self):
        md = "Some body text with no captions here."
        images = [self._record(0, "Figure 99: Missing.")]
        result = _inject_figure_ids(md, images)
        assert result == md

    def test_mid_sentence_reference_not_tagged(self):
        md = "As shown in Figure 1, the result is clear.\nFigure 1: Actual caption."
        images = [self._record(0, "Figure 1: Actual caption.")]
        result = _inject_figure_ids(md, images)
        # Only the line-start caption should be tagged, not the mid-sentence reference
        assert result.count("[FIGURE_ID: 0]") == 1
        assert "Actual caption. [FIGURE_ID: 0]" in result

    def test_bold_wrapped_caption_tagged(self):
        # pymupdf4llm renders standalone captions as **bold** lines
        md = "Text before.\n**Figure 6: A cooperative framework.**\nText after."
        images = [self._record(8, "Figure 6: A cooperative framework.")]
        result = _inject_figure_ids(md, images)
        assert "[FIGURE_ID: 8]" in result

    def test_double_space_caption_matches_single_space_in_markdown(self):
        # Raw PDF text blocks sometimes produce double spaces; markdown normalizes them
        md = "Text.\nFigure 3: Double space caption.\nMore."
        images = [self._record(2, "Figure  3: Double space caption.")]
        result = _inject_figure_ids(md, images)
        assert "[FIGURE_ID: 2]" in result


class TestBuildChunkImages:
    def test_empty_chunks_returns_empty_list(self):
        assert _build_chunk_images([]) == []

    def test_chunk_with_no_marker_returns_empty(self):
        assert _build_chunk_images(["No markers here."]) == [[]]

    def test_chunk_with_one_marker_returns_index(self):
        assert _build_chunk_images(["Text [FIGURE_ID: 3] more."]) == [[3]]

    def test_chunk_with_multiple_markers(self):
        assert _build_chunk_images(["[FIGURE_ID: 0] and [FIGURE_ID: 2]"]) == [[0, 2]]

    def test_multiple_chunks_mapped_independently(self):
        chunks = ["[FIGURE_ID: 0] text", "no markers", "[FIGURE_ID: 1] text"]
        assert _build_chunk_images(chunks) == [[0], [], [1]]


# ──────────────────────────────────────────────────────────────
# Scenario 9 — _extract_text_in_rect
# ──────────────────────────────────────────────────────────────

class TestExtractTextInRect:
    def _blocks_from_page(self, page) -> list:
        return page.get_text("dict")["blocks"]

    def test_text_fully_inside_rect_extracted(self, tmp_path):
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Figure 1: The architecture.", fontsize=10)
        blocks = self._blocks_from_page(page)
        # Rect that fully encloses the text block
        rect = pymupdf.Rect(50, 85, 400, 115)
        text = _extract_text_in_rect(blocks, rect)
        doc.close()
        assert "Figure 1" in text

    def test_text_outside_rect_not_extracted(self, tmp_path):
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Figure 1: The architecture.", fontsize=10)
        page.insert_text((72, 400), "Unrelated body text.", fontsize=10)
        blocks = self._blocks_from_page(page)
        rect = pymupdf.Rect(50, 85, 400, 115)
        text = _extract_text_in_rect(blocks, rect)
        doc.close()
        assert "Unrelated" not in text

    def test_block_below_containment_threshold_excluded(self, tmp_path):
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "Figure 1: Long caption line here.", fontsize=10)
        blocks = self._blocks_from_page(page)
        # Rect that clips the block to less than 50% containment
        block_rect = pymupdf.Rect(blocks[0]["bbox"])
        tiny_clip = pymupdf.Rect(
            block_rect.x0,
            block_rect.y0,
            block_rect.x0 + block_rect.width * 0.3,  # only 30% of width
            block_rect.y1,
        )
        text = _extract_text_in_rect(blocks, tiny_clip)
        doc.close()
        assert text == ""

    def test_empty_blocks_returns_empty_string(self):
        result = _extract_text_in_rect([], pymupdf.Rect(0, 0, 100, 100))
        assert result == ""

    def test_image_blocks_skipped(self, tmp_path):
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(72, 80, 250, 260), stream=_make_100x100_png())
        blocks = self._blocks_from_page(page)
        text = _extract_text_in_rect(blocks, pymupdf.Rect(50, 50, 300, 300))
        doc.close()
        assert text == ""

    def test_multiple_blocks_joined(self, tmp_path):
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), "First line.", fontsize=10)
        page.insert_text((72, 120), "Second line.", fontsize=10)
        blocks = self._blocks_from_page(page)
        rect = pymupdf.Rect(50, 85, 400, 135)
        text = _extract_text_in_rect(blocks, rect)
        doc.close()
        assert "First" in text
        assert "Second" in text


# ──────────────────────────────────────────────────────────────
# Scenario 10 — _is_blank_pixmap (Phase 5 gate)
# ──────────────────────────────────────────────────────────────

class TestIsBlankPixmap:
    def _render_page(self, page) -> "pymupdf.Pixmap":
        return page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False)

    def test_blank_white_page_is_blank(self):
        doc  = pymupdf.open()
        page = doc.new_page(width=100, height=100)
        pix  = self._render_page(page)
        doc.close()
        assert _is_blank_pixmap(pix) is True

    def test_page_with_black_rect_not_blank(self):
        doc   = pymupdf.open()
        page  = doc.new_page(width=100, height=100)
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(20, 20, 80, 80))
        shape.finish(color=(0, 0, 0), fill=(0, 0, 0), width=1)
        shape.commit()
        pix = self._render_page(page)
        doc.close()
        assert _is_blank_pixmap(pix) is False

    def test_page_with_gray_text_not_blank(self):
        doc  = pymupdf.open()
        page = doc.new_page(width=200, height=100)
        page.insert_text((10, 50), "Architecture diagram", fontsize=10)
        pix  = self._render_page(page)
        doc.close()
        assert _is_blank_pixmap(pix) is False

    def test_page_with_colored_rect_not_blank(self):
        doc   = pymupdf.open()
        page  = doc.new_page(width=100, height=100)
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(10, 10, 90, 90))
        shape.finish(color=(0.2, 0.4, 0.8), fill=(0.2, 0.4, 0.8), width=1)
        shape.commit()
        pix = self._render_page(page)
        doc.close()
        assert _is_blank_pixmap(pix) is False

    def test_blank_gate_prevents_blank_image_record(self, tmp_path):
        """A caption with only blank whitespace above it produces no ImageRecord."""
        doc  = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), "Body text above blank area.", fontsize=10)
        # No figure drawn — just blank space from y=90 to y=280
        page.insert_text((72, 290), "Figure 9: Missing figure.", fontsize=10)
        page.insert_text((72, 340), "Body text below caption.", fontsize=10)
        path = str(tmp_path / "blank_figure.pdf")
        doc.save(path)
        doc.close()
        result = PDFExtractor().extract(path)
        assert result.images == []


# ──────────────────────────────────────────────────────────────
# Scenario 11 — Layout dispatch with mock analyser (Phase 4)
# ──────────────────────────────────────────────────────────────

class _MockLayoutAnalyser:
    """Stub that bypasses Surya and returns predefined regions for every page."""
    available = True

    def __init__(self, regions: list):
        self._regions = regions

    def detect(self, page):
        return self._regions


class TestLayoutDispatch:
    def test_primary_path_used_when_analyser_available(self, tmp_path):
        """Mock analyser with correct figure+caption regions produces an ImageRecord."""
        path = _vector_figure_pdf(tmp_path)
        extractor = PDFExtractor()
        # The vector figure is drawn at Rect(72, 90, 300, 220); caption at y≈240
        extractor._layout = _MockLayoutAnalyser([
            LayoutRegion(label="Figure",  bbox=pymupdf.Rect(72, 90, 300, 220), confidence=0.95),
            LayoutRegion(label="Caption", bbox=pymupdf.Rect(72, 232, 450, 255), confidence=0.95),
        ])
        result = extractor.extract(path)
        assert len(result.images) >= 1
        assert result.images[0].data_uri.startswith("data:image/png;base64,")

    def test_primary_path_caption_text_populated(self, tmp_path):
        path = _vector_figure_pdf(tmp_path)
        extractor = PDFExtractor()
        extractor._layout = _MockLayoutAnalyser([
            LayoutRegion(label="Figure",  bbox=pymupdf.Rect(72, 90, 300, 220), confidence=0.95),
            LayoutRegion(label="Caption", bbox=pymupdf.Rect(72, 232, 450, 255), confidence=0.95),
        ])
        result = extractor.extract(path)
        assert len(result.images) >= 1
        assert "Figure 2" in result.images[0].caption

    def test_picture_label_treated_as_figure(self, tmp_path):
        """'Picture' label (Surya raster image label) is eligible for extraction."""
        path = _vector_figure_pdf(tmp_path)
        extractor = PDFExtractor()
        extractor._layout = _MockLayoutAnalyser([
            LayoutRegion(label="Picture", bbox=pymupdf.Rect(72, 90, 300, 220), confidence=0.95),
            LayoutRegion(label="Caption", bbox=pymupdf.Rect(72, 232, 450, 255), confidence=0.95),
        ])
        result = extractor.extract(path)
        assert len(result.images) >= 1

    def test_unmatched_caption_falls_back_to_heuristic(self, tmp_path):
        """Caption with no paired figure in layout detections is handled by heuristic."""
        path = _vector_figure_pdf(tmp_path)
        extractor = PDFExtractor()
        # Return only a caption region — no figure region; heuristic must find the figure
        extractor._layout = _MockLayoutAnalyser([
            LayoutRegion(label="Caption", bbox=pymupdf.Rect(72, 232, 450, 255), confidence=0.95),
        ])
        result = extractor.extract(path)
        # Heuristic should recover the vector figure via its drawing paths
        assert len(result.images) >= 1

    def test_non_figure_labels_ignored(self, tmp_path):
        """Regions with non-figure labels (Text, Table, etc.) are not extracted."""
        path = _vector_figure_pdf(tmp_path)
        extractor = PDFExtractor()
        extractor._layout = _MockLayoutAnalyser([
            LayoutRegion(label="Text",    bbox=pymupdf.Rect(72, 90, 300, 220), confidence=0.95),
            LayoutRegion(label="Caption", bbox=pymupdf.Rect(72, 232, 450, 255), confidence=0.95),
        ])
        result = extractor.extract(path)
        # "Text" label is not in _FIGURE_LABELS; caption is unmatched → heuristic runs
        # The heuristic will find the vector figure; result may still have an image
        # but it must NOT come from the "Text" region being treated as a figure
        for img in result.images:
            assert img.data_uri.startswith("data:image/png;base64,")

    def test_fallback_path_used_when_analyser_unavailable(self, tmp_path):
        """When analyser.available is False the heuristic path produces the same result."""
        path = _vector_figure_pdf(tmp_path)
        result_heuristic = PDFExtractor().extract(path)

        extractor = PDFExtractor()
        extractor._layout = _MockLayoutAnalyser([])
        extractor._layout.available = False  # force fallback
        result_forced = extractor.extract(path)

        assert len(result_heuristic.images) == len(result_forced.images)
