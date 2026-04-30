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

from extractors.pdf import ExtractionResult, PDFExtractor, TocItem


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
    """100×100 white RGB PNG — exceeds MIN_IMAGE_DIM (32 px)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    w, h = 100, 100
    raw_row = b"\x00" + b"\xff\xff\xff" * w
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
