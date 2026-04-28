"""
Milestone 1 — PDFExtractor tests.

All PDF fixtures are created programmatically with PyMuPDF; no external files.

Three scenarios:
  1. Inline figures  — [FIGURE EXCLUDED] placeholders appear with caption text.
  2. Data tables     — [TABLE EXCLUDED] placeholders include column count.
  3. Bold-only headers — headers detected by weight alone when all font sizes match.
"""

import struct
import zlib

import pymupdf
import pytest

from extractors.pdf import PDFExtractor


# ──────────────────────────────────────────────────────────────
# Fixture helpers
# ──────────────────────────────────────────────────────────────

def _make_1x1_png() -> bytes:
    """Minimal valid 1×1 white RGB PNG — used to insert a real image block."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr      = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    # filter byte (0) + one white RGB pixel
    idat      = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend      = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _figure_pdf(tmp_path) -> str:
    """
    Single-page PDF:  body text → image block → caption starting with 'Figure'.
    The caption text immediately follows the image so _read_pdf can detect it.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 80),  "Body text that precedes the figure.", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 100, 220, 200), stream=_make_1x1_png())
    page.insert_text((72, 215), "Figure 1: A simple test diagram.", fontsize=10)
    page.insert_text((72, 240), "More body text that follows the figure.", fontsize=10)

    path = str(tmp_path / "figure.pdf")
    doc.save(path)
    doc.close()
    return path


def _table_pdf(tmp_path) -> str:
    """
    Single-page PDF: a drawn 4-row × 3-col grid table with column headers,
    plus one line of text outside the table so we can confirm it is preserved.
    The grid is drawn with shape lines so find_tables() can detect it.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    x0, y0  = 72, 72
    col_w   = 120
    row_h   = 22
    n_cols  = 3
    n_rows  = 4          # 1 header row + 3 data rows

    # Draw the table grid
    shape = page.new_shape()
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        shape.draw_line(pymupdf.Point(x0, y),
                        pymupdf.Point(x0 + n_cols * col_w, y))
    for c in range(n_cols + 1):
        x = x0 + c * col_w
        shape.draw_line(pymupdf.Point(x, y0),
                        pymupdf.Point(x, y0 + n_rows * row_h))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    # Insert cell text
    headers = ["Species", "Count", "Region"]
    data    = [["Fox", "42", "Forest"],
               ["Bear", "7",  "Mountains"],
               ["Deer", "105", "Plains"]]
    for c, h in enumerate(headers):
        page.insert_text((x0 + c * col_w + 4, y0 + 15), h, fontsize=9)
    for r, row in enumerate(data, 1):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * col_w + 4, y0 + r * row_h + 15),
                              val, fontsize=9)

    # Text clearly outside the table
    page.insert_text((72, 175), "Some text outside the table.", fontsize=10)

    path = str(tmp_path / "table.pdf")
    doc.save(path)
    doc.close()
    return path


def _bold_headers_pdf(tmp_path) -> str:
    """
    Single-page PDF: every span is 10 pt.  Section headings use the bold face
    ('hebo' = Helvetica-Bold); body text uses the regular face ('helv').
    Font size alone cannot distinguish headers from body — only weight can.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72,  55), "Introduction",
                     fontsize=10, fontname="hebo")
    page.insert_text((72,  85), "Body paragraph explaining the introduction.",
                     fontsize=10, fontname="helv")
    page.insert_text((72, 115), "Another sentence of regular body text here.",
                     fontsize=10, fontname="helv")
    page.insert_text((72, 160), "Methods",
                     fontsize=10, fontname="hebo")
    page.insert_text((72, 190), "Body text describing the methods used.",
                     fontsize=10, fontname="helv")

    path = str(tmp_path / "bold_headers.pdf")
    doc.save(path)
    doc.close()
    return path


# ──────────────────────────────────────────────────────────────
# Scenario 1 — Inline figure placeholders
# ──────────────────────────────────────────────────────────────

class TestFigurePlaceholders:
    def test_figure_placeholder_present(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_figure_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        assert "[FIGURE EXCLUDED:" in text

    def test_figure_caption_captured(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_figure_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        # Caption begins with "Figure", so the extractor should include it.
        assert 'FIGURE EXCLUDED: "Figure 1' in text

    def test_body_text_preserved_around_figure(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_figure_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        assert "Body text that precedes" in text
        assert "More body text that follows" in text


# ──────────────────────────────────────────────────────────────
# Scenario 2 — Data table placeholders
# ──────────────────────────────────────────────────────────────

class TestTablePlaceholders:
    def test_table_placeholder_present(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_table_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        assert "[TABLE EXCLUDED:" in text

    def test_table_placeholder_includes_column_count(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_table_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        # Our table has 3 columns — the placeholder must state this.
        assert "\u00d7 3 cols" in text   # × 3 cols

    def test_outside_text_preserved(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_table_pdf(tmp_path))
        text   = " ".join(result["chunks"])
        assert "Some text outside the table" in text


# ──────────────────────────────────────────────────────────────
# Scenario 3 — Bold-weight-only header detection
# ──────────────────────────────────────────────────────────────

class TestBoldOnlyHeaders:
    def test_headers_list_is_non_empty(self, tmp_path):
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_bold_headers_pdf(tmp_path))
        assert result["headers"], "Expected non-empty headers list for bold-only PDF"

    def test_bold_heading_text_detected(self, tmp_path):
        result  = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_bold_headers_pdf(tmp_path))
        headers = result["headers"]
        assert any("Introduction" in h for h in headers)
        assert any("Methods" in h for h in headers)

    def test_regular_body_not_detected_as_header(self, tmp_path):
        result  = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(_bold_headers_pdf(tmp_path))
        headers = result["headers"]
        # "paragraph" only appears in regular-weight body text
        assert not any("paragraph" in h.lower() for h in headers)


# ──────────────────────────────────────────────────────────────
# Equation PDF fixture helpers
# ──────────────────────────────────────────────────────────────

def _equation_heuristic_pdf(tmp_path) -> str:
    """
    Single-page PDF: narrow (60px) and tall (120px) image block, no figure
    caption following it.  Width < 0.8*595 and height > width, so the size
    heuristic should classify it as an equation.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 70), "The cost function is computed as follows:", fontsize=10)
    # width=60, height=120 → height > width; 60 < 0.8*595=476
    page.insert_image(pymupdf.Rect(267, 90, 327, 210), stream=_make_1x1_png())
    page.insert_text((72, 225), "This yields the minimum loss.", fontsize=10)

    path = str(tmp_path / "equation_heuristic.pdf")
    doc.save(path)
    doc.close()
    return path


def _equation_context_pdf(tmp_path) -> str:
    """
    Single-page PDF: image block followed by 'where ...' variable description.
    Context detection should classify this as an equation regardless of size.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 70),  "The attention score is:", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 90, 300, 150), stream=_make_1x1_png())
    page.insert_text((72, 165), "where Q, K, and V are the query, key, and value matrices.", fontsize=10)

    path = str(tmp_path / "equation_context.pdf")
    doc.save(path)
    doc.close()
    return path


def _figure_and_equation_pdf(tmp_path) -> str:
    """
    Single-page PDF: a wide figure with 'Figure N' caption, then a narrow
    tall equation image.  Both placeholder types must appear in document order.
    """
    doc  = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Figure block (wide) followed by a figure caption
    page.insert_text((72,  40),  "Context before figure.", fontsize=10)
    page.insert_image(pymupdf.Rect(72,  55, 500, 160), stream=_make_1x1_png())
    page.insert_text((72, 175), "Figure 1: Network architecture overview.", fontsize=10)

    # Equation block (narrow + tall) with no figure caption
    page.insert_text((72, 230), "The loss is computed as:", fontsize=10)
    page.insert_image(pymupdf.Rect(267, 248, 327, 368), stream=_make_1x1_png())
    page.insert_text((72, 383), "This gives the final optimisation target.", fontsize=10)

    path = str(tmp_path / "figure_and_equation.pdf")
    doc.save(path)
    doc.close()
    return path


# ──────────────────────────────────────────────────────────────
# Scenario 4 — Equation placeholders
# ──────────────────────────────────────────────────────────────

class TestEquationPlaceholders:
    def test_equation_placeholder_via_heuristic(self, tmp_path):
        """Narrow, tall image with no figure caption → [EQUATION: ...]."""
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(
            _equation_heuristic_pdf(tmp_path)
        )
        text = " ".join(result["chunks"])
        assert "[EQUATION:" in text

    def test_equation_placeholder_via_context(self, tmp_path):
        """Image followed by 'where ...' → [EQUATION: ...]."""
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(
            _equation_context_pdf(tmp_path)
        )
        text = " ".join(result["chunks"])
        assert "[EQUATION:" in text

    def test_figure_caption_not_labelled_as_equation(self, tmp_path):
        """Image immediately followed by 'Figure N' caption → [FIGURE EXCLUDED: ...], not equation."""
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(
            _figure_pdf(tmp_path)
        )
        text = " ".join(result["chunks"])
        assert "[FIGURE EXCLUDED:" in text
        assert "[EQUATION:" not in text

    def test_both_placeholder_types_in_document_order(self, tmp_path):
        """Page with a figure then an equation → both placeholders, figure before equation."""
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(
            _figure_and_equation_pdf(tmp_path)
        )
        text = " ".join(result["chunks"])
        assert "[FIGURE EXCLUDED:" in text
        assert "[EQUATION:" in text
        assert text.index("[FIGURE EXCLUDED:") < text.index("[EQUATION:")

    def test_equation_placeholder_position_in_chunk(self, tmp_path):
        """Placeholder must sit between the text before and after the block."""
        result = PDFExtractor(chunk_size=8_000, overlap_size=1_500).extract(
            _equation_context_pdf(tmp_path)
        )
        text = " ".join(result["chunks"])
        before = "The attention score is:"
        after  = "where Q, K"
        eq_pos     = text.index("[EQUATION:")
        before_pos = text.index(before)
        after_pos  = text.index(after)
        assert before_pos < eq_pos < after_pos
