"""
Tests for MineruExtractor.

MinerU itself (GB-scale weights) is never invoked here: _run_mineru is stubbed
to point at a hand-built fixture output dir, so these exercise the
content_list.json -> ExtractionResult mapping deterministically and offline.
A separate real-MinerU smoke run is required to confirm the CLI invocation and
block-field names against the installed version.
"""

import json

import pymupdf
import pytest

from extractors.base import ExtractionResult
from extractors.factory import make_extractor
from extractors.mineru import (
    MineruExtractor,
    _html_table_to_md,
    _join_caption,
)

_IMG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def _make_pdf(tmp_path, title="My Paper", pages=2):
    """Minimal real PDF for the pymupdf metadata path (title/toc/page_count)."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        if i == 0:
            page.insert_text((72, 72), title, fontsize=24)  # largest font => title
        page.insert_text((72, 120), f"Body of page {i + 1}", fontsize=11)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _write_out_dir(tmp_path, content, crops=None):
    """Build a MinerU-style output dir from a content_list + optional crops."""
    out = tmp_path / "mineru_out"
    (out / "images").mkdir(parents=True)
    for name, data in (crops or {}).items():
        (out / "images" / name).write_bytes(data)
    (out / "doc_content_list.json").write_text(
        json.dumps(content), encoding="utf-8"
    )
    return out


def _make_fixture_out_dir(tmp_path):
    """Build a MinerU-style output dir: content_list.json + images/ crop."""
    out = tmp_path / "mineru_out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "fig1.jpg").write_bytes(_IMG_BYTES)
    content = [
        {"type": "text", "text": "Introduction", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Body text here.", "page_idx": 0},
        {"type": "equation", "text": "$$E=mc^2$$", "page_idx": 0},
        {
            "type": "image",
            "img_path": "images/fig1.jpg",
            "image_caption": ["Figure 1: A plot."],
            "page_idx": 0,
        },
        {
            "type": "table",
            "img_path": "",
            "table_caption": ["Table 1: Results."],
            "table_body": (
                "<table><tr><td>a</td><td>b</td></tr>"
                "<tr><td>1</td><td>2</td></tr></table>"
            ),
            "page_idx": 1,
        },
    ]
    (out / "doc_content_list.json").write_text(json.dumps(content), encoding="utf-8")
    return out


def _stubbed_extractor(tmp_path, out_dir):
    ext = MineruExtractor(chunk_size=8000, overlap_size=1500)
    ext._available = lambda: True
    ext._run_mineru = lambda file_path, out_root: out_dir
    return ext


class TestMapping:
    def _run(self, tmp_path):
        out_dir = _make_fixture_out_dir(tmp_path)
        pdf = _make_pdf(tmp_path)
        ext = _stubbed_extractor(tmp_path, out_dir)
        return ext.extract(str(pdf))

    def test_returns_extraction_result(self, tmp_path):
        assert isinstance(self._run(tmp_path), ExtractionResult)

    def test_name(self):
        assert MineruExtractor(chunk_size=1, overlap_size=0).name == "mineru"

    def test_heading_becomes_markdown_heading(self, tmp_path):
        assert "# Introduction" in self._run(tmp_path).markdown

    def test_equation_preserved(self, tmp_path):
        assert "$$E=mc^2$$" in self._run(tmp_path).markdown

    def test_table_converted_to_pipe_markdown(self, tmp_path):
        md = self._run(tmp_path).markdown
        assert "| a | b |" in md
        assert "| 1 | 2 |" in md
        assert "Table 1: Results." in md

    def test_captioned_image_becomes_figure(self, tmp_path):
        result = self._run(tmp_path)
        assert len(result.images) == 1
        img = result.images[0]
        assert img.caption == "Figure 1: A plot."
        assert img.data_uri.startswith("data:image/jpeg;base64,")
        assert img.page == 1

    def test_figure_id_tag_recovered_into_chunk_images(self, tmp_path):
        result = self._run(tmp_path)
        all_ids = [fid for chunk in result.chunk_images for fid in chunk]
        assert 0 in all_ids

    def test_uncaptioned_table_emits_no_figure(self, tmp_path):
        # The fixture table has no img_path, so it is data-only, never a figure.
        assert all("Table" not in img.caption for img in self._run(tmp_path).images)

    def test_metadata_from_pymupdf(self, tmp_path):
        result = self._run(tmp_path)
        assert result.page_count == 2
        assert result.pdf_title == "My Paper"


class TestTableWithCrop:
    def test_captioned_table_with_img_path_becomes_figure(self, tmp_path):
        # The diverging branch: a captioned table that also has a crop yields
        # BOTH a figure (ImageRecord + FIGURE_ID) and the Markdown table data.
        content = [
            {
                "type": "table",
                "img_path": "images/tbl1.jpg",
                "table_caption": ["Table 2: Scores."],
                "table_body": "<table><tr><td>x</td></tr></table>",
                "page_idx": 0,
            },
        ]
        out = _write_out_dir(tmp_path, content, {"tbl1.jpg": _IMG_BYTES})
        pdf = _make_pdf(tmp_path)
        result = _stubbed_extractor(tmp_path, out).extract(str(pdf))
        assert len(result.images) == 1
        assert result.images[0].caption == "Table 2: Scores."
        assert 0 in [fid for chunk in result.chunk_images for fid in chunk]
        assert "| x |" in result.markdown


class TestHeadingLevelCap:
    def test_deep_heading_capped_at_six(self, tmp_path):
        content = [{"type": "text", "text": "Deep", "text_level": 9, "page_idx": 0}]
        out = _write_out_dir(tmp_path, content)
        pdf = _make_pdf(tmp_path)
        md = _stubbed_extractor(tmp_path, out).extract(str(pdf)).markdown
        assert "###### Deep" in md
        assert "####### Deep" not in md


class TestMultiCaptionInContext:
    def test_multi_element_caption_joined(self, tmp_path):
        content = [
            {
                "type": "image",
                "img_path": "images/f.jpg",
                "image_caption": ["Figure 3:", "A diagram."],
                "page_idx": 0,
            }
        ]
        out = _write_out_dir(tmp_path, content, {"f.jpg": _IMG_BYTES})
        pdf = _make_pdf(tmp_path)
        result = _stubbed_extractor(tmp_path, out).extract(str(pdf))
        assert result.images[0].caption == "Figure 3: A diagram."
        assert "Figure 3: A diagram. [FIGURE_ID: 0]" in result.markdown


class TestEquationWrapping:
    def test_bare_latex_is_wrapped(self, tmp_path):
        content = [{"type": "equation", "text": "\\frac{a}{b}", "page_idx": 0}]
        out = _write_out_dir(tmp_path, content)
        pdf = _make_pdf(tmp_path)
        md = _stubbed_extractor(tmp_path, out).extract(str(pdf)).markdown
        assert "$$" in md
        assert "\\frac{a}{b}" in md

    def test_delimited_latex_left_alone(self, tmp_path):
        content = [{"type": "equation", "text": "$$x^2$$", "page_idx": 0}]
        out = _write_out_dir(tmp_path, content)
        pdf = _make_pdf(tmp_path)
        md = _stubbed_extractor(tmp_path, out).extract(str(pdf)).markdown
        assert "$$x^2$$" in md


class TestChartBlocks:
    def test_chart_block_becomes_figure(self, tmp_path):
        # Live MinerU 3.4 emits "chart" blocks (caption in chart_caption) for
        # plots/diagrams — these are figures we want, not droppable text.
        content = [
            {
                "type": "chart",
                "img_path": "images/c.jpg",
                "chart_caption": ["Figure 3: Latency plot."],
                "content": "",
                "page_idx": 1,
            }
        ]
        out = _write_out_dir(tmp_path, content, {"c.jpg": _IMG_BYTES})
        pdf = _make_pdf(tmp_path)
        result = _stubbed_extractor(tmp_path, out).extract(str(pdf))
        assert len(result.images) == 1
        assert result.images[0].caption == "Figure 3: Latency plot."
        assert "Figure 3: Latency plot. [FIGURE_ID: 0]" in result.markdown
        assert 0 in [fid for chunk in result.chunk_images for fid in chunk]


class TestColspanTable:
    def test_colspan_expands_to_keep_columns_aligned(self):
        # Real MinerU table_body uses colspan; cells must pad so rows align.
        md = _html_table_to_md(
            "<table>"
            "<tr><td colspan=\"3\">Spanning header</td></tr>"
            "<tr><td>a</td><td>b</td><td>c</td></tr>"
            "</table>"
        )
        lines = md.splitlines()
        # First row's single colspan=3 cell pads to 3 columns (2 trailing empty).
        assert lines[0] == "| Spanning header |  |  |"
        assert lines[2] == "| a | b | c |"


class TestNoiseBlocksSkipped:
    def test_header_footer_pagenumber_stripped(self, tmp_path):
        # Live MinerU 3.4 emits header/footer/page_number blocks; the lite path
        # strips this furniture, so we must too (regression for the skip fix).
        content = [
            {"type": "header", "text": "Check for updates", "page_idx": 0},
            {"type": "text", "text": "Real body content.", "page_idx": 0},
            {"type": "footer", "text": "MICRO 23 Proceedings ISBN 999", "page_idx": 0},
            {"type": "page_number", "text": "9907", "page_idx": 0},
        ]
        out = _write_out_dir(tmp_path, content)
        pdf = _make_pdf(tmp_path)
        md = _stubbed_extractor(tmp_path, out).extract(str(pdf)).markdown
        assert "Real body content." in md
        assert "Check for updates" not in md
        assert "MICRO 23 Proceedings" not in md
        assert "9907" not in md


class TestAvailabilityGate:
    def test_unavailable_raises_with_hint(self, tmp_path):
        ext = MineruExtractor(chunk_size=8000, overlap_size=1500)
        ext._available = lambda: False
        pdf = _make_pdf(tmp_path)
        with pytest.raises(RuntimeError) as exc:
            ext.extract(str(pdf))
        assert "pymupdf4llm" in str(exc.value)  # points user at the lite fallback


class TestFactoryRegistration:
    def test_make_extractor_returns_mineru(self):
        ext = make_extractor("mineru", chunk_size=8000, overlap_size=1500)
        assert isinstance(ext, MineruExtractor)


class TestHelpers:
    def test_join_caption_list(self):
        assert _join_caption(["Figure 1.", "Continued."]) == "Figure 1. Continued."

    def test_join_caption_empty(self):
        assert _join_caption(None) == ""
        assert _join_caption([]) == ""

    def test_join_caption_str(self):
        assert _join_caption("  hi  ") == "hi"

    def test_html_table_to_md_basic(self):
        md = _html_table_to_md(
            "<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        lines = md.splitlines()
        assert lines[0] == "| x | y |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 2 |"

    def test_html_table_to_md_empty(self):
        assert _html_table_to_md("") == ""
        assert _html_table_to_md("<table></table>") == ""

    def test_html_table_to_md_escapes_pipes(self):
        md = _html_table_to_md("<table><tr><td>a|b</td></tr></table>")
        assert r"a\|b" in md
