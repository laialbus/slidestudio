"""
MinerU-based extractor — the "quality" path.

MinerU bundles layout detection, formula recognition (UniMERNet), and table
recognition in one pipeline and emits a content_list.json describing the
document as ordered blocks (text / title / image / table / equation) with page
indices, captions, image crops, and table HTML.

This extractor drives entirely off content_list.json rather than MinerU's
rendered Markdown.  Building the Markdown ourselves means we control heading
levels, [FIGURE_ID] placement, and table formatting, so the result feeds the
exact same downstream contract (heading-boundary chunking + figure catalog) as
the lite pymupdf4llm path — only the source of structure changes.

MinerU is an OPTIONAL dependency (GB-scale model weights downloaded on first
run).  It is invoked lazily via its CLI; if unavailable, extract() raises a
clear RuntimeError pointing at the install path or the lite extractor.

IMPLEMENTATION NOTE (verify against the installed MinerU before trusting):
  - _run_mineru shells out to the `mineru` CLI. The CLI is the most
    version-stable interface, but its exact flags / output-dir layout have
    changed across MinerU releases. content_list.json itself (the block schema
    consumed below) is MinerU's documented, stable "middle format".
  - The block field names below (type, text, text_level, page_idx, img_path,
    image_caption, table_caption, table_body) are pinned to that format. A smoke
    run on a real PDF is required to confirm them for the installed version.
"""

import base64
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

from extractors.base import (
    BaseExtractor,
    ExtractionResult,
    ImageRecord,
    TocItem,
)
from extractors.chunking import _build_chunk_images, _chunk_by_headings
from extractors.pdf import _clean_tables, _extract_pdf_title

_INSTALL_HINT = (
    "MinerU is not installed or its CLI is not on PATH. Install it with "
    "`pip install -r requirements-mineru.txt` (downloads model weights on "
    'first run), or set PIPELINE["extractor"] = "pymupdf4llm" to use the lite '
    "extractor."
)

# MinerU writes a "<stem>_content_list.json" somewhere under the output dir; the
# exact nesting varies by version, so locate it by glob rather than fixed path.
_CONTENT_LIST_GLOB = "*_content_list.json"

# MinerU's CLI default backend is "hybrid-engine" (needs the VLM stack). We
# install only mineru[pipeline] (layout + formula + table + OCR, local), so the
# backend MUST be forced to "pipeline" or the run fails / pulls VLM models.
_MINERU_BACKEND = "pipeline"

# Markdown heading depth cap — chunking only splits on # / ##, but deeper levels
# are still emitted as valid headings for readability.
_MAX_HEADING_LEVEL = 6

_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})

# content_list block types that are page furniture, not content. The lite path
# strips these via to_markdown(header=False, footer=False); we drop them too so
# they don't pollute chunks. Verified present in live MinerU 3.4 output.
_SKIP_TYPES = frozenset(
    {"header", "footer", "page_number", "page_footnote", "discarded"}
)


class MineruExtractor(BaseExtractor):
    """Quality extractor backed by MinerU's content_list.json output."""

    @property
    def name(self) -> str:
        return "mineru"

    def extract(self, file_path: str) -> ExtractionResult:
        if not self._available():
            raise RuntimeError(_INSTALL_HINT)

        # Run MinerU into a temp dir and read its crops back into memory, so the
        # "no file I/O between steps" contract holds — nothing persists on disk.
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = self._run_mineru(file_path, tmp)
            content_list = self._load_content_list(out_dir)
            md, images = self._build_markdown_and_images(content_list, out_dir)

        # Validate/clean tables with the same policy as the lite path.
        md = _clean_tables(md)
        chunks = _chunk_by_headings(md, self._chunk_size, self._overlap_size)
        chunk_images = _build_chunk_images(chunks)

        # TOC / title / page count come from pymupdf on the original PDF — the
        # same zero-cost metadata path the lite extractor uses.
        with pymupdf.open(file_path) as doc:
            page_count = doc.page_count
            toc_items = [
                TocItem(level=level, heading=title, page=page)
                for level, title, page in doc.get_toc()
                if title and title.strip()
            ]
            pdf_title = _extract_pdf_title(doc)

        char_count = sum(len(c) for c in chunks)
        return ExtractionResult(
            markdown=md,
            toc_items=toc_items,
            chunks=chunks,
            chunk_images=chunk_images,
            images=images,
            page_count=page_count,
            char_count=char_count,
            ocr_used=True,   # MinerU runs OCR/layout models on every page
            pdf_title=pdf_title,
        )

    # ── MinerU invocation (verify against installed version) ──────────────

    def _available(self) -> bool:
        return shutil.which("mineru") is not None

    def _run_mineru(self, file_path: str, out_root: str) -> Path:
        """Run the MinerU CLI and return the dir holding content_list.json."""
        try:
            subprocess.run(
                ["mineru", "-p", file_path, "-o", out_root,
                 "-b", _MINERU_BACKEND],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"MinerU failed on {file_path!r}: {e.stderr or e.stdout}"
            ) from e
        matches = list(Path(out_root).rglob(_CONTENT_LIST_GLOB))
        if not matches:
            raise RuntimeError(
                "MinerU produced no content_list.json — its output layout may "
                "have changed; verify the installed version."
            )
        return matches[0].parent

    def _load_content_list(self, out_dir: Path) -> list[dict]:
        path = next(out_dir.glob(_CONTENT_LIST_GLOB))
        return json.loads(path.read_text(encoding="utf-8"))

    # ── content_list.json → markdown + images ─────────────────────────────

    def _build_markdown_and_images(
        self, content_list: list[dict], out_dir: Path
    ) -> tuple[str, list[ImageRecord]]:
        """Render ordered blocks to Markdown and collect captioned figures.

        Mirrors the lite path's contract: only captioned figures become
        ImageRecords (catalog membership == referenceability), and each gets a
        [FIGURE_ID: N] tag inline so _build_chunk_images recovers ownership.
        Tables are emitted as Markdown data regardless of caption; a captioned
        table also yields a figure crop, matching lite-path behaviour.
        """
        parts: list[str] = []
        images: list[ImageRecord] = []
        idx = 0

        for block in content_list:
            btype = block.get("type")
            if btype in _SKIP_TYPES:
                continue
            page = block.get("page_idx", 0) + 1

            if btype in ("image", "chart"):
                # MinerU classifies plots/diagrams as "chart" (caption in
                # chart_caption) and photos/figures as "image" (image_caption).
                # Both are figures we want in slides.
                cap_field = (
                    "chart_caption" if btype == "chart" else "image_caption"
                )
                caption = _join_caption(block.get(cap_field))
                if not caption:
                    continue  # uncaptioned figure is unreferenceable — skip
                record = _make_image_record(block, out_dir, idx, page, caption)
                if record is None:
                    # Crop missing/unreadable: drop the whole block. Unlike the
                    # lite path (whose caption text comes from pymupdf4llm,
                    # independent of figure rendering), our caption is sourced
                    # from this same block, so nothing remains to keep.
                    continue
                images.append(record)
                parts.append(f"{caption} [FIGURE_ID: {idx}]")
                idx += 1

            elif btype == "table":
                caption = _join_caption(block.get("table_caption"))
                record = (
                    _make_image_record(block, out_dir, idx, page, caption)
                    if caption else None
                )
                if record is not None:
                    images.append(record)
                    parts.append(f"{caption} [FIGURE_ID: {idx}]")
                    idx += 1
                elif caption:
                    parts.append(caption)
                table_md = _html_table_to_md(block.get("table_body") or "")
                if table_md:
                    parts.append(table_md)

            elif btype == "equation":
                text = (block.get("text") or "").strip()
                if text:
                    # MinerU usually emits $$-delimited LaTeX; some versions
                    # emit bare LaTeX, so wrap it to render as math, not prose.
                    if "$" not in text:
                        text = f"$$\n{text}\n$$"
                    parts.append(text)

            else:
                # text, title, list, and any other text-bearing block
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                level = block.get("text_level")
                if level:
                    hashes = "#" * min(int(level), _MAX_HEADING_LEVEL)
                    parts.append(f"{hashes} {text}")
                else:
                    parts.append(text)

        return "\n\n".join(parts), images


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _join_caption(value) -> str:
    """MinerU caption fields are lists of strings; join into one line."""
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip()).strip()
    return str(value).strip()


def _make_image_record(
    block: dict, out_dir: Path, idx: int, page: int, caption: str
) -> ImageRecord | None:
    """Read a MinerU crop from disk and encode it as a base64 data URI."""
    img_path = block.get("img_path")
    if not img_path:
        return None
    crop = out_dir / img_path
    if not crop.is_file():
        return None
    is_jpeg = crop.suffix.lower() in _JPEG_SUFFIXES
    mime = "image/jpeg" if is_jpeg else "image/png"
    b64 = base64.b64encode(crop.read_bytes()).decode("ascii")
    return ImageRecord(
        index=idx,
        caption=caption,
        data_uri=f"data:{mime};base64,{b64}",
        page=page,
    )


class _TableHTMLParser(HTMLParser):
    """Collect rows of cell text from a MinerU <table> HTML string."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._colspan = 1
            for key, value in attrs:
                if key == "colspan":
                    try:
                        self._colspan = max(1, int(value))
                    except (TypeError, ValueError):
                        pass

    def handle_endtag(self, tag):
        is_cell = tag in ("td", "th")
        if is_cell and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            # Emit empty cells for a colspan so columns stay aligned.
            self._row.extend([""] * (self._colspan - 1))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _html_table_to_md(table_html: str) -> str:
    """Convert a MinerU HTML table to a GitHub-flavoured Markdown pipe table."""
    if not table_html.strip():
        return ""
    parser = _TableHTMLParser()
    parser.feed(table_html)
    rows = [r for r in parser.rows if r]
    if not rows:
        return ""

    width = max(len(r) for r in rows)

    def _fmt(cells: list[str]) -> str:
        padded = cells + [""] * (width - len(cells))
        escaped = [c.replace("|", r"\|") for c in padded]
        return "| " + " | ".join(escaped) + " |"

    lines = [_fmt(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(_fmt(r) for r in rows[1:])
    return "\n".join(lines)
