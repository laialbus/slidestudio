import base64
import re
import warnings

import pymupdf
import pymupdf.layout  # registers the GNN layout engine inside pymupdf4llm
import pymupdf4llm.helpers.document_layout as _dl
import pymupdf4llm.helpers.pymupdf_rag as _rag
from pydantic import BaseModel

# Sub-chunk limits: used when a heading-boundary section exceeds the limit
# or when no headings are found (character-overlap fallback).
_CHUNK_CHAR_LIMIT = 8_000
_OVERLAP_SIZE     = 1_500

# Image filters
_MAX_IMAGE_BYTES = 512 * 1_024   # 512 KB — likely page backgrounds above this
_MIN_IMAGE_DIM   = 32            # px — ignore tiny icons/decorations

# Caption detection (checked in adjacent text blocks)
_CAPTION_RE = re.compile(r"^(fig\.?|figure|chart|table)\b", re.IGNORECASE)

# Heading boundary: only # and ## (not ### or deeper)
_HEADING_LINE_RE = re.compile(r"^(#{1,2})(?!#)\s", re.MULTILINE)

# Table validation thresholds
_MAX_TABLE_ROWS   = 70
_MIN_COL_FRACTION = 0.8   # ≥80% of rows must match the modal column count


# ──────────────────────────────────────────────────────────────
# Output models — live in extractors/, never imported by schemas/,
# agents/, or providers/.
# ──────────────────────────────────────────────────────────────

class TocItem(BaseModel):
    level:   int
    heading: str
    page:    int


class ImageRecord(BaseModel):
    index:    int
    caption:  str
    data_uri: str
    page:     int


class ExtractionResult(BaseModel):
    markdown:   str
    toc_items:  list[TocItem]
    chunks:     list[str]
    images:     list[ImageRecord]
    page_count: int
    char_count: int
    ocr_used:   bool


# ──────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────

class PDFExtractor:
    def __init__(self, chunk_size: int = _CHUNK_CHAR_LIMIT,
                 overlap_size: int = _OVERLAP_SIZE):
        self._chunk_size   = chunk_size
        self._overlap_size = overlap_size

    def extract(self, file_path: str) -> ExtractionResult:
        doc = pymupdf.open(file_path)
        page_count = doc.page_count

        # TOC items from the PDF's embedded table of contents (zero API cost)
        raw_toc  = doc.get_toc()
        toc_items = [
            TocItem(level=level, heading=title, page=page)
            for level, title, page in raw_toc
            if title and title.strip()
        ]

        # Parse once with the layout engine — gives markdown + OCR status
        # without reopening the file.  header=False/footer=False strips
        # repeating page decorations that pollute every chunk.
        parsed   = _dl.parse_document(doc)
        ocr_used = parsed.use_ocr != _dl.OCRMode.NEVER
        md_layout = parsed.to_markdown(header=False, footer=False)

        # Fall back to the RAG extractor when the layout engine captures
        # less than 50 % of what it would produce (e.g. simple
        # programmatically-created PDFs that have no complex layout).
        # Guard against RAG crashes on tables with empty cell lists (pymupdf bug).
        try:
            md_rag = _rag.to_markdown(doc)
        except Exception:
            md_rag = ""
        md = md_layout if len(md_layout.strip()) >= 0.5 * max(len(md_rag.strip()), 1) else md_rag

        # Warn (not crash) on suspiciously short extraction (possible scanned PDF)
        if len(md.strip()) < 200 and page_count > 1:
            warnings.warn(
                f"Extracted only {len(md.strip())} characters from {page_count} "
                "pages. The PDF may be scanned. Install Tesseract OCR for better "
                "results: https://tesseract-ocr.github.io/tessdoc/Installation.html",
                UserWarning,
                stacklevel=2,
            )

        # Extract images as base64 data URIs
        images = _extract_images(doc)

        doc.close()

        # Validate and clean malformed Markdown tables before chunking
        md = _clean_tables(md)

        # Chunk by heading boundaries (primary) or character overlap (fallback)
        chunks = _chunk_by_headings(md, self._chunk_size, self._overlap_size)

        char_count = sum(len(c) for c in chunks)
        return ExtractionResult(
            markdown=md,
            toc_items=toc_items,
            chunks=chunks,
            images=images,
            page_count=page_count,
            char_count=char_count,
            ocr_used=ocr_used,
        )


# ──────────────────────────────────────────────────────────────
# Heading-boundary chunking
# ──────────────────────────────────────────────────────────────

def _chunk_by_headings(md: str, chunk_size: int, overlap_size: int) -> list[str]:
    """
    Primary: split on # / ## lines. Each section is one chunk.
    If a section exceeds chunk_size, sub-split with overlap while preserving
    the heading at the top of every sub-chunk.
    Fallback: character overlap chunking when no # or ## headings exist.
    """
    positions = [m.start() for m in _HEADING_LINE_RE.finditer(md)]

    if not positions:
        return _chunk_by_chars(md, chunk_size, overlap_size)

    # Slice sections: [pos[0]..pos[1]), [pos[1]..pos[2]), …, [pos[-1]..end)
    sections: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(md)
        sections.append(md[start:end].rstrip())

    result: list[str] = []

    # Preamble — text before the first heading
    preamble = md[:positions[0]].strip()
    if preamble:
        result.extend(_split_section("", preamble, chunk_size, overlap_size))

    for section in sections:
        # Extract the heading line and the body separately
        nl = section.find("\n")
        if nl == -1:
            heading, body = section, ""
        else:
            heading, body = section[:nl], section[nl + 1:]

        result.extend(_split_section(heading, body, chunk_size, overlap_size))

    return [c for c in result if c.strip()] or [md]


def _split_section(heading: str, body: str, chunk_size: int, overlap_size: int) -> list[str]:
    """
    If heading+body fits in chunk_size, return as one chunk.
    Otherwise sub-split the body with overlap and prefix every sub-chunk with heading.
    """
    full = (heading + "\n" + body).strip() if heading else body.strip()
    if len(full) <= chunk_size:
        return [full] if full else []

    sub_bodies = _chunk_by_chars(body, chunk_size - len(heading) - 1, overlap_size)
    prefix = heading + "\n" if heading else ""
    return [(prefix + sb).strip() for sb in sub_bodies if sb.strip()]


def _chunk_by_chars(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """Character-overlap chunking with paragraph-snap boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            pb = text.rfind("\n\n", start, end)
            if pb != -1:
                end = pb
        chunks.append(text[start:end].strip())
        start = max(end - overlap_size, start + 1)

    return [c for c in chunks if c]


# ──────────────────────────────────────────────────────────────
# Table validation
# ──────────────────────────────────────────────────────────────

def _clean_tables(md: str) -> str:
    """
    Scan the Markdown for tables. Replace malformed tables — those with
    inconsistent column counts, more than _MAX_TABLE_ROWS rows, or mostly
    empty cells — with a structured placeholder that includes the caption if
    one was found in the surrounding text.
    """
    lines  = md.splitlines(keepends=True)
    result: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if "|" in line and _is_table_delimiter_or_row(line):
            table_lines: list[str] = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            # Look for a caption in the line immediately preceding or following
            caption = _find_nearby_caption(lines, i - len(table_lines) - 1, i)
            if _is_malformed_table(table_lines):
                cap_str = f": {caption}" if caption else ""
                result.append(f"[TABLE{cap_str}]\n")
            else:
                result.extend(table_lines)
        else:
            result.append(line)
            i += 1

    return "".join(result)


def _is_table_delimiter_or_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") or ("|" in stripped and stripped.endswith("|"))


def _is_malformed_table(table_lines: list[str]) -> bool:
    rows = [l for l in table_lines if "|" in l and not _is_separator_row(l)]
    if len(rows) > _MAX_TABLE_ROWS:
        return True
    if not rows:
        return False
    col_counts = [l.count("|") for l in rows]
    modal = max(set(col_counts), key=col_counts.count)
    consistent = sum(1 for c in col_counts if c == modal) / len(col_counts)
    if consistent < _MIN_COL_FRACTION:
        return True
    # Mostly empty cells
    cells = [cell.strip() for l in rows for cell in l.split("|") if cell.strip() or True]
    non_empty = sum(1 for c in cells if c)
    if cells and non_empty / len(cells) < 0.2:
        return True
    return False


def _is_separator_row(line: str) -> bool:
    return bool(re.match(r"^\s*\|[\s\-:|]+\|\s*$", line))


def _find_nearby_caption(lines: list[str], start: int, end: int) -> str:
    for idx in (start, end):
        if 0 <= idx < len(lines):
            text = lines[idx].strip()
            if _CAPTION_RE.match(text):
                return text[:120]
    return ""


# ──────────────────────────────────────────────────────────────
# Image extraction
# ──────────────────────────────────────────────────────────────

def _extract_images(doc) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_xrefs: set[int] = set()
    idx = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                img_data = doc.extract_image(xref)
            except Exception:
                continue
            if img_data is None:
                continue

            raw    = img_data.get("image", b"")
            width  = img_data.get("width",  0)
            height = img_data.get("height", 0)

            if len(raw) > _MAX_IMAGE_BYTES:
                continue
            if width < _MIN_IMAGE_DIM or height < _MIN_IMAGE_DIM:
                continue

            caption = _find_image_caption(page, img_info, blocks)
            ext     = img_data.get("ext", "png")
            b64     = base64.b64encode(raw).decode("ascii")
            data_uri = f"data:image/{ext};base64,{b64}"

            records.append(ImageRecord(
                index=idx,
                caption=caption,
                data_uri=data_uri,
                page=page_num + 1,
            ))
            idx += 1

    return records


def _find_image_caption(page, img_info, blocks: list) -> str:
    """
    Locate the bounding box of the image on the page, then search for a
    caption in the nearest text block below it whose text starts with a
    figure/table marker.
    """
    # Find the image's bounding box via page image-info list
    try:
        for info in page.get_image_info():
            if info.get("xref") == img_info[0]:
                img_rect = pymupdf.Rect(info["bbox"])
                break
        else:
            return ""
    except Exception:
        return ""

    best_text  = ""
    best_dist  = float("inf")

    for block in blocks:
        if block["type"] != 0:
            continue
        block_rect = pymupdf.Rect(block["bbox"])
        # Caption should be below the image
        if block_rect.y0 < img_rect.y1 - 5:
            continue
        dist = block_rect.y0 - img_rect.y1
        if dist > 80:   # more than ~1 inch below — probably unrelated
            continue
        text = " ".join(
            span["text"].strip()
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if _CAPTION_RE.match(text) and dist < best_dist:
            best_dist  = dist
            best_text  = text

    return best_text[:120]
