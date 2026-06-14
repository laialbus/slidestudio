import base64
import re
import warnings

import pymupdf
import pymupdf.layout  # registers the GNN layout engine inside pymupdf4llm
import pymupdf4llm.helpers.document_layout as _dl
import pymupdf4llm.helpers.pymupdf_rag as _rag
from pydantic import BaseModel

from extractors.layout import (
    LayoutAnalyser,
    _CAPTION_LABELS,
    _FIGURE_LABELS,
    _merge_figure_regions,
    _pair_figures_captions,
)

# Sub-chunk limits: used when a heading-boundary section exceeds the limit
# or when no headings are found (character-overlap fallback).
_CHUNK_CHAR_LIMIT = 8_000
_OVERLAP_SIZE     = 1_500

# Image filters / rendering
_MAX_IMAGE_BYTES       = 1 * 1_024 * 1_024  # 1 MB — cap rendered figure size
_FIGURE_SEARCH_WINDOW  = 500              # pt — vertical scan window from caption edge
_HAIRLINE_MAX_HEIGHT   = 2.0              # pt — stroke-only paths below this are decorative
_CAPTION_PADDING       = 3.0              # pt — safety margin around rendered figure rect
_SURYA_FIGURE_PADDING  = 8.0              # pt — Surya bboxes are tight; pad before render to keep axis labels/edges
_CAPTION_MERGE_GAP     = 8.0              # pt — max y-gap to absorb continuation caption blocks
_DRAWING_OVERLAP_FRAC  = 0.4              # drawings must x-overlap caption by this fraction
_RENDER_DPI            = 200
_RENDER_DPI_FALLBACK   = 100
_MIN_FIGURE_HEIGHT     = 20.0         # pt — discard rects too short to be a real figure
_BLANK_MIN_CHANNEL_VALUE = 250        # pixels below this value in any channel count as content
_TEXT_CONTAINMENT_FRAC   = 0.5        # min fraction of block area inside rect to include its text

# Caption pattern — anchored to the start of a text block
_CAPTION_RE = re.compile(r"^(fig\.?|figure|chart|table)\b", re.IGNORECASE)

# Title extraction: accept spans within this fraction of the max font size on page 1
_TITLE_FONT_TOLERANCE = 0.95

# Heading boundary: only # and ## (not ### or deeper)
_HEADING_LINE_RE = re.compile(r"^(#{1,2})(?!#)\s", re.MULTILINE)

# Figure ID tag — injected into markdown, scanned back from chunks
_FIGURE_ID_RE = re.compile(r"\[FIGURE_ID:\s*(\d+)\]")

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
    markdown:     str
    toc_items:    list[TocItem]
    chunks:       list[str]
    chunk_images: list[list[int]]   # chunk_idx → figure indices in that chunk
    images:       list[ImageRecord]
    page_count:   int
    char_count:   int
    ocr_used:     bool
    pdf_title:    str = ""          # largest-font text on page 1; empty if undetermined


# ──────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────

class PDFExtractor:
    def __init__(self, chunk_size: int = _CHUNK_CHAR_LIMIT,
                 overlap_size: int = _OVERLAP_SIZE):
        self._chunk_size   = chunk_size
        self._overlap_size = overlap_size
        self._layout       = LayoutAnalyser()

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
        images = _extract_images(doc, self._layout)

        pdf_title = _extract_pdf_title(doc)
        doc.close()

        # Validate and clean malformed Markdown tables before chunking
        md = _clean_tables(md)

        # Inject [FIGURE_ID: N] markers at each extracted caption so that
        # chunks carry figure ownership without any LLM involvement.
        md = _inject_figure_ids(md, images)

        # Chunk by heading boundaries (primary) or character overlap (fallback)
        chunks = _chunk_by_headings(md, self._chunk_size, self._overlap_size)

        # Build the chunk→figure index mapping by scanning for the injected tags.
        chunk_images = _build_chunk_images(chunks)

        char_count = sum(len(c) for c in chunks)
        return ExtractionResult(
            markdown=md,
            toc_items=toc_items,
            chunks=chunks,
            chunk_images=chunk_images,
            images=images,
            page_count=page_count,
            char_count=char_count,
            ocr_used=ocr_used,
            pdf_title=pdf_title,
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

def _extract_images(doc, analyser: LayoutAnalyser | None = None) -> list[ImageRecord]:
    """
    Primary path (Surya available): run layout detection per page, pair
    detected figure regions with detected caption regions by nearest-neighbour
    matching, then render each figure region directly.  Captions the model
    could not pair fall through to the caption-first heuristic below.

    Fallback path (Surya unavailable or no detections): caption-first
    heuristic — scan for Figure/Table captions, infer the figure region via
    vector drawing union, raster block detection, or whitespace gap walk, then
    render that region.
    """
    records: list[ImageRecord] = []
    idx = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        if analyser is not None and analyser.available:
            regions = analyser.detect(page)
            figures  = [r for r in regions if r.label in _FIGURE_LABELS]
            captions = [r for r in regions if r.label in _CAPTION_LABELS]

            # Fuse multi-panel composites into one region before pairing so the
            # caption matches the whole figure, not a single sub-panel.
            figures = _merge_figure_regions(figures)

            matched_pairs, unmatched_caps = _pair_figures_captions(figures, captions)

            for fig_region, cap_region in matched_pairs:
                caption_text = (
                    _extract_text_in_rect(blocks, cap_region.bbox)
                    if cap_region is not None else ""
                )
                # A figure with no caption is unreferenceable (the Planner picks
                # figures from the captioned catalog), so skip it rather than
                # spend bytes encoding dead weight.
                if not caption_text.strip():
                    continue
                # Surya bboxes are tight — pad before rendering so axis labels
                # and panel edges are not clipped, then clamp to the page.
                fig_rect = _pad_and_clip(
                    fig_region.bbox, _SURYA_FIGURE_PADDING, page.rect
                )
                raw = _render_figure(page, fig_rect)
                if raw is None:
                    continue
                b64 = base64.b64encode(raw).decode("ascii")
                records.append(ImageRecord(
                    index=idx,
                    caption=caption_text,
                    data_uri=f"data:image/png;base64,{b64}",
                    page=page_num + 1,
                ))
                idx += 1

            # Heuristic fallback for captions the model could not pair
            if unmatched_caps:
                body_size = _page_body_size(blocks)
                for cap_region in unmatched_caps:
                    caption_text = _extract_text_in_rect(blocks, cap_region.bbox)
                    cap_dict = {
                        "text":      caption_text,
                        "full_bbox": cap_region.bbox,
                        "page_num":  page_num,
                    }
                    figure_rect = _infer_figure_rect(page, blocks, cap_dict, body_size)
                    if figure_rect is None or figure_rect.is_empty:
                        continue
                    raw = _render_figure(page, figure_rect)
                    if raw is None:
                        continue
                    b64 = base64.b64encode(raw).decode("ascii")
                    records.append(ImageRecord(
                        index=idx,
                        caption=caption_text,
                        data_uri=f"data:image/png;base64,{b64}",
                        page=page_num + 1,
                    ))
                    idx += 1
        else:
            # Surya unavailable: caption-first heuristic path (unchanged)
            body_size = _page_body_size(blocks)
            captions  = _scan_figure_captions(blocks, page_num)
            for cap in captions:
                figure_rect = _infer_figure_rect(page, blocks, cap, body_size)
                if figure_rect is None or figure_rect.is_empty:
                    continue
                raw = _render_figure(page, figure_rect)
                if raw is None:
                    continue
                b64 = base64.b64encode(raw).decode("ascii")
                records.append(ImageRecord(
                    index=idx,
                    caption=cap["text"],
                    data_uri=f"data:image/png;base64,{b64}",
                    page=page_num + 1,
                ))
                idx += 1

    return records


def _extract_text_in_rect(blocks: list, rect: pymupdf.Rect) -> str:
    """
    Collect and join text from type-0 blocks whose bounding box is contained
    within rect by at least _TEXT_CONTAINMENT_FRAC of the block's own area.
    """
    parts: list[str] = []
    for block in blocks:
        if block["type"] != 0:
            continue
        br = pymupdf.Rect(block["bbox"])
        intersection = br & rect
        if intersection.is_empty:
            continue
        block_area = br.get_area()
        if block_area > 0 and intersection.get_area() / block_area >= _TEXT_CONTAINMENT_FRAC:
            text = _block_text(block)
            if text:
                parts.append(text)
    return " ".join(parts).strip()


# ── helpers ───────────────────────────────────────────────────

def _block_text(block: dict) -> str:
    return " ".join(
        span["text"].strip()
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ).strip()


def _page_body_size(blocks: list) -> float:
    """Median font size across all text spans — proxy for body-text size."""
    sizes = [
        span.get("size", 0.0)
        for block in blocks
        if block["type"] == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("size", 0.0) > 0
    ]
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _scan_figure_captions(blocks: list, page_num: int) -> list[dict]:
    """
    Return one dict per figure/table caption found on the page.
    Adjacent continuation blocks (within _CAPTION_MERGE_GAP) are merged into
    a single caption record, with end-of-block hyphens stripped.
    Each dict: text, full_bbox (pymupdf.Rect), page_num.
    Direction is detected from drawing evidence in _infer_figure_rect.
    """
    captions: list[dict] = []
    consumed: set[int] = set()

    for i, block in enumerate(blocks):
        if block["type"] != 0 or i in consumed:
            continue
        text = _block_text(block)
        if not _CAPTION_RE.match(text):
            continue

        full_bbox = pymupdf.Rect(block["bbox"])
        merged_text = text
        consumed.add(i)

        # Absorb immediately adjacent continuation blocks
        for j in range(i + 1, len(blocks)):
            nb = blocks[j]
            if nb["type"] != 0:
                break
            nb_rect = pymupdf.Rect(nb["bbox"])
            if nb_rect.y0 - full_bbox.y1 > _CAPTION_MERGE_GAP:
                break
            nb_text = _block_text(nb)
            if _CAPTION_RE.match(nb_text):
                break
            # Strip trailing hyphen (word split across blocks)
            if merged_text.endswith("-"):
                merged_text = merged_text[:-1] + nb_text
            else:
                merged_text = merged_text + " " + nb_text
            full_bbox = full_bbox | nb_rect
            consumed.add(j)

        captions.append({
            "text": merged_text.strip(),
            "full_bbox": full_bbox,
            "page_num": page_num,
        })

    return captions


def _infer_figure_rect(
    page, blocks: list, caption: dict, body_size: float
) -> pymupdf.Rect | None:
    """
    Priority 1 — vector drawings: union bounding boxes of drawing paths on
    either side of the caption; larger side wins.
    Priority 2 — raster images: type-1 blocks (PNG/JPEG XObjects) invisible to
    get_drawings(); use their page bbox directly.
    Priority 3 — whitespace gap walk on both sides; larger rect wins.
    Direction is always inferred from evidence, not assumed from caption keyword.
    """
    full_bbox: pymupdf.Rect = caption["full_bbox"]
    page_rect = page.rect
    drawings  = page.get_drawings()

    above_cands = [
        d["rect"] for d in drawings
        if d["rect"].y1 <= full_bbox.y0 + 2
        and d["rect"].y0 >= max(page_rect.y0, full_bbox.y0 - _FIGURE_SEARCH_WINDOW)
        and _x_overlaps(d["rect"], full_bbox)
        and not _is_hairline(d)
    ]
    below_cands = [
        d["rect"] for d in drawings
        if d["rect"].y0 >= full_bbox.y1 - 2
        and d["rect"].y1 <= min(page_rect.y1, full_bbox.y1 + _FIGURE_SEARCH_WINDOW)
        and _x_overlaps(d["rect"], full_bbox)
        and not _is_hairline(d)
    ]

    def _union(rects: list) -> pymupdf.Rect:
        u = rects[0]
        for r in rects[1:]:
            u = u | r
        return u

    if above_cands or below_cands:
        if above_cands and below_cands:
            au, bu = _union(above_cands), _union(below_cands)
            best = au if au.get_area() >= bu.get_area() else bu
        else:
            best = _union(above_cands or below_cands)
        padded = pymupdf.Rect(
            best.x0 - _CAPTION_PADDING,
            best.y0 - _CAPTION_PADDING,
            best.x1 + _CAPTION_PADDING,
            best.y1 + _CAPTION_PADDING,
        )
        result = padded & page_rect
        if not result.is_empty and result.height >= _MIN_FIGURE_HEIGHT:
            return result

    # Raster image fallback — type-1 blocks are PNG/JPEG XObjects invisible to
    # get_drawings(). Their bbox gives the exact page position directly.
    above_img = [
        pymupdf.Rect(b["bbox"]) for b in blocks
        if b["type"] == 1
        and pymupdf.Rect(b["bbox"]).y1 <= full_bbox.y0 + 2
        and pymupdf.Rect(b["bbox"]).y0 >= max(page_rect.y0, full_bbox.y0 - _FIGURE_SEARCH_WINDOW)
        and _x_overlaps(pymupdf.Rect(b["bbox"]), full_bbox)
    ]
    below_img = [
        pymupdf.Rect(b["bbox"]) for b in blocks
        if b["type"] == 1
        and pymupdf.Rect(b["bbox"]).y0 >= full_bbox.y1 - 2
        and pymupdf.Rect(b["bbox"]).y1 <= min(page_rect.y1, full_bbox.y1 + _FIGURE_SEARCH_WINDOW)
        and _x_overlaps(pymupdf.Rect(b["bbox"]), full_bbox)
    ]

    raster_candidate: pymupdf.Rect | None = None
    if above_img or below_img:
        if above_img and below_img:
            au, bu = _union(above_img), _union(below_img)
            best = au if au.get_area() >= bu.get_area() else bu
        else:
            best = _union(above_img or below_img)
        padded = pymupdf.Rect(
            best.x0 - _CAPTION_PADDING,
            best.y0 - _CAPTION_PADDING,
            best.x1 + _CAPTION_PADDING,
            best.y1 + _CAPTION_PADDING,
        )
        candidate = padded & page_rect
        if not candidate.is_empty and candidate.height >= _MIN_FIGURE_HEIGHT:
            raster_candidate = candidate

    # Whitespace fallback on both sides, now column-aware (see _whitespace_fallback).
    # Evaluate alongside raster and return the largest: a raster sub-element embedded
    # inside a Form XObject figure would have a smaller area than the full-column
    # whitespace estimate, so the correct region wins.
    above_rect = _whitespace_fallback(page, blocks, caption, body_size, "above")
    below_rect = _whitespace_fallback(page, blocks, caption, body_size, "below")
    candidates = [r for r in (raster_candidate, above_rect, below_rect) if r is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get_area())


def _x_overlaps(drawing_rect: pymupdf.Rect, caption_bbox: pymupdf.Rect) -> bool:
    """True when drawing_rect x-range overlaps caption_bbox by >= _DRAWING_OVERLAP_FRAC."""
    cap_width = max(caption_bbox.x1 - caption_bbox.x0, 1.0)
    overlap = min(drawing_rect.x1, caption_bbox.x1) - max(drawing_rect.x0, caption_bbox.x0)
    return (overlap / cap_width) >= _DRAWING_OVERLAP_FRAC


def _is_hairline(drawing: dict) -> bool:
    """Stroke-only paths with negligible height are decorative rules, not figure content."""
    return not drawing.get("fill") and drawing["rect"].height < _HAIRLINE_MAX_HEIGHT


def _whitespace_fallback(
    page, blocks: list, caption: dict, body_size: float, direction: str
) -> pymupdf.Rect | None:
    """
    Find the figure region by locating the nearest body-text block on the
    other side of the whitespace gap that separates figure from prose.
    x-range is expanded to encompass all blocks (including embedded images)
    found within the vertical window.
    """
    full_bbox: pymupdf.Rect = caption["full_bbox"]
    page_rect = page.rect

    if direction == "above":
        ref_y = full_bbox.y0
        search_limit = max(page_rect.y0, ref_y - _FIGURE_SEARCH_WINDOW)
        best_edge = search_limit  # default: top of search window

        for block in blocks:
            if block["type"] != 0:
                continue
            br = pymupdf.Rect(block["bbox"])
            if br.y1 > ref_y - 2:
                continue
            # Only consider text blocks in the caption's column — excludes
            # body text from an adjacent column that would collapse best_edge
            # to ref_y and produce an empty figure rect.
            if min(br.x1, full_bbox.x1) - max(br.x0, full_bbox.x0) <= 0:
                continue
            if _CAPTION_RE.match(_block_text(block)):
                continue
            sz = max(
                (span.get("size", 0.0)
                 for line in block.get("lines", [])
                 for span in line.get("spans", [])),
                default=0.0,
            )
            if sz < body_size * 0.8:
                continue
            if br.y1 > best_edge:
                best_edge = br.y1

        x0 = full_bbox.x0 - _CAPTION_PADDING
        x1 = full_bbox.x1 + _CAPTION_PADDING
        for block in blocks:
            br = pymupdf.Rect(block["bbox"])
            if best_edge <= br.y0 and br.y1 <= ref_y:
                x0 = min(x0, br.x0 - _CAPTION_PADDING)
                x1 = max(x1, br.x1 + _CAPTION_PADDING)

        figure_rect = pymupdf.Rect(x0, best_edge, x1, ref_y)

    else:  # "below" for tables
        ref_y = full_bbox.y1
        search_limit = min(page_rect.y1, ref_y + _FIGURE_SEARCH_WINDOW)
        best_edge = search_limit  # default: bottom of search window

        for block in blocks:
            if block["type"] != 0:
                continue
            br = pymupdf.Rect(block["bbox"])
            if br.y0 < ref_y + 2:
                continue
            if min(br.x1, full_bbox.x1) - max(br.x0, full_bbox.x0) <= 0:
                continue
            if _CAPTION_RE.match(_block_text(block)):
                continue
            sz = max(
                (span.get("size", 0.0)
                 for line in block.get("lines", [])
                 for span in line.get("spans", [])),
                default=0.0,
            )
            if sz < body_size * 0.8:
                continue
            if br.y0 < best_edge:
                best_edge = br.y0

        x0 = full_bbox.x0 - _CAPTION_PADDING
        x1 = full_bbox.x1 + _CAPTION_PADDING
        for block in blocks:
            br = pymupdf.Rect(block["bbox"])
            if ref_y <= br.y0 and br.y1 <= best_edge:
                x0 = min(x0, br.x0 - _CAPTION_PADDING)
                x1 = max(x1, br.x1 + _CAPTION_PADDING)

        figure_rect = pymupdf.Rect(x0, ref_y, x1, best_edge)

    if figure_rect.is_empty or figure_rect.height < _MIN_FIGURE_HEIGHT:
        return None
    return figure_rect & page_rect


def _is_blank_pixmap(pix: pymupdf.Pixmap) -> bool:
    """
    True when the pixmap contains no detectable content — all sampled pixels
    have every channel at or above _BLANK_MIN_CHANNEL_VALUE (near-white).
    Samples ~500 evenly-spaced pixels to keep cost O(1) regardless of size.
    """
    total_pixels = pix.width * pix.height
    if total_pixels == 0:
        return True
    n = pix.n
    samples = pix.samples
    stride = max(1, total_pixels // 500) * n
    for i in range(0, len(samples) - n + 1, stride):
        for c in range(n):
            if samples[i + c] < _BLANK_MIN_CHANNEL_VALUE:
                return False
    return True


def _pad_and_clip(
    rect: pymupdf.Rect, padding: float, page_rect: pymupdf.Rect
) -> pymupdf.Rect:
    """Expand rect by padding on all sides, clamped to the page rectangle."""
    padded = pymupdf.Rect(
        rect.x0 - padding,
        rect.y0 - padding,
        rect.x1 + padding,
        rect.y1 + padding,
    )
    return padded & page_rect


def _render_figure(page, figure_rect: pymupdf.Rect) -> bytes | None:
    """
    Render a page region as a PNG at _RENDER_DPI, halving DPI once if over
    budget.  Returns None if the rendered pixmap is blank (all near-white
    pixels), which indicates the inferred region captured empty whitespace
    rather than actual figure content.
    """
    for dpi in (_RENDER_DPI, _RENDER_DPI_FALLBACK):
        try:
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(clip=figure_rect, matrix=mat, alpha=False)
            if _is_blank_pixmap(pix):
                return None
            raw = pix.tobytes("png")
            if len(raw) <= _MAX_IMAGE_BYTES:
                return raw
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────────────────────
# PDF title extraction
# ──────────────────────────────────────────────────────────────

def _extract_pdf_title(doc) -> str:
    """
    Heuristic: the document title is the largest-font text on page 1.
    Collects all spans at or near the max font size (within _TITLE_FONT_TOLERANCE)
    and joins them. Returns "" if the document has no pages or no text.
    """
    if doc.page_count == 0:
        return ""
    spans = [
        span
        for block in doc[0].get_text("dict")["blocks"]
        if block["type"] == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    if not spans:
        return ""
    max_size = max(s.get("size", 0.0) for s in spans)
    title = " ".join(
        s["text"].strip()
        for s in spans
        if s.get("size", 0.0) >= max_size * _TITLE_FONT_TOLERANCE
    )
    return title[:120].strip()


# ──────────────────────────────────────────────────────────────
# Figure-ID injection and chunk-image mapping
# ──────────────────────────────────────────────────────────────

def _inject_figure_ids(md: str, images: list[ImageRecord]) -> str:
    """
    Append [FIGURE_ID: N] to the first line in the markdown that matches
    each extracted image's leading 'Figure N' / 'Table N' prefix.
    Anchoring to ^ (MULTILINE) avoids tagging mid-sentence references.
    Handles optional bold (**) wrapping added by pymupdf4llm, and normalizes
    whitespace so captions like 'Figure  3' (double-space from raw PDF blocks)
    still match 'Figure 3' in the markdown.
    """
    for img in images:
        m = re.match(
            r"((?:fig\.?\s*|figure\s*|chart\s*|table\s*)\d+)",
            img.caption,
            re.IGNORECASE,
        )
        if not m:
            continue
        # Split on whitespace so "Figure  3" → ["Figure", "3"], then join with
        # \s+ so the pattern matches any run of spaces between keyword and number.
        # Group 1 captures optional leading ** (bold markdown); group 2 captures
        # the rest of the line so the substitution can append FIGURE_ID after it.
        parts = [re.escape(p) for p in re.split(r'\s+', m.group(1).strip()) if p]
        prefix_pattern = r'\s+'.join(parts)
        pattern = re.compile(
            r"^(\*{0,2})(" + prefix_pattern + r"\b[^\n]*)",
            re.IGNORECASE | re.MULTILINE,
        )
        md = pattern.sub(
            lambda match, n=img.index: match.group(1) + match.group(2) + f" [FIGURE_ID: {n}]",
            md,
            count=1,
        )
    return md


def _build_chunk_images(chunks: list[str]) -> list[list[int]]:
    """Return a list parallel to chunks; each entry is the figure IDs found in that chunk."""
    return [
        [int(m.group(1)) for m in _FIGURE_ID_RE.finditer(chunk)]
        for chunk in chunks
    ]
