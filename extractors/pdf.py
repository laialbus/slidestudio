import re
from statistics import mode, StatisticsError

import pymupdf

# CHUNK_SIZE     = 8_000
# OVERLAP_SIZE     = 1_500
BLOCK_TYPE_IMAGE = 1

# Equation-signal patterns for context-based classification.
_EQ_LABEL_RE   = re.compile(r"\b(eq\.?\s*\d+|equation\s*\d+)\b", re.IGNORECASE)
_EQ_CONTEXT_RE = re.compile(r"\b(where|in which)\b", re.IGNORECASE)


class PDFExtractor:
    def __init__(self, chunk_size: int, overlap_size: int):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def extract(self, file_path: str) -> dict:
        """
        Returns {"headers": list[str], "chunks": list[str]}.
        Headers are extracted via dynamic font-size detection.
        Chunks are overlapping with paragraph-snap boundaries.
        Image and table blocks are replaced with structured placeholders.
        """
        doc     = pymupdf.open(file_path)
        text    = self._read_pdf(doc)
        headers = self._extract_headers(doc)
        chunks  = self._chunk(text)
        doc.close()
        return {"headers": headers, "chunks": chunks}

    def _read_pdf(self, doc) -> str:
        pages_text = []

        for page in doc:
            blocks    = page.get_text("dict")["blocks"]
            page_text = []

            # Detect table regions via find_tables (PyMuPDF 1.23+).
            # Store (bbox, placeholder) per table so we can emit the placeholder
            # once when the first overlapping text block is encountered, then
            # suppress the raw cell text to avoid duplication.
            table_regions: dict[int, tuple] = {}
            emitted_tables: set[int]        = set()
            try:
                for idx, table in enumerate(page.find_tables().tables):
                    rows   = table.extract()
                    n_rows = len(rows)
                    n_cols = len(rows[0]) if rows else 0
                    hint   = ""
                    if rows:
                        hint = " | ".join(str(cell or "").strip() for cell in rows[0])
                    ph = (
                        f"[TABLE EXCLUDED: {n_rows} rows \u00d7 {n_cols} cols"
                        + (f" | columns: {hint}" if hint else "")
                        + "]"
                    )
                    table_regions[idx] = (table.bbox, ph)
            except Exception:
                pass

            for i, block in enumerate(blocks):
                if block["type"] == BLOCK_TYPE_IMAGE:
                    ph = _classify_image_block(block, i, blocks, page.rect.width)
                    page_text.append(ph)

                elif block["type"] == 0:
                    bbx = block["bbox"]
                    # Emit table placeholder on first text block inside the table,
                    # then skip all text blocks inside the same table region.
                    table_hit = next(
                        (tid for tid, (tbx, _) in table_regions.items()
                         if _overlaps(bbx, tbx)),
                        None,
                    )
                    if table_hit is not None:
                        if table_hit not in emitted_tables:
                            page_text.append(table_regions[table_hit][1])
                            emitted_tables.add(table_hit)
                        continue

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                page_text.append(text)

            pages_text.append(" ".join(page_text))

        return "\n".join(pages_text)

    def _extract_headers(self, doc) -> list[str]:
        """
        Dynamically detects headers by comparing each span's font size to the
        statistical mode of all font sizes on the page (the body text size).
        A span is a header if its size exceeds the mode, OR if its size equals
        the mode but the font is bold — catching documents that use weight rather
        than size to mark headings.
        """
        headers = []

        for page in doc:
            blocks = page.get_text("dict")["blocks"]

            all_sizes = [
                span["size"]
                for block in blocks
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]

            if not all_sizes:
                continue

            try:
                body_size = mode(all_sizes)
            except StatisticsError:
                body_size = min(all_sizes)

            for block in blocks:
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    is_header = any(
                        s["size"] > body_size or (s["size"] == body_size and "bold" in s.get("font", "").lower())
                        for s in spans
                    )
                    if is_header:
                        line_text = " ".join(s["text"].strip() for s in spans)
                        if line_text:
                            headers.append(line_text)

        return headers

    def _chunk(self, text: str) -> list[str]:
        """
        Splits text into overlapping chunks of at most self.chunk_size characters.
        Each boundary snaps to the nearest paragraph break (double newline).
        The next chunk begins self.overlap_size characters before the previous
        chunk's end, creating a sliding window that prevents severing
        continuous thoughts at hard edges.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start  = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1:
                    end = paragraph_break

            chunks.append(text[start:end].strip())

            # Move start back by self.overlap_size for the sliding window.
            # The max() guard prevents an infinite loop when no paragraph
            # break was found and the window cannot advance.
            start = max(end - self.overlap_size, start + 1)

        return [c for c in chunks if c]


def _get_block_text(block: dict) -> str:
    return " ".join(
        span["text"].strip()
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )


def _classify_image_block(block: dict, idx: int, blocks: list, page_width: float) -> str:
    """
    Returns [FIGURE EXCLUDED: ...] or [EQUATION: ...] for an image block.

    Decision order (false negatives preferred over false positives):
    1. Next text block starts with a figure marker → figure.
    2. Surrounding text contains an equation label or variable description → equation.
    3. Block is narrower than 80 % of page width AND taller than wide → equation.
    4. Default → figure.
    """
    # 1 — Figure caption in the immediately following text block
    if idx + 1 < len(blocks) and blocks[idx + 1]["type"] == 0:
        candidate = _get_block_text(blocks[idx + 1])
        if candidate.lower().startswith(("fig", "figure", "chart", "plate")):
            return f'[FIGURE EXCLUDED: "{candidate[:80]}"]'

    # 2 — Equation signals in surrounding text
    prev_text = (
        _get_block_text(blocks[idx - 1])
        if idx > 0 and blocks[idx - 1]["type"] == 0
        else ""
    )
    next_text = (
        _get_block_text(blocks[idx + 1])
        if idx + 1 < len(blocks) and blocks[idx + 1]["type"] == 0
        else ""
    )
    context = f"{prev_text} {next_text}".strip()

    label_match   = _EQ_LABEL_RE.search(context)
    context_match = _EQ_CONTEXT_RE.search(context)

    if label_match or context_match:
        caption = (label_match.group(0) if label_match else context)[:80]
        return f"[EQUATION: {caption}]"

    # 3 — Size heuristic: narrow and taller than it is wide
    x0, y0, x1, y1 = block["bbox"]
    width  = x1 - x0
    height = y1 - y0
    if page_width > 0 and width < 0.8 * page_width and height > width:
        return "[EQUATION: no caption detected]"

    # 4 — Default: figure
    return "[FIGURE EXCLUDED: no caption detected]"


def _overlaps(a, b) -> bool:
    """True when rectangles a and b share any area."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
