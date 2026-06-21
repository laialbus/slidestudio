"""
Format-agnostic chunking and figure-tag helpers shared by all extractors.

These operate purely on Markdown text and the [FIGURE_ID: N] tag convention, so
both the pymupdf4llm-based extractor and the MinerU-based extractor reuse them
unchanged.  Each extractor is responsible for injecting [FIGURE_ID: N] tags into
its Markdown in whatever way suits its caption format; _build_chunk_images then
recovers per-chunk figure ownership by scanning for those tags.
"""

import re

# Heading boundary: only # and ## (not ### or deeper)
_HEADING_LINE_RE = re.compile(r"^(#{1,2})(?!#)\s", re.MULTILINE)

# Figure ID tag — injected into markdown, scanned back from chunks
_FIGURE_ID_RE = re.compile(r"\[FIGURE_ID:\s*(\d+)\]")


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


def _build_chunk_images(chunks: list[str]) -> list[list[int]]:
    """Return a list parallel to chunks; each entry is the figure IDs found in that chunk."""
    return [
        [int(m.group(1)) for m in _FIGURE_ID_RE.finditer(chunk)]
        for chunk in chunks
    ]
