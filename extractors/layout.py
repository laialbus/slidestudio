"""
Layout analysis via Surya (surya-ocr).

Surya is the primary figure detection path. If the package is missing or
fails to load, LayoutAnalyser.available is False and pdf.py falls back to
the caption-first heuristic.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import pymupdf

_LAYOUT_DPI = 96
_MIN_LAYOUT_CONFIDENCE = 0.5
_MAX_CAPTION_FIGURE_GAP_PT = 80.0
_PANEL_MERGE_GAP_PT = 18.0  # pt — max gap on BOTH axes to fuse composite-figure panels

# Surya label mappings:
#   "Figure"  ← <complex-block> (architecture diagrams, charts, schematics)
#   "Picture" ← <image>         (photographs, raster images)
#   "Caption" ← <caption>       (figure/table caption text)
_FIGURE_LABELS: frozenset[str] = frozenset({"Figure", "Picture"})
_CAPTION_LABELS: frozenset[str] = frozenset({"Caption"})


@dataclass
class LayoutRegion:
    label: str
    bbox: pymupdf.Rect  # PDF point coordinates (72pt = 1 inch)
    confidence: float


class LayoutAnalyser:
    """
    Wraps the Surya layout predictor. Instantiation always succeeds; the
    available property reports whether Surya loaded correctly.
    """

    def __init__(self) -> None:
        self._available = False
        self._predictor = None
        self._load_error: str | None = None
        try:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings as _s
            _fp = FoundationPredictor(checkpoint=_s.LAYOUT_MODEL_CHECKPOINT)
            self._predictor = LayoutPredictor(_fp)
            self._available = True
        except Exception as e:
            # Surface the degradation loudly instead of silently dropping to the
            # heuristic figure path. A silent fallback hides dependency/version
            # breakage (e.g. an incompatible transformers or pillow) — the lite
            # path keeps "working" at lower quality and nobody notices. Capture
            # the reason so the cause is diagnosable, not just the symptom.
            self._load_error = f"{type(e).__name__}: {e}"
            warnings.warn(
                "Surya layout model unavailable — falling back to the "
                "lower-quality heuristic figure-detection path. This usually "
                "means a dependency/version mismatch in the extractor "
                f"environment. Reason: {self._load_error}",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def load_error(self) -> str | None:
        """The captured reason Surya failed to load, or None if it loaded."""
        return self._load_error

    def detect(self, page: pymupdf.Page) -> list[LayoutRegion]:
        """
        Render page at _LAYOUT_DPI, run Surya layout inference, and return
        regions transformed back to PDF point coordinates.

        Returns [] on any error or when Surya is unavailable.
        """
        if not self._available or self._predictor is None:
            return []
        try:
            from PIL import Image
            mat = pymupdf.Matrix(_LAYOUT_DPI / 72, _LAYOUT_DPI / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            results = self._predictor([img])
            scale = 72.0 / _LAYOUT_DPI
            regions: list[LayoutRegion] = []
            for box in results[0].bboxes:
                conf = box.confidence
                if conf is not None and conf < _MIN_LAYOUT_CONFIDENCE:
                    continue
                x0, y0, x1, y1 = box.bbox
                regions.append(LayoutRegion(
                    label=box.label,
                    bbox=pymupdf.Rect(
                        x0 * scale, y0 * scale, x1 * scale, y1 * scale
                    ),
                    confidence=conf if conf is not None else 1.0,
                ))
            return regions
        except Exception:
            return []


def _merge_figure_regions(
    figures: list[LayoutRegion],
    gap_pt: float = _PANEL_MERGE_GAP_PT,
) -> list[LayoutRegion]:
    """
    Fuse figure regions that are adjacent or overlapping on BOTH axes into a
    single region.

    Surya detects each panel of a composite figure (e.g. Figure 2a / 2b / 2c)
    as its own Figure/Picture region. Without merging, the greedy 1:1 caption
    pairing in _pair_figures_captions attaches the caption to one panel and
    orphans the rest, so the composite renders as a single sub-panel.

    Two regions merge when the gap between their bounding boxes is <= gap_pt on
    BOTH the x and y axis (the gap is 0 when they overlap on that axis).
    Requiring a small gap on both axes keeps genuinely separate figures
    elsewhere on the page — which are divided by a caption and whitespace —
    apart. Merging is transitive (union-find), so a row or grid of panels
    collapses into one region.
    """
    n = len(figures)
    if n < 2:
        return list(figures)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = figures[i].bbox, figures[j].bbox
            x_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
            y_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
            if x_gap <= gap_pt and y_gap <= gap_pt:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[LayoutRegion] = []
    for members in groups.values():
        box = pymupdf.Rect(figures[members[0]].bbox)
        conf = figures[members[0]].confidence
        for k in members[1:]:
            box |= figures[k].bbox
            conf = max(conf, figures[k].confidence)
        merged.append(LayoutRegion(label="Figure", bbox=box, confidence=conf))
    return merged


def _pair_figures_captions(
    figures: list[LayoutRegion],
    captions: list[LayoutRegion],
    max_gap_pt: float = _MAX_CAPTION_FIGURE_GAP_PT,
) -> tuple[list[tuple[LayoutRegion, LayoutRegion | None]], list[LayoutRegion]]:
    """
    Nearest-neighbour bipartite matching of figure regions to caption regions.

    Eligibility gate: figure and caption must have positive x-range overlap
    (same column). Distance metric: vertical gap in PDF points between their
    bounding boxes (0 when they touch or overlap vertically). Pairs are
    assigned greedily in ascending gap order; each region is claimed at most
    once.

    Returns:
        matched_pairs  — (figure, caption | None) list; figures with no
                         eligible caption are included as (figure, None)
        unmatched_caps — captions that found no eligible figure; the caller
                         routes these through the heuristic fallback
    """
    if not figures:
        return [], list(captions)

    candidates: list[tuple[float, int, int]] = []
    for fi, fig in enumerate(figures):
        for ci, cap in enumerate(captions):
            x_overlap = (
                min(fig.bbox.x1, cap.bbox.x1) - max(fig.bbox.x0, cap.bbox.x0)
            )
            if x_overlap <= 0:
                continue
            gap = max(
                0.0,
                fig.bbox.y0 - cap.bbox.y1,  # caption above figure
                cap.bbox.y0 - fig.bbox.y1,  # caption below figure
            )
            if gap > max_gap_pt:
                continue
            candidates.append((gap, fi, ci))

    candidates.sort()

    matched_figs: set[int] = set()
    matched_caps: set[int] = set()
    pairs: list[tuple[LayoutRegion, LayoutRegion | None]] = []

    for _, fi, ci in candidates:
        if fi in matched_figs or ci in matched_caps:
            continue
        pairs.append((figures[fi], captions[ci]))
        matched_figs.add(fi)
        matched_caps.add(ci)

    for fi, fig in enumerate(figures):
        if fi not in matched_figs:
            pairs.append((fig, None))

    unmatched_caps = [
        cap for ci, cap in enumerate(captions) if ci not in matched_caps
    ]
    return pairs, unmatched_caps
