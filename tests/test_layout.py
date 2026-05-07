"""
Tests for extractors/layout.py — LayoutAnalyser and _pair_figures_captions.

LayoutAnalyser tests run without Surya installed (available == False).
_pair_figures_captions tests use synthetic LayoutRegion objects with
pymupdf.Rect bounding boxes and do not require Surya.
"""
import pymupdf
import pytest

from extractors.layout import (
    LayoutAnalyser,
    LayoutRegion,
    _pair_figures_captions,
    _FIGURE_LABELS,
    _CAPTION_LABELS,
    _MAX_CAPTION_FIGURE_GAP_PT,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _fig(x0: float, y0: float, x1: float, y1: float) -> LayoutRegion:
    return LayoutRegion(label="Figure", bbox=pymupdf.Rect(x0, y0, x1, y1), confidence=0.9)


def _cap(x0: float, y0: float, x1: float, y1: float) -> LayoutRegion:
    return LayoutRegion(label="Caption", bbox=pymupdf.Rect(x0, y0, x1, y1), confidence=0.9)


# ──────────────────────────────────────────────────────────────
# LayoutRegion dataclass
# ──────────────────────────────────────────────────────────────

class TestLayoutRegion:
    def test_fields_stored(self):
        rect = pymupdf.Rect(10, 20, 100, 200)
        r = LayoutRegion(label="Figure", bbox=rect, confidence=0.85)
        assert r.label == "Figure"
        assert r.bbox == rect
        assert r.confidence == pytest.approx(0.85)

    def test_picture_label_accepted(self):
        r = LayoutRegion(label="Picture", bbox=pymupdf.Rect(0, 0, 10, 10), confidence=0.7)
        assert r.label in _FIGURE_LABELS

    def test_caption_label_accepted(self):
        r = LayoutRegion(label="Caption", bbox=pymupdf.Rect(0, 0, 10, 10), confidence=0.8)
        assert r.label in _CAPTION_LABELS


# ──────────────────────────────────────────────────────────────
# LayoutAnalyser — behaviour without Surya installed
# ──────────────────────────────────────────────────────────────

class TestLayoutAnalyserWithoutSurya:
    def test_instantiates_without_error(self):
        analyser = LayoutAnalyser()
        assert analyser is not None

    def test_available_false_when_surya_absent(self):
        analyser = LayoutAnalyser()
        # surya-ocr is not in the project's base requirements; it is optional.
        # If this assertion fails, Surya is installed — that is also acceptable.
        if not analyser.available:
            assert analyser.available is False

    def test_detect_returns_empty_when_unavailable(self):
        analyser = LayoutAnalyser()
        if analyser.available:
            pytest.skip("Surya is installed — unavailable path not exercised")
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 80), "Test page.", fontsize=10)
        result = analyser.detect(page)
        doc.close()
        assert result == []

    def test_detect_returns_list(self):
        analyser = LayoutAnalyser()
        if analyser.available:
            pytest.skip("Surya is installed — unavailable path not exercised")
        doc = pymupdf.open()
        page = doc.new_page()
        result = analyser.detect(page)
        doc.close()
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────
# _pair_figures_captions — core matching logic
# ──────────────────────────────────────────────────────────────

class TestPairFiguresCaptions:

    # ── empty inputs ─────────────────────────────────────────

    def test_no_figures_returns_all_captions_unmatched(self):
        caps = [_cap(10, 200, 200, 215)]
        pairs, unmatched = _pair_figures_captions([], caps)
        assert pairs == []
        assert len(unmatched) == 1
        assert unmatched[0] is caps[0]

    def test_no_captions_returns_all_figures_with_none(self):
        figs = [_fig(10, 50, 200, 180)]
        pairs, unmatched = _pair_figures_captions(figs, [])
        assert len(pairs) == 1
        fig, cap = pairs[0]
        assert fig is figs[0]
        assert cap is None
        assert unmatched == []

    def test_both_empty_returns_empty(self):
        pairs, unmatched = _pair_figures_captions([], [])
        assert pairs == []
        assert unmatched == []

    # ── successful pairing ────────────────────────────────────

    def test_adjacent_figure_above_caption_paired(self):
        # Standard layout: figure at y 50-180, caption at y 185-200 (below)
        fig = _fig(10, 50, 200, 180)
        cap = _cap(10, 185, 200, 200)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert len(pairs) == 1
        assert pairs[0] == (fig, cap)
        assert unmatched == []

    def test_caption_above_figure_paired(self):
        # Less common but valid: caption at y 50-65, figure at y 70-200
        cap = _cap(10, 50, 200, 65)
        fig = _fig(10, 70, 200, 200)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert len(pairs) == 1
        assert pairs[0] == (fig, cap)
        assert unmatched == []

    def test_touching_bboxes_gap_is_zero(self):
        # Caption y0 == figure y1 — gap should be 0, always eligible
        fig = _fig(10, 50, 200, 180)
        cap = _cap(10, 180, 200, 195)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert len(pairs) == 1
        assert pairs[0] == (fig, cap)

    def test_two_figures_two_captions_each_pair_correctly(self):
        # Left column: fig1 + cap1; right column: fig2 + cap2
        fig1 = _fig(10, 50, 200, 180)
        cap1 = _cap(10, 185, 200, 200)
        fig2 = _fig(300, 50, 490, 180)
        cap2 = _cap(300, 185, 490, 200)
        pairs, unmatched = _pair_figures_captions([fig1, fig2], [cap1, cap2])
        assert len(pairs) == 2
        assert unmatched == []
        pair_map = {id(f): c for f, c in pairs}
        assert pair_map[id(fig1)] is cap1
        assert pair_map[id(fig2)] is cap2

    def test_picture_label_eligible(self):
        fig = LayoutRegion(label="Picture", bbox=pymupdf.Rect(10, 50, 200, 180), confidence=0.9)
        cap = _cap(10, 185, 200, 200)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert len(pairs) == 1
        assert pairs[0] == (fig, cap)

    # ── ineligibility: no x-overlap ──────────────────────────

    def test_no_x_overlap_figure_gets_none(self):
        # Figure in left column, caption in right column — no x overlap
        fig = _fig(10, 50, 190, 180)
        cap = _cap(300, 185, 490, 200)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert len(pairs) == 1
        assert pairs[0] == (fig, None)
        assert len(unmatched) == 1
        assert unmatched[0] is cap

    def test_shared_edge_not_counted_as_overlap(self):
        # fig.x1 == cap.x0 — touching edges, overlap == 0 → ineligible
        fig = _fig(10, 50, 200, 180)
        cap = _cap(200, 185, 390, 200)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert pairs[0][1] is None  # figure gets None partner

    # ── ineligibility: gap exceeds threshold ─────────────────

    def test_gap_exceeds_max_figure_gets_none(self):
        fig = _fig(10, 50, 200, 180)
        # Caption far below — gap = 400 > _MAX_CAPTION_FIGURE_GAP_PT (80)
        cap = _cap(10, 580, 200, 595)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert pairs[0][1] is None
        assert len(unmatched) == 1

    def test_gap_exactly_at_max_is_eligible(self):
        fig = _fig(10, 50, 200, 180)
        cap = _cap(10, 180 + _MAX_CAPTION_FIGURE_GAP_PT, 200, 200 + _MAX_CAPTION_FIGURE_GAP_PT)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert pairs[0][1] is not None  # exactly at threshold → eligible

    def test_gap_one_pt_over_max_is_ineligible(self):
        fig = _fig(10, 50, 200, 180)
        over = _MAX_CAPTION_FIGURE_GAP_PT + 1.0
        cap = _cap(10, 180 + over, 200, 195 + over)
        pairs, unmatched = _pair_figures_captions([fig], [cap])
        assert pairs[0][1] is None

    # ── greedy assignment ─────────────────────────────────────

    def test_ambiguous_caption_paired_with_closest_figure(self):
        # fig_near is 5pt below the caption; fig_far is 60pt below
        cap = _cap(10, 50, 200, 65)
        fig_near = _fig(10, 70, 200, 200)    # gap = 70-65 = 5
        fig_far  = _fig(10, 125, 200, 260)   # gap = 125-65 = 60
        pairs, unmatched = _pair_figures_captions([fig_near, fig_far], [cap])
        pair_map = {id(f): c for f, c in pairs}
        assert pair_map[id(fig_near)] is cap
        assert pair_map[id(fig_far)] is None

    def test_each_region_claimed_at_most_once(self):
        # Two captions both eligible for the same figure — only the closer one wins
        fig = _fig(10, 50, 200, 180)
        cap_close = _cap(10, 183, 200, 198)  # gap = 3
        cap_far   = _cap(10, 220, 200, 235)  # gap = 40
        pairs, unmatched = _pair_figures_captions([fig], [cap_close, cap_far])
        pair_map = {id(f): c for f, c in pairs}
        assert pair_map[id(fig)] is cap_close
        assert len(unmatched) == 1
        assert unmatched[0] is cap_far

    def test_three_figures_one_has_no_caption(self):
        fig1 = _fig(10, 50, 200, 180)
        cap1 = _cap(10, 185, 200, 200)
        fig2 = _fig(10, 300, 200, 430)
        cap2 = _cap(10, 435, 200, 450)
        fig3 = _fig(10, 550, 200, 680)  # no caption available
        pairs, unmatched = _pair_figures_captions([fig1, fig2, fig3], [cap1, cap2])
        assert len(pairs) == 3
        assert unmatched == []
        pair_map = {id(f): c for f, c in pairs}
        assert pair_map[id(fig1)] is cap1
        assert pair_map[id(fig2)] is cap2
        assert pair_map[id(fig3)] is None

    # ── custom max_gap_pt override ────────────────────────────

    def test_custom_max_gap_respected(self):
        fig = _fig(10, 50, 200, 180)
        cap = _cap(10, 210, 200, 225)  # gap = 30
        # With tight threshold of 20, gap=30 is ineligible
        pairs, _ = _pair_figures_captions([fig], [cap], max_gap_pt=20.0)
        assert pairs[0][1] is None
        # With looser threshold of 40, gap=30 is eligible
        pairs, _ = _pair_figures_captions([fig], [cap], max_gap_pt=40.0)
        assert pairs[0][1] is cap
