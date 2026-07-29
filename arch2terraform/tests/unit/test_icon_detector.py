"""
Unit tests for Stage 2a — icon_detector.py

Three real bugs found and fixed 2026-07-27, reproduced first against a
synthetic diagram matching a real user-reported failure (EC2/ALB/Amazon-MQ
icons vanishing entirely from a diagram that used a different, valid AWS
icon rendering style than the solid-square style this detector was
originally tuned against):

1. Thin-outline "ring" icons (a coloured stroke with a plain interior — a
   common modern AWS icon style, not just the solid-fill squares this was
   built for) failed the old raw-pixel-count fill-ratio check even though
   they're unambiguously real icons.
2. A solid icon sitting on top of a TINTED container background fill
   (e.g. a light-blue subnet box) merged with that fill into one oversized
   blob that failed the max-size filter, losing the icon completely.
3. Fixing (1) via convex-hull fill measurement introduced a NEW problem:
   `cv2.findContours(..., RETR_LIST, ...)` returns two nested contours for
   one ring (outer + inner edge of the stroke), both now passing every
   filter — so one physical icon became two candidates. Fixed with a
   greedy IOU-based dedup pass.

All tests create synthetic BGR images with cv2 primitives, matching the
style of test_layout_detector.py — no binary fixture files needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2

from arch2terraform.adapters.image.icon_detector import detect_icon_candidates
from arch2terraform.schemas.diagram import BoundingBox


def _white_canvas(width: int = 400, height: int = 400) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def _draw_ring_icon(img: np.ndarray, cx: int, cy: int, r: int = 45,
                     color: tuple = (190, 80, 130), thickness: int = 4) -> None:
    """Thin-outline circular icon with a plain white interior — the ALB-style
    icon that used to vanish entirely."""
    cv2.circle(img, (cx, cy), r, color, thickness)
    # small internal pictogram so it's not a perfectly empty ring
    cv2.line(img, (cx - 15, cy - 10), (cx + 15, cy - 10), (20, 20, 20), 2)
    cv2.line(img, (cx, cy - 10), (cx, cy + 15), (20, 20, 20), 2)


def _draw_solid_icon(img: np.ndarray, x: int, y: int, size: int = 90,
                      color: tuple = (34, 126, 230)) -> None:
    """Solid-fill square icon — the original style this detector was tuned for."""
    cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)


class TestThinOutlineIcons:
    def test_ring_icon_alone_is_detected(self):
        img = _white_canvas()
        _draw_ring_icon(img, 200, 200)

        candidates = detect_icon_candidates(img)

        assert len(candidates) == 1

    def test_ring_icon_is_not_duplicated_by_outer_and_inner_contour(self):
        """Real bug (3) above: RETR_LIST finds both edges of the stroke —
        must collapse to exactly one candidate, not two overlapping ones."""
        img = _white_canvas()
        _draw_ring_icon(img, 200, 200, thickness=6)

        candidates = detect_icon_candidates(img)

        assert len(candidates) == 1

    def test_solid_icon_still_detected_unaffected_by_hull_change(self):
        """Regression guard: switching the fill check to convex-hull area
        must not break detection of the original solid-square style."""
        img = _white_canvas()
        _draw_solid_icon(img, 150, 150)

        candidates = detect_icon_candidates(img)

        assert len(candidates) == 1
        c = candidates[0]
        assert abs(c.bbox.width - 90) <= 2
        assert abs(c.bbox.height - 90) <= 2

    def test_sparse_dash_fragment_still_rejected(self):
        """Regression guard: the hull-based fill fix must not make the
        detector newly permissive about genuinely sparse noise (a lone
        dash fragment, not an icon)."""
        img = _white_canvas()
        cv2.line(img, (100, 100), (114, 102), (0, 113, 237), 2)  # ~14x2 dash

        candidates = detect_icon_candidates(img)

        assert len(candidates) == 0


class TestContainerFillExclusion:
    def test_icon_on_tinted_container_background_is_recovered(self):
        """Real bug (2) above: without telling the detector about the
        container's own bbox, a solid icon sitting on a tinted fill merges
        with that fill into one oversized blob and vanishes."""
        img = _white_canvas(600, 600)
        # Tinted container background (light blue, well within the
        # foreground luminance band — this is the actual failure mode)
        cv2.rectangle(img, (50, 50), (550, 550), (250, 236, 224), -1)
        _draw_solid_icon(img, 250, 250, size=90, color=(34, 126, 230))

        container_bbox = BoundingBox(x=50, y=50, width=500, height=500)

        without_exclusion = detect_icon_candidates(img)
        with_exclusion = detect_icon_candidates(img, container_bboxes=[container_bbox])

        # The icon must NOT be independently recoverable without telling
        # the detector about the container (pins down the bug this test
        # guards against actually existing before the fix).
        icon_sized = [c for c in without_exclusion if 70 <= c.bbox.width <= 110]
        assert icon_sized == []

        icon_sized_fixed = [c for c in with_exclusion if 70 <= c.bbox.width <= 110]
        assert len(icon_sized_fixed) == 1

    def test_no_container_bboxes_behaves_exactly_as_before(self):
        """Backward-compat guard: omitting container_bboxes (the default)
        must not change behavior for diagrams with no tinted containers."""
        img = _white_canvas()
        _draw_solid_icon(img, 150, 150)

        candidates = detect_icon_candidates(img, container_bboxes=None)

        assert len(candidates) == 1


class TestNoNumericWarnings:
    def test_container_exclusion_does_not_raise_numeric_warnings(self, recwarn):
        """Real bug found during testing: the original int16 color-distance
        calculation overflowed and produced NaN via sqrt() of a spurious
        negative value, surfaced as a RuntimeWarning. Must stay fixed."""
        img = _white_canvas(600, 600)
        cv2.rectangle(img, (50, 50), (550, 550), (250, 236, 224), -1)
        _draw_solid_icon(img, 250, 250, size=90, color=(34, 126, 230))
        container_bbox = BoundingBox(x=50, y=50, width=500, height=500)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            detect_icon_candidates(img, container_bboxes=[container_bbox])
