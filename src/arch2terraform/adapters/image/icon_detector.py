"""
Stage 2a — Service icon locator.

Finds candidate service icon bounding boxes within the diagram by detecting
distinct non-background, non-container-colour blobs. Each candidate is
returned as an IconCandidate (bbox + BGR crop) for downstream phash matching.

Algorithm
---------
1. Threshold grayscale to isolate foreground (non-white background).
2. Mask out known container-border colours already processed by Stage 1.
3. Morphological close to unify fragmented icon bodies.
4. Find external contours → filter by size, aspect ratio, fill.
5. Return crops ready for hash_matcher.match_icon().

Tuning notes
------------
* min_side / max_side bracket the expected icon size in pixels. Standard AWS
  Architecture diagrams render service icons at 40–80 px on a 1920px canvas;
  adjust for higher-DPI exports.
* container border colours are taken from layout_detector.ColorProfile HSV ranges
  and dilated by 5 px to catch anti-aliased fringe pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from arch2terraform.schemas.diagram import BoundingBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IconCandidate:
    """One candidate service icon extracted from the diagram."""
    bbox: BoundingBox
    crop: np.ndarray  # BGR, as loaded by cv2 — convert to PIL before phash


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_icon_candidates(
    bgr: np.ndarray,
    *,
    min_side: int = 28,
    max_side: int = 160,
    min_fill: float = 0.25,
    bg_threshold: int = 240,
    arrow_threshold: int = 60,
    max_aspect: float = 2.0,
) -> list[IconCandidate]:
    """
    Detect candidate service icon bounding boxes in a BGR diagram image.

    Foreground definition: pixels with grayscale in (arrow_threshold, bg_threshold).
    This excludes both:
      • white/near-white background (gray ≥ bg_threshold = 240)
      • near-black arrows/text     (gray ≤ arrow_threshold = 60)

    AWS icon backgrounds are coloured squares in the 80-220 grayscale range,
    so they fall cleanly inside this band. Excluding dark pixels prevents arrows
    from bridging icon blobs into one large connected component.

    Parameters
    ----------
    bgr             : BGR image array as returned by cv2.imread().
    min_side        : minimum side length (px).
    max_side        : maximum side length (px).
    min_fill        : minimum foreground fraction of bounding-box area.
    bg_threshold    : upper grayscale bound for foreground (default 240).
    arrow_threshold : lower grayscale bound for icon foreground (default 60).
                      Near-black arrows (gray ≈ 30) are excluded; coloured
                      icon backgrounds (gray ≈ 100-200) are included.
    max_aspect      : maximum long:short side ratio (icons are roughly square).

    Returns
    -------
    List of IconCandidate, each with .bbox and .crop (BGR array).
    """
    H, W = bgr.shape[:2]

    # ── Step 1: foreground mask — coloured midtone pixels only ───────────
    # Excludes white background (too bright) AND near-black arrows (too dark).
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg_mask = ((gray > arrow_threshold) & (gray < bg_threshold)).astype(np.uint8) * 255

    # ── Step 2: morphological close to unify icon bodies ─────────────────
    # Kernel 5 px bridges small internal gaps (e.g., gaps between icon
    # graphic elements) without merging icons that are ≥ 10 px apart.
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ── Step 3: find all contours (RETR_LIST, not RETR_EXTERNAL) ─────────
    # RETR_EXTERNAL returns only the outermost contour — here that would be
    # the solid navy AWS-Cloud border ring, making every icon an interior hole
    # and invisible. RETR_LIST retrieves every contour at all nesting levels;
    # the size/aspect/fill filters below handle the rest.
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[IconCandidate] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ── Size filter ───────────────────────────────────────────────────
        if w < min_side or h < min_side:
            continue
        if w > max_side or h > max_side:
            continue

        # ── Aspect-ratio filter (icons ≈ square) ─────────────────────────
        if max(w, h) / max(min(w, h), 1) > max_aspect:
            continue

        # ── Fill check ────────────────────────────────────────────────────
        # Fraction of the bbox that is non-background.
        # Dash segments: ~14 px long × 2 px wide → fill ≈ 2/28 = 0.07 (fails).
        # Service icons: dense coloured square → fill ≥ 0.30 (passes).
        crop_mask = fg_mask[y : y + h, x : x + w]
        fill = np.count_nonzero(crop_mask) / (w * h)
        if fill < min_fill:
            continue

        # ── Clamp to image bounds and build crop ──────────────────────────
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        crop_bgr = bgr[y1:y2, x1:x2].copy()

        candidates.append(
            IconCandidate(
                bbox=BoundingBox(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                ),
                crop=crop_bgr,
            )
        )

    logger.info(
        "Icon detector: %d candidates in %dx%d image "
        "(min_side=%d max_side=%d min_fill=%.2f max_aspect=%.1f)",
        len(candidates), W, H, min_side, max_side, min_fill, max_aspect,
    )
    return candidates
