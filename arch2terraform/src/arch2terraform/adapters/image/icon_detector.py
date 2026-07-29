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
    container_bboxes: "list | None" = None,
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
    container_bboxes: bounding boxes (objects with .x/.y/.width/.height, e.g.
                      Stage 1's DiagramNode.bbox) of already-detected
                      containers (VPC/subnet/etc). Real bug found 2026-07-27:
                      a container that's rendered with a TINTED FILL (not
                      just an outline) — e.g. a light-blue "Public Subnet"
                      box — has a grayscale value that also falls inside the
                      foreground band, so its fill and any icon sitting
                      inside it merge into ONE contour whose bounding box is
                      the whole container, which then fails `max_side` and
                      the icon is lost entirely (confirmed by reproducing:
                      an EC2 icon sitting on a tinted subnet fill vanished
                      completely, even though checked in isolation its own
                      fill/grayscale stats were fine). Passing the container
                      boxes lets this function sample each one's own
                      background colour and subtract it locally, so the
                      icon inside is freed from a single mega-blob rather
                      than being swallowed by it.

    Returns
    -------
    List of IconCandidate, each with .bbox and .crop (BGR array).
    """
    H, W = bgr.shape[:2]

    # ── Step 1: foreground mask — coloured midtone pixels only ───────────
    # Excludes white background (too bright) AND near-black arrows (too dark).
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg_mask = ((gray > arrow_threshold) & (gray < bg_threshold)).astype(np.uint8) * 255

    # ── Step 1b: subtract each container's own tinted background fill ────
    # (see container_bboxes docstring above). Sample a small patch near
    # each corner of the container (most likely to be plain background,
    # not an icon, a label, or a border line) and treat any pixel in that
    # container's box within a tolerance of that colour as background too.
    for box in (container_bboxes or []):
        bx, by = int(box.x), int(box.y)
        bw, bh = int(box.width), int(box.height)
        bx1, by1 = max(0, bx), max(0, by)
        bx2, by2 = min(W, bx + bw), min(H, by + bh)
        if bx2 - bx1 < 10 or by2 - by1 < 10:
            continue
        inset = 6
        samples = [
            bgr[min(by1 + inset, H - 1), min(bx1 + inset, W - 1)],
            bgr[min(by1 + inset, H - 1), max(bx2 - inset - 1, 0)],
            bgr[max(by2 - inset - 1, 0), min(bx1 + inset, W - 1)],
        ]
        bg_color = np.median(np.stack(samples), axis=0)
        # float32, not int16: a per-channel diff up to 255 squares to 65025,
        # which overflows int16's ±32767 range and wraps to a spurious
        # negative value — sqrt() of a sum containing one then produces NaN
        # (observed as a real RuntimeWarning during testing, not just a
        # theoretical concern).
        region = bgr[by1:by2, bx1:bx2].astype(np.float32)
        dist = np.sqrt(np.sum((region - bg_color.astype(np.float32)) ** 2, axis=2))
        close_to_bg = dist < 18  # small tolerance — real icons differ far more than this
        region_mask = fg_mask[by1:by2, bx1:bx2]
        region_mask[close_to_bg] = 0
        fg_mask[by1:by2, bx1:bx2] = region_mask

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
        # Real bug found 2026-07-27: raw pixel-count fill fraction rejects
        # THIN-OUTLINE icons (e.g. a purple-ring ALB icon with a plain
        # white interior — common in newer/flat AWS icon sets, not just
        # the solid-square style this was originally tuned for) even
        # though they're unambiguously real icons — a thin ring only
        # covers a few percent of its own bounding box. Using the
        # contour's CONVEX HULL area instead of raw pixel count fixes this:
        # a hollow ring's hull is (almost) the full disc it traces out
        # (~78% of a circle inscribed in a square bbox), so it clears
        # min_fill easily, while genuinely sparse noise (dash fragments,
        # stray anti-aliasing) still has a small hull and correctly fails.
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        fill = hull_area / (w * h)
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

    deduped = _dedupe_overlapping(candidates)
    logger.info(
        "Icon detector: %d candidates (%d before dedup) in %dx%d image "
        "(min_side=%d max_side=%d min_fill=%.2f max_aspect=%.1f)",
        len(deduped), len(candidates), W, H, min_side, max_side, min_fill, max_aspect,
    )
    return deduped


def _dedupe_overlapping(candidates: list[IconCandidate], iou_threshold: float = 0.5) -> list[IconCandidate]:
    """
    Real bug found 2026-07-27, introduced by this same session's own
    convex-hull fill fix: `cv2.findContours(..., cv2.RETR_LIST, ...)` on a
    RING-shaped icon (thin outline, hollow interior — see the fill-check
    docstring above) returns TWO nested contours for the one ring — the
    outer edge and the inner edge of the stroke — and both now pass every
    filter (hull-based fill fixed exactly so this class of icon would stop
    being rejected). Without this step, one physical ring icon on the
    diagram would silently become TWO IconCandidates at nearly the same
    bbox, and downstream two DiagramNodes / two classified resources for
    what a human looking at the diagram sees as one ALB. Greedy IOU-based
    suppression: sort by area descending (keep the OUTER contour, whose
    bbox better captures the icon's true visual extent for phash matching),
    drop anything overlapping an already-kept box past `iou_threshold`.
    """
    def _iou(a: IconCandidate, b: IconCandidate) -> float:
        ax1, ay1, ax2, ay2 = a.bbox.x, a.bbox.y, a.bbox.x + a.bbox.width, a.bbox.y + a.bbox.height
        bx1, by1, bx2, by2 = b.bbox.x, b.bbox.y, b.bbox.x + b.bbox.width, b.bbox.y + b.bbox.height
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = a.bbox.width * a.bbox.height
        area_b = b.bbox.width * b.bbox.height
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    ordered = sorted(candidates, key=lambda c: c.bbox.width * c.bbox.height, reverse=True)
    kept: list[IconCandidate] = []
    for cand in ordered:
        if any(_iou(cand, k) > iou_threshold for k in kept):
            continue
        kept.append(cand)
    return kept
