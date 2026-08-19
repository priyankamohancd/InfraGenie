"""
Stage 1 — Layout / boundary-box detector.

Detects the container hierarchy in a raster architecture diagram using
classical computer vision (OpenCV).

COLOR-AGNOSTIC REDESIGN (2026-07-28)
-------------------------------------
Root cause found 2026-07-27/28: a real user diagram's container borders were
verified (via direct pixel sampling guided by Canny edge detection) to be a
light, low-saturation blue (BGR≈200,150,130 → HSV H≈109-111, S≈50-70,
V≈200-234) that falls OUTSIDE every one of the 5 hardcoded
``DEFAULT_COLOR_PROFILES`` HSV ranges below — so Stage 1 found zero
containers on that diagram (and, downstream, the classifier never saw a VPC,
subnets, or the EKS cluster boundary at all). AWS diagrams "in the wild" are
routinely re-themed, hand-drawn, or exported from tools with different
default palettes, so any FIXED set of color ranges is inherently brittle.

Detection is now primarily STRUCTURAL, not color-based: Stage 1 finds
straight border segments with a Canny + probabilistic Hough line transform,
merges collinear fragments, and assembles matching horizontal/vertical line
quadruples into rectangle candidates — completely independent of hue. This
was verified empirically against the real failing diagram: the Hough-line
approach correctly recovered the true VPC boundary (and nested subnet / EKS
boundaries) that the color-profile approach missed entirely.

Color is now used only as a SOFT hint, applied after a rectangle is already
structurally confirmed: sampled border-pixel colors are matched against the
well-known AWS palette below (``DEFAULT_COLOR_PROFILES``) to assign a
specific ``ContainerType`` (VPC/AWS_CLOUD/AZ/SUBNET) when they match a known
convention. When they don't match any known color, the region is still kept
— never silently dropped — as ``ContainerType.UNKNOWN``, to be resolved
later by OCR label-text matching in the classifier (mirroring how
unclassified icon nodes are already handled, never dropped, elsewhere in the
pipeline).

Known AWS diagram color/style conventions (used only for the soft type hint):

    Container type       Color                Hex       Style
    ───────────────────  ───────────────────  ────────  ──────
    AWS Cloud            Dark navy            #242F3E   solid
    VPC                  Green  (classic)     #7AA116   solid
    VPC                  Purple (new style)   #8C4FFF   solid
    Availability Zone    Teal                 #00A4A6   dashed
    Subnet (any)         Orange               #ED7100   dashed
    Security Group       Orange               #ED7100   dashed, smaller

Color values sourced from Architecture-Group-Icons SVG files (April 2026 pack).

Output
------
``detect_containers(image_path)`` returns a list of ``DiagramNode`` objects
with ``shape=NodeShape.CONTAINER``, ``bbox`` in pixel coordinates, and
``parent_id`` set to the ID of the immediately enclosing container.

These nodes are ready to be merged with the icon-detection output from
Stages 2/3 into a complete ``ParsedDiagram``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from arch2terraform.schemas.diagram import BoundingBox, DiagramNode, NodeShape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Container type taxonomy
# ---------------------------------------------------------------------------

class ContainerType(str, Enum):
    AWS_CLOUD          = "aws_cloud"
    VPC                = "vpc"
    AVAILABILITY_ZONE  = "availability_zone"
    SUBNET             = "subnet"
    SECURITY_GROUP     = "security_group"
    UNKNOWN            = "unknown"


# Map container type → image_ref string used in DiagramNode.
# The classifier downstream uses this to decide which Terraform resource
# to emit (VPC, subnet, etc.).
_CONTAINER_IMAGE_REF: dict[ContainerType, str] = {
    ContainerType.AWS_CLOUD:         "AWS-Cloud",
    ContainerType.VPC:               "Virtual-private-cloud-VPC",
    ContainerType.AVAILABILITY_ZONE: "Availability-Zone",
    ContainerType.SUBNET:            "Subnet",
    ContainerType.SECURITY_GROUP:    "Security-Group",
    ContainerType.UNKNOWN:           "Unknown-Container",
}


# ---------------------------------------------------------------------------
# Color profiles
# ---------------------------------------------------------------------------
# Each profile defines an HSV range (OpenCV: H 0-179, S 0-255, V 0-255)
# and the container type it implies.
#
# Two profiles can map to the same ContainerType to handle both old and new
# AWS diagram styles in a single pass.

@dataclass
class ColorProfile:
    name:           str
    hsv_lower:      np.ndarray
    hsv_upper:      np.ndarray
    container_type: ContainerType
    expected_style: str = "solid"   # "solid" | "dashed" | "either"


# Default profiles — derived from the AWS icon pack SVG colors.
DEFAULT_COLOR_PROFILES: list[ColorProfile] = [
    # AWS Cloud outer boundary — dark navy #242F3E
    ColorProfile(
        name           = "dark_cloud",
        hsv_lower      = np.array([95,  30,  15]),
        hsv_upper      = np.array([125, 180, 100]),
        container_type = ContainerType.AWS_CLOUD,
        expected_style = "solid",
    ),
    # VPC — classic green #7AA116
    ColorProfile(
        name           = "green_vpc",
        hsv_lower      = np.array([28,  80,  50]),
        hsv_upper      = np.array([52, 255, 230]),
        container_type = ContainerType.VPC,
        expected_style = "solid",
    ),
    # VPC — newer purple style #8C4FFF
    ColorProfile(
        name           = "purple_vpc",
        hsv_lower      = np.array([120, 80, 120]),
        hsv_upper      = np.array([155, 255, 255]),
        container_type = ContainerType.VPC,
        expected_style = "solid",
    ),
    # Availability Zone — teal #00A4A6
    ColorProfile(
        name           = "teal_az",
        hsv_lower      = np.array([80, 100,  60]),
        hsv_upper      = np.array([105, 255, 230]),
        container_type = ContainerType.AVAILABILITY_ZONE,
        expected_style = "dashed",
    ),
    # Subnet / Security Group / Auto-Scaling — orange #ED7100
    ColorProfile(
        name           = "orange_subnet",
        hsv_lower      = np.array([ 4, 120,  80]),
        hsv_upper      = np.array([25, 255, 255]),
        container_type = ContainerType.SUBNET,
        expected_style = "dashed",
    ),
]


# ---------------------------------------------------------------------------
# Internal detection result (richer than DiagramNode, for ranking/filtering)
# ---------------------------------------------------------------------------

@dataclass
class ContainerRegion:
    """Raw detection result before hierarchy assignment."""
    node_id:        str
    bbox:           BoundingBox
    container_type: ContainerType
    color_profile:  str             # name of matching ColorProfile
    border_style:   str             # "solid" | "dashed"
    fill_ratio:     float           # fraction of border pixels that are colored
    area:           float           # bbox area in pixels²
    parent_id:      str | None = None
    extra:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """All tunable knobs in one place."""

    # ---- Preprocessing ----
    blur_kernel_size: int = 3       # Gaussian blur before color thresholding

    # ---- Morphological closing (connects dash gaps) ----
    # Morphological close kernel and iteration count.
    # kernel=15 bridges dash gaps up to ~14px (typical AWS diagram dash gap ≈10-12px).
    # Keep iterations=1: a second close pass extends bridging reach enough to merge
    # adjacent containers that are 30-40px apart, which we must NOT do.
    close_kernel_size: int = 15
    close_iterations:  int = 1

    # ---- Contour filtering ----
    min_bbox_area:     int = 5_000  # px² — ignore tiny fragments
    min_side_length:   int = 40     # px — ignore very thin strips
    max_bbox_fraction: float = 0.98 # ignore rectangles that fill almost the whole image

    # ---- Icon/container color collision guard ----
    # VPC (green/purple) and AWS Cloud (navy) all use *solid* borders — and so
    # do several individual AWS4 resource icons (e.g. Application Load
    # Balancer, Internet Gateway, and Route 53 all render with purple or
    # green circular strokes). A single icon glyph can satisfy every other
    # contour filter above (area, side length) on a large enough source
    # image, and gets misdetected as its own VPC/Cloud container — found
    # 2026-07-24 via a real hand-composited test PNG where 3 of 5 "VPC"
    # detections turned out to be 90x90px squares (individual icon
    # outlines), producing 3 phantom aws_vpc resources downstream.
    # Real container boundaries always enclose other diagram content, so
    # they are (a) far larger than any icon — Stage 2a's own icon detector
    # caps icon side length at 160px, so 200x200=40,000px² is a safe floor
    # — and (b) rarely near-square, unlike icon glyphs which are always
    # drawn in a square bounding box. A region is treated as icon-shaped
    # (and excluded from VPC/AWS_CLOUD classification) only when BOTH
    # conditions hold, so a genuinely square-but-large VPC is never dropped.
    icon_collision_max_area:       float = 40_000  # px² (~200x200) — below this, could be an icon
    icon_collision_aspect_tolerance: float = 0.35  # |w-h|/max(w,h) <= this counts as "square"

    # ---- Dashed-line detection ----
    # Sample a strip this wide along each bbox edge in the original (pre-dilation) mask.
    dash_sample_width: int = 6      # px on each side of the nominal edge
    # Fill ratio below this → dashed; above this → solid
    dash_threshold:    float = 0.60

    # ---- Polygon approximation ----
    poly_epsilon_frac: float = 0.02  # fraction of perimeter for approxPolyDP

    # ---- Containment hierarchy ----
    # A box B is considered "inside" box A when B's bbox overlaps A's bbox by
    # at least this fraction of B's area (handles slight pixel overshoot).
    containment_overlap_frac: float = 0.85

    # Size ratio: a box must be this much smaller than its parent candidate.
    containment_size_ratio: float = 0.90

    color_profiles: list[ColorProfile] = field(
        default_factory=lambda: list(DEFAULT_COLOR_PROFILES)
    )

    # ---- Structural (color-agnostic) edge detection ----
    # See module docstring "COLOR-AGNOSTIC REDESIGN". Canny edge map fed to
    # a probabilistic Hough transform to find straight border segments,
    # independent of border hue/saturation/value.
    edge_canny_lo:            int = 30
    edge_canny_hi:            int = 100
    edge_hough_threshold:     int = 60    # HoughLinesP accumulator threshold
    edge_hough_min_len:       int = 80    # px — minimum raw segment length
    edge_hough_max_gap:       int = 25    # px — max gap to bridge into one segment
    edge_line_axis_tol:       int = 4     # px — max perpendicular deviation to count as "straight"
    # Only consider merged lines at least this fraction of the image's
    # width/height as candidate container walls — filters out short
    # fragments from icons, text, and arrows before the (expensive)
    # rectangle-assembly pass.
    edge_min_line_frac:       float = 0.15
    edge_merge_gap:           int = 30    # px — bridge gaps this size when merging collinear segments
    edge_merge_axis_tol:      int = 6     # px — bucket tolerance across the perpendicular axis
    # When assembling a top/bottom horizontal pair into a rectangle, a
    # vertical line is accepted as that rectangle's left/right wall if it
    # passes within this many px of the target x (or y, for horizontals)...
    edge_corner_tol:          int = 25
    # ...AND its span overlaps the opposite pair's span by at least this
    # fraction (handles Hough fragmenting near corners rather than
    # requiring the wall to reach the exact corner pixel).
    edge_corner_overlap_frac: float = 0.6
    # Minimum "weakest side" edge-pixel density (on a dilated edge map)
    # required to accept an assembled rectangle — real container walls
    # score well above this; spurious line-quadruple combinations
    # (unrelated lines that happen to loosely align) score far lower,
    # verified empirically against the real failing diagram.
    edge_fill_min_ratio:      float = 0.30
    edge_dilate_kernel:       int = 3


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def _build_color_mask(hsv: np.ndarray, profile: ColorProfile) -> np.ndarray:
    """Binary mask for pixels matching this color profile's HSV range."""
    mask = cv2.inRange(hsv, profile.hsv_lower, profile.hsv_upper)
    # Orange wraps around H=0 in some lighting conditions — add a second range.
    if profile.name == "orange_subnet":
        wrap = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([4, 255, 255]))
        mask = cv2.bitwise_or(mask, wrap)
    return mask


def _close_mask(mask: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Morphological close: connects dash gaps so dashed rectangles become solid shapes."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (cfg.close_kernel_size, cfg.close_kernel_size),
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iterations)


def _detect_edge_lines(
    gray: np.ndarray,
    cfg: DetectorConfig,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Structural, color-agnostic line detection (see module docstring).

    Returns (horiz, vert):
      horiz: list of (x_start, x_end, y)   — merged near-horizontal segments
      vert:  list of (y_start, y_end, x)   — merged near-vertical segments
    Only segments at least ``edge_min_line_frac`` of the image's relevant
    dimension survive — short fragments from icons/text/arrows are dropped
    here so the O(n²) rectangle-assembly pass stays cheap and clean.
    """
    img_h, img_w = gray.shape
    edges = cv2.Canny(gray, cfg.edge_canny_lo, cfg.edge_canny_hi)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=cfg.edge_hough_threshold,
        minLineLength=cfg.edge_hough_min_len,
        maxLineGap=cfg.edge_hough_max_gap,
    )
    raw_h: list[tuple[float, float, float]] = []
    raw_v: list[tuple[float, float, float]] = []
    if lines is not None:
        for line in lines:
            # cv2.HoughLinesP's return shape changed between OpenCV versions
            # (N,1,4) pre-5.0 vs (N,4) from 5.0 onward -- reshape(-1) flattens
            # either shape to a plain length-4 array so unpacking is safe
            # regardless of which OpenCV is installed.
            x1, y1, x2, y2 = line.reshape(-1)[:4]
            if abs(y2 - y1) <= cfg.edge_line_axis_tol and abs(x2 - x1) >= cfg.edge_hough_min_len:
                raw_h.append((float(min(x1, x2)), float(max(x1, x2)), float((y1 + y2) / 2)))
            elif abs(x2 - x1) <= cfg.edge_line_axis_tol and abs(y2 - y1) >= cfg.edge_hough_min_len:
                raw_v.append((float(min(y1, y2)), float(max(y1, y2)), float((x1 + x2) / 2)))

    horiz = _merge_collinear(raw_h, tol=cfg.edge_merge_axis_tol, gap=cfg.edge_merge_gap)
    vert = _merge_collinear(raw_v, tol=cfg.edge_merge_axis_tol, gap=cfg.edge_merge_gap)

    # Length floor: the SMALLER of "a legitimate container wall
    # (min_side_length)" and "a meaningful fraction of the image". Using
    # only the fraction-based cutoff would (and, caught by
    # test_layout_detector.py, did) filter out genuinely small containers
    # like Security Groups on a large canvas; using only min_side_length
    # lets through too much icon/text/arrow noise on dense real diagrams.
    # Taking the smaller of the two keeps small-but-real containers while
    # still pruning short fragments on big images.
    min_h_len = min(cfg.min_side_length, cfg.edge_min_line_frac * img_w)
    min_v_len = min(cfg.min_side_length, cfg.edge_min_line_frac * img_h)
    horiz = [l for l in horiz if (l[1] - l[0]) >= min_h_len]
    vert = [l for l in vert if (l[1] - l[0]) >= min_v_len]
    return horiz, vert


def _merge_collinear(
    lines: list[tuple[float, float, float]],
    tol: int,
    gap: int,
) -> list[tuple[float, float, float]]:
    """
    Merge near-duplicate/fragmented segments that lie on (roughly) the same
    line — buckets by the perpendicular coordinate (within ``tol``), then
    within each bucket merges runs whose spans are within ``gap`` px of each
    other. Works identically for horizontal (x_start,x_end,y) and vertical
    (y_start,y_end,x) tuples since both are (span_start, span_end, axis).
    """
    if not lines:
        return []
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    for s0, s1, axis in lines:
        buckets.setdefault(round(axis / max(tol, 1)), []).append((s0, s1, axis))

    merged: list[tuple[float, float, float]] = []
    for group in buckets.values():
        group.sort(key=lambda l: l[0])
        cur_s0, cur_s1, axes = group[0][0], group[0][1], [group[0][2]]
        for s0, s1, axis in group[1:]:
            if s0 <= cur_s1 + gap:
                cur_s1 = max(cur_s1, s1)
                axes.append(axis)
            else:
                merged.append((cur_s0, cur_s1, sum(axes) / len(axes)))
                cur_s0, cur_s1, axes = s0, s1, [axis]
        merged.append((cur_s0, cur_s1, sum(axes) / len(axes)))
    return merged


def _edge_fill_ratio(
    dilated_edges: np.ndarray,
    x: int, y: int, w: int, h: int,
    band: int,
) -> float:
    """
    Fraction of edge-map pixels along each of the 4 sides of (x,y,w,h) that
    are actually edge pixels, sampled every 2px. Returns the WEAKEST side's
    score — a genuine rectangle wall has strong edges on all 4 sides; a
    spurious combination assembled from unrelated lines typically has at
    least one side with little/no real edge support.
    """
    img_h, img_w = dilated_edges.shape
    x2, y2 = x + w, y + h

    def _side_ratio(pts: list[tuple[int, int]]) -> float:
        hits = 0
        total = 0
        for px, py in pts:
            if 0 <= px < img_w and 0 <= py < img_h:
                total += 1
                if dilated_edges[py, px] > 0:
                    hits += 1
        return hits / total if total else 0.0

    top = [(px, py) for px in range(x, x2, 2) for py in range(max(0, y - band), y + band + 1)]
    bottom = [(px, py) for px in range(x, x2, 2) for py in range(max(0, y2 - band), y2 + band + 1)]
    left = [(px, py) for py in range(y, y2, 2) for px in range(max(0, x - band), x + band + 1)]
    right = [(px, py) for py in range(y, y2, 2) for px in range(max(0, x2 - band), x2 + band + 1)]

    return min(_side_ratio(top), _side_ratio(bottom), _side_ratio(left), _side_ratio(right))


def _assemble_rectangles(
    horiz: list[tuple[float, float, float]],
    vert: list[tuple[float, float, float]],
    cfg: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    """
    Pair up (top, bottom) horizontal lines with matching (left, right)
    vertical walls into rectangle candidates — the structural core of the
    color-agnostic redesign. A vertical line is accepted as a wall when it
    passes within ``edge_corner_tol`` px of the target x AND its y-span
    covers at least ``edge_corner_overlap_frac`` of the candidate box's
    height (Hough fragments near corners rather than reaching the exact
    corner pixel, so exact-endpoint matching is too strict).
    """
    rects: list[tuple[int, int, int, int]] = []
    for i, h1 in enumerate(horiz):
        for h2 in horiz[i + 1:]:
            top, bottom = (h1, h2) if h1[2] < h2[2] else (h2, h1)
            box_h = bottom[2] - top[2]
            if box_h < cfg.min_side_length:
                continue
            x_lo = max(top[0], bottom[0])
            x_hi = min(top[1], bottom[1])
            if x_hi - x_lo < cfg.min_side_length:
                continue
            left_x = min(top[0], bottom[0])
            right_x = max(top[1], bottom[1])

            def _has_wall(target_x: float) -> bool:
                for vy0, vy1, vx in vert:
                    if abs(vx - target_x) > cfg.edge_corner_tol:
                        continue
                    overlap = max(0.0, min(vy1, bottom[2]) - max(vy0, top[2]))
                    if overlap >= cfg.edge_corner_overlap_frac * box_h:
                        return True
                return False

            if _has_wall(left_x) and _has_wall(right_x):
                rects.append((int(left_x), int(top[2]), int(right_x - left_x), int(bottom[2] - top[2])))
    return rects


def _min_side_color_ratio(
    color_mask: np.ndarray,
    x: int, y: int, w: int, h: int,
    cfg: DetectorConfig,
) -> float:
    """
    Like ``_measure_fill_ratio`` but returns the WEAKEST of the 4 sides
    instead of a single pooled fraction across all of them.

    Real bug found during the color-agnostic redesign: the structural
    rectangle-assembly pass can legitimately close a box using an INTERIOR
    divider line as one side (e.g. an AZ/Subnet divider inside a VPC) paired
    with the VPC's own long outer walls as its other two sides — this
    produces a geometrically real but semantically spurious nested
    rectangle. A pooled fill_ratio across all sides scores this fairly high
    (3 of its 4 sides genuinely are VPC-green), so it was being confidently
    (mis)classified as its own separate VPC — 4 phantom "vpc" detections
    from one real VPC, confirmed by reproducing against the aws_icon_diagram
    fixture. Requiring every side individually — not just the pooled
    average — to show the color kills these partial/composite matches,
    since the one side that's actually a different container's border
    (or an interior divider) won't carry that color at all.
    """
    sw = cfg.dash_sample_width
    img_h, img_w = color_mask.shape

    def _safe_slice(r0, r1, c0, c1):
        r0, r1 = max(0, r0), min(img_h, r1)
        c0, c1 = max(0, c0), min(img_w, c1)
        return color_mask[r0:r1, c0:c1]

    sides = [
        _safe_slice(y - sw, y + sw, x, x + w),
        _safe_slice(y + h - sw, y + h + sw, x, x + w),
        _safe_slice(y, y + h, x - sw, x + sw),
        _safe_slice(y, y + h, x + w - sw, x + w + sw),
    ]
    ratios = []
    for s in sides:
        if s.size == 0:
            ratios.append(0.0)
        else:
            ratios.append(float(np.count_nonzero(s)) / s.size)
    return min(ratios)


def _classify_region_color(
    hsv: np.ndarray,
    x: int, y: int, w: int, h: int,
    cfg: DetectorConfig,
) -> tuple[ContainerType, str]:
    """
    Soft color hint (see module docstring). For SOLID-style profiles
    (VPC/AWS Cloud), requires ALL FOUR sides to individually show that color
    above a low bar — see ``_min_side_color_ratio``'s docstring: pooling
    across sides let a box built from 3 real VPC walls + 1 unrelated
    interior divider line score high enough to be misclassified as its own
    separate VPC (4 phantom VPCs from 1 real one, confirmed against the
    aws_icon_diagram fixture). DASHED-style profiles (AZ/Subnet) keep the
    original pooled fill_ratio instead: dashed strokes routinely have a gap
    fall exactly on one sampled side, so requiring every side to
    individually clear the bar is too strict and was observed to
    misclassify real subnets/AZs as UNKNOWN. If no profile clears its bar,
    the region stays UNKNOWN — a hint, not a gate; nothing is ever rejected
    outright for failing to match a known color, it just doesn't get a
    specific type.
    """
    best_type = ContainerType.UNKNOWN
    best_name = "generic_edge"
    best_margin = 0.0  # score - threshold, so different scales stay comparable
    for profile in cfg.color_profiles:
        mask = _build_color_mask(hsv, profile)
        if profile.expected_style == "solid":
            score = _min_side_color_ratio(mask, x, y, w, h, cfg)
            threshold = 0.12
        else:
            score = _measure_fill_ratio(mask, x, y, w, h, cfg)
            threshold = 0.15
        margin = score - threshold
        if margin > 0 and margin > best_margin:
            best_margin = margin
            best_type = profile.container_type
            best_name = profile.name
    return best_type, best_name


def _contours_as_rects(
    closed_mask: np.ndarray,
    img_h: int,
    img_w: int,
    cfg: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    """
    Find contours in the closed mask, approximate each to a polygon, and
    return bounding rects (x, y, w, h) for those that look like rectangles.
    """
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    img_area = img_h * img_w

    for cnt in contours:
        # Approximate to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, cfg.poly_epsilon_frac * peri, True)

        # Accept 4-sided polygons OR bounding rects of larger blobs
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < cfg.min_bbox_area:
            continue
        if area > cfg.max_bbox_fraction * img_area:
            continue
        if w < cfg.min_side_length or h < cfg.min_side_length:
            continue

        rects.append((x, y, w, h))

    return rects


def _measure_fill_ratio(
    original_mask: np.ndarray,
    x: int, y: int, w: int, h: int,
    cfg: DetectorConfig,
) -> float:
    """
    Sample the original (pre-dilation) color mask along the four edges of the
    bounding rect and return the fraction of pixels that are colored.

    A solid border will be nearly 1.0; a dashed border will be < 0.6.
    """
    sw = cfg.dash_sample_width
    img_h, img_w = original_mask.shape

    def _safe_slice(r0, r1, c0, c1):
        r0, r1 = max(0, r0), min(img_h, r1)
        c0, c1 = max(0, c0), min(img_w, c1)
        return original_mask[r0:r1, c0:c1]

    strips = [
        _safe_slice(y - sw, y + sw, x, x + w),         # top edge
        _safe_slice(y + h - sw, y + h + sw, x, x + w), # bottom edge
        _safe_slice(y, y + h, x - sw, x + sw),          # left edge
        _safe_slice(y, y + h, x + w - sw, x + w + sw), # right edge
    ]
    all_px = np.concatenate([s.flatten() for s in strips if s.size > 0])
    if all_px.size == 0:
        return 0.0
    return float(np.count_nonzero(all_px)) / all_px.size


def _classify_container_by_size(
    region: ContainerRegion,
    all_regions: list[ContainerRegion],
) -> ContainerType:
    """
    Disambiguate SUBNET vs SECURITY_GROUP for orange detections:
    if a region is significantly smaller than another SUBNET-typed region
    at the same nesting level, re-classify it as SECURITY_GROUP.
    This is a heuristic — OCR/label matching will refine it later.
    """
    if region.container_type != ContainerType.SUBNET:
        return region.container_type

    peer_subnets = [
        r for r in all_regions
        if r.container_type == ContainerType.SUBNET
        and r.node_id != region.node_id
        and r.parent_id == region.parent_id
    ]
    if not peer_subnets:
        return region.container_type

    median_peer_area = float(np.median([p.area for p in peer_subnets]))
    if region.area < 0.25 * median_peer_area:
        return ContainerType.SECURITY_GROUP

    return region.container_type


def _bbox_overlap_fraction(inner: BoundingBox, outer: BoundingBox) -> float:
    """
    Fraction of `inner`'s area that overlaps with `outer`.
    Returns 0.0 if there is no overlap.
    """
    ix1 = max(inner.x, outer.x)
    iy1 = max(inner.y, outer.y)
    ix2 = min(inner.x + inner.width,  outer.x + outer.width)
    iy2 = min(inner.y + inner.height, outer.y + outer.height)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    inner_area = inner.width * inner.height
    return intersection / inner_area if inner_area > 0 else 0.0


def _build_containment_hierarchy(
    regions: list[ContainerRegion],
    cfg: DetectorConfig,
) -> list[ContainerRegion]:
    """
    Assign parent_id to each region.

    A region B is a child of A when:
    - A's area > B's area * containment_size_ratio  (A is meaningfully larger)
    - B's bbox overlaps A's bbox by >= containment_overlap_frac of B's area
    - Among all valid parents, prefer a KNOWN (non-UNKNOWN) container_type
      over an UNKNOWN one, then pick the smallest (immediate parent).

    The known-over-unknown preference matters for the color-agnostic
    structural detector: it can produce a spurious UNKNOWN composite
    rectangle that happens to sit strictly between a real child and its real
    (KNOWN-typed) parent in size — e.g. built from the real parent's own
    top/left/right walls plus an unrelated interior line as its 4th side.
    Picking "smallest valid parent" alone would wire the child to that
    artifact instead of the real container it visually belongs to (caught
    by test_inner_subnet_parent_is_vpc). An UNKNOWN region is only used as
    a parent when no KNOWN region qualifies at all — it's still kept as a
    fallback so orphaned children never silently lose their nesting.
    """
    # Sort largest → smallest so we iterate parents before children.
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)

    for i, child in enumerate(sorted_regions):
        best_known_parent: ContainerRegion | None = None
        best_known_area: float = float("inf")
        best_unknown_parent: ContainerRegion | None = None
        best_unknown_area: float = float("inf")

        for j, parent in enumerate(sorted_regions):
            if i == j:
                continue
            if parent.area <= child.area * cfg.containment_size_ratio:
                continue
            overlap = _bbox_overlap_fraction(child.bbox, parent.bbox)
            if overlap < cfg.containment_overlap_frac:
                continue
            if parent.container_type == ContainerType.UNKNOWN:
                if parent.area < best_unknown_area:
                    best_unknown_area = parent.area
                    best_unknown_parent = parent
            else:
                if parent.area < best_known_area:
                    best_known_area = parent.area
                    best_known_parent = parent

        best_parent = best_known_parent if best_known_parent is not None else best_unknown_parent
        child.parent_id = best_parent.node_id if best_parent else None

    return sorted_regions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_containers(
    image_path: str | Path,
    cfg: DetectorConfig | None = None,
) -> list[DiagramNode]:
    """
    Main entry point for Stage 1.

    Parameters
    ----------
    image_path : path to a PNG/JPG architecture diagram.
    cfg        : optional ``DetectorConfig`` to override defaults.

    Returns
    -------
    List of ``DiagramNode`` objects with ``shape=NodeShape.CONTAINER``,
    sorted largest-first (outermost containers come first).
    Coordinates are in image pixels.
    """
    if cfg is None:
        cfg = DetectorConfig()

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Diagram image not found: {path}")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return detect_containers_from_array(img_bgr, cfg=cfg)


def detect_containers_from_array(
    img_bgr: np.ndarray,
    cfg: DetectorConfig | None = None,
) -> list[DiagramNode]:
    """
    Same as ``detect_containers`` but accepts a BGR numpy array directly.
    Useful for testing without touching the filesystem.
    """
    if cfg is None:
        cfg = DetectorConfig()

    img_h, img_w = img_bgr.shape[:2]
    logger.info("Stage 1: detecting containers in %dx%d image", img_w, img_h)

    # Preprocess
    blurred = cv2.GaussianBlur(img_bgr, (cfg.blur_kernel_size, cfg.blur_kernel_size), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # ── Structural (color-agnostic) candidate generation ──────────────────
    # See module docstring "COLOR-AGNOSTIC REDESIGN": borders are found by
    # shape (straight edges forming closed rectangles), not by hue, so this
    # works regardless of the diagram's actual color palette.
    horiz, vert = _detect_edge_lines(gray, cfg)
    logger.debug("  structural: %d horizontal / %d vertical candidate walls", len(horiz), len(vert))

    edges = cv2.Canny(gray, cfg.edge_canny_lo, cfg.edge_canny_hi)
    dilated_edges = cv2.dilate(
        edges, np.ones((cfg.edge_dilate_kernel, cfg.edge_dilate_kernel), np.uint8), iterations=1
    )

    candidate_rects = _assemble_rectangles(horiz, vert, cfg)
    img_area = img_h * img_w
    candidate_rects = [
        (x, y, w, h) for (x, y, w, h) in candidate_rects
        if w * h <= cfg.max_bbox_fraction * img_area
        and w * h >= cfg.min_bbox_area
        and w >= cfg.min_side_length and h >= cfg.min_side_length
    ]
    logger.debug("  structural: %d assembled rectangle candidates", len(candidate_rects))

    raw_regions: list[ContainerRegion] = []

    for (x, y, w, h) in candidate_rects:
        # Structural confidence guard: a genuine rectangular wall has strong
        # edge support on all 4 sides; a spurious pairing of unrelated lines
        # (loose corner-tolerance matches) typically has at least one weak
        # side. Verified empirically against the real diagram that motivated
        # this redesign — see module docstring.
        edge_fill = _edge_fill_ratio(dilated_edges, x, y, w, h, band=cfg.edge_dilate_kernel + 1)
        if edge_fill < cfg.edge_fill_min_ratio:
            logger.debug(
                "    Skip: bbox=(%d,%d,%d,%d) edge_fill=%.2f below threshold — likely spurious",
                x, y, w, h, edge_fill,
            )
            continue

        # Color is now a soft hint applied AFTER structural confirmation —
        # never a gate. Unmatched borders stay ContainerType.UNKNOWN and are
        # still kept (resolved later via OCR label matching downstream).
        container_type, color_profile = _classify_region_color(hsv, x, y, w, h, cfg)

        # Sliver guard, UNKNOWN regions only: a >8:1 aspect ratio box that
        # doesn't even match a known color is a leftover line-pairing
        # artifact (e.g. a thin strip formed between two nearby dashed
        # borders), not a real container — no real AWS Cloud/VPC/AZ/Subnet
        # is drawn that elongated. Confirmed against the aws_icon_diagram
        # fixture: a spurious 1223x61 (20:1) sliver directly above the real
        # Subnet was surviving every other filter and being classified as a
        # phantom aws_vpc downstream by the classifier's generic container
        # fallback. Regions with a KNOWN color match are exempt — a real
        # narrow Subnet band is legitimate and must never be dropped.
        if container_type == ContainerType.UNKNOWN:
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 8.0:
                logger.debug(
                    "    Skip: bbox=(%d,%d,%d,%d) aspect=%.1f — unclassified sliver, not a container",
                    x, y, w, h, aspect,
                )
                continue

        # Border style: reuse the known profile's expected style when we have
        # a color match; otherwise fall back to the generic edge-density
        # comparison against dash_threshold.
        matched_profile = next((p for p in cfg.color_profiles if p.name == color_profile), None)
        if matched_profile is not None and matched_profile.expected_style in ("solid", "dashed"):
            border_style = matched_profile.expected_style
        else:
            border_style = "dashed" if edge_fill < cfg.dash_threshold else "solid"

        # Icon/container collision guard — see DetectorConfig's
        # icon_collision_max_area docstring. Restored to its original scope:
        # only SOLID-style regions are prone to being an individual icon's
        # own outline (VPC/AWS Cloud render as solid strokes, and so do
        # several AWS4 icon glyphs like ALB/IGW/Route53). Dashed regions
        # (subnets, security groups, AZs) are deliberately exempt — a small
        # dashed square is a legitimate Security Group, not a phantom icon.
        # Applying this guard to dashed regions too (tried during this
        # redesign) broke SECURITY_GROUP disambiguation, since small SGs are
        # exactly the shape/size this guard is built to reject — caught by
        # the existing test_layout_detector.py suite.
        if border_style == "solid":
            area = w * h
            aspect_delta = abs(w - h) / max(w, h)
            if area < cfg.icon_collision_max_area and aspect_delta <= cfg.icon_collision_aspect_tolerance:
                logger.debug(
                    "    Skip: bbox=(%d,%d,%d,%d) area=%d aspect_delta=%.2f — icon-shaped, not a container",
                    x, y, w, h, area, aspect_delta,
                )
                continue

        bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
        region = ContainerRegion(
            node_id        = str(uuid.uuid4()),
            bbox           = bbox,
            container_type = container_type,
            color_profile  = color_profile,
            border_style   = border_style,
            fill_ratio     = edge_fill,
            area           = float(w * h),
        )
        raw_regions.append(region)
        logger.debug(
            "    + %s  bbox=(%d,%d,%d,%d)  edge_fill=%.2f  style=%s  type=%s",
            color_profile, x, y, w, h, edge_fill, border_style, container_type.value,
        )

    if not raw_regions:
        logger.warning("Stage 1: no container regions detected")
        return []

    # De-duplicate near-identical-bbox composites BEFORE type-based dedup.
    # Real bug found during this redesign: the structural rectangle
    # assembly can close a box using one container's true outer walls
    # paired with an unrelated nearby line as its 4th side (e.g. the AWS
    # Cloud's own walls plus the VPC's top edge one side over) — the result
    # is a "container-shaped" bbox that's almost, but not exactly, the same
    # as a real one. It usually resolves to UNKNOWN (its color doesn't
    # match on every side) and so survives the type-keyed _deduplicate
    # below untouched, then corrupts the containment hierarchy by acting as
    # a same-size sibling/parent for the real container. This pass merges
    # any two regions whose bboxes are near-identical (>=90% overlap, sizes
    # within 1.5x of each other) REGARDLESS of resolved type, keeping the
    # one with a known (non-UNKNOWN) type, or higher fill_ratio on ties.
    raw_regions = _suppress_near_duplicate_composites(raw_regions)

    # Collapse "wall-reuse chains": on diagrams where NOTHING matches a
    # known color (Stage 1 stays fully structural — every region UNKNOWN),
    # _suppress_near_duplicate_composites above can't help, since it only
    # ever drops UNKNOWN vs. a KNOWN anchor. Real bug found 2026-07-28
    # against a dense, non-standard-color diagram: the same 3 real walls
    # (e.g. a container's left, right, and top edge) can each pair with
    # SEVERAL different plausible 4th walls (different nearby interior
    # lines), producing a whole chain of nested candidates that share those
    # 3 walls exactly and differ only in how far the 4th one extends — 21
    # "subnets" from what was really a handful of real ones, confirmed by
    # inspecting the raw bboxes (multiple entries sharing identical x/y/w
    # with only height differing). Ordinary sibling containers (e.g. two
    # real subnets stacked in the same VPC) also often share a wall, but
    # their OTHER two edges never overlap each other — only a genuine
    # same-source chain has one candidate's bbox fully nested inside
    # another's while still sharing 3 exact walls, which is what this
    # targets.
    raw_regions = _collapse_wall_reuse_chains(raw_regions, tol=cfg.edge_corner_tol)

    # De-duplicate: two regions of the SAME resolved container_type with
    # very similar bboxes are almost certainly the same physical container
    # found twice (e.g. via slightly different line-quadruple combinations).
    # container_type (not color_profile) is now the identity key, since
    # unmatched regions all share color_profile="generic_edge" but may still
    # be genuinely different nested containers.
    raw_regions = _deduplicate(raw_regions, overlap_threshold=0.85)

    # Refine SUBNET vs SECURITY_GROUP by relative size.
    for region in raw_regions:
        region.container_type = _classify_container_by_size(region, raw_regions)

    # Build the containment hierarchy.
    regions_with_parents = _build_containment_hierarchy(raw_regions, cfg)

    logger.info(
        "Stage 1: detected %d container regions", len(regions_with_parents)
    )
    for r in regions_with_parents:
        logger.debug(
            "  %s  type=%s  parent=%s  bbox=(%.0f,%.0f,%.0f,%.0f)",
            r.node_id[:8], r.container_type.value, r.parent_id[:8] if r.parent_id else "None",
            r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height,
        )

    return [_to_diagram_node(r) for r in regions_with_parents]


def _collapse_wall_reuse_chains(
    regions: list[ContainerRegion],
    tol: int,
) -> list[ContainerRegion]:
    """
    See call-site comment. Groups regions that share the same LEFT+RIGHT
    walls (within ``tol`` px) but differ in vertical extent, and separately
    groups regions sharing the same TOP+BOTTOM walls but differing in
    horizontal extent. Within each such group, the ORIGINAL rule kept only
    the LARGEST member (assumed to be the real, complete boundary — Hough
    line assembly tends to surface the true far wall as one of the
    candidates, typically the outermost) and dropped the smaller ones IF
    AND ONLY IF the smaller one's bbox is (near-)fully contained within
    the larger one's, on the theory that "real side-by-side siblings
    sharing a wall never satisfy this, since their other two edges don't
    overlap each other."

    That theory holds for pairs of siblings, but breaks for the UNION of
    several siblings: N real, non-overlapping containers stacked along the
    varying axis (e.g. 3 real subnet bands stacked inside one VPC) all
    share the VPC's left/right walls with each other AND, critically, with
    a spurious "top-of-band-1-to-bottom-of-band-N" composite that the same
    Hough line assembly also produces (by pairing band 1's top wall with
    band N's bottom wall and skipping the real dividers in between). That
    composite trivially satisfies "largest, and every real sibling is
    fully contained within it" — so the original rule silently collapsed
    all 3 real subnets into that one spurious union. Confirmed via a
    direct repro against tests/fixtures/images/aws_icon_diagram.png
    (2026-08-19): before this fix, all 3 orange_subnet bands collapsed
    into a single region spanning the full stack.

    Fix: before dropping everything nested inside the group's largest
    member, check whether those nested candidates themselves tile — i.e.
    form a set of MUTUALLY NON-OVERLAPPING regions that together cover
    most of the largest member's extent. Real, distinct siblings tile this
    way almost perfectly (each occupies its own slice with only small gaps
    between them). Genuine wall-reuse duplicates of a single container do
    the opposite: they're all near-restatements of the same box with only
    the ambiguous 4th wall's position varying, so they overlap each other
    heavily and no non-trivial tiling exists among them. When a real
    tiling is found, the "largest" is the spurious one — drop it (and any
    other nested candidate not part of the tiling) and keep the tiling
    siblings instead. Otherwise, fall back to the original behaviour.
    """
    def _bucket(v: float) -> int:
        return round(v / max(tol, 1))

    def _axis_span(r: ContainerRegion, axis: str) -> tuple[float, float]:
        if axis == "y":
            return r.bbox.y, r.bbox.y + r.bbox.height
        return r.bbox.x, r.bbox.x + r.bbox.width

    def _tiling_subset(nested: list[ContainerRegion]) -> list[ContainerRegion]:
        """Greedily builds the finest-grained set of mutually
        non-overlapping candidates from `nested`, smallest-area first, so
        real distinct siblings (small, disjoint) are preferred over
        coarser partial composites that happen to also fit the group."""
        chosen: list[ContainerRegion] = []
        for r in sorted(nested, key=lambda r: r.area):
            if all(_bbox_overlap_fraction(r.bbox, p.bbox) <= 0.15 for p in chosen):
                chosen.append(r)
        return chosen

    def _tiling_coverage(chosen: list[ContainerRegion], largest: ContainerRegion, axis: str) -> float:
        largest_start, largest_end = _axis_span(largest, axis)
        largest_len = max(largest_end - largest_start, 1.0)
        covered = sum(
            max(min(e, largest_end) - max(s, largest_start), 0.0)
            for s, e in (_axis_span(r, axis) for r in chosen)
        )
        return covered / largest_len

    def _collapse_pass(current: list[ContainerRegion], axis: str) -> list[ContainerRegion]:
        groups: dict[tuple[int, int], list[ContainerRegion]] = {}
        for r in current:
            if axis == "y":
                # same left AND right x -> compare vertical extent.
                key = (_bucket(r.bbox.x), _bucket(r.bbox.x + r.bbox.width))
            else:
                # same top AND bottom y -> compare horizontal extent.
                key = (_bucket(r.bbox.y), _bucket(r.bbox.y + r.bbox.height))
            groups.setdefault(key, []).append(r)

        to_drop: set[str] = set()
        for group in groups.values():
            if len(group) < 2:
                continue
            largest = max(group, key=lambda r: r.area)
            nested = [
                r for r in group
                if r.node_id != largest.node_id
                and _bbox_overlap_fraction(r.bbox, largest.bbox) >= 0.95
            ]
            if not nested:
                continue

            tiling = _tiling_subset(nested)
            if len(tiling) >= 2 and _tiling_coverage(tiling, largest, axis) >= 0.6:
                # `largest` is a spurious union of real, distinct siblings
                # that happen to share its walls -- drop the union and any
                # other redundant nested candidate, keep the tiling.
                to_drop.add(largest.node_id)
                keep_ids = {r.node_id for r in tiling}
                to_drop.update(r.node_id for r in nested if r.node_id not in keep_ids)
            else:
                # Genuine wall-reuse chain: one real container, several
                # redundant near-duplicate candidates for its ambiguous
                # 4th wall. Keep the largest, drop the rest (original
                # behaviour).
                to_drop.update(r.node_id for r in nested)

        return [r for r in current if r.node_id not in to_drop]

    # Run the vertical-extent pass (shared left/right walls) fully first,
    # THEN run the horizontal-extent pass (shared top/bottom walls) over
    # only what survived. Doing both passes against the same original,
    # unfiltered region list (as a single earlier version of this function
    # did) let a region the first pass had already identified as spurious
    # still "win" a second-pass pairing against a region the first pass
    # wanted to keep -- e.g. two near-identical (~1-2px apart) candidates
    # for the real middle subnet band landing in the same top/bottom
    # bucket, one of which the vertical pass had already dropped as a
    # redundant duplicate; comparing it against the other in the
    # horizontal pass caused BOTH to be dropped and the real band vanished
    # entirely (confirmed via direct repro against aws_icon_diagram.png,
    # 2026-08-19). Sequencing the passes so the second only ever compares
    # among first-pass survivors removes that cross-pass contamination.
    working = _collapse_pass(list(regions), "y")
    working = _collapse_pass(working, "x")
    return working


def _suppress_near_duplicate_composites(
    regions: list[ContainerRegion],
    overlap_threshold: float = 0.90,
    max_size_ratio: float = 1.5,
) -> list[ContainerRegion]:
    """
    See call-site comment. Drops an UNKNOWN-typed region when its bbox is
    near-identical to a KNOWN-typed region's — that combination is the
    actual failure signature of a spurious composite (real AWS Cloud +
    unrelated line forming a slightly-off duplicate of the Cloud itself,
    still unclassifiable by color so it stays UNKNOWN).

    Deliberately restricted to KNOWN-vs-UNKNOWN pairs only. A first version
    of this suppressed any near-identical pair regardless of type and broke
    real nesting: AWS Cloud and its VPC are often drawn with only a modest
    margin between them, so a real Cloud/VPC pair can be just as close in
    size/overlap as a spurious composite is — the VPC (correctly classified
    as ContainerType.VPC) was being merged away as a "duplicate" of the
    Cloud, and `test_three_level_nesting` (VPC not detected at all) caught
    it. Two KNOWN regions are always structurally real, different
    containers (already handled correctly by the containment hierarchy +
    type-keyed `_deduplicate`), so they're never touched here.
    """
    known = [r for r in regions if r.container_type != ContainerType.UNKNOWN]
    unknown = [r for r in regions if r.container_type == ContainerType.UNKNOWN]

    def _shares_full_span(u_bbox: BoundingBox, k_bbox: BoundingBox, axis_tol: int = 10) -> bool:
        """
        True when u's bbox reuses BOTH of k's walls along one axis (same
        left+right x, or same top+bottom y, within a small pixel
        tolerance) while being shorter along the other axis. This is the
        signature of a composite built from a real container's own outer
        walls plus one unrelated interior line as the 4th side — e.g. a
        real VPC's left+right verticals paired with an internal divider
        instead of the VPC's own top or bottom. Confirmed against the
        aws_icon_diagram fixture: every spurious leftover UNKNOWN fragment
        shared its x-span exactly with the real VPC/AWS-Cloud bbox.
        """
        same_x_span = (
            abs(u_bbox.x - k_bbox.x) <= axis_tol
            and abs((u_bbox.x + u_bbox.width) - (k_bbox.x + k_bbox.width)) <= axis_tol
        )
        same_y_span = (
            abs(u_bbox.y - k_bbox.y) <= axis_tol
            and abs((u_bbox.y + u_bbox.height) - (k_bbox.y + k_bbox.height)) <= axis_tol
        )
        return same_x_span or same_y_span

    kept_unknown: list[ContainerRegion] = []
    for u in unknown:
        is_spurious = False
        for k in known:
            # Case 1: near-identical bbox to a known region (close in size,
            # very high overlap) — see docstring above.
            size_ratio = max(u.area, k.area) / max(min(u.area, k.area), 1.0)
            if size_ratio <= max_size_ratio:
                smaller, larger = (u, k) if u.area <= k.area else (k, u)
                if _bbox_overlap_fraction(smaller.bbox, larger.bbox) >= overlap_threshold:
                    is_spurious = True
                    break
            # Case 2: fully contained within a known region AND reuses that
            # region's full span on one axis — a "partial crop" composite,
            # regardless of how much smaller it is on the other axis.
            if (
                _bbox_overlap_fraction(u.bbox, k.bbox) >= 0.95
                and _shares_full_span(u.bbox, k.bbox)
            ):
                is_spurious = True
                break
        if not is_spurious:
            kept_unknown.append(u)

    return known + kept_unknown


def _deduplicate(
    regions: list[ContainerRegion],
    overlap_threshold: float = 0.85,
) -> list[ContainerRegion]:
    """
    Remove duplicate detections of the SAME container.

    Two regions are considered duplicates only when:
    - They resolve to the SAME container_type (different types = different
      containers → never duplicates, even if one is inside the other)
    - Their bboxes overlap by >= overlap_threshold of the smaller region's area

    The type guard is critical: a VPC that completely surrounds a Subnet
    would otherwise score 100% overlap and be wrongly eliminated. A child
    container is never a duplicate of its parent. Uses container_type
    (not color_profile) as the identity key since the color-agnostic
    structural detector resolves many regions to the same
    color_profile="generic_edge" even when they're genuinely different
    nested containers — container_type is the real semantic identity.
    """
    kept: list[ContainerRegion] = []
    # Process highest-confidence (fill_ratio) first so we keep the better detection.
    for region in sorted(regions, key=lambda r: r.fill_ratio, reverse=True):
        is_duplicate = False
        for existing in kept:
            # Cross-type detections are NEVER duplicates — they represent
            # different container types that may legitimately overlap (parent/child).
            if region.container_type != existing.container_type:
                continue
            # Size-ratio guard within the same profile: an orange security group
            # sitting inside an orange subnet is NOT a duplicate — it's a smaller
            # child of the same color family. Only deduplicate regions of similar
            # size (within max_area_ratio of each other).
            area_ratio = max(region.area, existing.area) / max(min(region.area, existing.area), 1.0)
            if area_ratio > 4.0:
                continue
            overlap = _bbox_overlap_fraction(region.bbox, existing.bbox)
            if overlap >= overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(region)
    return kept


def _to_diagram_node(region: ContainerRegion) -> DiagramNode:
    """Convert a ContainerRegion to the canonical DiagramNode schema."""
    return DiagramNode(
        id            = region.node_id,
        raw_label     = "",   # populated by Stage 4 OCR
        shape         = NodeShape.CONTAINER,
        bbox          = region.bbox,
        style_raw     = f"color_profile={region.color_profile};fill_ratio={region.fill_ratio:.2f}",
        image_ref     = _CONTAINER_IMAGE_REF[region.container_type],
        parent_id     = region.parent_id,
        source_format = "image",
        extra         = {
            "container_type": region.container_type.value,
            "border_style":   region.border_style,
            "fill_ratio":     region.fill_ratio,
        },
    )
