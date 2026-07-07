"""
Stage 1 — Layout / boundary-box detector.

Detects the container hierarchy in a raster architecture diagram using
classical computer vision (OpenCV). No ML involved — this stage relies on
the fact that AWS Architecture diagrams always use specific, well-defined
colors and shapes for boundary boxes:

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
    - Among all valid parents, pick the smallest one (immediate parent).
    """
    # Sort largest → smallest so we iterate parents before children.
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)

    for i, child in enumerate(sorted_regions):
        best_parent: ContainerRegion | None = None
        best_parent_area: float = float("inf")

        for j, parent in enumerate(sorted_regions):
            if i == j:
                continue
            if parent.area <= child.area * cfg.containment_size_ratio:
                continue
            overlap = _bbox_overlap_fraction(child.bbox, parent.bbox)
            if overlap < cfg.containment_overlap_frac:
                continue
            if parent.area < best_parent_area:
                best_parent_area = parent.area
                best_parent = parent

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

    raw_regions: list[ContainerRegion] = []

    for profile in cfg.color_profiles:
        original_mask = _build_color_mask(hsv, profile)
        # Only apply morphological close for dashed-border profiles — it bridges
        # dash gaps to form a solid closed contour. For solid profiles, close
        # expands the thin border ring to fill the entire image boundary, causing
        # the bounding rect to span the full canvas and get filtered out.
        if profile.expected_style == "dashed":
            closed_mask = _close_mask(original_mask, cfg)
        else:
            closed_mask = original_mask

        candidate_rects = _contours_as_rects(closed_mask, img_h, img_w, cfg)
        logger.debug(
            "  profile '%s': %d candidate rects", profile.name, len(candidate_rects)
        )

        for (x, y, w, h) in candidate_rects:
            fill_ratio = _measure_fill_ratio(original_mask, x, y, w, h, cfg)

            # Border style classification:
            # The profile's expected_style is used as the primary indicator,
            # because fill_ratio alone is unreliable for thin solid borders
            # (sampling a 12px strip around a 4px border dilutes the fill signal).
            # fill_ratio is only used to catch clearly wrong detections.
            if profile.expected_style == "solid":
                # Very low fill_ratio on a "solid" profile → likely a noise contour.
                if fill_ratio < 0.20:
                    logger.debug(
                        "    Skip: %s fill_ratio=%.2f too low for solid", profile.name, fill_ratio
                    )
                    continue
                border_style = "solid"
            elif profile.expected_style == "dashed":
                # Very high fill_ratio on a "dashed" profile → likely a solid rect
                # mismatched to this profile (e.g. filled background artifacts).
                if fill_ratio > 0.85:
                    logger.debug(
                        "    Skip: %s fill_ratio=%.2f too high for dashed", profile.name, fill_ratio
                    )
                    continue
                border_style = "dashed"
            else:
                # "either" — let the raw fill_ratio decide
                border_style = "dashed" if fill_ratio < cfg.dash_threshold else "solid"

            bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))
            region = ContainerRegion(
                node_id        = str(uuid.uuid4()),
                bbox           = bbox,
                container_type = profile.container_type,
                color_profile  = profile.name,
                border_style   = border_style,
                fill_ratio     = fill_ratio,
                area           = float(w * h),
            )
            raw_regions.append(region)
            logger.debug(
                "    + %s  bbox=(%d,%d,%d,%d)  fill=%.2f  style=%s",
                profile.name, x, y, w, h, fill_ratio, border_style,
            )

    if not raw_regions:
        logger.warning("Stage 1: no container regions detected")
        return []

    # De-duplicate: if two regions from different profiles have very similar
    # bboxes, keep the one with the larger fill_ratio (more confident).
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


def _deduplicate(
    regions: list[ContainerRegion],
    overlap_threshold: float = 0.85,
) -> list[ContainerRegion]:
    """
    Remove duplicate detections of the SAME container.

    Two regions are considered duplicates only when:
    - They come from the SAME color profile (different profiles = different
      container types → never duplicates, even if one is inside the other)
    - Their bboxes overlap by >= overlap_threshold of the smaller region's area

    The profile guard is critical: a green VPC that completely surrounds an
    orange Subnet would otherwise score 100% overlap and be wrongly eliminated.
    A child container is never a duplicate of its parent.
    """
    kept: list[ContainerRegion] = []
    # Process highest-confidence (fill_ratio) first so we keep the better detection.
    for region in sorted(regions, key=lambda r: r.fill_ratio, reverse=True):
        is_duplicate = False
        for existing in kept:
            # Cross-profile detections are NEVER duplicates — they represent
            # different container types that may legitimately overlap (parent/child).
            if region.color_profile != existing.color_profile:
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
