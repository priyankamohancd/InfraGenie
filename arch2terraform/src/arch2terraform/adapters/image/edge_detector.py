"""
Stage 5 — Arrow / edge detector.

Detects directed connections between diagram nodes by finding dark arrow lines
in the image after masking out all known elements (icons and container borders).

Algorithm
---------
1. Build an "arrow mask": dark pixels (grayscale < dark_threshold) excluding
   • Each service-icon bounding box (±icon_pad + 30 px below for text labels).
   • Navy AWS-Cloud border pixels (HSV-based mask; navy is dark enough to
     pass the grayscale threshold).

2. Detect arrowhead blobs via connected components on the arrow mask.
   cv2.arrowedLine leaves a 1–2 px gap between the shaft and the filled
   arrowhead triangle, so heads appear as separate small connected components
   (span 8–55 px, area 8–250 px²).

3. Detect arrow shafts via HoughLinesP — not connected components.
   Rationale: when multiple arrows share a source or target node, the shaft
   pixels form a connected graph in the mask and would merge into one giant
   blob if we used CC. HoughLinesP finds each segment independently.

4. For each Hough segment:
   a. Match both endpoints to the nearest non-container node within
      max_match_dist (120 px).
   b. Skip if both endpoints match the same node or neither can be matched.
   c. Check for a nearby arrowhead blob at each endpoint to determine
      direction (head end = target).

5. Deduplicate: keep one edge per (source_id, target_id) pair.

Edge style is DASHED if the supporting segment is short relative to the
node-to-node distance (fill_ratio < 0.25); SOLID otherwise.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

import cv2
import numpy as np

from arch2terraform.schemas.diagram import BoundingBox, DiagramEdge, DiagramNode, EdgeStyle, NodeShape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DARK_THRESHOLD      = 80     # grayscale upper bound for "dark enough to be arrow"
ICON_PAD            = 6      # extra pixels beyond each icon bbox to exclude
LABEL_PAD_BELOW     = 30     # extra downward padding to cover text labels below icons
MIN_HEAD            = 8      # arrowhead span lower bound (px)
MAX_HEAD            = 55     # arrowhead span upper bound (px)
MIN_HEAD_AREA       = 8      # arrowhead pixel count lower bound
MAX_HEAD_AREA       = 300    # arrowhead pixel count upper bound
HEAD_SEARCH_RADIUS  = 40     # px from segment endpoint to look for an arrowhead
MAX_MATCH_DIST      = 120    # max px from segment endpoint to node bbox edge
HOUGH_THRESHOLD     = 20     # HoughLinesP accumulator threshold (votes)
HOUGH_MIN_LENGTH    = 30     # minimum line length (px) to keep
HOUGH_MAX_GAP       = 8      # max gap between collinear segments to bridge (px)


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------

def _build_arrow_mask(
    bgr: np.ndarray,
    nodes: list[DiagramNode],
    dark_threshold: int = DARK_THRESHOLD,
    icon_pad: int = ICON_PAD,
) -> np.ndarray:
    """Return binary mask of dark pixels that are NOT part of any known element."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Exclude navy AWS-Cloud border (HSV range)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    navy_mask = cv2.inRange(hsv, np.array([95, 30, 15]), np.array([125, 180, 100]))
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(navy_mask))

    # Exclude each service-icon bbox (containers are NOT excluded — arrows run through them)
    H, W = bgr.shape[:2]
    for node in nodes:
        if node.shape == NodeShape.CONTAINER:
            continue
        b = node.bbox
        x1 = max(0, int(b.x)      - icon_pad)
        y1 = max(0, int(b.y)      - icon_pad)
        x2 = min(W, int(b.right)  + icon_pad)
        y2 = min(H, int(b.bottom) + icon_pad + LABEL_PAD_BELOW)
        dark[y1:y2, x1:x2] = 0

    return dark


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist_point_to_bbox(px: float, py: float, bbox: BoundingBox) -> float:
    """Minimum Euclidean distance from (px, py) to the rectangle."""
    cx = max(bbox.x, min(px, bbox.right))
    cy = max(bbox.y, min(py, bbox.bottom))
    return float(np.hypot(px - cx, py - cy))


def _nearest_icon_node(
    x: float,
    y: float,
    icon_nodes: list[DiagramNode],
    max_dist: float,
) -> DiagramNode | None:
    """Return the icon node with the smallest bbox-edge distance to (x, y)."""
    best_node: DiagramNode | None = None
    best_dist = max_dist
    for node in icon_nodes:
        d = _dist_point_to_bbox(x, y, node.bbox)
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node


# ---------------------------------------------------------------------------
# Arrowhead detection (connected components on the arrow mask)
# ---------------------------------------------------------------------------

def _detect_heads(
    arrow_mask: np.ndarray,
) -> list[tuple[float, float]]:
    """
    Return centroids of arrowhead blobs in the arrow mask.

    Arrowheads from cv2.arrowedLine appear as small filled triangles with
    span in [MIN_HEAD, MAX_HEAD] and area in [MIN_HEAD_AREA, MAX_HEAD_AREA].
    """
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        arrow_mask, connectivity=8
    )
    heads: list[tuple[float, float]] = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        bw   = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh   = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        span = max(bw, bh)
        if MIN_HEAD <= span <= MAX_HEAD and MIN_HEAD_AREA <= area <= MAX_HEAD_AREA:
            heads.append((float(centroids[lbl, 0]), float(centroids[lbl, 1])))
    return heads


def _head_near(
    ep: tuple[float, float],
    heads: list[tuple[float, float]],
    radius: float = HEAD_SEARCH_RADIUS,
) -> bool:
    """True if any arrowhead centroid is within `radius` px of endpoint `ep`."""
    ex, ey = ep
    return any(np.hypot(ex - hx, ey - hy) <= radius for hx, hy in heads)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_edges(
    bgr: np.ndarray,
    nodes: list[DiagramNode],
    *,
    dark_threshold: int    = DARK_THRESHOLD,
    icon_pad: int          = ICON_PAD,
    max_match_dist: float  = MAX_MATCH_DIST,
    hough_threshold: int   = HOUGH_THRESHOLD,
    hough_min_length: int  = HOUGH_MIN_LENGTH,
    hough_max_gap: int     = HOUGH_MAX_GAP,
    head_search_radius: float = HEAD_SEARCH_RADIUS,
) -> list[DiagramEdge]:
    """
    Detect directed edges between nodes in a BGR diagram image.

    Parameters
    ----------
    bgr               : BGR image (from cv2.imread).
    nodes             : all DiagramNodes (containers + icons).
    dark_threshold    : grayscale ceiling for arrow pixels (default 80).
    icon_pad          : extra px beyond icon bbox to mask (default 6).
    max_match_dist    : max px from segment endpoint to node bbox edge (default 120).
    hough_threshold   : HoughLinesP accumulator votes to accept a line (default 20).
    hough_min_length  : minimum Hough segment length in px (default 30).
    hough_max_gap     : maximum gap in px that HoughLinesP may bridge (default 8).
    head_search_radius: px radius to search for an arrowhead at each endpoint (default 40).

    Returns
    -------
    List of DiagramEdge. Direction is determined by arrowhead proximity;
    style is SOLID unless the segment is broken/sparse (DASHED).
    """
    arrow_mask = _build_arrow_mask(bgr, nodes, dark_threshold, icon_pad)

    # Only non-container nodes are valid edge endpoints
    icon_nodes = [n for n in nodes if n.shape != NodeShape.CONTAINER]

    # ── Arrowhead detection (small CC blobs on the mask) ─────────────────
    heads = _detect_heads(arrow_mask)
    logger.info("Edge detector: %d arrowhead blobs found", len(heads))

    # ── Hough line segment detection ──────────────────────────────────────
    raw_lines = cv2.HoughLinesP(
        arrow_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap,
    )
    if raw_lines is None:
        logger.info("Edge detector: HoughLinesP found no segments")
        return []

    logger.info("Edge detector: %d Hough segments before filtering", len(raw_lines))

    # ── Per-segment: match endpoints to icon nodes ────────────────────────
    # Accumulate (src_id, tgt_id, directed, segment_length) per node pair
    pair_votes: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_directed: dict[tuple[str, str], bool] = {}

    for seg in raw_lines:
        # Same OpenCV-version shape difference as layout_detector.py's
        # _detect_edge_lines -- reshape(-1) is safe under both the pre-5.0
        # (N,1,4) and 5.0+ (N,4) HoughLinesP return shapes.
        x1, y1, x2, y2 = seg.reshape(-1)[:4]
        ep_a = (float(x1), float(y1))
        ep_b = (float(x2), float(y2))

        node_a = _nearest_icon_node(ep_a[0], ep_a[1], icon_nodes, max_match_dist)
        node_b = _nearest_icon_node(ep_b[0], ep_b[1], icon_nodes, max_match_dist)

        if node_a is None or node_b is None:
            continue
        if node_a.id == node_b.id:
            continue

        # Determine direction from arrowhead proximity
        a_has_head = _head_near(ep_a, heads, head_search_radius)
        b_has_head = _head_near(ep_b, heads, head_search_radius)

        if a_has_head and not b_has_head:
            src_id, tgt_id, directed = node_b.id, node_a.id, True
        elif b_has_head and not a_has_head:
            src_id, tgt_id, directed = node_a.id, node_b.id, True
        else:
            # No clear direction — canonical order by id for dedup
            src_id, tgt_id, directed = (
                (node_a.id, node_b.id, False)
                if node_a.id < node_b.id
                else (node_b.id, node_a.id, False)
            )

        seg_len = float(np.hypot(x2 - x1, y2 - y1))
        pair_votes[(src_id, tgt_id)].append(seg_len)
        # A directed vote wins over undirected
        if directed:
            pair_directed[(src_id, tgt_id)] = True
        else:
            pair_directed.setdefault((src_id, tgt_id), False)

    logger.info("Edge detector: %d unique node pairs after segment matching", len(pair_votes))

    # ── Build DiagramEdge per pair (deduplicated) ─────────────────────────
    edges: list[DiagramEdge] = []
    seen: set[frozenset[str]] = set()   # undirected pair for reverse-dedup

    for (src_id, tgt_id), lengths in pair_votes.items():
        pair_fs = frozenset({src_id, tgt_id})
        if pair_fs in seen:
            continue
        seen.add(pair_fs)

        directed = pair_directed.get((src_id, tgt_id), False)

        # Style: treat as dashed if total detected length is short relative
        # to euclidean distance between node centres
        src_node = next(n for n in icon_nodes if n.id == src_id)
        tgt_node = next(n for n in icon_nodes if n.id == tgt_id)
        src_cx = src_node.bbox.x + src_node.bbox.width  / 2
        src_cy = src_node.bbox.y + src_node.bbox.height / 2
        tgt_cx = tgt_node.bbox.x + tgt_node.bbox.width  / 2
        tgt_cy = tgt_node.bbox.y + tgt_node.bbox.height / 2
        expected_len = float(np.hypot(tgt_cx - src_cx, tgt_cy - src_cy))
        fill_ratio = sum(lengths) / max(expected_len, 1.0)
        style = EdgeStyle.DASHED if fill_ratio < 0.25 else EdgeStyle.SOLID

        edges.append(DiagramEdge(
            id=str(uuid.uuid4()),
            source_id=src_id,
            target_id=tgt_id,
            label="",
            style=style,
            extra={
                "directed":    directed,
                "fill_ratio":  round(fill_ratio, 3),
                "hough_votes": len(lengths),
            },
        ))
        logger.debug(
            "Edge: %s→%s  directed=%s  fill=%.2f  votes=%d",
            src_id[:6], tgt_id[:6], directed, fill_ratio, len(lengths),
        )

    logger.info("Edge detector: %d edges emitted", len(edges))
    return edges
