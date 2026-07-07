"""
Stage 4 — OCR text label extractor.

Runs Tesseract on the diagram image to find all text bounding boxes, then
spatially assigns each text block to the closest diagram node. The result is
a dict mapping node_id → cleaned label string, which the adapter uses to
populate DiagramNode.raw_label.

Design decisions
----------------
* We preprocess with CLAHE + mild sharpening to improve OCR accuracy on
  anti-aliased text rendered over coloured container backgrounds.
* Only blocks with Tesseract confidence ≥ min_conf and word length ≥ 2 are
  kept — single characters are almost always noise on architecture diagrams.
* Association is by nearest-centroid: for each text block we find the node
  whose centre is closest. If multiple blocks map to the same node, their
  text is joined with a space (handles multi-line labels like "Auto\\nScaling").
* Container nodes (VPC, AZ, Subnet) often have no icon, so OCR is their only
  source of label text — we don't skip them during assignment.

Graceful degradation
--------------------
If Tesseract is not installed, the function returns an empty dict and logs a
warning rather than crashing the whole adapter pipeline. The adapter emits the
service_name from phash as raw_label in that case.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """One word/phrase detected by Tesseract."""
    text:  str
    x:     int
    y:     int
    w:     int
    h:     int
    conf:  int   # Tesseract confidence 0-100

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Return a grayscale image optimised for Tesseract 4 LSTM.

    Steps:
    1. Convert to grayscale.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalisation) — lifts
       low-contrast text on coloured backgrounds.
    3. Unsharp mask — sharpens edges, helps Tesseract's segmentation.
    4. Scale to 2× — Tesseract is calibrated for ~300 DPI; many diagram
       exports are 96 DPI, so upscaling improves recall.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Unsharp mask (σ=1, strength=1.5)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 2× upscale
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    return gray


# ---------------------------------------------------------------------------
# Tesseract wrapper
# ---------------------------------------------------------------------------

_CLEAN_RE = re.compile(r"[^\w\s\-./]")   # keep word chars, spaces, hyphens, dots, slashes


def _clean(text: str) -> str:
    """Strip OCR noise characters and collapse whitespace."""
    text = _CLEAN_RE.sub("", text)
    return " ".join(text.split())


def run_tesseract(bgr: np.ndarray, min_conf: int = 50, min_height: int = 6) -> list[TextBlock]:
    """
    Run Tesseract on `bgr` and return filtered TextBlock list.

    Parameters
    ----------
    bgr        : BGR image (as loaded by cv2.imread).
    min_conf   : Minimum Tesseract word-level confidence to keep (0-100).
    min_height : Minimum block height in px (original image space) to keep.
                 AWS diagrams draw AZ/Subnet borders as dashes, and Tesseract's
                 PSM 11 (sparse text) reliably hallucinates 2-4px-tall "words"
                 (e.g. "ee", "mm", "SS") out of a run of collinear dash marks —
                 confirmed empirically: every real-world diagram tested produces
                 dozens of these, some clearing the confidence bar. Real label
                 text is never this short vertically even at low DPI, so a
                 height floor removes the whole failure mode cheaply.

    Returns
    -------
    List of TextBlock, one per accepted word/phrase. Empty list if
    Tesseract is not installed or returns no usable text.
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning(
            "pytesseract not installed — Stage 4 OCR skipped. "
            "Install with: pip install pytesseract (and ensure tesseract binary is in PATH)."
        )
        return []

    preprocessed = _preprocess_for_ocr(bgr)

    try:
        data = pytesseract.image_to_data(
            preprocessed,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",   # sparse text — finds text anywhere, no column assumptions
        )
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract binary not found — Stage 4 OCR skipped.")
        return []

    scale = 0.5   # we upscaled 2×; map coordinates back to original image space

    blocks: list[TextBlock] = []
    n = len(data["text"])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        conf = int(data["conf"][i])
        if conf < min_conf:
            continue
        cleaned = _clean(raw)
        if len(cleaned) < 2:   # single chars are almost always noise
            continue

        # Map coordinates from preprocessed (2×) back to original space
        x = int(data["left"][i] * scale)
        y = int(data["top"][i] * scale)
        w = int(data["width"][i] * scale)
        h = int(data["height"][i] * scale)

        if h < min_height:   # dash-line artifact, not real text — see docstring
            continue

        blocks.append(TextBlock(text=cleaned, x=x, y=y, w=w, h=h, conf=conf))

    logger.info(
        "OCR: %d text blocks detected (min_conf=%d, min_height=%d)",
        len(blocks), min_conf, min_height,
    )
    return blocks


# ---------------------------------------------------------------------------
# Node → label assignment
# ---------------------------------------------------------------------------

def assign_labels(
    blocks: list[TextBlock],
    node_centres: dict[str, tuple[float, float]],
    max_distance: float | None = None,
) -> dict[str, str]:
    """
    Assign each TextBlock to the nearest node centre.

    Parameters
    ----------
    blocks        : list of TextBlock from run_tesseract().
    node_centres  : dict of node_id → (cx, cy) in image coordinates.
    max_distance  : if set, text blocks farther than this many pixels from
                    every node centre are discarded (useful to drop legend text).

    Returns
    -------
    dict mapping node_id → concatenated label string (multi-word blocks joined
    with spaces in document order).
    """
    if not blocks or not node_centres:
        return {}

    # Build ordered list for distance calculations
    ids   = list(node_centres.keys())
    cxs   = np.array([node_centres[nid][0] for nid in ids], dtype=float)
    cys   = np.array([node_centres[nid][1] for nid in ids], dtype=float)

    assignment: dict[str, list[TextBlock]] = {nid: [] for nid in ids}

    for block in blocks:
        dx = cxs - block.cx
        dy = cys - block.cy
        dists = np.sqrt(dx * dx + dy * dy)
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])

        if max_distance is not None and nearest_dist > max_distance:
            logger.debug(
                "OCR block '%s' (%.0f, %.0f) too far from any node (dist=%.0f > %.0f) — discarded",
                block.text, block.cx, block.cy, nearest_dist, max_distance,
            )
            continue

        assignment[ids[nearest_idx]].append(block)

    # Merge multiple blocks per node: sort by reading order (top-to-bottom,
    # left-to-right within ±20 px vertical band) then join with space.
    result: dict[str, str] = {}
    for nid, node_blocks in assignment.items():
        if not node_blocks:
            continue
        node_blocks.sort(key=lambda b: (b.y // 20, b.x))
        merged = " ".join(b.text for b in node_blocks)
        result[nid] = _clean(merged)
        logger.debug("OCR label for node %s: %r", nid[:8], result[nid])

    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def extract_labels(
    bgr: np.ndarray,
    node_centres: dict[str, tuple[float, float]],
    *,
    min_conf: int = 50,
    min_height: int = 6,
    max_distance: float | None = None,
) -> dict[str, str]:
    """
    One-call API: run OCR on `bgr` and return node_id → label mapping.

    Parameters
    ----------
    bgr           : BGR image.
    node_centres  : dict of node_id → (cx, cy) in image coordinates.
    min_conf      : minimum Tesseract word confidence to keep.
    min_height    : minimum text block height (px) to keep — filters dash-line
                    OCR artifacts, see run_tesseract().
    max_distance  : maximum pixel distance from a node centre to accept a
                    text block. Pass None to accept the nearest node regardless
                    of distance (safe for diagrams where every text belongs to
                    some element).

    Returns
    -------
    dict[str, str] — may be empty if Tesseract is unavailable or finds nothing.
    """
    blocks = run_tesseract(bgr, min_conf=min_conf, min_height=min_height)
    if not blocks:
        return {}
    return assign_labels(blocks, node_centres, max_distance=max_distance)
