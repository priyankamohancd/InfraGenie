"""
Image adapter (PNG/JPG) — Phase 3 detection cascade.

Pipeline
--------
Stage 1  layout_detector   — find VPC/AZ/Subnet container boundaries
Stage 2a icon_detector     — locate candidate service icon blobs
Stage 2b hash_matcher      — identify each icon via perceptual hash (phash)
Stage 3  stage3_matcher    — NCC template matching for phash near-misses
Stage 4  ocr_extractor     — Tesseract OCR text labels (populates raw_label)

The diagram is the single source of truth: every container, service, and label
is derived from the pixels — no user-supplied metadata required.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from arch2terraform.adapters.base import BaseAdapter
from arch2terraform.adapters.image.icon_detector import detect_icon_candidates
from arch2terraform.adapters.image.layout_detector import (
    ContainerType,
    detect_containers_from_array,
)
from arch2terraform.adapters.image.hash_matcher import match_icon, load_table
from arch2terraform.adapters.image.ocr_extractor import extract_labels
from arch2terraform.adapters.image.stage3_matcher import Stage3Matcher
from arch2terraform.adapters.image.edge_detector import detect_edges
from arch2terraform.schemas.diagram import (
    BoundingBox,
    DiagramNode,
    NodeShape,
    ParsedDiagram,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public error type (kept for callers that may import it)
# ---------------------------------------------------------------------------

class ImageAdapterNotImplemented(NotImplementedError):
    """Raised when a sub-stage that is still a stub is invoked."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bgr_crop_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR OpenCV array to an RGB PIL Image suitable for imagehash."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _assign_parent(
    bbox: BoundingBox,
    container_nodes: list[DiagramNode],
) -> str | None:
    """
    Return the id of the smallest container node that fully encloses `bbox`.
    Mirrors the containment logic in layout_detector but operates on already-
    built DiagramNodes rather than raw ContainerRegions.
    """
    best_node: DiagramNode | None = None
    best_area: float = float("inf")

    for node in container_nodes:
        cb = node.bbox
        if (
            cb.x <= bbox.x
            and cb.y <= bbox.y
            and cb.right >= bbox.right
            and cb.bottom >= bbox.bottom
        ):
            area = cb.width * cb.height
            if area < best_area:
                best_area = area
                best_node = node

    return best_node.id if best_node else None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ImageAdapter(BaseAdapter):
    """
    Parses raster architecture diagrams (PNG / JPG) into the canonical
    ParsedDiagram IR used by the classifier and HCL generator.

    Parameters
    ----------
    max_hamming : int
        Hamming distance ceiling for a confident phash match (default 10).
        Icons exceeding this threshold are included as UNKNOWN nodes so the
        Stage 3/4 stubs can be wired in later without schema changes.
    table_path : str | Path | None
        Override the default reference hash table location. Useful for tests.
    """

    format_name = "image"

    def __init__(
        self,
        max_hamming: int = 10,
        table_path: str | Path | None = None,
        icons_dir: str | Path | None = None,
    ) -> None:
        self._max_hamming = max_hamming
        self._table_path  = table_path
        self._icons_dir   = Path(icons_dir) if icons_dir else None

    # ------------------------------------------------------------------
    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".png", ".jpg", ".jpeg"}

    # ------------------------------------------------------------------
    def parse(self, file_path: str) -> ParsedDiagram:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Diagram file not found: {path}")

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cv2.imread failed to decode '{path}'")

        warnings: list[str] = []

        # ── Stage 1: container / boundary detection ──────────────────────
        logger.info("[Stage 1] Detecting containers in '%s'", path.name)
        container_nodes = detect_containers_from_array(bgr)
        logger.info("[Stage 1] Found %d container nodes", len(container_nodes))

        # ── Stage 2a: icon candidate detection ───────────────────────────
        logger.info("[Stage 2a] Detecting icon candidates")
        candidates = detect_icon_candidates(bgr)
        logger.info("[Stage 2a] Found %d icon candidates", len(candidates))

        # ── Stage 2b: phash matching ──────────────────────────────────────
        service_nodes: list[DiagramNode] = []
        stage3_pending: list[tuple[DiagramNode, np.ndarray]] = []  # (node, crop_bgr)

        # Pre-load the hash table once for all candidates
        try:
            table = load_table(self._table_path)
        except FileNotFoundError:
            warnings.append(
                "Reference hash table not found — service icons will not be identified. "
                "Run scripts/build_hash_table.py to generate it."
            )
            table = {}

        for cand in candidates:
            pil_crop  = _bgr_crop_to_pil(cand.crop)
            result    = match_icon(pil_crop, table=table, max_hamming=self._max_hamming)
            parent_id = _assign_parent(cand.bbox, container_nodes)

            if result.confident:
                service_name = result.service_name or "unknown"
                node = DiagramNode(
                    id=str(uuid.uuid4()),
                    raw_label=service_name,
                    shape=NodeShape.ICON,
                    bbox=cand.bbox,
                    image_ref=service_name,
                    parent_id=parent_id,
                    source_format="image",
                    extra={
                        "phash_hamming": result.hamming,
                        "category": result.category,
                        "stage": "phash",
                    },
                )
                service_nodes.append(node)
            else:
                # Emit a provisional UNKNOWN node; Stage 3 may upgrade it.
                node = DiagramNode(
                    id=str(uuid.uuid4()),
                    raw_label="",
                    shape=NodeShape.UNKNOWN,
                    bbox=cand.bbox,
                    image_ref=None,
                    parent_id=parent_id,
                    source_format="image",
                    extra={
                        "phash_hamming": result.hamming,
                        "stage": "unmatched",
                    },
                )
                service_nodes.append(node)
                stage3_pending.append((node, cand.crop))

        logger.info(
            "[Stage 2b] %d phash-matched, %d forwarded to Stage 3",
            len(service_nodes) - len(stage3_pending),
            len(stage3_pending),
        )

        # ── Stage 3: NCC template matching for phash near-misses ─────────
        if stage3_pending:
            stage3 = Stage3Matcher(table, icons_dir=self._icons_dir)
            stage3_matched = 0
            for node, crop_bgr in stage3_pending:
                s3_result = stage3.match(crop_bgr)
                if s3_result.confident:
                    service_name = s3_result.service_name or "unknown"
                    node.raw_label  = service_name
                    node.image_ref  = service_name
                    node.shape      = NodeShape.ICON
                    node.extra.update({
                        "ncc_score": round(s3_result.ncc_score, 4),
                        "category":  s3_result.category,
                        "stage":     "ncc",
                    })
                    stage3_matched += 1
                else:
                    node.extra["ncc_score"] = round(s3_result.ncc_score, 4)

            still_unmatched = len(stage3_pending) - stage3_matched
            logger.info(
                "[Stage 3] NCC matched %d / %d; %d still unmatched",
                stage3_matched, len(stage3_pending), still_unmatched,
            )

            if self._icons_dir is None and stage3_pending:
                warnings.append(
                    f"{len(stage3_pending)} icon(s) forwarded to Stage 3 but icons_dir "
                    "was not set — NCC matching skipped. Pass icons_dir= to ImageAdapter."
                )
            elif still_unmatched:
                warnings.append(
                    f"{still_unmatched} icon(s) could not be identified by phash or NCC template "
                    "matching. OCR (Stage 4) will populate their labels where text is visible."
                )

        # ── Stage 4: OCR text labels ──────────────────────────────────────
        # Build a centroid map for every node we have so far so OCR blocks
        # can be assigned to the nearest element (icon or container alike).
        all_nodes_so_far = container_nodes + service_nodes
        node_centres: dict[str, tuple[float, float]] = {
            n.id: (n.bbox.x + n.bbox.width / 2, n.bbox.y + n.bbox.height / 2)
            for n in all_nodes_so_far
        }

        # Estimate a sensible max_distance: 1.5 × median icon side length,
        # or 120 px if no icons were found. This prevents legend / title text
        # far from any diagram element from being assigned to a random node.
        if icons_found := [n for n in service_nodes if n.shape == NodeShape.ICON]:
            median_side = float(np.median([max(n.bbox.width, n.bbox.height) for n in icons_found]))
            max_dist = max(median_side * 2.5, 80.0)
        else:
            max_dist = 120.0

        labels = extract_labels(bgr, node_centres, max_distance=max_dist)

        for node in all_nodes_so_far:
            ocr_label = labels.get(node.id, "").strip()
            if not ocr_label:
                continue
            if node.shape == NodeShape.ICON and node.raw_label:
                # For icons already identified by phash, append OCR text as
                # a hint (the phash name is authoritative; OCR may refine it).
                node.extra["ocr_label"] = ocr_label
            else:
                # For containers and unmatched icons, OCR is the primary label.
                node.raw_label = ocr_label

        logger.info("[Stage 4] OCR assigned labels to %d / %d nodes", len(labels), len(all_nodes_so_far))

        # ── Stage 5: edge / arrow detection ──────────────────────────────
        all_nodes = container_nodes + service_nodes
        logger.info("[Stage 5] Detecting edges")
        edges = detect_edges(bgr, all_nodes)
        logger.info("[Stage 5] Found %d edges", len(edges))

        # ── Assemble ParsedDiagram ────────────────────────────────────────
        return ParsedDiagram(
            nodes=all_nodes,
            edges=edges,
            source_format="image",
            source_file=str(path),
            warnings=warnings,
        )
