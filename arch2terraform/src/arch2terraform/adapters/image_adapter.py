"""
Image adapter (PNG/JPG) — Phase 3 detection cascade.

Pipeline
--------
Stage 1  layout_detector       — find VPC/AZ/Subnet container boundaries
Stage 2a icon_detector         — locate candidate service icon blobs
Stage 2b hash_matcher          — identify each icon via perceptual hash (phash)
Stage 3  stage3_matcher        — NCC template matching for phash near-misses
Stage 3b stage3b_cnn_classifier — CNN fallback for icons that fail both phash
                                   and NCC (optional — requires torch + a
                                   trained checkpoint; degrades gracefully)
Stage 4  ocr_extractor         — Tesseract OCR text labels (populates raw_label)

The diagram is the single source of truth: every container, service, and label
is derived from the pixels — no user-supplied metadata required.
"""

from __future__ import annotations

import logging
import os
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
from arch2terraform.adapters.image.vision_llm_detector import detect_via_vision_llm
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
    enable_cnn_fallback : bool
        Whether to run Stage 3b (CNN) on icons that fail both phash and NCC
        (default True). Automatically degrades to a no-op with a debug log
        if torch isn't installed or the checkpoint hasn't been trained yet —
        safe to leave enabled everywhere.
    cnn_checkpoint_path : str | Path | None
        Override the default trained CNN checkpoint location. Useful for tests.
    cnn_min_confidence : float
        Softmax-probability floor for a confident Stage 3b match (default 0.85).
    use_vision_llm : bool | None
        When True, replaces the entire classical Stage 1-4 cascade (layout
        detector, icon detector, phash/NCC/CNN matching, OCR) with a single
        vision-LLM call (see vision_llm_detector.py's module docstring for
        the rationale — the classical cascade is a stack of hand-tuned
        heuristics that each only generalize to the diagrams they were tuned
        against). Edge/arrow detection (Stage 5) still runs classically
        either way, since it was never implicated in the detection-quality
        issues this replaces. Defaults to reading the
        ARCH2TERRAFORM_USE_VISION_LLM env var (any of "1"/"true"/"yes",
        case-insensitive) when not explicitly passed, so it can be toggled
        per-environment without a code change.
    vision_llm_model : str | None
        Override the default vision-capable model identifier. Falls back to
        the ARCH2TERRAFORM_VISION_LLM_MODEL env var when not explicitly
        passed (e.g. set to a cheaper/faster model like Haiku for testing
        without a code change), then to DEFAULT_MODEL in
        vision_llm_detector.py.
    vision_llm_api_key : str | None
        Explicit API key; falls back to the SDK's normal env var resolution.
    """

    format_name = "image"

    def __init__(
        self,
        max_hamming: int = 10,
        table_path: str | Path | None = None,
        icons_dir: str | Path | None = None,
        enable_cnn_fallback: bool = True,
        cnn_checkpoint_path: str | Path | None = None,
        cnn_min_confidence: float = 0.85,
        use_vision_llm: bool | None = None,
        vision_llm_model: str | None = None,
        vision_llm_api_key: str | None = None,
    ) -> None:
        self._max_hamming = max_hamming
        self._table_path  = table_path
        self._icons_dir   = Path(icons_dir) if icons_dir else None
        self._enable_cnn_fallback = enable_cnn_fallback
        self._cnn_checkpoint_path = cnn_checkpoint_path
        self._cnn_min_confidence  = cnn_min_confidence
        if use_vision_llm is None:
            use_vision_llm = os.environ.get("ARCH2TERRAFORM_USE_VISION_LLM", "").strip().lower() in (
                "1", "true", "yes",
            )
        self._use_vision_llm = use_vision_llm
        self._vision_llm_model = vision_llm_model or os.environ.get("ARCH2TERRAFORM_VISION_LLM_MODEL") or None
        self._vision_llm_api_key = vision_llm_api_key

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

        if self._use_vision_llm:
            return self._parse_via_vision_llm(path, bgr)

        warnings: list[str] = []

        # ── Stage 1: container / boundary detection ──────────────────────
        logger.info("[Stage 1] Detecting containers in '%s'", path.name)
        container_nodes = detect_containers_from_array(bgr)
        logger.info("[Stage 1] Found %d container nodes", len(container_nodes))

        # ── Stage 2a: icon candidate detection ───────────────────────────
        # Passing Stage 1's container boxes lets the icon detector subtract
        # each container's own tinted background fill before contour
        # detection — otherwise an icon sitting on a filled/tinted
        # container background (e.g. a light-blue subnet box) merges with
        # that fill into one oversized blob and is lost entirely (real bug
        # found 2026-07-27 — see icon_detector.py's docstring).
        logger.info("[Stage 2a] Detecting icon candidates")
        candidates = detect_icon_candidates(
            bgr, container_bboxes=[n.bbox for n in container_nodes]
        )
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

        # ── Stage 3b: CNN fallback for icons that failed phash AND NCC ───
        # Narrow by design — only the residual that both Stage 2 and Stage 3
        # couldn't confidently identify ever reaches this stage. See
        # stage3b_cnn_classifier.py's module docstring for the reasoning.
        cnn_pending = [(node, crop) for node, crop in stage3_pending if node.shape == NodeShape.UNKNOWN]
        if cnn_pending and self._enable_cnn_fallback:
            try:
                from arch2terraform.adapters.image.stage3b_cnn_classifier import Stage3bCNNClassifier

                cnn_matcher = Stage3bCNNClassifier(
                    table,
                    checkpoint_path=self._cnn_checkpoint_path,
                    min_confidence=self._cnn_min_confidence,
                )
                cnn_matched = 0
                for node, crop_bgr in cnn_pending:
                    cnn_result = cnn_matcher.match(crop_bgr)
                    if cnn_result.confident:
                        service_name = cnn_result.service_name or "unknown"
                        node.raw_label = service_name
                        node.image_ref = service_name
                        node.shape = NodeShape.ICON
                        node.extra.update({
                            "cnn_confidence": round(cnn_result.confidence, 4),
                            "category": cnn_result.category,
                            "stage": "cnn",
                        })
                        cnn_matched += 1
                    else:
                        node.extra["cnn_confidence"] = round(cnn_result.confidence, 4)

                logger.info(
                    "[Stage 3b] CNN matched %d / %d",
                    cnn_matched, len(cnn_pending),
                )
            except (ImportError, FileNotFoundError) as exc:
                logger.debug("[Stage 3b] CNN fallback unavailable (%s) — skipping", exc)

        still_unidentified = [n for n, _ in stage3_pending if n.shape == NodeShape.UNKNOWN]
        if still_unidentified:
            warnings.append(
                f"{len(still_unidentified)} icon(s) could not be identified by phash, NCC, or the "
                "CNN fallback. OCR (Stage 4) will populate their labels where text is visible."
            )

        # ── Stage 4: OCR text labels ──────────────────────────────────────
        # Build an anchor-point map for every node so OCR blocks can be
        # assigned to the nearest element (icon or container alike).
        #
        # Containers use a top-left inset anchor, not their geometric
        # centroid. AWS diagram convention draws a container's own title
        # (e.g. "Public Subnet") right at its top-left corner — but a large
        # container's centroid can sit much closer to an unrelated floating
        # annotation (a protocol/CIDR callout like "HTTP  0.0.0.0/0" placed
        # between an ALB and the subnet it points to) than to that corner
        # text. Nearest-centroid matching then lets the floating annotation
        # win and hijack the container's label — confirmed 2026-07-24 via a
        # real test image where a Public Subnet was mislabeled "HTTP",
        # producing a bogus "aws_subnet HTTP" resource. Icons keep the
        # centroid anchor: their labels are conventionally centered directly
        # below the icon glyph, so centroid is already the right target.
        all_nodes_so_far = container_nodes + service_nodes

        def _label_anchor(n: DiagramNode) -> tuple[float, float]:
            if n.shape == NodeShape.CONTAINER:
                inset_x = min(40.0, n.bbox.width * 0.15)
                inset_y = min(30.0, n.bbox.height * 0.15)
                return (n.bbox.x + inset_x, n.bbox.y + inset_y)
            return (n.bbox.x + n.bbox.width / 2, n.bbox.y + n.bbox.height / 2)

        node_centres: dict[str, tuple[float, float]] = {
            n.id: _label_anchor(n) for n in all_nodes_so_far
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

    # ------------------------------------------------------------------
    def _parse_via_vision_llm(self, path: Path, bgr: np.ndarray) -> ParsedDiagram:
        """
        Vision-LLM path (see vision_llm_detector.py's module docstring).
        Replaces Stages 1-4 entirely. Edge/connection detection is now ALSO
        handled by the same vision-LLM call (added 2026-07-31, per explicit
        follow-up request for the model to actually understand what a
        connection between two resources means, not just that a line exists
        between them) — classical `detect_edges()` runs only as a fallback
        when the model reported zero connections (e.g. a diagram with
        genuinely no drawn arrows, or a response that omitted the
        "connections" key), so a diagram never silently ends up with no
        edges at all just because the model didn't return any.

        Any VisionLLMError is deliberately allowed to propagate (not
        swallowed into a stub/empty result) — same reasoning as
        arch2tf-product's diagram_parser.py fix, 2026-07-28: a silently
        wrong result is worse than a visible failure the caller can act on.
        """
        logger.info("[Vision LLM] Parsing '%s' via vision-LLM detection path", path.name)
        parsed = detect_via_vision_llm(
            path,
            model=self._vision_llm_model or "claude-sonnet-5",
            api_key=self._vision_llm_api_key,
        )

        if parsed.edges:
            logger.info("[Vision LLM] Using %d model-detected connections", len(parsed.edges))
            edges = parsed.edges
        else:
            logger.info("[Stage 5] Vision LLM reported no connections — falling back to classical edge detection")
            edges = detect_edges(bgr, parsed.nodes)
            logger.info("[Stage 5] Found %d edges", len(edges))

        return ParsedDiagram(
            nodes=parsed.nodes,
            edges=edges,
            source_format="image",
            source_file=str(path),
            warnings=parsed.warnings,
        )
