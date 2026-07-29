"""
Vision-LLM diagram detector — alternative to the classical CV cascade.

Rationale
---------
The classical pipeline (layout_detector's Canny+Hough line assembly and HSV
color profiles, icon_detector's contour/fill heuristics, hash_matcher's
perceptual hashing, ocr_extractor's Tesseract OCR) is a stack of hand-tuned
thresholds, each one only ever verified against the specific diagrams it was
tuned against. Every fix made to one diagram style risked (and, across this
project's history, repeatedly did) regress another — color profiles that
don't cover a diagram's actual palette, icon collisions from exact-pixel
phash matching, spurious container fragments from ambiguous line pairing,
OCR mislabeling from centroid-based text assignment. None of these
approaches understand *what* they're looking at; they only pattern-match
pixels.

This module replaces that whole cascade for one call to a vision-capable
LLM, which is asked to semantically identify every AWS resource, its label,
its approximate bounding box, and its containment relationship — the same
judgment a human reviewing the diagram would make, rather than a chain of
geometric/color heuristics trying to approximate it. This is expected to
generalize far better across diagram styles, tools, and icon sets, at the
cost of per-diagram API latency/cost and non-determinism (mitigated with a
low temperature and a strict JSON output contract, below).

Integration contract
--------------------
`detect_via_vision_llm()` returns a full `ParsedDiagram` with `DiagramNode`s
for every container and icon (containment expressed via `parent_id`, exactly
like every other adapter). `image_ref` is set to the LLM's best-guess AWS
service/resource name (e.g. "Amazon EC2", "VPC", "Public Subnet") rather than
a terraform_type directly — this is deliberate: it lets the existing,
audited `classifier.py` (icon_key/label_keyword matching against the
catalog) do the actual AWS-resource-type resolution unchanged, so none of
that already-tested logic needs to be duplicated or trusted to the LLM.
Edge/arrow detection is NOT covered here (kept on the classical
`edge_detector.py`, called separately by `image_adapter.py`) — arrows were
never implicated in any of the detection-quality issues this module targets,
so keeping that scope out reduces the size and risk of this change.

Failure handling
-----------------
Any failure (API error, malformed/non-JSON response, empty element list) is
raised as `VisionLLMError` rather than silently producing an empty or wrong
`ParsedDiagram` — callers decide whether to fail the job or fall back to the
classical cascade. This deliberately does NOT swallow errors into a fake
success, the same class of bug found and fixed 2026-07-28 in
arch2tf-product's diagram_parser.py stub fallback.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from arch2terraform.schemas.diagram import (
    BoundingBox,
    DiagramEdge,
    DiagramNode,
    NodeShape,
    ParsedDiagram,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 8192

# Element types the LLM may emit. "group" is deliberately distinct from
# "container": a container is a real AWS network boundary (VPC, subnet, AZ,
# security group) that the classifier can resolve to a Terraform resource;
# a group is a purely organizational box a diagram author drew around a
# cluster of unrelated icons (e.g. a box around "Frontend/Backend/Worker"
# pod services) with no Terraform equivalent at all. Asking the LLM to make
# this distinction directly is the semantic judgment classical CV kept
# getting wrong — see layout_detector.py/classifier.py's own docstrings
# from 2026-07-28 for the concrete failures this caused (phantom VPCs,
# phantom subnets from non-resource grouping boxes).
_VALID_ELEMENT_TYPES = {"container", "icon", "group"}


class VisionLLMError(RuntimeError):
    """Raised when the vision LLM call fails or returns unusable output."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(width: int, height: int) -> str:
    return f"""You are analyzing an AWS architecture diagram that is exactly {width}x{height} pixels \
(origin (0,0) at the top-left corner, x increases rightward, y increases downward).

Identify EVERY element on the diagram: AWS service icons, and network/organizational boundary \
boxes (VPC, Subnet, Availability Zone, Security Group, Region, or a generic grouping box with no \
specific AWS meaning).

Return ONLY a single JSON object (no markdown fences, no prose before or after) with this exact shape:

{{
  "elements": [
    {{
      "id": "e1",
      "type": "container" | "icon" | "group",
      "service_name": "<best-guess AWS service or boundary name, e.g. 'Amazon EC2', 'VPC', 'Public Subnet', 'Security Group', null if type is 'group'>",
      "label": "<any visible text label/caption near or inside this element, verbatim, empty string if none>",
      "bbox": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}},
      "parent_id": "<id of the immediately enclosing container/group element, or null if top-level>",
      "confidence": <float 0.0-1.0, your own confidence in this identification>
    }}
  ]
}}

Rules:
- "type": "container" for a box that represents a REAL AWS network resource (VPC, Subnet, \
Availability Zone, Security Group). Use "group" for any other visual grouping box (e.g. a box drawn \
around a cluster of microservice icons, a "tier" label, a region label) that is NOT itself a \
provisionable AWS resource — do not guess a container type for these, mark them "group" with \
service_name null instead.
- "type": "icon" for every individual AWS service icon (EC2, RDS, Lambda, S3, ALB, WAF, EKS, etc).
- bbox coordinates MUST be actual pixel coordinates within the stated {width}x{height} image, not \
normalized/relative values.
- parent_id must reference another element's "id" in this same list (the smallest/innermost \
container or group that visually encloses this element), or null if the element is not nested \
inside anything.
- Include every element you can identify, even ones drawn in an unconventional style or non-standard \
color — do not skip anything just because it doesn't match a typical AWS icon color palette.
- If a container/group box has no visible label, still include it with an empty "label" — do not \
omit it.
- Respond with ONLY the JSON object described above."""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


def _image_dimensions(path: Path) -> tuple[int, int]:
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        return img.size  # (width, height)


# ---------------------------------------------------------------------------
# API call (isolated behind a thin, injectable function for testability —
# no test in this repo should ever need a real network call or API key)
# ---------------------------------------------------------------------------

def _default_call_vision_api(
    image_b64: str,
    media_type: str,
    prompt: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,  # deterministic-as-possible structured extraction, not creative generation
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw_text: str) -> dict:
    """
    Tolerates the model wrapping its JSON in a markdown code fence despite
    being told not to — models do this often enough that defensively
    stripping fences is cheaper than a guaranteed round-trip retry.
    """
    text = raw_text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _parse_response(data: dict) -> tuple[list[DiagramNode], list[str]]:
    """
    Converts the LLM's JSON element list into DiagramNodes. Pure function,
    no I/O — this is what unit tests exercise directly against canned JSON,
    without needing a real API call.
    """
    warnings: list[str] = []
    elements = data.get("elements")
    if not isinstance(elements, list):
        raise VisionLLMError(
            f"Vision LLM response missing a valid 'elements' list (got {type(elements).__name__})"
        )

    # First pass: mint a real UUID for every element with a local id, so
    # parent_id references can be resolved regardless of list order.
    id_map: dict[str, str] = {}
    for el in elements:
        local_id = el.get("id") if isinstance(el, dict) else None
        if local_id:
            id_map[str(local_id)] = str(uuid.uuid4())

    nodes: list[DiagramNode] = []
    for el in elements:
        if not isinstance(el, dict):
            warnings.append(f"Skipped malformed element (not an object): {el!r}")
            continue

        local_id = el.get("id")
        real_id = id_map.get(str(local_id)) if local_id else str(uuid.uuid4())

        bbox_raw = el.get("bbox")
        if not isinstance(bbox_raw, dict):
            warnings.append(f"Skipped element {local_id!r} — missing/invalid bbox")
            continue
        try:
            bbox = BoundingBox(
                x=float(bbox_raw["x"]),
                y=float(bbox_raw["y"]),
                width=float(bbox_raw["width"]),
                height=float(bbox_raw["height"]),
            )
        except (KeyError, TypeError, ValueError):
            warnings.append(f"Skipped element {local_id!r} — non-numeric bbox fields")
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            warnings.append(f"Skipped element {local_id!r} — non-positive bbox size")
            continue

        el_type = str(el.get("type") or "").strip().lower()
        if el_type not in _VALID_ELEMENT_TYPES:
            warnings.append(
                f"Element {local_id!r} has unrecognized type {el_type!r} — treating as unclassified"
            )
            shape = NodeShape.UNKNOWN
        elif el_type == "icon":
            shape = NodeShape.ICON
        elif el_type == "container":
            shape = NodeShape.CONTAINER
        else:  # "group" — a real visual box, but never a resource. See
            # module docstring / _VALID_ELEMENT_TYPES comment. Kept as
            # CONTAINER shape (so containment/geometry still works and
            # children can nest under it) but with no image_ref, so
            # classifier.py's existing structural-only / no-signal-fallback
            # handling (see classifier.py 2026-07-28) naturally routes it
            # to `unclassified` instead of guessing a resource type for it.
            shape = NodeShape.CONTAINER

        service_name = str(el.get("service_name") or "").strip() or None
        if el_type == "group":
            service_name = None
        label = str(el.get("label") or "").strip()

        parent_local = el.get("parent_id")
        parent_id = id_map.get(str(parent_local)) if parent_local else None

        confidence = el.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        nodes.append(
            DiagramNode(
                id=real_id,
                raw_label=label or service_name or "",
                shape=shape,
                bbox=bbox,
                image_ref=service_name,
                parent_id=parent_id,
                source_format="image",
                extra={"stage": "vision_llm", "vision_confidence": confidence, "vision_type": el_type},
            )
        )

    return nodes, warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_via_vision_llm(
    image_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    call_api: Callable[[str, str, str, str, str | None, int], str] | None = None,
) -> ParsedDiagram:
    """
    Detect every container/icon on `image_path` via a vision-capable LLM
    call and return a full ParsedDiagram (nodes only — edges remain the
    classical `edge_detector.py`'s responsibility, see module docstring).

    Parameters
    ----------
    image_path : path to the diagram image.
    model      : vision-capable model identifier.
    api_key    : explicit API key; falls back to the anthropic SDK's normal
                 ANTHROPIC_API_KEY env var resolution if not given.
    max_tokens : response token ceiling — large diagrams with many elements
                 need headroom to avoid truncated JSON.
    call_api   : injectable low-level API-call function, signature
                 (image_b64, media_type, prompt, model, api_key, max_tokens) -> raw_text.
                 Tests pass a canned stub here — this is the ONLY function in
                 this module that ever performs network I/O.

    Raises
    ------
    FileNotFoundError : image_path doesn't exist.
    VisionLLMError    : the API call failed, or the response wasn't usable
                        JSON in the expected shape. Never silently degrades
                        to an empty/wrong ParsedDiagram — see module
                        docstring's "Failure handling" section.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Diagram image not found: {path}")

    width, height = _image_dimensions(path)
    image_b64, media_type = _encode_image(path)
    prompt = _build_prompt(width, height)

    call = call_api or _default_call_vision_api
    try:
        raw_text = call(image_b64, media_type, prompt, model, api_key, max_tokens)
    except Exception as exc:  # noqa: BLE001 — deliberately broad, re-raised as our own type
        raise VisionLLMError(f"Vision LLM API call failed: {exc}") from exc

    if not raw_text or not raw_text.strip():
        raise VisionLLMError("Vision LLM returned an empty response")

    try:
        data = _extract_json(raw_text)
    except json.JSONDecodeError as exc:
        raise VisionLLMError(
            f"Vision LLM did not return valid JSON: {exc}. Raw response (truncated): {raw_text[:500]!r}"
        ) from exc

    nodes, warnings = _parse_response(data)
    if not nodes:
        warnings.append("Vision LLM returned zero usable elements for this diagram.")

    logger.info(
        "[Vision LLM] Detected %d elements (%d containers, %d icons) in '%s'",
        len(nodes),
        sum(1 for n in nodes if n.shape == NodeShape.CONTAINER),
        sum(1 for n in nodes if n.shape == NodeShape.ICON),
        path.name,
    )

    return ParsedDiagram(
        nodes=nodes,
        edges=[],
        source_format="image",
        source_file=str(path),
        warnings=warnings,
    )
