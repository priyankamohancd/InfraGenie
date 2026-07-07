"""
Classifier: turns each DiagramNode into a ClassifiedResource by matching
against the AWS resource catalog.

Matching priority:
  1. icon_ref match (image_ref from draw.io stencils / Lucidchart shape library)
     — high confidence, this is metadata the diagram tool itself attached.
  2. label keyword match — lower confidence, used for Excalidraw and as
     fallback when icon match fails or is absent.
  3. No match -> node goes into unclassified_nodes, never silently dropped.

Confidence scores feed Phase 2's clarification-question logic: low-confidence
matches get flagged in needs_clarification so the user can confirm/correct
before Terraform is generated.
"""

from __future__ import annotations

import re

from arch2terraform.classifier.catalog import CATALOG, ResourceDefinition
from arch2terraform.schemas.diagram import DiagramNode, NodeShape, ParsedDiagram
from arch2terraform.schemas.resources import ClassifiedResource

_ICON_MATCH_CONFIDENCE = 0.95
_LABEL_MATCH_CONFIDENCE = 0.65
_CONTAINER_SHAPE_FALLBACK_CONFIDENCE = 0.5

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

# Container types that are AWS's implicit structural boundaries rather than
# provisionable resources:
#   - AWS Cloud is just the outer partition boundary every diagram has — there
#     is no corresponding Terraform resource at all.
#   - An Availability Zone is never created as its own resource; it's an
#     attribute (`availability_zone = "..."`) set on the zonal resources
#     inside it (subnets, EBS volumes, etc.).
# These come from the image adapter's layout_detector (see
# adapters/image/layout_detector.py::_CONTAINER_IMAGE_REF). Before this set
# existed, both fell through to the generic "unmatched container -> aws_vpc"
# fallback below, silently emitting phantom VPC resources that don't belong
# in the diagram's actual infrastructure — a correctness bug for a tool whose
# whole point is trustworthy IaC from the diagram. They're skipped here
# (not classified, not flagged as unclassified/needs-review) since they are
# expected, recognized structure, not something the user needs to fix.
_STRUCTURAL_ONLY_IMAGE_REFS = {"aws-cloud", "availability-zone"}


def _normalize_ref(ref: str) -> str:
    """Lowercase and strip separators so 'Security-Group' == 'securitygroup'.

    Different adapters format image_ref differently (draw.io stencil names
    have no separators, e.g. 'mxgraph.aws4.securityGroup'; the image adapter's
    layout_detector emits hyphenated names, e.g. 'Security-Group'). Catalog
    icon_keys are written without separators, so without this normalization,
    any hyphenated image_ref with the key split across a hyphen (e.g.
    'security-group' vs. key 'securitygroup') silently fails to match and
    falls through to the container fallback -> phantom aws_vpc bug above.
    """
    return _NON_ALNUM_RE.sub("", ref.lower())


def classify_diagram(diagram: ParsedDiagram) -> tuple[list[ClassifiedResource], list[str]]:
    """Returns (classified resources, ids of nodes that couldn't be classified)."""
    classified: list[ClassifiedResource] = []
    unclassified: list[str] = []
    used_names: set[str] = set()

    for node in diagram.nodes:
        if node.shape == NodeShape.CONTAINER and (node.image_ref or "").lower() in _STRUCTURAL_ONLY_IMAGE_REFS:
            continue  # recognized structural boundary, not a resource — see comment above

        result = _classify_node(node)
        if result is None:
            unclassified.append(node.id)
            continue

        definition, confidence = result
        tf_name = _unique_terraform_name(node, used_names)
        needs_clarification = [] if confidence >= _ICON_MATCH_CONFIDENCE else ["resource_type"]

        classified.append(
            ClassifiedResource(
                node_id=node.id,
                resource_type=definition.terraform_type,
                terraform_name=tf_name,
                display_label=node.raw_label or definition.terraform_type,
                confidence=confidence,
                attributes=dict(definition.default_attributes),
                is_container=definition.is_container,
                needs_clarification=needs_clarification,
            )
        )

    return classified, unclassified


def _classify_node(node: DiagramNode) -> tuple[ResourceDefinition, float] | None:
    icon_match = _match_by_icon(node)
    if icon_match:
        return icon_match, _ICON_MATCH_CONFIDENCE

    label_match = _match_by_label(node)
    if label_match:
        return label_match, _LABEL_MATCH_CONFIDENCE

    # Generic container shapes with no specific match (e.g. an unlabeled box
    # drawn as a container) still get treated as a container with low confidence,
    # since dropping them would break containment relationships downstream.
    if node.shape == NodeShape.CONTAINER:
        vpc_def = next(d for d in CATALOG if d.terraform_type == "aws_vpc")
        return vpc_def, _CONTAINER_SHAPE_FALLBACK_CONFIDENCE

    return None


def _match_by_icon(node: DiagramNode) -> ResourceDefinition | None:
    if not node.image_ref:
        return None
    ref = _normalize_ref(node.image_ref)

    # Collect every (definition, matched_key) pair across the whole catalog rather
    # than stopping at the first list entry that matches anything. Generic keys like
    # "instance" (EC2) are substrings of more specific labels like "RDS Instance", so
    # without this, catalog order alone would decide the result. Picking the longest
    # matched key makes the more specific catalog entry win regardless of list position.
    candidates: list[tuple[ResourceDefinition, str]] = []
    for definition in CATALOG:
        for key in definition.icon_keys:
            if key in ref:
                candidates.append((definition, key))

    if not candidates:
        return None

    best_definition, _ = max(candidates, key=lambda pair: len(pair[1]))
    return best_definition


def _match_by_label(node: DiagramNode) -> ResourceDefinition | None:
    label = (node.raw_label or "").lower().strip()
    if not label:
        return None

    candidates: list[tuple[ResourceDefinition, str]] = []
    for definition in CATALOG:
        for kw in definition.label_keywords:
            if kw in label:
                candidates.append((definition, kw))

    if not candidates:
        return None

    best_definition, _ = max(candidates, key=lambda pair: len(pair[1]))
    return best_definition


def _unique_terraform_name(node: DiagramNode, used_names: set[str]) -> str:
    base = node.raw_label or node.id
    slug = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_") or "resource"
    if not re.match(r"^[a-z_]", slug):
        slug = f"r_{slug}"

    name = slug
    suffix = 1
    while name in used_names:
        suffix += 1
        name = f"{slug}_{suffix}"
    used_names.add(name)
    return name
