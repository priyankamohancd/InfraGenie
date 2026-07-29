"""
arch2terraform Bridge
-----------------------
The single integration point between Phase 2 (this FastAPI backend) and
Phase 3 (the `arch2terraform` package, ~/work/thesis/arch2terraform/).

Why this exists: Phase 2 used to have its own, unaudited reimplementations
of "classify a diagram node into an AWS resource type"
(services/parser/icon_resource_map.py) and "generate HCL for a resource"
(services/planner/terraform_planner.py's old `_resource_block`/`_format_attr`)
that never went through arch2terraform's real-`terraform validate` audit —
missing required arguments, wrong ARN formats, wrong resource type names,
phantom VPCs from unmatched containers, and (as of 2026-07-08) full nested
HCL block support all had to be found and fixed once already in
arch2terraform's catalog/classifier/generator. Reimplementing them here
separately meant none of that correctness work carried over, and any diagram
using `.drawio`/`.excalidraw`/lucidchart formats hit a bare stub — only
image uploads got real parsing, via arch2terraform's ImageAdapter, and even
then classification itself was still done locally via icon_resource_map.py
rather than arch2terraform's audited catalog.

This module routes ALL diagram formats through arch2terraform's full
front-end pipeline (parse -> classify -> resolve relationships) and maps the
result into Phase 2's `shared.schemas.models` contracts, so Phase 2 only
adds what's genuinely new: multi-module grouping, the clarification UI flow,
sandbox validation, and packaging. See terraform_planner.py for how HCL
generation itself now also delegates to arch2terraform (via
`arch2terraform.generator.hcl_format.resource_block`), and its
`_wire_containment_attrs` for the one piece of NEW work this integration
requires that arch2terraform itself never needed: arch2terraform always
emits one flat `main.tf`, so a containment reference like
`subnet_id = aws_subnet.x.id` always resolves. Phase 2 splits resources
across multiple Terraform modules (networking/compute/database/...), so a
containment reference is only safe to emit directly when parent and child
land in the *same* module — see terraform_planner.py for the cross-module
fallback.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── Path setup ──────────────────────────────────────────────────────────────
# arch2terraform package lives two levels above arch2tf-product/
_ARCH2TF_SRC = Path(__file__).resolve().parents[5] / "arch2terraform" / "src"
if str(_ARCH2TF_SRC) not in sys.path:
    sys.path.insert(0, str(_ARCH2TF_SRC))

_PRODUCT_ROOT = Path(__file__).resolve().parents[4]
if str(_PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRODUCT_ROOT))

from arch2terraform.adapters.registry import parse_diagram as _a2tf_parse_diagram
from arch2terraform.classifier.classifier import classify_diagram as _a2tf_classify_diagram
from arch2terraform.resolver.resolver import resolve_relationships as _a2tf_resolve_relationships
from arch2terraform.schemas.resources import ResourceGraph as A2TFResourceGraph

from shared.schemas.models import DiagramFormat, ParsedConnection, ParsedDiagram, ParsedResource

# arch2terraform's ParsedDiagram.source_format is a plain string set per-adapter
# (see each adapter's `source_format` value) — map to Phase 2's enum.
_FORMAT_MAP: dict[str, DiagramFormat] = {
    "drawio": DiagramFormat.DRAWIO,
    "excalidraw": DiagramFormat.EXCALIDRAW,
    "lucidchart": DiagramFormat.LUCIDCHART,
    "image": DiagramFormat.IMAGE,
}

# arch2terraform's resolver.py relationship_type -> Phase 2's connection_type.
# "containment" passes through unchanged (see models.py's ParsedConnection
# docstring for why it's handled specially rather than folded into a generic
# bucket).
_RELATIONSHIP_TYPE_MAP: dict[str, str] = {
    "containment": "containment",
    "network_ingress": "security",
    "routes_to": "dependency",
    "iam_attachment": "iam",
}

# arch2terraform/classifier/classifier.py's confidence tiers — mirrored here
# only to derive Phase 2's `match_source` label for the UI, not to
# re-implement any matching logic.
_ICON_MATCH_CONFIDENCE = 0.95
_LABEL_MATCH_CONFIDENCE = 0.65


def run_arch2terraform_pipeline(file_path: str, original_filename: str) -> ParsedDiagram:
    """
    Parse + classify + resolve relationships for `file_path` using
    arch2terraform's full pipeline (works for .drawio/.xml, .excalidraw,
    lucidchart .svg/.csv, and image formats — whichever adapter
    arch2terraform's registry picks for the file), then map the result into
    Phase 2's ParsedDiagram contract.

    Raises whatever arch2terraform's registry/adapters raise (e.g.
    UnsupportedFormatError, FileNotFoundError) — diagram_parser.py is
    responsible for catching and converting these into a job failure.
    """
    diagram = _a2tf_parse_diagram(file_path)
    classified, unclassified = _a2tf_classify_diagram(diagram)
    graph = _a2tf_resolve_relationships(diagram, classified)

    if unclassified:
        log.warning(
            "%s: %d diagram node(s) could not be classified by arch2terraform: %s",
            original_filename, len(unclassified), unclassified,
        )
    for w in diagram.warnings:
        log.warning("%s: arch2terraform adapter warning: %s", original_filename, w)

    resources = [_to_parsed_resource(r) for r in graph.resources]
    connections = _to_parsed_connections(graph)

    type_summary: dict[str, int] = {}
    for r in resources:
        type_summary[r.aws_resource_type] = type_summary.get(r.aws_resource_type, 0) + 1

    return ParsedDiagram(
        source_format=_FORMAT_MAP.get(diagram.source_format, DiagramFormat.UNKNOWN),
        resources=resources,
        connections=connections,
        total_resources=len(resources),
        total_connections=len(connections),
        resource_type_summary=type_summary,
    )


def _to_parsed_resource(classified) -> ParsedResource:
    """Maps an arch2terraform ClassifiedResource -> Phase 2's ParsedResource.
    node_id maps 1:1 to ParsedResource.id since nothing downstream of the
    classifier regenerates IDs."""
    if classified.confidence >= _ICON_MATCH_CONFIDENCE:
        match_source = "style"
    elif classified.confidence >= _LABEL_MATCH_CONFIDENCE:
        match_source = "label"
    else:
        match_source = "fallback"

    return ParsedResource(
        id=classified.node_id,
        aws_resource_type=classified.resource_type,
        logical_name=classified.terraform_name,
        label=classified.display_label,
        properties=dict(classified.attributes),
        nested_blocks=dict(classified.nested_blocks),
        confidence=classified.confidence,
        match_source=match_source,
        # Diagram-native custom-data tags (draw.io Edit Data, Excalidraw
        # customData) — see arch2terraform's DiagramNode.tags docstring.
        # ParsedResource.tags already existed on this model but nothing
        # populated it until now.
        tags=dict(classified.tags),
        # Extra top-level HCL resources (e.g. aws_mq_broker's
        # random_password/Secrets Manager pair) that must land in the same
        # module as this resource — see ParsedResource.companion_blocks.
        companion_blocks=list(classified.companion_blocks),
    )


def _to_parsed_connections(graph: A2TFResourceGraph) -> list[ParsedConnection]:
    connections: list[ParsedConnection] = []
    for rel in graph.relationships:
        # arch2terraform's ResourceRelationship carries the diagram edge's
        # original label (e.g. "PostgreSQL", "Read"), but nothing downstream
        # was reading it - carried through under the "_"-prefixed convention
        # terraform_planner.py's conn_attrs loop already uses to mean
        # "metadata, not a literal resource attribute" (see _resource_block_lines
        # / _generate_module_hcl). The security engine bridge (security_bridge.py)
        # is the first consumer: it needs edge labels for its traffic-flow
        # Level-1 inference (label -> protocol/port), which silently never
        # fired before this since attribute_map was always empty.
        attribute_map = {"_label": rel.label} if rel.label else {}
        connections.append(ParsedConnection(
            source_id=rel.source_node_id,
            target_id=rel.target_node_id,
            connection_type=_RELATIONSHIP_TYPE_MAP.get(rel.relationship_type, "dependency"),
            attribute_map=attribute_map,
        ))
    return connections
