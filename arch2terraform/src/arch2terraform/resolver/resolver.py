"""
Relationship resolver: turns DiagramEdges + geometric/explicit containment
into a ResourceGraph of typed ResourceRelationships.

Two sources of relationships:
  1. Explicit edges (arrows/connectors) -> classified by simple heuristics
     into network_ingress, routes_to, or iam_attachment based on the two
     endpoint resource types.
  2. Containment -> a node whose parent_id points to a container resource,
     or whose bbox sits inside a container's bbox, becomes a "containment"
     relationship (e.g. EC2 instance inside a VPC/subnet box).

This is deliberately rule-based rather than ML-based: relationships in
architecture diagrams are conventionally drawn, and the diagram is the
source of truth, so we trust its explicit structure over inference.
"""

from __future__ import annotations

from arch2terraform.schemas.diagram import ParsedDiagram
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph, ResourceRelationship

_IAM_TYPES = {"aws_iam_role", "aws_iam_policy", "aws_iam_user"}
_NETWORK_EDGE_TYPES = {
    "aws_security_group", "aws_network_acl", "aws_route_table",
    "aws_internet_gateway", "aws_nat_gateway",
}


def resolve_relationships(
    diagram: ParsedDiagram, classified: list[ClassifiedResource]
) -> ResourceGraph:
    by_node_id = {r.node_id: r for r in classified}
    relationships: list[ResourceRelationship] = []

    relationships.extend(_resolve_explicit_edges(diagram, by_node_id))
    relationships.extend(_resolve_containment(diagram, by_node_id))

    unclassified = [n.id for n in diagram.nodes if n.id not in by_node_id]

    return ResourceGraph(
        resources=classified,
        relationships=relationships,
        unclassified_nodes=unclassified,
    )


def _resolve_explicit_edges(
    diagram: ParsedDiagram, by_node_id: dict[str, ClassifiedResource]
) -> list[ResourceRelationship]:
    relationships: list[ResourceRelationship] = []

    for edge in diagram.edges:
        source = by_node_id.get(edge.source_id)
        target = by_node_id.get(edge.target_id)
        if source is None or target is None:
            continue  # edge touches an unclassified node; skip rather than guess

        rel_type = _classify_edge_relationship(source, target)
        relationships.append(
            ResourceRelationship(
                source_node_id=edge.source_id,
                target_node_id=edge.target_id,
                relationship_type=rel_type,
                label=edge.label,
            )
        )

    return relationships


def _classify_edge_relationship(source: ClassifiedResource, target: ClassifiedResource) -> str:
    if source.resource_type in _IAM_TYPES or target.resource_type in _IAM_TYPES:
        return "iam_attachment"
    if source.resource_type in _NETWORK_EDGE_TYPES or target.resource_type in _NETWORK_EDGE_TYPES:
        return "network_ingress"
    return "routes_to"


def _resolve_containment(
    diagram: ParsedDiagram, by_node_id: dict[str, ClassifiedResource]
) -> list[ResourceRelationship]:
    relationships: list[ResourceRelationship] = []
    seen_pairs: set[tuple[str, str]] = set()

    for node in diagram.nodes:
        if node.id not in by_node_id:
            continue

        container_id = _find_container_id(node, diagram, by_node_id)
        if container_id and container_id != node.id:
            pair = (container_id, node.id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                relationships.append(
                    ResourceRelationship(
                        source_node_id=container_id,
                        target_node_id=node.id,
                        relationship_type="containment",
                    )
                )

    return relationships


def _find_container_id(node, diagram: ParsedDiagram, by_node_id: dict[str, ClassifiedResource]) -> str | None:
    # 1. Explicit parent reference from the format (most reliable)
    if node.parent_id and node.parent_id in by_node_id:
        candidate = by_node_id[node.parent_id]
        if candidate.is_container:
            return node.parent_id

    # 2. Geometric containment fallback (covers formats/diagrams without
    #    explicit parent linkage, e.g. loosely grouped Excalidraw shapes)
    best_container = None
    best_area = None
    for other in diagram.nodes:
        if other.id == node.id or other.id not in by_node_id:
            continue
        if not by_node_id[other.id].is_container:
            continue
        if other.bbox.contains(node.bbox):
            area = other.bbox.width * other.bbox.height
            if best_area is None or area < best_area:
                best_area = area
                best_container = other.id

    return best_container
