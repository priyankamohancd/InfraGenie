"""
Schema for classified AWS resources — the output of the classifier stage.

A DiagramNode becomes a ClassifiedResource once we know which Terraform
resource type it maps to. The relationship resolver then links these
together using DiagramEdges + containment, producing a ResourceGraph that
the HCL generator can walk directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassifiedResource:
    """One AWS resource, identified from a diagram node."""

    node_id: str                      # back-reference to DiagramNode.id
    resource_type: str                # e.g. "aws_instance", "aws_s3_bucket"
    terraform_name: str               # sanitized local name, e.g. "web_server_1"
    display_label: str                # original human label, for comments/README
    confidence: float                  # 0.0–1.0, from classifier matching
    attributes: dict = field(default_factory=dict)   # inferred/defaulted TF attributes
    is_container: bool = False        # true for VPC/subnet/security-group boxes
    needs_clarification: list[str] = field(default_factory=list)  # ambiguous attrs flagged for Phase 2's clarifier


@dataclass
class ResourceRelationship:
    """A resolved relationship between two classified resources."""

    source_node_id: str
    target_node_id: str
    relationship_type: str    # e.g. "network_ingress", "containment", "iam_attachment", "routes_to"
    label: str = ""


@dataclass
class ResourceGraph:
    """Fully resolved graph: classified resources + relationships, ready for HCL generation."""

    resources: list[ClassifiedResource]
    relationships: list[ResourceRelationship]
    unclassified_nodes: list[str] = field(default_factory=list)  # node ids the classifier couldn't map

    def resource_by_node_id(self, node_id: str) -> ClassifiedResource | None:
        for r in self.resources:
            if r.node_id == node_id:
                return r
        return None

    def children_of(self, container_node_id: str) -> list[ClassifiedResource]:
        """Resources contained within a given container (VPC, subnet, SG box)."""
        child_ids = {
            rel.target_node_id
            for rel in self.relationships
            if rel.source_node_id == container_node_id and rel.relationship_type == "containment"
        }
        return [r for r in self.resources if r.node_id in child_ids]
