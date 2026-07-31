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
    nested_blocks: dict = field(default_factory=dict)  # required HCL nested blocks, e.g. vpc_config
    is_container: bool = False        # true for VPC/subnet/security-group boxes
    needs_clarification: list[str] = field(default_factory=list)  # ambiguous attrs flagged for Phase 2's clarifier
    # Carried through unchanged from DiagramNode.tags — the diagram's own
    # native custom-data metadata (draw.io Edit Data, Excalidraw customData).
    # Never interpreted here; classification/relationship-resolution doesn't
    # need it. It exists on this schema purely as a pass-through so
    # downstream consumers (Phase 2's ParsedResource.tags, and eventually
    # policy-pack selection) don't need their own separate lookup back to
    # the original DiagramNode.
    tags: dict[str, str] = field(default_factory=dict)
    # Extra, pre-rendered top-level HCL resource blocks that must be emitted
    # alongside this resource in the same file/module (e.g. a random_password
    # + aws_secretsmanager_secret pair backing an aws_mq_broker's password,
    # since unlike RDS/Aurora there's no manage_master_user_password-style
    # AWS-managed-secret flag for MQ — see classifier.py's
    # _build_mq_broker_companion_blocks()). Empty for every other resource
    # type. Rendered as raw HCL text (not another ResourceDefinition) since
    # these companions don't need their own classification/confidence/
    # nested_blocks machinery — they exist purely to back one attribute on
    # the resource that declares them.
    companion_blocks: list[str] = field(default_factory=list)


@dataclass
class ResourceRelationship:
    """A resolved relationship between two classified resources."""

    source_node_id: str
    target_node_id: str
    relationship_type: str    # e.g. "network_ingress", "containment", "iam_attachment", "routes_to"
    label: str = ""
    # Added 2026-07-31: the Vision-LLM's own semantic read of what this
    # connection MEANS ("read"/"write"/"manage"/"read_write"/"network"),
    # using the whole diagram's context — not just this edge's label text in
    # isolation. Empty when the classical (non-Vision-LLM) pipeline produced
    # this edge, or when the model wasn't confident enough to commit to one;
    # arch2tf-product's dynamic_iam_generator.py prefers this over its own
    # keyword-substring inference from the edge label when set. See
    # vision_llm_detector.py's module docstring.
    operation_hint: str = ""


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
