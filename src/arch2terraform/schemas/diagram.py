"""
Canonical intermediate representation (IR) for parsed architecture diagrams.

Every format adapter (draw.io, Lucidchart, Excalidraw, image-cascade stub)
normalizes its raw input into this schema. Nothing downstream — classifier,
resolver, HCL generator — ever touches a format-specific structure again.
This is what makes the diagram the single source of truth: one contract,
many producers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeShape(str, Enum):
    RECTANGLE = "rectangle"
    CYLINDER = "cylinder"
    CLOUD = "cloud"
    DIAMOND = "diamond"
    CIRCLE = "circle"
    ICON = "icon"
    CONTAINER = "container"  # VPC/subnet boxes that other nodes sit inside
    UNKNOWN = "unknown"


class EdgeStyle(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, other: "BoundingBox") -> bool:
        """True if `other` sits geometrically inside this box (for container detection)."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )


@dataclass
class DiagramNode:
    """A single shape/icon on the diagram, before any AWS resource classification."""

    id: str
    raw_label: str
    shape: NodeShape
    bbox: BoundingBox
    style_raw: str = ""              # original style string, format-specific, kept for debugging
    image_ref: str | None = None     # icon/image identifier if the shape carries one (e.g. mxgraph.aws4.ec2)
    fill_color: str | None = None
    parent_id: str | None = None     # explicit parent from the format (e.g. draw.io group/container)
    source_format: str = "unknown"
    extra: dict = field(default_factory=dict)  # format-specific leftovers, never read downstream
    # Native custom-data metadata attached to the shape in the diagramming
    # tool itself — draw.io's "Edit Data" (serialized as an <object>/
    # <UserObject> wrapper around the <mxCell>), Excalidraw's per-element
    # `customData` field. Deliberately a real, first-class field (not folded
    # into `extra`, which is documented as unread downstream) since this IS
    # meant to be read downstream: it's how a diagram stays the source of
    # truth for policy-relevant intent (tier=prod, pii=true, public=true)
    # without that intent living in a second file. Empty for untagged shapes
    # and for formats that can't carry it (Lucidchart's CSV export has no
    # custom-data column — see lucidchart_adapter.py).
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class DiagramEdge:
    """A connector between two nodes."""

    id: str
    source_id: str
    target_id: str
    label: str = ""
    style: EdgeStyle = EdgeStyle.UNKNOWN
    extra: dict = field(default_factory=dict)


@dataclass
class ParsedDiagram:
    """Top-level output of any adapter. This is the only thing the classifier consumes."""

    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    source_format: str
    source_file: str
    warnings: list[str] = field(default_factory=list)

    def node_by_id(self, node_id: str) -> DiagramNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None
