"""
Excalidraw adapter.

Excalidraw exports a flat JSON file: {"type": "excalidraw", "elements": [...]}.
Elements are absolute-positioned rectangles/diamonds/ellipses/text/arrows.
Text elements often act as floating labels rather than bound text, so we
attach the nearest unbound text element to a shape by containment/proximity.

There's no AWS icon stencil system in Excalidraw, so resource hints come
entirely from text labels — the classifier leans more heavily on label
matching for diagrams from this adapter.
"""

from __future__ import annotations

import json

from arch2terraform.adapters.base import BaseAdapter
from arch2terraform.schemas.diagram import (
    BoundingBox,
    DiagramEdge,
    DiagramNode,
    EdgeStyle,
    NodeShape,
    ParsedDiagram,
)

_SHAPE_MAP = {
    "rectangle": NodeShape.RECTANGLE,
    "diamond": NodeShape.DIAMOND,
    "ellipse": NodeShape.CIRCLE,
}


class ExcalidrawAdapter(BaseAdapter):
    format_name = "excalidraw"

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(".excalidraw") or file_path.lower().endswith(".excalidraw.json")

    def parse(self, file_path: str) -> ParsedDiagram:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        elements = data.get("elements", [])
        warnings: list[str] = []

        shape_elements = [e for e in elements if e.get("type") in _SHAPE_MAP and not e.get("isDeleted")]
        text_elements = [e for e in elements if e.get("type") == "text" and not e.get("isDeleted")]
        arrow_elements = [e for e in elements if e.get("type") == "arrow" and not e.get("isDeleted")]

        nodes: list[DiagramNode] = []
        for el in shape_elements:
            bbox = BoundingBox(
                x=float(el.get("x", 0)),
                y=float(el.get("y", 0)),
                width=float(el.get("width", 0)),
                height=float(el.get("height", 0)),
            )
            label = self._resolve_label(el, text_elements)
            container_id = el.get("frameId")

            nodes.append(
                DiagramNode(
                    id=el.get("id", ""),
                    raw_label=label,
                    shape=_SHAPE_MAP.get(el.get("type"), NodeShape.UNKNOWN),
                    bbox=bbox,
                    fill_color=el.get("backgroundColor"),
                    parent_id=container_id,
                    source_format=self.format_name,
                )
            )

        # Containers in Excalidraw are usually just bigger rectangles drawn first
        # and geometrically containing others. Mark those as CONTAINER post-hoc.
        self._mark_geometric_containers(nodes)

        edges: list[DiagramEdge] = []
        for el in arrow_elements:
            source = el.get("startBinding", {}).get("elementId") if el.get("startBinding") else None
            target = el.get("endBinding", {}).get("elementId") if el.get("endBinding") else None
            if not source or not target:
                warnings.append(f"Skipped unbound arrow {el.get('id')}: missing start/end binding")
                continue

            edge_style = EdgeStyle.DASHED if el.get("strokeStyle") == "dashed" else EdgeStyle.SOLID
            edges.append(
                DiagramEdge(
                    id=el.get("id", ""),
                    source_id=source,
                    target_id=target,
                    label=self._resolve_label(el, text_elements),
                    style=edge_style,
                )
            )

        return ParsedDiagram(
            nodes=nodes,
            edges=edges,
            source_format=self.format_name,
            source_file=file_path,
            warnings=warnings,
        )

    # -- internals -----------------------------------------------------

    def _resolve_label(self, el: dict, text_elements: list[dict]) -> str:
        bound = el.get("boundElements") or []
        for b in bound:
            if b.get("type") == "text":
                for t in text_elements:
                    if t.get("id") == b.get("id"):
                        return t.get("text", "") or ""
        return ""

    def _mark_geometric_containers(self, nodes: list[DiagramNode]) -> None:
        for outer in nodes:
            outer_box = outer.bbox
            for inner in nodes:
                if inner.id == outer.id:
                    continue
                if outer_box.contains(inner.bbox) and outer_box.width * outer_box.height > inner.bbox.width * inner.bbox.height:
                    outer.shape = NodeShape.CONTAINER
                    inner.parent_id = inner.parent_id or outer.id
