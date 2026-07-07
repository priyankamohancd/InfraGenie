"""
draw.io (diagrams.net) adapter.

draw.io files are mxGraph XML. Two on-disk shapes are common:
  1. Plain XML: <mxGraphModel><root><mxCell .../></root></mxGraphModel>
  2. Compressed: <mxfile><diagram>BASE64+DEFLATE...</diagram></mxfile>

Each <mxCell> is either a vertex (a node) or an edge (a connector), and
style strings carry the AWS shape stencil reference we use downstream
for classification, e.g. style="shape=mxgraph.aws4.resourceIcon;
resIcon=mxgraph.aws4.ec2;..."
"""

from __future__ import annotations

import base64
import re
import urllib.parse
import zlib
import xml.etree.ElementTree as ET

from arch2terraform.adapters.base import BaseAdapter
from arch2terraform.schemas.diagram import (
    BoundingBox,
    DiagramEdge,
    DiagramNode,
    EdgeStyle,
    NodeShape,
    ParsedDiagram,
)

_CONTAINER_STYLE_HINTS = ("group", "container=1", "mxgraph.aws4.group")


class DrawioAdapter(BaseAdapter):
    format_name = "drawio"

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith((".drawio", ".xml"))

    def parse(self, file_path: str) -> ParsedDiagram:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        xml_text = self._extract_xml(raw)
        root = ET.fromstring(xml_text)

        nodes: list[DiagramNode] = []
        edges: list[DiagramEdge] = []
        warnings: list[str] = []

        cells = root.findall(".//mxCell")
        for cell in cells:
            cell_id = cell.get("id", "")
            if cell_id in ("0", "1"):
                # mxGraph's implicit root layers, never real shapes
                continue

            is_edge = cell.get("edge") == "1"
            if is_edge:
                edge = self._parse_edge(cell)
                if edge:
                    edges.append(edge)
                continue

            is_vertex = cell.get("vertex") == "1"
            if not is_vertex:
                continue

            node = self._parse_vertex(cell)
            if node:
                nodes.append(node)
            else:
                warnings.append(f"Skipped vertex cell {cell_id}: missing geometry")

        return ParsedDiagram(
            nodes=nodes,
            edges=edges,
            source_format=self.format_name,
            source_file=file_path,
            warnings=warnings,
        )

    # -- internals -----------------------------------------------------

    def _extract_xml(self, raw: str) -> str:
        """Handle both plain and compressed draw.io exports."""
        if "<mxGraphModel" in raw:
            return raw

        # Compressed form: <diagram ...>PAYLOAD</diagram>
        match = re.search(r"<diagram[^>]*>([^<]+)</diagram>", raw)
        if not match:
            raise ValueError("Could not locate <mxGraphModel> or <diagram> payload in draw.io file")

        payload = match.group(1).strip()
        compressed = base64.b64decode(payload)
        decompressed = zlib.decompress(compressed, -15)  # raw deflate, no zlib header
        xml_text = urllib.parse.unquote(decompressed.decode("utf-8"))
        return xml_text

    def _parse_vertex(self, cell) -> DiagramNode | None:
        geometry = cell.find("mxGeometry")
        if geometry is None:
            return None

        try:
            bbox = BoundingBox(
                x=float(geometry.get("x", 0)),
                y=float(geometry.get("y", 0)),
                width=float(geometry.get("width", 0)),
                height=float(geometry.get("height", 0)),
            )
        except (TypeError, ValueError):
            return None

        style = cell.get("style", "")
        shape = self._infer_shape(style)
        image_ref = self._extract_image_ref(style)

        return DiagramNode(
            id=cell.get("id", ""),
            raw_label=cell.get("value", "") or "",
            shape=shape,
            bbox=bbox,
            style_raw=style,
            image_ref=image_ref,
            fill_color=self._extract_style_prop(style, "fillColor"),
            parent_id=cell.get("parent"),
            source_format=self.format_name,
        )

    def _parse_edge(self, cell) -> DiagramEdge | None:
        source = cell.get("source")
        target = cell.get("target")
        if not source or not target:
            return None

        style = cell.get("style", "")
        edge_style = EdgeStyle.DASHED if "dashed=1" in style else EdgeStyle.SOLID

        return DiagramEdge(
            id=cell.get("id", ""),
            source_id=source,
            target_id=target,
            label=cell.get("value", "") or "",
            style=edge_style,
        )

    def _infer_shape(self, style: str) -> NodeShape:
        if any(hint in style for hint in _CONTAINER_STYLE_HINTS):
            return NodeShape.CONTAINER
        if "mxgraph.aws4" in style or "shape=image" in style:
            return NodeShape.ICON
        if "cylinder" in style:
            return NodeShape.CYLINDER
        if "cloud" in style:
            return NodeShape.CLOUD
        if "rhombus" in style:
            return NodeShape.DIAMOND
        if "ellipse" in style:
            return NodeShape.CIRCLE
        if "rounded" in style or "whiteSpace" in style:
            return NodeShape.RECTANGLE
        return NodeShape.UNKNOWN

    def _extract_image_ref(self, style: str) -> str | None:
        # resIcon=mxgraph.aws4.ec2  OR  shape=mxgraph.aws4.ec2
        for key in ("resIcon", "shape"):
            val = self._extract_style_prop(style, key)
            if val and "aws4" in val:
                return val
        return None

    def _extract_style_prop(self, style: str, key: str) -> str | None:
        match = re.search(rf"{key}=([^;]+)", style)
        return match.group(1) if match else None
