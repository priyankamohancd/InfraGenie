"""
Lucidchart adapter.

Lucidchart's standard (non-Enterprise-API) export path is a CSV of shape
data: File > Export > CSV. Each row is one shape with position/size/text/
shape-library metadata; connector rows reference source/target shape IDs.
This avoids requiring Lucidchart API credentials, which most users won't have.

Expected columns (Lucidchart's standard CSV export schema):
  Id, Name, Shape Library, Page ID, Contained By, Group, Source, Target,
  Text Area 1, Line Source, Line Destination, "Width", "Height", "X", "Y"

Column names vary slightly by Lucidchart export version, so lookups are
case-insensitive and tolerant of missing columns.
"""

from __future__ import annotations

import csv

from arch2terraform.adapters.base import BaseAdapter
from arch2terraform.schemas.diagram import (
    BoundingBox,
    DiagramEdge,
    DiagramNode,
    EdgeStyle,
    NodeShape,
    ParsedDiagram,
)

_CONTAINER_SHAPE_HINTS = ("container", "group", "vpc", "subnet", "boundary")


class LucidchartAdapter(BaseAdapter):
    format_name = "lucidchart"

    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(".csv")

    def parse(self, file_path: str) -> ParsedDiagram:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [self._normalize_row(row) for row in reader]

        warnings: list[str] = []
        nodes: list[DiagramNode] = []
        edge_rows: list[dict] = []

        for row in rows:
            source = row.get("source")
            target = row.get("target")
            if source and target:
                edge_rows.append(row)
                continue

            node = self._row_to_node(row, warnings)
            if node:
                nodes.append(node)

        edges: list[DiagramEdge] = []
        for i, row in enumerate(edge_rows):
            edges.append(
                DiagramEdge(
                    id=row.get("id") or f"edge-{i}",
                    source_id=row["source"],
                    target_id=row["target"],
                    label=row.get("text area 1", ""),
                    style=EdgeStyle.SOLID,
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

    def _normalize_row(self, row: dict) -> dict:
        """Lowercase keys so we're tolerant of Lucidchart's export-version header variance."""
        return {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}

    def _row_to_node(self, row: dict, warnings: list[str]) -> DiagramNode | None:
        node_id = row.get("id") or row.get("name")
        if not node_id:
            warnings.append("Skipped row with no Id/Name column")
            return None

        try:
            bbox = BoundingBox(
                x=float(row.get("x", 0) or 0),
                y=float(row.get("y", 0) or 0),
                width=float(row.get("width", 0) or 0),
                height=float(row.get("height", 0) or 0),
            )
        except ValueError:
            warnings.append(f"Skipped row {node_id}: non-numeric geometry")
            return None

        shape_lib = row.get("shape library", "")
        shape_name = row.get("name", "")
        # The AWS resource type signal lives in the "Name" column (e.g. "RDS Instance",
        # "EC2 Instance"), not in the free-text "Text Area 1" business label (e.g. "Orders DB").
        # Combine both into the icon-matching surface so the classifier's icon_keys lookup
        # (which expects substrings like "rds", "ec2", "s3") actually has something to match.
        icon_surface = f"{shape_lib} {shape_name}".strip()
        label = row.get("text area 1") or shape_name or ""
        shape = self._infer_shape(icon_surface, label)
        parent = row.get("contained by") or row.get("group") or None

        return DiagramNode(
            id=node_id,
            raw_label=label,
            shape=shape,
            bbox=bbox,
            style_raw=shape_lib,
            image_ref=icon_surface if "aws" in icon_surface.lower() else None,
            parent_id=parent or None,
            source_format=self.format_name,
        )

    def _infer_shape(self, shape_lib: str, label: str) -> NodeShape:
        haystack = f"{shape_lib} {label}".lower()
        if any(hint in haystack for hint in _CONTAINER_SHAPE_HINTS):
            return NodeShape.CONTAINER
        if "aws" in shape_lib.lower():
            return NodeShape.ICON
        return NodeShape.RECTANGLE
