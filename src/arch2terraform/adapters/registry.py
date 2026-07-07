"""
Registry that picks the correct adapter for a given input file.

This is the single entry point Phase 2's pipeline will call — it never
needs to know which adapter handled the file, only that it gets back a
ParsedDiagram.
"""

from __future__ import annotations

import os

from arch2terraform.adapters.base import BaseAdapter
from arch2terraform.adapters.drawio_adapter import DrawioAdapter
from arch2terraform.adapters.excalidraw_adapter import ExcalidrawAdapter
from arch2terraform.adapters.image_adapter import ImageAdapter
from arch2terraform.adapters.lucidchart_adapter import LucidchartAdapter
from arch2terraform.schemas.diagram import ParsedDiagram

# Root of the AWS icon pack (contains Architecture-Service-Icons_* /
# Architecture-Group-Icons_* subdirectories). Stage 2 (phash) works without
# this — it only needs the pre-built reference_hashes.pkl — but Stage 3 (NCC
# template-match fallback) needs the actual icon files to build templates
# from, so without this set, any icon that phash can't confidently match
# falls straight through to OCR/UNKNOWN instead of getting the NCC pass.
# Set via env var rather than hardcoding a path so this works across machines.
_ICONS_DIR = os.environ.get("ARCH2TERRAFORM_ICONS_DIR")

_ADAPTERS: list[BaseAdapter] = [
    DrawioAdapter(),
    ExcalidrawAdapter(),
    LucidchartAdapter(),
    ImageAdapter(icons_dir=_ICONS_DIR),
]


class UnsupportedFormatError(ValueError):
    pass


def get_adapter(file_path: str) -> BaseAdapter:
    for adapter in _ADAPTERS:
        if adapter.can_parse(file_path):
            return adapter
    raise UnsupportedFormatError(
        f"No adapter found for '{file_path}'. Supported extensions: "
        ".drawio, .xml, .excalidraw, .csv, .png, .jpg, .jpeg"
    )


def parse_diagram(file_path: str) -> ParsedDiagram:
    adapter = get_adapter(file_path)
    return adapter.parse(file_path)
