"""
Base interface every format adapter must implement.

Keeping this tiny and strict is what lets us add new diagram formats later
without touching the classifier/resolver/generator at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from arch2terraform.schemas.diagram import ParsedDiagram


class BaseAdapter(ABC):
    """Contract: raw file path in, ParsedDiagram out. Nothing else leaks through."""

    format_name: str = "unknown"

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Cheap check (extension/magic bytes) — used by the adapter registry to pick a parser."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDiagram:
        """Parse the file fully into the canonical ParsedDiagram IR."""
        raise NotImplementedError
