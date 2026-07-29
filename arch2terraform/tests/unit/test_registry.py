import os

import pytest

from arch2terraform.adapters.registry import UnsupportedFormatError, get_adapter, parse_diagram
from arch2terraform.adapters.drawio_adapter import DrawioAdapter
from arch2terraform.adapters.excalidraw_adapter import ExcalidrawAdapter
from arch2terraform.adapters.lucidchart_adapter import LucidchartAdapter
from arch2terraform.adapters.image_adapter import ImageAdapter

DRAWIO_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "drawio", "sample_architecture.drawio")


def test_get_adapter_routes_by_extension():
    assert isinstance(get_adapter("foo.drawio"), DrawioAdapter)
    assert isinstance(get_adapter("foo.excalidraw"), ExcalidrawAdapter)
    assert isinstance(get_adapter("foo.csv"), LucidchartAdapter)
    assert isinstance(get_adapter("foo.png"), ImageAdapter)


def test_get_adapter_raises_for_unknown_extension():
    with pytest.raises(UnsupportedFormatError):
        get_adapter("foo.unknownformat")


def test_parse_diagram_end_to_end():
    diagram = parse_diagram(DRAWIO_FIXTURE)
    assert diagram.source_format == "drawio"
    assert len(diagram.nodes) == 5
