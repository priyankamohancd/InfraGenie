import pytest

from arch2terraform.adapters.image_adapter import ImageAdapter, ImageAdapterNotImplemented


def test_can_parse_image_extensions():
    adapter = ImageAdapter()
    assert adapter.can_parse("foo.png")
    assert adapter.can_parse("foo.jpg")
    assert adapter.can_parse("foo.jpeg")
    assert not adapter.can_parse("foo.drawio")


def test_parse_raises_file_not_found_for_missing_file():
    """ImageAdapter now runs the full detection pipeline; a missing file raises FileNotFoundError."""
    adapter = ImageAdapter()
    with pytest.raises(FileNotFoundError, match="Diagram file not found"):
        adapter.parse("nonexistent_diagram.png")


def test_image_adapter_not_implemented_still_importable():
    """ImageAdapterNotImplemented must stay importable so any downstream callers don't break."""
    assert issubclass(ImageAdapterNotImplemented, NotImplementedError)
