import os

import pytest

from arch2terraform.adapters.excalidraw_adapter import ExcalidrawAdapter
from arch2terraform.schemas.diagram import NodeShape

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "excalidraw", "sample_architecture.excalidraw")


@pytest.fixture
def parsed():
    return ExcalidrawAdapter().parse(FIXTURE)


def test_can_parse_extension():
    adapter = ExcalidrawAdapter()
    assert adapter.can_parse("foo.excalidraw")
    assert not adapter.can_parse("foo.drawio")


def test_parses_expected_shape_node_count(parsed):
    # vpc-box, ec2-box, lambda-box (text elements aren't nodes)
    assert len(parsed.nodes) == 3


def test_text_labels_bound_to_shapes(parsed):
    ec2 = parsed.node_by_id("ec2-box")
    lambda_box = parsed.node_by_id("lambda-box")
    assert ec2.raw_label == "EC2 instance"
    assert lambda_box.raw_label == "Lambda function"


def test_geometric_container_detected(parsed):
    vpc = parsed.node_by_id("vpc-box")
    assert vpc.shape == NodeShape.CONTAINER


def test_contained_nodes_get_parent_id(parsed):
    ec2 = parsed.node_by_id("ec2-box")
    assert ec2.parent_id == "vpc-box"


def test_bound_arrow_becomes_edge(parsed):
    assert len(parsed.edges) == 1
    edge = parsed.edges[0]
    assert edge.source_id == "ec2-box"
    assert edge.target_id == "lambda-box"


def test_untagged_shapes_have_empty_tags(parsed):
    for node in parsed.nodes:
        assert node.tags == {}


# ─────────────────────────────────────────────────────────────────────────────
# customData — 2026-07-08, same intent-carrying mechanism as draw.io's Edit
# Data, native to Excalidraw's own element format.
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_data_extracted_as_tags(tmp_path):
    import json
    data = {
        "type": "excalidraw",
        "elements": [
            {
                "id": "rect1", "type": "rectangle", "x": 0, "y": 0, "width": 80, "height": 80,
                "isDeleted": False, "customData": {"tier": "prod", "public": True},
            },
        ],
    }
    f = tmp_path / "tagged.excalidraw"
    f.write_text(json.dumps(data))
    parsed = ExcalidrawAdapter().parse(str(f))

    rect = parsed.node_by_id("rect1")
    assert rect is not None
    # Non-string values (True) are stringified, not dropped — customData is
    # arbitrary JSON and DiagramNode.tags is dict[str, str].
    assert rect.tags == {"tier": "prod", "public": "True"}


def test_missing_or_non_dict_custom_data_yields_empty_tags(tmp_path):
    import json
    data = {
        "type": "excalidraw",
        "elements": [
            {"id": "rect1", "type": "rectangle", "x": 0, "y": 0, "width": 80, "height": 80, "isDeleted": False},
            {"id": "rect2", "type": "rectangle", "x": 0, "y": 0, "width": 80, "height": 80, "isDeleted": False, "customData": "not-a-dict"},
        ],
    }
    f = tmp_path / "tagged.excalidraw"
    f.write_text(json.dumps(data))
    parsed = ExcalidrawAdapter().parse(str(f))

    assert parsed.node_by_id("rect1").tags == {}
    assert parsed.node_by_id("rect2").tags == {}
