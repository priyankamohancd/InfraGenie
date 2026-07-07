import os

import pytest

from arch2terraform.adapters.lucidchart_adapter import LucidchartAdapter
from arch2terraform.schemas.diagram import NodeShape

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "lucidchart", "sample_architecture.csv")


@pytest.fixture
def parsed():
    return LucidchartAdapter().parse(FIXTURE)


def test_can_parse_extension():
    adapter = LucidchartAdapter()
    assert adapter.can_parse("foo.csv")
    assert not adapter.can_parse("foo.drawio")


def test_parses_expected_node_count(parsed):
    # sg-1, rds-1, sns-1 (edge-1 row is a connector, not a node)
    assert len(parsed.nodes) == 3


def test_parses_expected_edge_count(parsed):
    assert len(parsed.edges) == 1
    edge = parsed.edges[0]
    assert edge.source_id == "rds-1"
    assert edge.target_id == "sns-1"
    assert edge.label == "replicates to"


def test_security_group_detected_as_container(parsed):
    sg = parsed.node_by_id("sg-1")
    assert sg.shape == NodeShape.CONTAINER


def test_contained_by_maps_to_parent_id(parsed):
    rds = parsed.node_by_id("rds-1")
    assert rds.parent_id == "sg-1"


def test_labels_extracted(parsed):
    rds = parsed.node_by_id("rds-1")
    assert rds.raw_label == "Orders DB"


def test_name_column_flows_into_image_ref_for_classifier_matching(parsed):
    """Regression test: the AWS resource type signal lives in the 'Name' column
    ('RDS Instance'), not in the free-text business label ('Orders DB'). The
    classifier's icon matching relies on image_ref carrying that signal."""
    rds = parsed.node_by_id("rds-1")
    assert rds.image_ref is not None
    assert "rds" in rds.image_ref.lower() or "RDS" in rds.image_ref
