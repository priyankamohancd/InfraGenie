import os

import pytest

from arch2terraform.adapters.drawio_adapter import DrawioAdapter
from arch2terraform.schemas.diagram import NodeShape

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "drawio", "sample_architecture.drawio")


@pytest.fixture
def parsed():
    return DrawioAdapter().parse(FIXTURE)


def test_can_parse_extension():
    adapter = DrawioAdapter()
    assert adapter.can_parse("foo.drawio")
    assert adapter.can_parse("foo.xml")
    assert not adapter.can_parse("foo.csv")


def test_parses_expected_node_count(parsed):
    # vpc1, subnet1, ec2_1, igw_1, s3_1 = 5 vertices (mxCell id 0/1 excluded)
    assert len(parsed.nodes) == 5


def test_parses_expected_edge_count(parsed):
    assert len(parsed.edges) == 2


def test_container_shape_detected(parsed):
    vpc = parsed.node_by_id("vpc1")
    subnet = parsed.node_by_id("subnet1")
    assert vpc.shape == NodeShape.CONTAINER
    assert subnet.shape == NodeShape.CONTAINER


def test_icon_ref_extracted_for_aws_shapes(parsed):
    ec2 = parsed.node_by_id("ec2_1")
    s3 = parsed.node_by_id("s3_1")
    assert ec2.image_ref is not None and "ec2" in ec2.image_ref
    assert s3.image_ref is not None and "s3" in s3.image_ref


def test_parent_id_preserved_for_nested_nodes(parsed):
    ec2 = parsed.node_by_id("ec2_1")
    subnet = parsed.node_by_id("subnet1")
    assert ec2.parent_id == "subnet1"
    assert subnet.parent_id == "vpc1"


def test_edge_endpoints_correct(parsed):
    edge_ids = {(e.source_id, e.target_id) for e in parsed.edges}
    assert ("ec2_1", "s3_1") in edge_ids
    assert ("ec2_1", "igw_1") in edge_ids


def test_labels_extracted(parsed):
    ec2 = parsed.node_by_id("ec2_1")
    assert ec2.raw_label == "Web Server"
