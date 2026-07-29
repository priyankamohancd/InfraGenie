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


def test_untagged_shapes_have_empty_tags(parsed):
    """The common case (no "Edit Data" ever used) must not regress — every
    node in the existing fixture has no custom data."""
    for node in parsed.nodes:
        assert node.tags == {}


# ─────────────────────────────────────────────────────────────────────────────
# Custom data ("Edit Data") — 2026-07-08. draw.io wraps a shape's <mxCell> in
# an <object>/<UserObject> element when it carries custom key/value data,
# moving `id` and the display label up to the wrapper. Before this was
# handled, a wrapped shape's id (and thus everything downstream keyed by it —
# containment, wiring, resource identity) silently went blank.
# ─────────────────────────────────────────────────────────────────────────────

_TAGGED_OBJECT_XML = """
<mxGraphModel>
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <object label="Web Server" tier="prod" pii="false" id="ec2_1">
      <mxCell style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.ec2;" vertex="1" parent="1">
        <mxGeometry x="40" y="40" width="60" height="60" as="geometry" />
      </mxCell>
    </object>
    <mxCell id="s3_1" value="Static Assets Bucket" style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.s3;" vertex="1" parent="1">
      <mxGeometry x="200" y="40" width="60" height="60" as="geometry" />
    </mxCell>
  </root>
</mxGraphModel>
"""

_TAGGED_USEROBJECT_XML = """
<mxGraphModel>
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <UserObject label="Database" tier="prod" id="db_1">
      <mxCell style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.rds;" vertex="1" parent="1">
        <mxGeometry x="40" y="40" width="60" height="60" as="geometry" />
      </mxCell>
    </UserObject>
  </root>
</mxGraphModel>
"""


def test_object_wrapped_shape_custom_data_extracted_as_tags(tmp_path):
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_OBJECT_XML)
    parsed = DrawioAdapter().parse(str(f))

    ec2 = parsed.node_by_id("ec2_1")
    assert ec2 is not None, "wrapped shape's id must not go blank (the regression this closes)"
    assert ec2.tags == {"tier": "prod", "pii": "false"}


def test_object_wrapped_shape_id_and_label_correct(tmp_path):
    """The actual regression: before reading the wrapper, a wrapped mxCell
    has no `id` of its own, so the old `cell.get('id', '')` silently
    produced an empty id for every custom-data-tagged shape."""
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_OBJECT_XML)
    parsed = DrawioAdapter().parse(str(f))

    ids = {n.id for n in parsed.nodes}
    assert "" not in ids
    assert "ec2_1" in ids
    ec2 = parsed.node_by_id("ec2_1")
    assert ec2.raw_label == "Web Server"
    assert ec2.image_ref is not None and "ec2" in ec2.image_ref


def test_object_wrapped_and_plain_shapes_coexist(tmp_path):
    """A diagram with SOME tagged shapes and some plain ones — both parsing
    paths must work in the same file without one clobbering the other."""
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_OBJECT_XML)
    parsed = DrawioAdapter().parse(str(f))

    assert len(parsed.nodes) == 2
    s3 = parsed.node_by_id("s3_1")
    assert s3 is not None
    assert s3.raw_label == "Static Assets Bucket"
    assert s3.tags == {}


def test_userobject_wrapper_also_supported(tmp_path):
    """Newer draw.io exports use <UserObject> instead of <object> — same
    custom-data mechanism, different tag name."""
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_USEROBJECT_XML)
    parsed = DrawioAdapter().parse(str(f))

    db = parsed.node_by_id("db_1")
    assert db is not None
    assert db.raw_label == "Database"
    assert db.tags == {"tier": "prod"}
