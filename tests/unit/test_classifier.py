from arch2terraform.classifier.catalog import CATALOG
from arch2terraform.classifier.classifier import classify_diagram
from arch2terraform.schemas.diagram import BoundingBox, DiagramNode, NodeShape, ParsedDiagram


def _node(node_id, label="", image_ref=None, shape=NodeShape.ICON, parent_id=None):
    return DiagramNode(
        id=node_id,
        raw_label=label,
        shape=shape,
        bbox=BoundingBox(0, 0, 10, 10),
        image_ref=image_ref,
        parent_id=parent_id,
        source_format="test",
    )


def test_catalog_has_at_least_45_resource_types():
    assert len(CATALOG) >= 45


def test_catalog_terraform_types_are_unique():
    types = [d.terraform_type for d in CATALOG]
    assert len(types) == len(set(types)), "Duplicate terraform_type entries in catalog"


def test_classify_by_icon_ref_high_confidence():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Web Server", image_ref="mxgraph.aws4.ec2")],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_instance"
    assert classified[0].confidence >= 0.9
    assert classified[0].needs_clarification == []


def test_classify_by_label_lower_confidence():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Lambda function", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_lambda_function"
    assert classified[0].confidence < 0.9
    assert "resource_type" in classified[0].needs_clarification


def test_unrecognized_node_goes_to_unclassified():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Mystery Box", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 0
    assert unclassified == ["n1"]


def test_container_shape_without_label_defaults_to_vpc():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="", image_ref=None, shape=NodeShape.CONTAINER)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_vpc"
    assert classified[0].is_container


def test_terraform_names_are_unique_and_sanitized():
    diagram = ParsedDiagram(
        nodes=[
            _node("n1", label="Web Server!", image_ref="mxgraph.aws4.ec2"),
            _node("n2", label="Web Server!", image_ref="mxgraph.aws4.ec2"),
        ],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    names = {c.terraform_name for c in classified}
    assert len(names) == 2
    for name in names:
        assert name.replace("_", "a").isalnum()  # only [a-z0-9_]


def test_rds_instance_not_misclassified_as_ec2():
    """Regression test: 'RDS Instance' contains the substring 'instance', which is
    EC2's icon key. The classifier must prefer the more specific 'rds' match over
    the generic 'instance' match, regardless of catalog list order."""
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Orders DB", image_ref="AWS19 / Database RDS Instance")],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_db_instance"


def test_longest_label_keyword_wins_over_shorter_generic_one():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="aurora cluster", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert classified[0].resource_type == "aws_rds_cluster"
