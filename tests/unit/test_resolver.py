from arch2terraform.resolver.resolver import resolve_relationships
from arch2terraform.schemas.diagram import BoundingBox, DiagramEdge, DiagramNode, NodeShape, ParsedDiagram
from arch2terraform.schemas.resources import ClassifiedResource


def _node(node_id, bbox, parent_id=None):
    return DiagramNode(
        id=node_id, raw_label=node_id, shape=NodeShape.ICON, bbox=bbox, parent_id=parent_id, source_format="test"
    )


def _resource(node_id, resource_type, is_container=False):
    return ClassifiedResource(
        node_id=node_id,
        resource_type=resource_type,
        terraform_name=node_id,
        display_label=node_id,
        confidence=0.9,
        is_container=is_container,
    )


def test_explicit_containment_via_parent_id():
    vpc_node = _node("vpc1", BoundingBox(0, 0, 500, 500))
    ec2_node = _node("ec2_1", BoundingBox(10, 10, 50, 50), parent_id="vpc1")

    diagram = ParsedDiagram(nodes=[vpc_node, ec2_node], edges=[], source_format="test", source_file="test")
    classified = [_resource("vpc1", "aws_vpc", is_container=True), _resource("ec2_1", "aws_instance")]

    graph = resolve_relationships(diagram, classified)
    containment = [r for r in graph.relationships if r.relationship_type == "containment"]
    assert len(containment) == 1
    assert containment[0].source_node_id == "vpc1"
    assert containment[0].target_node_id == "ec2_1"


def test_geometric_containment_fallback_without_parent_id():
    vpc_node = _node("vpc1", BoundingBox(0, 0, 500, 500))
    ec2_node = _node("ec2_1", BoundingBox(10, 10, 50, 50))  # no parent_id set

    diagram = ParsedDiagram(nodes=[vpc_node, ec2_node], edges=[], source_format="test", source_file="test")
    classified = [_resource("vpc1", "aws_vpc", is_container=True), _resource("ec2_1", "aws_instance")]

    graph = resolve_relationships(diagram, classified)
    containment = [r for r in graph.relationships if r.relationship_type == "containment"]
    assert len(containment) == 1
    assert containment[0].source_node_id == "vpc1"


def test_picks_smallest_enclosing_container_when_nested():
    vpc_node = _node("vpc1", BoundingBox(0, 0, 500, 500))
    subnet_node = _node("subnet1", BoundingBox(10, 10, 200, 200), parent_id="vpc1")
    ec2_node = _node("ec2_1", BoundingBox(20, 20, 50, 50), parent_id="subnet1")

    diagram = ParsedDiagram(nodes=[vpc_node, subnet_node, ec2_node], edges=[], source_format="test", source_file="test")
    classified = [
        _resource("vpc1", "aws_vpc", is_container=True),
        _resource("subnet1", "aws_subnet", is_container=True),
        _resource("ec2_1", "aws_instance"),
    ]

    graph = resolve_relationships(diagram, classified)
    ec2_containment = [r for r in graph.relationships if r.relationship_type == "containment" and r.target_node_id == "ec2_1"]
    assert len(ec2_containment) == 1
    assert ec2_containment[0].source_node_id == "subnet1"  # closest container, not vpc1


def test_edge_between_iam_resources_classified_as_iam_attachment():
    role_node = _node("role1", BoundingBox(0, 0, 10, 10))
    lambda_node = _node("lambda1", BoundingBox(100, 100, 10, 10))
    edge = DiagramEdge(id="e1", source_id="lambda1", target_id="role1")

    diagram = ParsedDiagram(nodes=[role_node, lambda_node], edges=[edge], source_format="test", source_file="test")
    classified = [_resource("role1", "aws_iam_role"), _resource("lambda1", "aws_lambda_function")]

    graph = resolve_relationships(diagram, classified)
    edge_rels = [r for r in graph.relationships if r.relationship_type != "containment"]
    assert len(edge_rels) == 1
    assert edge_rels[0].relationship_type == "iam_attachment"


def test_edge_touching_unclassified_node_is_skipped():
    ec2_node = _node("ec2_1", BoundingBox(0, 0, 10, 10))
    mystery_node = _node("mystery1", BoundingBox(100, 100, 10, 10))
    edge = DiagramEdge(id="e1", source_id="ec2_1", target_id="mystery1")

    diagram = ParsedDiagram(nodes=[ec2_node, mystery_node], edges=[edge], source_format="test", source_file="test")
    classified = [_resource("ec2_1", "aws_instance")]  # mystery1 deliberately not classified

    graph = resolve_relationships(diagram, classified)
    assert graph.relationships == []
    assert "mystery1" in graph.unclassified_nodes
