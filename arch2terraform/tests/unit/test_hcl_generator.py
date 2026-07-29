from arch2terraform.generator.hcl_generator import (
    generate_main_tf,
    generate_outputs_tf,
    generate_provider_tf,
    generate_readme_md,
    generate_variables_tf,
)
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph, ResourceRelationship


def _vpc_subnet_ec2_graph():
    vpc = ClassifiedResource(
        node_id="vpc1", resource_type="aws_vpc", terraform_name="main_vpc",
        display_label="Main VPC", confidence=0.95, is_container=True,
        attributes={"cidr_block": "10.0.0.0/16"},
    )
    subnet = ClassifiedResource(
        node_id="subnet1", resource_type="aws_subnet", terraform_name="public_subnet",
        display_label="Public Subnet", confidence=0.95, is_container=True,
        attributes={"cidr_block": "10.0.1.0/24"},
    )
    ec2 = ClassifiedResource(
        node_id="ec2_1", resource_type="aws_instance", terraform_name="web_server",
        display_label="Web Server", confidence=0.95,
        attributes={"instance_type": "t3.micro"},
    )
    relationships = [
        ResourceRelationship("vpc1", "subnet1", "containment"),
        ResourceRelationship("subnet1", "ec2_1", "containment"),
    ]
    return ResourceGraph(resources=[vpc, subnet, ec2], relationships=relationships)


def test_provider_tf_has_required_blocks():
    content = generate_provider_tf()
    assert 'required_providers' in content
    assert 'source  = "hashicorp/aws"' in content
    assert 'provider "aws"' in content


def test_variables_tf_has_region_and_environment():
    content = generate_variables_tf()
    assert 'variable "aws_region"' in content
    assert 'variable "environment"' in content


def test_main_tf_contains_all_resource_blocks():
    graph = _vpc_subnet_ec2_graph()
    content = generate_main_tf(graph)
    assert 'resource "aws_vpc" "main_vpc"' in content
    assert 'resource "aws_subnet" "public_subnet"' in content
    assert 'resource "aws_instance" "web_server"' in content


def test_main_tf_wires_containment_as_references_not_comments():
    graph = _vpc_subnet_ec2_graph()
    content = generate_main_tf(graph)
    # subnet should reference the vpc's id, not just mention it in a comment
    assert "vpc_id" in content and "aws_vpc.main_vpc.id" in content
    assert "subnet_id" in content and "aws_subnet.public_subnet.id" in content


def test_main_tf_flags_low_confidence_resources():
    vpc = ClassifiedResource(
        node_id="vpc1", resource_type="aws_vpc", terraform_name="guessed_vpc",
        display_label="Some Box", confidence=0.5, is_container=True,
    )
    graph = ResourceGraph(resources=[vpc], relationships=[])
    content = generate_main_tf(graph)
    assert "low-confidence match" in content


def test_main_tf_handles_empty_graph():
    graph = ResourceGraph(resources=[], relationships=[])
    content = generate_main_tf(graph)
    assert "No classifiable resources" in content


def test_outputs_tf_generates_known_attributes():
    graph = _vpc_subnet_ec2_graph()
    content = generate_outputs_tf(graph)
    assert 'output "main_vpc_id"' in content
    assert "aws_vpc.main_vpc.id" in content
    assert 'output "web_server_id"' in content


def test_readme_lists_unclassified_and_low_confidence():
    vpc = ClassifiedResource(
        node_id="vpc1", resource_type="aws_vpc", terraform_name="main_vpc",
        display_label="Main VPC", confidence=0.5, is_container=True,
    )
    graph = ResourceGraph(resources=[vpc], relationships=[], unclassified_nodes=["weird_node"])
    readme = generate_readme_md(graph, "diagram.drawio", warnings=["Skipped vertex cell 99"])
    assert "Low-confidence matches" in readme
    assert "weird_node" in readme
    assert "Skipped vertex cell 99" in readme


def test_ec2_directly_in_vpc_does_not_get_wrong_subnet_id_reference():
    """Regression test: if a diagram shows an EC2 instance contained directly by a VPC
    (no subnet drawn), the generator must NOT wire subnet_id to point at the VPC's id —
    that's a real attribute name pointing at the wrong resource type. It should instead
    flag this in a comment so the user wires it correctly by hand."""
    vpc = ClassifiedResource(
        node_id="vpc1", resource_type="aws_vpc", terraform_name="vpc",
        display_label="VPC", confidence=0.95, is_container=True,
    )
    ec2 = ClassifiedResource(
        node_id="ec2_1", resource_type="aws_instance", terraform_name="ec2_instance",
        display_label="EC2 instance", confidence=0.95,
        attributes={"instance_type": "t3.micro"},
    )
    graph = ResourceGraph(
        resources=[vpc, ec2],
        relationships=[ResourceRelationship("vpc1", "ec2_1", "containment")],
    )
    content = generate_main_tf(graph)
    assert "subnet_id = aws_vpc" not in content.replace(" ", "")  # no wrong wiring, with or without alignment spaces
    assert "wire subnet_id manually" in content


def test_ec2_in_actual_subnet_does_get_wired_correctly():
    subnet = ClassifiedResource(
        node_id="subnet1", resource_type="aws_subnet", terraform_name="subnet",
        display_label="Subnet", confidence=0.95, is_container=True,
    )
    ec2 = ClassifiedResource(
        node_id="ec2_1", resource_type="aws_instance", terraform_name="ec2_instance",
        display_label="EC2 instance", confidence=0.95,
        attributes={"instance_type": "t3.micro"},
    )
    graph = ResourceGraph(
        resources=[subnet, ec2],
        relationships=[ResourceRelationship("subnet1", "ec2_1", "containment")],
    )
    content = generate_main_tf(graph)
    assert "aws_subnet.subnet.id" in content
    assert "wire subnet_id manually" not in content
