"""
Cross-Module Containment Wiring — unit tests
-----------------------------------------------
Focused, synthetic-data tests for terraform_planner.py's cross-module
wiring (see _CrossModuleWire, _wire_containment_attrs, _generate_module_hcl,
_generate_root_module). Built directly against ParsedResource/ParsedConnection
objects rather than going through the full diagram-parsing pipeline, so these
run fast and pin down the exact generated shape independent of any specific
fixture diagram.

Complements backend/tests/test_terraform_validate_e2e.py, which exercises
this same code path end-to-end (through real fixtures + real `terraform
validate`) but doesn't assert on the specific variable/output/module-argument
names — that's what these tests are for.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import ParsedDiagram, ParsedResource, ParsedConnection, DiagramFormat
from app.services.planner.terraform_planner import build_terraform_plan


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _same_module_diagram() -> ParsedDiagram:
    """VPC + subnet: both land in the 'networking' module -> should wire
    with a direct resource reference, no cross-module plumbing needed."""
    vpc = ParsedResource(
        id="vpc-1", aws_resource_type="aws_vpc", logical_name="main_vpc",
        label="Main VPC", properties={"cidr_block": "10.0.0.0/16"},
    )
    subnet = ParsedResource(
        id="subnet-1", aws_resource_type="aws_subnet", logical_name="public_subnet",
        label="Public Subnet", properties={"cidr_block": "10.0.1.0/24"},
    )
    conn = ParsedConnection(source_id="vpc-1", target_id="subnet-1", connection_type="containment")
    return ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[vpc, subnet], connections=[conn])


def _cross_module_diagram() -> ParsedDiagram:
    """Subnet (networking module) containing an EC2 instance (compute
    module): different modules -> should produce real var./output/module-arg
    plumbing instead of a bare resource reference."""
    subnet = ParsedResource(
        id="subnet-1", aws_resource_type="aws_subnet", logical_name="public_subnet",
        label="Public Subnet", properties={"cidr_block": "10.0.1.0/24"},
    )
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0abcdef1234567890", "instance_type": "t3.micro"},
    )
    conn = ParsedConnection(source_id="subnet-1", target_id="ec2-1", connection_type="containment")
    return ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[subnet, instance], connections=[conn])


@pytest.mark.anyio
async def test_same_module_containment_uses_direct_resource_reference():
    plan = await build_terraform_plan(_same_module_diagram(), project_name="test")
    networking = next(m for m in plan.modules if m.name == "networking")

    main_tf_compact = " ".join(networking.files["main.tf"].split())
    assert "vpc_id = aws_vpc.main_vpc.id" in main_tf_compact
    # No cross-module plumbing should be generated for a same-module wire.
    assert "variable \"main_vpc" not in networking.files.get("variables.tf", "")
    for mod in plan.modules:
        assert "var.main_vpc" not in mod.files.get("main.tf", "")


@pytest.mark.anyio
async def test_cross_module_containment_generates_full_plumbing():
    plan = await build_terraform_plan(_cross_module_diagram(), project_name="test")
    networking = next(m for m in plan.modules if m.name == "networking")
    compute = next(m for m in plan.modules if m.name == "compute")

    # 1. Child resource attribute references the variable, not a bare
    #    cross-module resource reference (which terraform validate would
    #    reject as "reference to undeclared resource"). Padding varies with
    #    the longest attribute key in the block (e.g. vpc_security_group_ids,
    #    once a security group gets attached — see security_bridge.py's
    #    2026-07-24 SG-attachment fix), so match loosely rather than on
    #    exact alignment.
    assert re.search(r"\bsubnet_id\s*=\s*var\.networking_public_subnet_id\b", compute.files["main.tf"])

    # 2. Compute module declares the variable it consumes.
    assert 'variable "networking_public_subnet_id"' in compute.files["variables.tf"]

    # 3. Networking module exposes a matching output for the subnet's id.
    assert 'output "public_subnet_id"' in networking.files["outputs.tf"]
    assert "aws_subnet.public_subnet.id" in networking.files["outputs.tf"]

    # 4. Root module wires the two together in the compute module call.
    root_main = plan.root_module_files["main.tf"]
    assert "networking_public_subnet_id = module.networking.public_subnet_id" in root_main

    # 5. The wiring is documented in-line for anyone reading the generated code.
    assert "NOTE:" in compute.files["main.tf"]
    assert "var.networking_public_subnet_id" in compute.files["main.tf"]


@pytest.mark.anyio
async def test_cross_module_wire_is_deduped_across_multiple_children():
    """Two EC2 instances in the same subnet, subnet in a different module
    than both instances, should produce exactly one variable/output pair —
    not one per child resource."""
    subnet = ParsedResource(
        id="subnet-1", aws_resource_type="aws_subnet", logical_name="public_subnet",
        label="Public Subnet", properties={"cidr_block": "10.0.1.0/24"},
    )
    instance_a = ParsedResource(
        id="ec2-a", aws_resource_type="aws_instance", logical_name="web_server_a",
        label="Web Server A", properties={"ami": "ami-0abcdef1234567890", "instance_type": "t3.micro"},
    )
    instance_b = ParsedResource(
        id="ec2-b", aws_resource_type="aws_instance", logical_name="web_server_b",
        label="Web Server B", properties={"ami": "ami-0abcdef1234567890", "instance_type": "t3.micro"},
    )
    diagram = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[subnet, instance_a, instance_b],
        connections=[
            ParsedConnection(source_id="subnet-1", target_id="ec2-a", connection_type="containment"),
            ParsedConnection(source_id="subnet-1", target_id="ec2-b", connection_type="containment"),
        ],
    )
    plan = await build_terraform_plan(diagram, project_name="test")
    compute = next(m for m in plan.modules if m.name == "compute")
    networking = next(m for m in plan.modules if m.name == "networking")

    assert compute.files["variables.tf"].count('variable "networking_public_subnet_id"') == 1
    assert networking.files["outputs.tf"].count('output "public_subnet_id"') == 1
    assert plan.root_module_files["main.tf"].count(
        "networking_public_subnet_id = module.networking.public_subnet_id"
    ) == 1
