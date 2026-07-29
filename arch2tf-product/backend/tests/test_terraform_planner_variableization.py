"""
Required-Field Variable-ization — unit tests
------------------------------------------------
Covers _variableize_mandatory_fields / _hcl_type_for / _hcl_default_literal
in terraform_planner.py (added 2026-07-08, per her explicit request after a
real `terraform apply` failure): every MANDATORY_FIELDS-covered value (ami,
instance_type, engine, cidr_block, etc.) should be emitted as a real
Terraform `variable` with a sensible default, referenced via `var.<name>`
in the resource block — not baked in as a literal string — so it's
overridable later via a plain terraform.tfvars / -var without touching
generated code.

Synthetic ParsedResource/ParsedConnection data, same style as
test_cross_module_wiring.py, so these pin the exact generated shape
independent of any specific fixture diagram.
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
from app.services.parser.missing_info_detector import detect_missing_info


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_ami_and_instance_type_become_real_variables_not_literals():
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0realvalue1234567", "instance_type": "t3.micro"},
    )
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[instance], connections=[])
    plan = await build_terraform_plan(diagram, project_name="test")
    compute = next(m for m in plan.modules if m.name == "compute")

    main_tf = compute.files["main.tf"]
    # Padding varies with the longest attribute key in the block (e.g.
    # vpc_security_group_ids, once a security group gets attached — see
    # security_bridge.py's 2026-07-24 SG-attachment fix), so match loosely
    # on the reference itself rather than exact alignment.
    assert re.search(r"\bami\s*=\s*var\.ami\b", main_tf)
    assert '"ami-0realvalue1234567"' not in main_tf  # not baked as a literal anymore

    var_tf = compute.files["variables.tf"]
    assert 'variable "ami" {' in var_tf
    assert 'default     = "ami-0realvalue1234567"' in var_tf
    assert 'variable "instance_type" {' in var_tf
    assert 'default     = "t3.micro"' in var_tf


@pytest.mark.anyio
async def test_numeric_and_boolean_fields_get_unquoted_hcl_types():
    db = ParsedResource(
        id="db-1", aws_resource_type="aws_db_instance", logical_name="postgres_db",
        label="Postgres DB",
        properties={
            "engine": "postgres", "instance_class": "db.t3.micro",
            "allocated_storage": 20,  # real int, not a clarification-answer string
            "username": "admin", "manage_master_user_password": True,
        },
    )
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[db], connections=[])
    plan = await build_terraform_plan(diagram, project_name="test")
    database = next(m for m in plan.modules if m.name == "database")

    var_tf = database.files["variables.tf"]
    assert 'variable "allocated_storage" {' in var_tf
    assert "type        = number" in var_tf
    assert "default     = 20" in var_tf  # unquoted number, not "20"

    # username/manage_master_user_password are NOT in MANDATORY_FIELDS (the
    # secure default is intentional — see catalog.py) and must stay exactly
    # as arch2terraform's catalog emits them: real literals, untouched.
    main_tf = database.files["main.tf"]
    assert 'username                    = "admin"' in main_tf
    assert "manage_master_user_password = true" in main_tf
    assert 'variable "username"' not in var_tf


@pytest.mark.anyio
async def test_two_resources_same_type_different_values_get_disambiguated_names():
    """The real collision case flat/shared naming has to handle: a VPC and
    a subnet both need `cidr_block`, with genuinely different values.
    Sharing one `var.cidr_block` between them would be a hard bug (one
    resource's default silently overwritten by the other's), so the second
    one must get a disambiguated name instead."""
    vpc = ParsedResource(
        id="vpc-1", aws_resource_type="aws_vpc", logical_name="main_vpc",
        label="Main VPC", properties={"cidr_block": "10.0.0.0/16"},
    )
    subnet = ParsedResource(
        id="subnet-1", aws_resource_type="aws_subnet", logical_name="public_subnet",
        label="Public Subnet", properties={"cidr_block": "10.0.1.0/24", "availability_zone": "us-east-1a"},
    )
    conn = ParsedConnection(source_id="vpc-1", target_id="subnet-1", connection_type="containment")
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[vpc, subnet], connections=[conn])
    plan = await build_terraform_plan(diagram, project_name="test")
    networking = next(m for m in plan.modules if m.name == "networking")

    var_tf = networking.files["variables.tf"]
    main_tf = networking.files["main.tf"]

    # VPC keeps the flat/shared name (first claimant).
    assert 'variable "cidr_block" {' in var_tf
    assert 'default     = "10.0.0.0/16"' in var_tf
    # Subnet's DIFFERENT value gets disambiguated, not silently dropped/shared.
    assert 'variable "cidr_block_public_subnet" {' in var_tf
    assert 'default     = "10.0.1.0/24"' in var_tf
    # Exactly one declaration of each — no duplicate/colliding `variable "cidr_block"` blocks.
    assert var_tf.count('variable "cidr_block" {') == 1
    assert var_tf.count('variable "cidr_block_public_subnet" {') == 1
    assert "cidr_block = var.cidr_block\n" in main_tf or "cidr_block        = var.cidr_block\n" in main_tf


@pytest.mark.anyio
async def test_two_resources_same_type_same_value_share_one_variable():
    """Flat/shared naming's whole point: when two resources genuinely want
    the SAME value, they should share one variable, not get two redundant
    ones."""
    instance_a = ParsedResource(
        id="ec2-a", aws_resource_type="aws_instance", logical_name="web_a",
        label="Web A", properties={"ami": "ami-0shared00000000", "instance_type": "t3.micro"},
    )
    instance_b = ParsedResource(
        id="ec2-b", aws_resource_type="aws_instance", logical_name="web_b",
        label="Web B", properties={"ami": "ami-0shared00000000", "instance_type": "t3.micro"},
    )
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[instance_a, instance_b], connections=[])
    plan = await build_terraform_plan(diagram, project_name="test")
    compute = next(m for m in plan.modules if m.name == "compute")

    var_tf = compute.files["variables.tf"]
    assert var_tf.count('variable "ami" {') == 1
    assert 'variable "ami_web_a"' not in var_tf
    assert 'variable "ami_web_b"' not in var_tf
    # Both resources reference the same shared variable.
    assert compute.files["main.tf"].count("var.ami") == 2


@pytest.mark.anyio
async def test_generic_fallback_covered_resource_type_also_gets_variable_ized():
    """Extended 2026-07-08: resource types with NO MANDATORY_FIELDS entry
    (only caught by missing_info_detector's generic placeholder fallback)
    must ALSO end up tfvars-overridable, not just asked-and-baked. Requires
    running the resource through detect_missing_info() first — that's what
    actually populates resource.variableize_keys; build_terraform_plan alone
    has no static list to fall back on for a type like aws_ecr_repository."""
    ecr = ParsedResource(
        id="ecr-1", aws_resource_type="aws_ecr_repository", logical_name="my_repo",
        label="My ECR Repo", properties={"name": "replace-with-repository-name"},
    )
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[ecr], connections=[])
    detect_missing_info(diagram, "job1")  # mutates ecr.variableize_keys in place
    assert ecr.variableize_keys == ["name"]

    ecr.properties["name"] = "my-real-repo-name"  # simulate the answer being applied
    plan = await build_terraform_plan(diagram, project_name="test")
    containers = next(m for m in plan.modules if m.name == "containers")

    assert "name = var.name" in containers.files["main.tf"]
    assert '"my-real-repo-name"' not in containers.files["main.tf"]
    var_tf = containers.files["variables.tf"]
    assert 'variable "name" {' in var_tf
    assert 'default     = "my-real-repo-name"' in var_tf


@pytest.mark.anyio
async def test_resource_with_no_variableize_keys_falls_back_to_mandatory_fields_only():
    """Backward-compat guard: a ParsedResource built directly (never through
    detect_missing_info — e.g. every other synthetic test in this file) has
    an empty variableize_keys, and must behave exactly as before this
    extension: MANDATORY_FIELDS-covered fields still get variable-ized."""
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0realvalue1234567", "instance_type": "t3.micro"},
    )
    assert instance.variableize_keys == []
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[instance], connections=[])
    plan = await build_terraform_plan(diagram, project_name="test")
    compute = next(m for m in plan.modules if m.name == "compute")
    assert 'variable "ami" {' in compute.files["variables.tf"]
