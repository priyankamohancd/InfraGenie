"""
Security Engine Bridge — unit tests
-------------------------------------
Focused, synthetic-data tests for security_bridge.py's integration into
terraform_planner.py: security-group generation, least-privilege IAM policy
generation, and resource-role attachment wiring, all derived automatically
from a diagram's resources/connections with no explicit security-group or
IAM-role node required.

Complements the security engine's own standalone verification (Theisis/
implementation/security, now copied into app/services/security_engine/) and
test_cross_module_wiring.py, whose _CrossModuleWire/_generate_module_hcl
machinery this integration reuses (generalized with a `parent_attr` field —
see terraform_planner.py) rather than duplicating it.
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


def _three_tier_diagram() -> ParsedDiagram:
    """VPC -> subnet -> EC2 -> S3, with the EC2/S3 connection carrying an
    edge label the security engine should use for its policy naming (edge
    labels reach ParsedConnection.attribute_map["_label"] via
    arch2terraform_bridge.py's rel.label passthrough)."""
    vpc = ParsedResource(
        id="vpc-1", aws_resource_type="aws_vpc", logical_name="main_vpc",
        label="Main VPC", properties={"cidr_block": "10.0.0.0/16"},
    )
    subnet = ParsedResource(
        id="subnet-1", aws_resource_type="aws_subnet", logical_name="public_subnet",
        label="Public Subnet", properties={"cidr_block": "10.0.1.0/24"},
    )
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0abcdef1234567890", "instance_type": "t3.micro"},
    )
    bucket = ParsedResource(
        id="s3-1", aws_resource_type="aws_s3_bucket", logical_name="assets_bucket",
        label="Assets Bucket", properties={"bucket": "assets-bucket"},
    )
    connections = [
        ParsedConnection(source_id="vpc-1", target_id="subnet-1", connection_type="containment"),
        ParsedConnection(source_id="subnet-1", target_id="ec2-1", connection_type="containment"),
        ParsedConnection(source_id="ec2-1", target_id="s3-1", connection_type="dependency",
                          attribute_map={"_label": "Read"}),
    ]
    return ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[vpc, subnet, instance, bucket],
        connections=connections,
    )


@pytest.mark.anyio
async def test_generated_security_group_references_diagrams_own_vpc():
    plan = await build_terraform_plan(_three_tier_diagram(), project_name="test", environment="dev")
    networking = next(m for m in plan.modules if m.name == "networking")

    sg_tf = networking.files.get("generated_security_groups.tf", "")
    assert sg_tf, "expected a generated_security_groups.tf in the networking module"
    # Same-module reference to the diagram's own VPC, not the var.vpc_id
    # fallback (which is only for diagrams with no VPC node at all).
    assert "vpc_id      = aws_vpc.main_vpc.id" in sg_tf
    assert "var.vpc_id" not in sg_tf
    # No unquoted ${...} interpolation anywhere - the exact class of bug
    # found when this engine was first run for real (see security_engine's
    # own file history).
    assert "${aws_vpc" not in sg_tf
    assert "${aws_security_group" not in sg_tf


def _no_vpc_diagram() -> ParsedDiagram:
    """EC2 -> S3, with NO aws_vpc node anywhere in the diagram — exercises
    security_bridge.py's var.vpc_id fallback path (the diagram has no VPC
    for the generated security group to reference directly)."""
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0abcdef1234567890", "instance_type": "t3.micro"},
    )
    bucket = ParsedResource(
        id="s3-1", aws_resource_type="aws_s3_bucket", logical_name="assets_bucket",
        label="Assets Bucket", properties={"bucket": "assets-bucket"},
    )
    connections = [
        ParsedConnection(source_id="ec2-1", target_id="s3-1", connection_type="dependency",
                          attribute_map={"_label": "Read"}),
    ]
    return ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[instance, bucket],
        connections=connections,
    )


@pytest.mark.anyio
async def test_vpc_id_fallback_variable_has_a_default():
    """2026-07-24: a diagram with no aws_vpc node falls back to declaring
    `var.vpc_id` for the generated security group. That variable lives in
    its own file (generated_security_groups_variables.tf), outside the
    child_wires bookkeeping _generate_root_module() uses to decide what to
    pass into root main.tf's `module "networking" { ... }` call — so with
    no default, root's call never supplies it and a real
    `terraform init` fails with "Missing required argument" on that module
    block. Found via an actual diagram exercising this exact path. A
    placeholder default (same fake-but-format-valid pattern as every other
    catalog.py placeholder) keeps init/validate passing regardless."""
    plan = await build_terraform_plan(_no_vpc_diagram(), project_name="test", environment="dev")
    networking = next(m for m in plan.modules if m.name == "networking")

    vars_tf = networking.files.get("generated_security_groups_variables.tf", "")
    assert vars_tf, "expected a generated_security_groups_variables.tf in the networking module"
    assert 'variable "vpc_id"' in vars_tf
    assert "default" in vars_tf  # the actual bug: this used to be missing entirely
    assert 'default     = "vpc-00000000000000000"' in vars_tf

    sg_tf = networking.files.get("generated_security_groups.tf", "")
    assert "var.vpc_id" in sg_tf  # confirms this test actually exercises the fallback path

    # Root main.tf's module "networking" call has no reason to pass vpc_id
    # explicitly (it's not part of child_wires) — the whole point of the fix
    # is that the variable now tolerates that via its own default.
    root_main = plan.root_module_files.get("main.tf", "")
    assert 'module "networking"' in root_main


@pytest.mark.anyio
async def test_no_resource_module_declares_no_unused_standard_vars():
    """2026-07-24: the 'networking' module in a no-VPC diagram exists only
    to hold generated_security_groups.tf (forced into being — see
    terraform_planner.py's extra_module_files handling) with ZERO regular
    ParsedResources. Before pruning, it still declared aws_region/
    environment/project (root passes them into every module the same way)
    with nothing in the module's own files ever referencing them — flagged
    by tflint's terraform_unused_declarations rule in real testing.
    _prune_unused_standard_variables() must strip them, and root's
    `module "networking" { ... }` call must stop passing them too (or
    validate/init would fail on an unsupported argument instead)."""
    plan = await build_terraform_plan(_no_vpc_diagram(), project_name="test", environment="dev")
    networking = next(m for m in plan.modules if m.name == "networking")
    assert networking.source_resources == []  # confirms this is the zero-resource case

    vars_tf = networking.files.get("variables.tf", "")
    for var_name in ("aws_region", "environment", "project"):
        assert f'variable "{var_name}"' not in vars_tf, (
            f"'{var_name}' should have been pruned — nothing in the networking "
            f"module references it when the module has no real resources"
        )

    root_main = plan.root_module_files["main.tf"]
    networking_block = root_main.split('module "networking" {')[1].split("\n}")[0]
    for var_name in ("aws_region", "environment", "project"):
        assert f"= var.{var_name}" not in networking_block, (
            f"root should not pass '{var_name}' to a module that no longer declares it"
        )
    assert "common_tags = local.common_tags" in networking_block  # never pruned


@pytest.mark.anyio
async def test_generated_iam_role_gets_least_privilege_s3_policy():
    plan = await build_terraform_plan(_three_tier_diagram(), project_name="test", environment="dev")
    security = next(m for m in plan.modules if m.name == "security")

    iam_tf = security.files.get("generated_iam_roles.tf", "")
    assert iam_tf, "expected a generated_iam_roles.tf in the security module"
    assert 'resource "aws_iam_role" "role_web_server"' in iam_tf
    assert "s3:GetObject" in iam_tf
    # No placeholder garbage for edges the engine doesn't understand.
    assert "ACTION_PLACEHOLDER" not in iam_tf
    assert "arn:aws:PLACEHOLDER" not in iam_tf
    # The data sources its ARNs reference are actually declared somewhere
    # in this module (see security_bridge.py's _IAM_DATA_SOURCES).
    assert 'data "aws_caller_identity" "current"' in iam_tf
    assert 'data "aws_region" "current"' in iam_tf


@pytest.mark.anyio
async def test_role_attachment_uses_cross_module_wiring_not_duplicate_resource():
    """The security engine's own attachments.tf template renders a second,
    incomplete `resource "aws_instance" "web_server" {...}` skeleton (no
    ami/instance_type) - a guaranteed terraform validate failure once merged
    next to the REAL, fully-formed aws_instance in the compute module. This
    integration must never emit that; instead the attachment is a single
    attribute wired onto the real resource, cross-module, exactly like any
    other containment reference."""
    plan = await build_terraform_plan(_three_tier_diagram(), project_name="test", environment="dev")
    compute = next(m for m in plan.modules if m.name == "compute")
    security = next(m for m in plan.modules if m.name == "security")

    # No second aws_instance declaration anywhere in the security module.
    for content in security.files.values():
        assert 'resource "aws_instance"' not in content

    # Exactly one aws_instance in the whole plan (the real one, in compute).
    # NOTE: compute.files and security.files both use filenames like
    # "main.tf" — must NOT be merged into one dict keyed by filename (that
    # would silently drop one module's content), so count across both
    # modules' values as separate lists instead.
    total_instance_decls = sum(
        content.count('resource "aws_instance"')
        for content in list(compute.files.values()) + list(security.files.values())
    )
    assert total_instance_decls == 1

    # The real instance's iam_instance_profile is wired via a variable...
    # Padding varies with the longest attribute key in the block (e.g.
    # vpc_security_group_ids, once a security group gets attached — see
    # the 2026-07-24 SG-attachment fix), so match loosely rather than on
    # exact alignment.
    assert re.search(
        r"iam_instance_profile\s*=\s*var\.security_role_web_server_profile_name",
        compute.files["main.tf"],
    )
    assert 'variable "security_role_web_server_profile_name"' in compute.files["variables.tf"]

    # ...which the security module actually declares an output for (the
    # exact dangling-reference bug caught when this was first wired up: the
    # skeleton-module code path passed hardcoded empty wire sets and never
    # emitted this output at all).
    assert 'output "role_web_server_profile_name"' in security.files["outputs.tf"]
    assert "aws_iam_instance_profile.role_web_server_profile.name" in security.files["outputs.tf"]

    # ...and the root module actually passes it through.
    assert "security_role_web_server_profile_name = module.security.role_web_server_profile_name" \
        in plan.root_module_files["main.tf"]


def _alb_diagram(tags: dict | None = None, label: str = "Web ALB") -> ParsedDiagram:
    """A single internet-facing-shaped ALB, optionally tagged with a tier."""
    alb = ParsedResource(
        id="alb-1", aws_resource_type="aws_lb", logical_name="web_alb",
        label=label, properties={"load_balancer_type": "application"},
        tags=tags or {},
    )
    return ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[alb], connections=[])


@pytest.mark.anyio
async def test_tier_public_tag_opens_alb_to_internet_even_without_public_label():
    """2026-07-21: diagram custom-data tags (tier=public/private/internal) now
    drive network exposure — see security_group_generator.py's
    _resolve_tier_public(). A tier=public tag must open HTTP/HTTPS to
    0.0.0.0/0 even when the label itself doesn't contain "public"."""
    plan = await build_terraform_plan(
        _alb_diagram(tags={"tier": "public"}, label="Internal Gateway"),
        project_name="test", environment="dev",
    )
    networking = next(m for m in plan.modules if m.name == "networking")
    sg_tf = networking.files.get("generated_security_groups.tf", "")
    assert "0.0.0.0/0" in sg_tf


@pytest.mark.anyio
async def test_tier_private_tag_overrides_public_label_heuristic():
    """The inverse: a label that WOULD trigger the legacy "public" in label
    heuristic must be overridden to the internal CIDR by an explicit
    tier=private/internal tag, never left open to 0.0.0.0/0."""
    plan = await build_terraform_plan(
        _alb_diagram(tags={"tier": "private"}, label="Public Web ALB"),
        project_name="test", environment="dev",
    )
    networking = next(m for m in plan.modules if m.name == "networking")
    sg_tf = networking.files.get("generated_security_groups.tf", "")
    # Ingress must be scoped internal, not left open (egress legitimately
    # stays 0.0.0.0/0 - that's outbound traffic, unrelated to this feature).
    assert "HTTP from Internet" not in sg_tf
    assert "HTTPS from Internet" not in sg_tf
    assert "HTTP from internal network" in sg_tf
    assert "10.0.0.0/8" in sg_tf


@pytest.mark.anyio
async def test_untagged_alb_keeps_legacy_label_heuristic():
    """No tags at all (the common case for diagrams that never used custom
    data) must behave exactly as before this feature existed."""
    plan = await build_terraform_plan(
        _alb_diagram(tags={}, label="Public Web ALB"),
        project_name="test", environment="dev",
    )
    networking = next(m for m in plan.modules if m.name == "networking")
    sg_tf = networking.files.get("generated_security_groups.tf", "")
    assert "0.0.0.0/0" in sg_tf


@pytest.mark.anyio
async def test_no_security_module_created_for_diagram_with_no_iam_eligible_resources():
    """A diagram with only data/storage resources (no EC2/Lambda/etc., no
    outbound edges needing IAM) shouldn't get an empty 'security' module
    manufactured for no reason."""
    bucket = ParsedResource(
        id="s3-1", aws_resource_type="aws_s3_bucket", logical_name="assets_bucket",
        label="Assets Bucket", properties={"bucket": "assets-bucket"},
    )
    diagram = ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[bucket], connections=[])

    plan = await build_terraform_plan(diagram, project_name="test", environment="dev")
    assert not any(m.name == "security" for m in plan.modules)
