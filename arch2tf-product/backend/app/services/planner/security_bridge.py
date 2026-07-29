"""
Security Engine Bridge
-----------------------
Integration point between terraform_planner.py and the Terraform
Accelerators security engine (app/services/security_engine/) — the module
originally built and verified standalone under Theisis/implementation/security
(traffic flow analysis, security group generation, dynamic least-privilege
IAM policy generation, resource-to-role linking, orchestration). Copied here
verbatim (after fixing 6 real bugs found by actually running it: invalid
${...} HCL interpolation, a dangling security-group reference for Lambda
sources, duplicate aws_iam_role_policy resource addresses, a role-naming
mismatch that made every attachment reference dangling, placeholder IAM
policies generated for plain network edges, and the traffic analyzer's
Level-1/2/3 port inference never actually being consulted) — see that
module's own file history for details.

This is a genuinely separate pipeline stage from per-diagram-node resource
generation (_resource_block_lines in terraform_planner.py): it derives
security groups and IAM policies from ALL of a diagram's resources and
connections as a whole graph, the same way the security engine's own docs
describe it ("[PLAN] -> [SECURITY MODULE] -> [VALIDATE]"), so a diagram never
has to include explicit security-group or IAM-role nodes to get either.

Why security_engine/ has flat internal imports (`from models import ...`,
not `from .models import ...`): it's the same source used standalone via
`python complete_implementation_example.py`. Rather than rewrite every
internal import across 7 files (real risk of introducing a fresh bug into
code that was just carefully debugged and verified), this bridge inserts the
security_engine/ directory itself onto sys.path — the same "insert the
sibling package's own directory" approach terraform_planner.py already uses
for arch2terraform's src/.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

_SECURITY_ENGINE_SRC = Path(__file__).resolve().parents[1] / "security_engine"
if str(_SECURITY_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(_SECURITY_ENGINE_SRC))

from complete_security_orchestrator import CompleteSecurityOrchestrator  # type: ignore  # noqa: E402
from resource_role_linker import ResourceRoleLinker  # type: ignore  # noqa: E402

if TYPE_CHECKING:
    from shared.schemas.models import ParsedDiagram, ParsedResource

# Data sources the generated iam_roles.tf's ARNs reference
# (${data.aws_region.current.name}, ${data.aws_caller_identity.current.account_id})
# — declared once here rather than relying on the security engine's own
# variables.tf (which also declares vpc_id/namespace/region variables Phase 2
# doesn't use in this form, so that file is intentionally NOT merged in).
_IAM_DATA_SOURCES = (
    "# Data sources required by generated IAM policy ARNs\n"
    'data "aws_caller_identity" "current" {}\n'
    "\n"
    'data "aws_region" "current" {}\n'
)

# resource type -> (attribute to set on the resource, parent resource type
# that holds it, attribute of THAT parent to reference, logical-name suffix).
# Deliberately NOT reusing the security engine's own resource_role_linker.py
# attachment templates here (_generate_ec2_attachment etc.) - those render a
# full standalone `resource "aws_instance" "..." { ... }` skeleton with no
# ami/instance_type, meant as a stand-alone illustration when this engine is
# used on its own. In this integration the REAL aws_instance/aws_lambda_function
# already exists as a fully-formed resource in its own module (compute/
# serverless/containers) - so this maps to a single attribute set on THAT
# existing resource instead, wired across modules the same way containment
# attributes are (see terraform_planner.py's _CrossModuleWire).
ATTACHMENT_ATTR_BY_RESOURCE_TYPE: dict[str, tuple[str, str, str, str]] = {
    "aws_instance": ("iam_instance_profile", "aws_iam_instance_profile", "name", "_profile"),
    "aws_lambda_function": ("role", "aws_iam_role", "arn", ""),
    "aws_ecs_task_definition": ("execution_role_arn", "aws_iam_role", "arn", ""),
    "aws_sfn_state_machine": ("role_arn", "aws_iam_role", "arn", ""),
}

# Resource types security_group_generator.py (Theisis/implementation/security)
# creates a security group for — mirrors its own SG_ELIGIBLE_TYPES exactly
# (aws_instance/aws_lb/aws_db_instance/aws_elasticache_cluster) — mapped to
# the HCL attribute that actually attaches a security group to that resource
# type. Added 2026-07-24: checkov's CKV2_AWS_5 ("Security Groups are
# attached to another resource") was failing on every generated SG because,
# unlike the IAM role/instance-profile attachment above, nothing wired the
# generated aws_security_group onto the resource it was named after — the SG
# existed but nothing ever referenced its id, a real orphaned-resource bug,
# not a false positive.
SG_ATTACHMENT_ATTR_BY_RESOURCE_TYPE: dict[str, str] = {
    "aws_instance": "vpc_security_group_ids",
    "aws_lb": "security_groups",
    "aws_db_instance": "vpc_security_group_ids",
    "aws_elasticache_cluster": "security_group_ids",
}


def _sg_terraform_name(namespace: str, resource_label: str) -> str:
    """Mirrors security_group_generator.py's own
    _get_sg_resource_name() exactly — duplicated here (rather than imported)
    because it's an instance method requiring an already-constructed
    SecurityGroupGenerator, and this only needs the pure string transform to
    predict what name a given resource's SG WOULD have, so we can check
    whether the engine actually generated one for it and wire the real
    resource address rather than guess-and-hope."""
    return f"{namespace}_{resource_label.lower().replace(' ', '_').replace('-', '_')}_sg"


def _build_resource_graph(parsed: "ParsedDiagram") -> dict:
    """
    Maps Phase 2's ParsedDiagram into the {nodes, edges} shape
    complete_security_orchestrator.CompleteSecurityOrchestrator expects.
    Containment edges are excluded — they express diagram nesting (e.g. "this
    subnet is inside this VPC"), not traffic/API calls, so they carry no
    security-relevant signal (mirrors terraform_planner.py's own
    `relevant_conns` filter for the exact same reason).
    """
    nodes = [
        {
            "id": r.id,
            "label": r.label,
            "type": r.aws_resource_type,
            # Only pass through simple scalar properties (e.g. `engine` for
            # aws_db_instance/aws_elasticache_cluster) — the traffic flow
            # analyzer's Level-3 inference reads metadata.get('engine'), not
            # full resource properties, and some property values here are
            # already-rendered HCL reference strings (var.*, aws_x.y.id)
            # rather than plain data, which would be meaningless as metadata.
            "metadata": {
                k: v for k, v in r.properties.items()
                if isinstance(v, (str, int, float, bool))
                and not (isinstance(v, str) and (v.startswith("var.") or "." in v and v.split(".")[0].startswith("aws_")))
            },
            # Diagram-native custom-data tags (draw.io Edit Data, Excalidraw
            # customData — see arch2terraform's DiagramNode.tags docstring).
            # security_group_generator.py reads tags['tier'] (public/private/
            # internal) to decide network exposure; nothing else in the
            # security engine reads tags yet.
            "tags": dict(r.tags),
        }
        for r in parsed.resources
    ]

    edges = [
        {
            "from": c.source_id,
            "to": c.target_id,
            "label": c.attribute_map.get("_label", ""),
            "type": "connection",
        }
        for c in parsed.connections
        if c.connection_type != "containment"
    ]

    return {"nodes": nodes, "edges": edges}


def run_security_engine(
    parsed: "ParsedDiagram",
    module_resources: dict[str, list["ParsedResource"]],
    project_name: str,
    environment: str,
) -> tuple[dict[str, dict[str, str]], list[dict], list[dict]]:
    """
    Runs the security engine over the full diagram once and returns
    (extra_module_files, attachment_specs, sg_attachment_specs):

    - extra_module_files: EXTRA files to merge into (or create) the
      'networking' module (generated security groups + rules) and 'security'
      module (generated IAM roles/policies). Deliberately does NOT include
      the security engine's own attachments.tf — that template renders a
      standalone, incomplete `resource "aws_instance" "..." {...}` skeleton
      (no ami/instance_type) meant for standalone use of this engine; merging
      it here would produce a second, invalid declaration of a resource that
      already exists, fully-formed, in its own module. See
      ATTACHMENT_ATTR_BY_RESOURCE_TYPE instead.
    - attachment_specs: one dict per resource that got an IAM role -
      {"resource_id", "resource_type", "role_tf_id"} - for the caller
      (terraform_planner.py) to wire the actual attachment attribute
      (iam_instance_profile / role / execution_role_arn / role_arn) onto the
      REAL resource via the existing cross-module wiring mechanism.
    - sg_attachment_specs: one dict per resource that got a security group -
      {"resource_id", "resource_type", "sg_attr", "sg_terraform_name"} - same
      idea, for the caller to wire the generated aws_security_group's id onto
      the real resource's security-group attribute (vpc_security_group_ids /
      security_groups / security_group_ids depending on resource type). Added
      2026-07-24 after checkov's CKV2_AWS_5 caught every generated SG sitting
      unattached — the engine created it, but nothing referenced its id.

    Returns ({}, [], []) if there's nothing to generate (empty diagram) or if
    the security engine errors — logged, not raised, so a bug in this
    optional layer can never take down the rest of an otherwise-successful
    plan (same "degrade, don't crash the pipeline" pattern used elsewhere in
    this backend, e.g. job_store.py's Redis fallback).
    """
    resource_graph = _build_resource_graph(parsed)
    if not resource_graph["nodes"]:
        return {}, [], []

    # Prefer a direct same-module reference to the diagram's own VPC — the
    # generated security groups always land in the 'networking' module
    # alongside it (see below), so this resolves within one file exactly
    # like any other same-module containment reference
    # (_wire_containment_attrs's same-module case). Falls back to a plain
    # var.vpc_id the user supplies at apply time for a diagram with no VPC
    # node (e.g. a Lambda-only serverless diagram that still wants IAM
    # policies but has no networking layer at all).
    vpc_id_ref = "var.vpc_id"
    for r in module_resources.get("networking", []):
        if r.aws_resource_type == "aws_vpc":
            vpc_id_ref = f"aws_vpc.{r.logical_name}.id"
            break

    # Cosmetic identifier-hygiene fix, 2026-07-24: this used to hyphen-join
    # ("demo-dev"), which then got spliced raw into every generated SG/IAM
    # Terraform identifier (_sg_terraform_name only strips hyphens from
    # resource_label, not namespace) — e.g. `aws_security_group.demo-dev_
    # public_alb_sg`. Hyphenated HCL identifiers are legal Terraform (this
    # was never a `terraform validate` bug), but underscore-only keeps
    # every generated identifier consistent with resource_label's own
    # underscore convention. NOTE: this does NOT fix Checkov's CKV2_AWS_5
    # false positive below — that's a separate, confirmed graph-resolution
    # limitation (see the checkov:skip annotation in terraform_generator.py).
    namespace = f"{project_name}_{environment}"
    orchestrator = CompleteSecurityOrchestrator(namespace=namespace, vpc_id=vpc_id_ref)

    try:
        result = orchestrator.execute_complete_implementation(resource_graph)
    except Exception:
        log.exception(
            "Security engine failed to generate security groups/IAM policies "
            "for this diagram - continuing without them"
        )
        return {}, [], []

    if result.status != "success":
        log.warning("Security engine returned status=%r: %s", result.status, result.error_message)
        return {}, [], []

    tf = result.terraform_files or {}
    extra: dict[str, dict[str, str]] = defaultdict(dict)

    # The security engine's own generators always return a non-empty string
    # (at minimum a header comment) even with zero security groups/roles
    # generated - checking truthiness/`.strip()` alone would create an empty
    # 'security'/'networking' module for every diagram, not just ones the
    # engine actually found something to secure. `resource "` only appears
    # when there's a real block to emit.
    if 'resource "' in tf.get("security_groups.tf", ""):
        extra["networking"]["generated_security_groups.tf"] = tf["security_groups.tf"]
        if vpc_id_ref == "var.vpc_id":
            # Only declare the fallback variable when it's actually referenced -
            # a diagram that DOES have a VPC node never needs it, since the
            # generated SGs reference that VPC directly instead.
            #
            # `default` matters here, not just style: this variable is declared
            # in its own file, outside the child_wires bookkeeping that
            # _generate_root_module() (terraform_planner.py) consults to decide
            # which extra `key = value` lines to add to root main.tf's
            # `module "networking" { ... }` call. With no default, root's call
            # never supplies it and `terraform init` fails outright with
            # "Missing required argument" on the module block — a real bug
            # found 2026-07-24 via an actual diagram with no aws_vpc node.
            # A placeholder VPC ID (same fake-but-format-valid pattern as
            # catalog.py's ami-/subnet-/lt- placeholders elsewhere) lets
            # validate/init pass; only apply (correctly) fails until the user
            # supplies a real VPC ID.
            extra["networking"]["generated_security_groups_variables.tf"] = "\n".join([
                'variable "vpc_id" {',
                '  description = "VPC ID for the auto-generated security groups (this diagram has no aws_vpc node) — override via terraform.tfvars or -var"',
                "  type        = string",
                '  default     = "vpc-00000000000000000"',
                "}",
                "",
            ])

    if 'resource "' in tf.get("iam_roles.tf", ""):
        extra["security"]["generated_iam_roles.tf"] = _IAM_DATA_SOURCES + "\n" + tf["iam_roles.tf"]

    attachment_specs: list[dict] = []
    if extra.get("security"):
        # Only wire attachments if the security module is actually going to
        # exist - if generated_iam_roles.tf ended up empty (shouldn't happen
        # given resource_mappings come from the same successful result, but
        # defensive: a resource_id in resource_mappings with no matching
        # role declared would be exactly the dangling-reference bug this
        # whole redesign exists to avoid).
        for resource_id, mapping in (result.resource_mappings or {}).items():
            attachment_specs.append({
                "resource_id": resource_id,
                "resource_type": mapping.resource_type,
                "role_tf_id": ResourceRoleLinker._role_tf_id(mapping.resource_label),
            })

    sg_attachment_specs: list[dict] = []
    sg_tf = tf.get("security_groups.tf", "")
    if 'resource "' in sg_tf:
        for resource in parsed.resources:
            sg_attr = SG_ATTACHMENT_ATTR_BY_RESOURCE_TYPE.get(resource.aws_resource_type)
            if not sg_attr:
                continue
            expected_sg_name = _sg_terraform_name(namespace, resource.label or resource.logical_name)
            # Only wire it if the engine actually declared this exact SG —
            # a resource type being SG-eligible doesn't guarantee one was
            # generated (e.g. the engine could skip it for a reason this
            # bridge doesn't need to duplicate); checking the real output
            # instead of assuming avoids wiring a reference to a resource
            # that was never declared.
            if f'resource "aws_security_group" "{expected_sg_name}"' in sg_tf:
                sg_attachment_specs.append({
                    "resource_id": resource.id,
                    "resource_type": resource.aws_resource_type,
                    "sg_attr": sg_attr,
                    "sg_terraform_name": expected_sg_name,
                })

    return dict(extra), attachment_specs, sg_attachment_specs
