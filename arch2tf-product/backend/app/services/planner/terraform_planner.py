"""
Terraform Planner + Module Selection Engine
--------------------------------------------
Takes a ParsedDiagram (with answers applied) and produces a TerraformPlan:
  - Groups resources into logical modules (networking, compute, database, etc.)
  - Calls the HCL generator for each module
  - Builds root module that calls all child modules
  - Returns the full TerraformPlan with all files

As of 2026-07-08, per-resource HCL block generation itself is delegated to
arch2terraform.generator.hcl_format.resource_block() (see _resource_block_lines
below) instead of a local reimplementation — this is what actually carries
arch2terraform's real-`terraform validate`-audited catalog work (required
arguments, ARN-shaped placeholders, nested HCL blocks) into Phase 2's output.
Module grouping and cross-module containment wiring below are genuinely new
Phase 2 concerns arch2terraform itself never had to solve, since it only ever
emits one flat main.tf.
"""
from __future__ import annotations
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

log = logging.getLogger(__name__)

from app.core.config import get_settings

# Shared schemas
# services/planner/terraform_planner.py -> planner(0)/services(1)/app(2)/
# backend(3)/arch2tf-product(4). Was parents[5] (one level too far, lands on
# "thesis") — a pre-existing bug, same class as missing_info_detector.py's.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.schemas.models import (
    ParsedDiagram, ParsedResource, ParsedConnection, TerraformPlan, TerraformModule
)

# arch2terraform package (see arch2terraform_bridge.py for the parsing-side
# integration point — this is the generation-side one)
_ARCH2TF_SRC = Path(__file__).resolve().parents[5] / "arch2terraform" / "src"
if str(_ARCH2TF_SRC) not in sys.path:
    sys.path.insert(0, str(_ARCH2TF_SRC))
from arch2terraform.generator.hcl_format import resource_block as _a2tf_resource_block

# Same source of truth the clarification UI uses (missing_info_detector.py) —
# reused here so "every field this system ever asked the user to fill in"
# and "every field that gets variable-ized instead of hardcoded" can never
# silently drift apart into two different lists.
from app.services.parser.missing_info_detector import MANDATORY_FIELDS
from app.services.planner.security_bridge import (
    ATTACHMENT_ATTR_BY_RESOURCE_TYPE,
    run_security_engine,
)

# ── Containment wiring rules ─────────────────────────────────────────────────
# Mirrors arch2terraform/generator/hcl_generator.py's _CONTAINMENT_WIRING_RULES
# exactly (child_resource_type -> (attribute_name, required_parent_resource_type)).
# arch2terraform's catalog intentionally leaves these attributes unset in
# default_attributes (e.g. aws_subnet has no default vpc_id) so the generator
# wires them from the diagram's actual containment structure instead of a
# fake placeholder — Phase 2 must replicate that wiring or these attributes
# would simply be missing from the generated HCL.
_CONTAINMENT_WIRING_RULES: dict[str, tuple[str, str]] = {
    "aws_subnet": ("vpc_id", "aws_vpc"),
    "aws_internet_gateway": ("vpc_id", "aws_vpc"),
    "aws_security_group": ("vpc_id", "aws_vpc"),
    "aws_instance": ("subnet_id", "aws_subnet"),
    "aws_route_table": ("vpc_id", "aws_vpc"),
    "aws_network_acl": ("vpc_id", "aws_vpc"),
    "aws_lb_target_group": ("vpc_id", "aws_vpc"),
    "aws_nat_gateway": ("subnet_id", "aws_subnet"),
}

@dataclass(frozen=True)
class _CrossModuleWire:
    """
    One cross-module containment reference that needs real Terraform
    plumbing: an `output` in the parent's module, a `variable` in the
    child's module, and a module-call argument in the root module
    connecting the two. See _wire_containment_attrs, and where this type is
    consumed in _generate_module_hcl (both as a "child module" needing the
    variable declared, and as a "parent module" needing the output to
    exist) and _generate_root_module (needing the module-call argument).

    Frozen + hashable so callers can collect these in a set and dedupe
    automatically when multiple child resources reference the same parent.
    """
    parent_module: str
    parent_resource_type: str
    parent_logical_name: str
    child_module: str
    # Which attribute of the parent resource is being wired through - "id"
    # for every containment case (subnet_id, vpc_id, ...), but also used by
    # the security-engine role-attachment wiring below with "name" (an
    # instance profile's .name) or "arn" (a role's .arn). Defaulting to "id"
    # keeps every existing containment call site's variable_name/output_name
    # strings byte-for-byte identical to before this field existed.
    parent_attr: str = "id"

    @property
    def variable_name(self) -> str:
        # Namespaced by parent module + parent resource so two different
        # cross-module references (e.g. two different subnets) in the same
        # child module never collide.
        return f"{self.parent_module}_{self.parent_logical_name}_{self.parent_attr}"

    @property
    def output_name(self) -> str:
        return f"{self.parent_logical_name}_{self.parent_attr}"

# ── Module grouping rules ────────────────────────────────────────────────────
# resource_type_prefix → module name
MODULE_ASSIGNMENT: dict[str, str] = {
    # Networking
    "aws_vpc":                    "networking",
    "aws_subnet":                 "networking",
    "aws_internet_gateway":       "networking",
    "aws_nat_gateway":            "networking",
    "aws_route_table":            "networking",
    "aws_route_table_association":"networking",
    "aws_security_group":         "networking",
    "aws_network_acl":            "networking",
    "aws_vpc_peering_connection": "networking",
    "aws_vpn_gateway":            "networking",
    "aws_dx_connection":          "networking",
    "aws_eip":                    "networking",
    # Compute
    "aws_instance":               "compute",
    "aws_autoscaling_group":      "compute",
    "aws_launch_template":        "compute",
    "aws_lb":                     "compute",
    "aws_lb_listener":            "compute",
    "aws_lb_target_group":        "compute",
    # Containers
    "aws_ecs_cluster":            "containers",
    "aws_ecs_service":            "containers",
    "aws_ecs_task_definition":    "containers",
    "aws_eks_cluster":            "containers",
    "aws_eks_node_group":         "containers",
    "aws_ecr_repository":         "containers",
    "aws_batch_job_definition":   "containers",
    # Serverless
    "aws_lambda_function":        "serverless",
    "aws_api_gateway_rest_api":   "serverless",
    "aws_appsync_graphql_api":    "serverless",
    "aws_sfn_state_machine":      "serverless",
    # Database
    "aws_db_instance":            "database",
    "aws_rds_cluster":            "database",
    "aws_rds_cluster_instance":   "database",
    "aws_dynamodb_table":         "database",
    "aws_elasticache_cluster":    "database",
    "aws_elasticache_replication_group": "database",
    "aws_opensearch_domain":      "database",
    # Storage
    "aws_s3_bucket":              "storage",
    "aws_s3_bucket_versioning":   "storage",
    "aws_efs_file_system":        "storage",
    "aws_ebs_volume":             "storage",
    # Messaging
    "aws_sqs_queue":              "messaging",
    "aws_sns_topic":              "messaging",
    "aws_kinesis_stream":         "messaging",
    "aws_kinesis_firehose_delivery_stream": "messaging",
    "aws_mq_broker":              "messaging",
    "aws_msk_cluster":            "messaging",
    "aws_eventbridge_event_bus":  "messaging",
    # Security & IAM
    "aws_iam_role":               "security",
    "aws_iam_policy":             "security",
    "aws_iam_role_policy_attachment": "security",
    "aws_kms_key":                "security",
    "aws_secretsmanager_secret":  "security",
    "aws_acm_certificate":        "security",
    "aws_wafv2_web_acl":          "security",
    "aws_cognito_user_pool":      "security",
    # CDN & DNS
    "aws_cloudfront_distribution":"cdn",
    "aws_route53_zone":           "cdn",
    "aws_route53_record":         "cdn",
    # Observability
    "aws_cloudwatch_log_group":   "observability",
    "aws_cloudwatch_metric_alarm":"observability",
    # CI/CD
    "aws_codepipeline":           "cicd",
    "aws_codebuild_project":      "cicd",
    "aws_codecommit_repository":  "cicd",
}

MODULE_DESCRIPTIONS: dict[str, str] = {
    "networking":    "VPC, subnets, internet gateway, NAT, security groups, route tables",
    "compute":       "EC2 instances, auto scaling groups, load balancers",
    "containers":    "ECS/EKS clusters, task definitions, ECR repositories",
    "serverless":    "Lambda functions, API Gateway, Step Functions",
    "database":      "RDS, DynamoDB, ElastiCache, OpenSearch",
    "storage":       "S3 buckets, EFS file systems, EBS volumes",
    "messaging":     "SQS queues, SNS topics, Kinesis, MSK, EventBridge",
    "security":      "IAM roles/policies, KMS keys, Secrets Manager, ACM, WAF",
    "cdn":           "CloudFront distributions, Route53 hosted zones and records",
    "observability": "CloudWatch log groups and alarms",
    "cicd":          "CodePipeline, CodeBuild, CodeCommit",
}


async def build_terraform_plan(
    parsed: ParsedDiagram,
    aws_region: str = "us-east-1",
    environment: str = "dev",
    project_name: str = "arch2terraform",
) -> TerraformPlan:
    """
    Main planner entry point.
    Groups resources into modules, generates HCL, builds root module.
    """
    # Group resources into modules
    module_resources: dict[str, list[ParsedResource]] = defaultdict(list)
    for resource in parsed.resources:
        module_name = MODULE_ASSIGNMENT.get(resource.aws_resource_type, "misc")
        module_resources[module_name].append(resource)

    # Wire containment relationships (e.g. subnet.vpc_id) into real Terraform
    # references now that every resource has a module assignment — must run
    # after module_resources is fully built (needs to know which module each
    # resource landed in) and before HCL generation for any module.
    wiring_warnings, cross_module_wires = _wire_containment_attrs(module_resources, parsed)

    # Security engine: derives security-group rules and least-privilege IAM
    # policies from the whole resource graph — genuinely separate from the
    # per-node generation below, since it needs no explicit security-group
    # or IAM-role node in the diagram to produce either. Must run before the
    # main HCL-generation loop below: it both mutates compute resources'
    # properties in place (wiring the real generated role/instance-profile
    # onto e.g. an EC2 instance's iam_instance_profile attribute) and adds
    # entries to cross_module_wires, both of which the loop needs to see
    # already settled — same reasoning as containment wiring just above. See
    # security_bridge.py's module docstring for the full design rationale.
    extra_module_files, attachment_specs, sg_attachment_specs = run_security_engine(
        parsed, module_resources, project_name, environment,
    )
    _wire_role_attachments(module_resources, attachment_specs, cross_module_wires)
    _wire_sg_attachments(module_resources, sg_attachment_specs, cross_module_wires)

    # Index cross-module wires both ways: by child module (needs the
    # `variable` declared + uses it in a resource attribute) and by parent
    # module (needs the `output` to exist so the root module can read it).
    # Built AFTER _wire_role_attachments so its wires (e.g. compute's EC2
    # instance needing the security module's generated instance profile) are
    # included alongside containment wires from the very first module that
    # consumes wires_by_child/wires_by_parent below.
    wires_by_child: dict[str, set[_CrossModuleWire]] = defaultdict(set)
    wires_by_parent: dict[str, set[_CrossModuleWire]] = defaultdict(set)
    for wire in cross_module_wires:
        wires_by_child[wire.child_module].add(wire)
        wires_by_parent[wire.parent_module].add(wire)

    # Force every module the security engine has extra files for to exist
    # even if no diagram resource landed there on its own (e.g. IAM
    # roles/policies generated for a diagram with no other
    # 'security'-assigned resource) - module_resources is a defaultdict(list),
    # so simply indexing it creates an empty entry. Must happen before the
    # main generation loop below so that module goes through the exact same
    # code path (correct wires_by_child/wires_by_parent/mandatory_vars
    # lookups) as every other module, rather than a separately-built skeleton
    # that - as first written and caught by actually running this end to end
    # - passed hardcoded empty wire sets and silently produced a plan whose
    # root module referenced an output the security module never declared.
    for mod_name in extra_module_files:
        module_resources[mod_name]  # noqa: B018 - intentional defaultdict touch

    # Turn every MANDATORY_FIELDS-covered value (ami, instance_type, engine,
    # cidr_block, etc. — whatever the clarification UI asked about) into a
    # real per-module `variable` instead of a baked literal. Must run after
    # module_resources/containment wiring are settled (same reasoning as
    # above) and before HCL generation for any module.
    mandatory_vars_by_module = _variableize_mandatory_fields(module_resources)

    # Only create modules for groups that have resources (or were forced
    # above because the security engine has extra files for them)
    modules: list[TerraformModule] = []
    module_std_vars: dict[str, set[str]] = {}
    for mod_name, resources in module_resources.items():
        tf_files = _generate_module_hcl(
            mod_name, resources, parsed, aws_region, environment, project_name,
            wiring_warnings,
            wires_by_child.get(mod_name, set()),
            wires_by_parent.get(mod_name, set()),
            mandatory_vars_by_module.get(mod_name, {}),
        )
        tf_files.update(extra_module_files.get(mod_name, {}))
        # Some modules exist with NO regular ParsedResources at all — only
        # forced into being because the security engine has extra files for
        # them (e.g. a diagram with no aws_vpc node still gets a
        # "networking" module purely to hold generated_security_groups.tf).
        # _resource_block_lines' tags block is what normally makes
        # aws_region/environment/project genuinely "used" inside a module —
        # with zero resources, that never runs, so those three would stay
        # declared-but-unreferenced. Prune whichever ones truly aren't used
        # anywhere in this module's files, and remember which survive so the
        # root module call below stays symmetric with what's actually
        # declared (passing an argument a module no longer declares would
        # trade one terraform error for another).
        tf_files, kept_std_vars = _prune_unused_standard_variables(tf_files)
        module_std_vars[mod_name] = kept_std_vars
        modules.append(TerraformModule(
            name=mod_name,
            source_resources=[r.id for r in resources],
            description=MODULE_DESCRIPTIONS.get(mod_name, mod_name),
            files=tf_files,
        ))

    # Build root module that wires child modules together
    root_files = _generate_root_module(
        modules, aws_region, environment, project_name, wires_by_child, module_std_vars,
    )

    total = sum(len(m.source_resources) for m in modules)

    return TerraformPlan(
        modules=modules,
        root_module_files=root_files,
        resource_count=total,
    )


def _generate_module_hcl(
    module_name: str,
    resources: list[ParsedResource],
    diagram: ParsedDiagram,
    aws_region: str,
    environment: str,
    project_name: str,
    wiring_warnings: dict[str, str],
    child_wires: set[_CrossModuleWire],
    parent_wires: set[_CrossModuleWire],
    mandatory_vars: dict[str, dict],
) -> dict[str, str]:
    """Generate all .tf files for one module."""
    files: dict[str, str] = {}

    # Build a cross-resource reference map for this module. "containment"
    # connections are excluded here — they're handled by _wire_containment_attrs
    # instead (folding them into this generic depends_on logic would be
    # backwards: a VPC doesn't depend_on its subnet, and doing so would create
    # a circular reference once the subnet's vpc_id attribute references the VPC).
    resource_ids = {r.id for r in resources}
    relevant_conns = [
        c for c in diagram.connections
        if c.source_id in resource_ids and c.connection_type != "containment"
    ]

    # ── main.tf ─────────────────────────────────────────────────────────────
    main_lines = [
        f"# Module: {module_name}",
        f"# {MODULE_DESCRIPTIONS.get(module_name, '')}",
        "# Generated by arch2terraform — do not edit manually",
        "",
    ]

    # Connection attribute lookup: resource_id → {attr: value}
    conn_attrs: dict[str, dict] = defaultdict(dict)
    for conn in relevant_conns:
        for k, v in conn.attribute_map.items():
            if not k.startswith("_"):
                conn_attrs[conn.source_id][k] = v

    # Build depends_on from connections
    dep_map: dict[str, list[str]] = defaultdict(list)
    for conn in relevant_conns:
        if conn.target_id in resource_ids:
            tgt = next((r for r in resources if r.id == conn.target_id), None)
            src = next((r for r in resources if r.id == conn.source_id), None)
            if tgt and src:
                dep_map[src.id].append(
                    f"{tgt.aws_resource_type}.{tgt.logical_name}"
                )

    for resource in resources:
        main_lines += _resource_block_lines(
            resource,
            conn_attrs.get(resource.id, {}),
            dep_map.get(resource.id, []),
            wiring_warnings.get(resource.id),
        )
        main_lines.append("")
        # Extra top-level resources this one depends on (e.g. aws_mq_broker's
        # random_password/Secrets Manager pair) — must land in the SAME
        # module as their owner, right after it for readability.
        for block in resource.companion_blocks:
            main_lines.append(block)
            main_lines.append("")

    files["main.tf"] = "\n".join(main_lines)

    # ── variables.tf ─────────────────────────────────────────────────────────
    var_lines = [
        f"# Module: {module_name} — input variables",
        "",
        'variable "aws_region" {',
        '  description = "AWS region"',
        '  type        = string',
        "}",
        "",
        'variable "environment" {',
        '  description = "Environment name"',
        '  type        = string',
        "}",
        "",
        'variable "project" {',
        '  description = "Project name"',
        '  type        = string',
        "}",
        "",
        'variable "common_tags" {',
        '  description = "Tags to apply to all resources"',
        '  type        = map(string)',
        '  default     = {}',
        "}",
        "",
    ]
    # Cross-module containment wiring: this module is the CHILD side for each
    # of these — it needs a variable declared to receive the parent module's
    # output, passed in via the root module's `module "..." { ... }` call.
    for wire in sorted(child_wires, key=lambda w: w.variable_name):
        var_lines += [
            f'variable "{wire.variable_name}" {{',
            f'  description = "Cross-module reference: {wire.parent_resource_type}.{wire.parent_logical_name}.{wire.parent_attr} '
            f'from the \'{wire.parent_module}\' module (wired via containment in the source diagram)"',
            '  type        = string',
            "}",
            "",
        ]
    # Required-field variables (see _variableize_mandatory_fields): every
    # value this system asked the user about for a resource in this module,
    # now overridable via terraform.tfvars / -var instead of hand-editing
    # main.tf. `default` is set to whatever value was already resolved
    # (clarification answer or catalog default) so `terraform plan` with no
    # tfvars at all still behaves exactly as it did before this change.
    for var_name, meta in sorted(mandatory_vars.items()):
        labels = ", ".join(meta["labels"])
        var_lines += [
            f'variable "{var_name}" {{',
            f'  description = "{meta["field_key"]} for {labels} — override via terraform.tfvars or -var"',
            f'  type        = {meta["type"]}',
            f'  default     = {_hcl_default_literal(meta["value"], meta["type"])}',
            "}",
            "",
        ]
    files["variables.tf"] = "\n".join(var_lines)

    # ── outputs.tf ───────────────────────────────────────────────────────────
    emitted_output_names: set[str] = set()
    out_lines = [f"# Module: {module_name} — outputs", ""]
    for resource in resources:
        for attr in _get_output_attrs(resource.aws_resource_type):
            out_name = f"{resource.logical_name}_{attr}"
            emitted_output_names.add(out_name)
            ref = f"{resource.aws_resource_type}.{resource.logical_name}.{attr}"
            out_lines += [
                f'output "{out_name}" {{',
                f'  description = "{attr} of {resource.label}"',
                f'  value       = {ref}',
                "}",
                "",
            ]

    # Cross-module containment wiring: this module is the PARENT side for
    # each of these — it needs an `.id` output so the child module's
    # variable (above) has something to actually receive. Usually already
    # covered by the loop above (aws_vpc/aws_subnet both have "id" in
    # _get_output_attrs), but this is a safety net for any future wiring
    # rule whose parent type isn't already in that map — better to emit a
    # duplicate-safe extra output than to generate a root-module argument
    # that references one that doesn't exist.
    for wire in sorted(parent_wires, key=lambda w: w.output_name):
        if wire.output_name in emitted_output_names:
            continue
        emitted_output_names.add(wire.output_name)
        ref = f"{wire.parent_resource_type}.{wire.parent_logical_name}.{wire.parent_attr}"
        out_lines += [
            f'output "{wire.output_name}" {{',
            f'  description = "{wire.parent_attr} of {wire.parent_logical_name} (needed for cross-module wiring)"',
            f'  value       = {ref}',
            "}",
            "",
        ]
    files["outputs.tf"] = "\n".join(out_lines)

    # ── versions.tf ──────────────────────────────────────────────────────────
    files["versions.tf"] = _versions_tf(
        needs_random_provider=any(r.companion_blocks for r in resources)
    )

    return files


def _resource_block_lines(
    resource: ParsedResource,
    conn_attrs: dict,
    depends_on: list[str],
    wiring_warning: str | None,
) -> list[str]:
    """
    Builds one resource block's lines by delegating flat-attribute and
    nested-block rendering to arch2terraform's hcl_format.resource_block() —
    this is what actually carries the real-`terraform validate`-audited
    catalog correctness (required arguments, ARN-shaped placeholders, nested
    HCL blocks) into Phase 2's output, instead of the old locally
    reimplemented `_format_attr`.

    Splices in the two things arch2terraform itself doesn't emit: a merged
    tags block and depends_on. Both are Phase 2-only concerns — arch2terraform
    has no tagging convention of its own, and no cross-resource depends_on
    since it only ever emits a single file where HCL's own implicit
    reference-based dependency graph is sufficient.
    """
    all_props = {**resource.properties, **conn_attrs}
    attrs = {k: v for k, v in all_props.items() if not k.startswith("_")}

    comment = f"From diagram node: {resource.label}"
    if resource.confidence < 0.95:
        comment += f" (low-confidence match, confidence={resource.confidence:.2f} — please review)"
    if wiring_warning:
        comment += f" — NOTE: {wiring_warning}"

    block_text = _a2tf_resource_block(
        resource.aws_resource_type, resource.logical_name, attrs, comment,
        nested_blocks=resource.nested_blocks,
    )

    lines = block_text.split("\n")
    assert lines[-1] == "}", f"unexpected resource_block() output shape: {lines[-1]!r}"
    body_lines = lines[:-1]

    # Tags. Environment/Project/Region are included directly here (not just
    # folded into root's local.common_tags) so that var.environment/
    # var.project/var.aws_region are each textually referenced inside every
    # module that has at least one real resource — before this, each
    # module's variables.tf declared these three unconditionally (root
    # passes them into every module the same way) but nothing in the
    # module's own files ever used them, which is exactly what tflint's
    # terraform_unused_declarations rule flagged in every single generated
    # module. This is a real tagging improvement in its own right (standard
    # AWS cost-allocation practice), not just linter appeasement — it just
    # also happens to resolve the warning as a side effect wherever a
    # module has real resources to tag.
    body_lines += [
        "",
        "  tags = merge(var.common_tags, {",
        f'    Name        = "{resource.label}"',
        f'    Module      = "{resource.aws_resource_type}"',
        "    Environment = var.environment",
        "    Project     = var.project",
        "    Region      = var.aws_region",
        "  })",
    ]

    # depends_on
    if depends_on:
        unique_deps = list(dict.fromkeys(depends_on))  # preserve order, dedupe
        body_lines.append("")
        body_lines.append("  depends_on = [")
        for dep in unique_deps:
            body_lines.append(f"    {dep},")
        body_lines.append("  ]")

    body_lines.append("}")
    return body_lines


_STANDARD_MODULE_VARS = ("aws_region", "environment", "project")


def _prune_unused_standard_variables(tf_files: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """
    Strip aws_region/environment/project declarations from a module's
    variables.tf when nothing in that module's own files references them.

    _resource_block_lines' tags block makes these three genuinely "used" in
    the common case, but a module can exist with zero regular
    ParsedResources — e.g. a "networking" module forced into being purely to
    hold the security engine's generated_security_groups.tf when a diagram
    has no aws_vpc node at all — where the tags loop never runs. Root main.tf
    passes all three into every module unconditionally (simplest, most
    consistent code path), so without this, those modules would declare all
    three but reference none of them: exactly what tflint's
    terraform_unused_declarations rule flagged in real testing.

    common_tags is deliberately never pruned — every resource's
    `tags = merge(var.common_tags, ...)` always references it.

    Returns the (possibly modified) file dict and the set of standard vars
    that are STILL declared after pruning — the caller must pass this
    through to _generate_root_module() so its module-call argument list
    stays symmetric with what's actually declared. Passing
    `aws_region = var.aws_region` into a module that no longer declares
    `aws_region` would trade "declared but not used" (a warning) for
    "Unsupported argument" (a real terraform init/validate error) — the
    exact class of bug already found once this session (see
    security_bridge.py's vpc_id fallback fix).
    """
    variables_tf = tf_files.get("variables.tf", "")
    if not variables_tf:
        return tf_files, set()

    other_content = "\n".join(c for fname, c in tf_files.items() if fname != "variables.tf")

    kept: set[str] = set()
    for var_name in _STANDARD_MODULE_VARS:
        if f"var.{var_name}" in other_content:
            kept.add(var_name)
            continue
        pattern = re.compile(rf'variable "{var_name}" \{{[^}}]*\}}\n\n?')
        variables_tf, n = pattern.subn("", variables_tf, count=1)
        if n == 0:
            # Declaration wasn't found in the expected shape — keep it
            # "declared" defensively rather than silently create a mismatch
            # between variables.tf and what root's module call supplies.
            kept.add(var_name)

    tf_files["variables.tf"] = variables_tf
    return tf_files, kept


def _wire_containment_attrs(
    module_resources: dict[str, list[ParsedResource]],
    parsed: ParsedDiagram,
) -> tuple[dict[str, str], set[_CrossModuleWire]]:
    """
    Mutates each ParsedResource's `properties` in place: for containment
    connections that match a wireable rule (mirrors arch2terraform's
    hcl_generator._wire_containment), sets the attribute to a real Terraform
    reference — `aws_vpc.x.id` when parent and child land in the *same*
    Phase 2 module (resolves directly within one file), or
    `var.<parent_module>_<parent_name>_id` when they land in *different*
    modules.

    Why the module split matters: arch2terraform always emits one flat
    main.tf, so `subnet_id = aws_subnet.x.id` always resolves. Phase 2 splits
    resources across multiple Terraform modules (networking/compute/...), and
    a bare resource reference like that is only valid within the same
    module/file — across modules it would be a reference to an undeclared
    resource, which fails `terraform validate`. The cross-module case is made
    real (not just flagged) by threading a `var.*` reference through here,
    an `output` in the parent's module (see `_generate_module_hcl`'s
    `parent_wires` handling), a `variable` in the child's module (same
    function's `child_wires` handling), and the module-call argument
    connecting them in the root module (`_generate_root_module`).

    Returns (warnings, cross_module_wires): `warnings` is
    {resource_id: comment} for the cross-module cases, surfaced in that
    resource's generated comment so the wiring is visible right where it's
    used even though it's a `var.*` reference rather than a literal one;
    `cross_module_wires` is the deduplicated set of wiring specs the caller
    needs to actually generate the output/variable/module-argument plumbing
    for.
    """
    resource_module: dict[str, str] = {
        r.id: mod for mod, rs in module_resources.items() for r in rs
    }
    by_id: dict[str, ParsedResource] = {
        r.id: r for rs in module_resources.values() for r in rs
    }

    warnings: dict[str, str] = {}
    cross_module_wires: set[_CrossModuleWire] = set()

    for conn in parsed.connections:
        if conn.connection_type != "containment":
            continue
        parent = by_id.get(conn.source_id)
        child = by_id.get(conn.target_id)
        if not parent or not child:
            continue

        rule = _CONTAINMENT_WIRING_RULES.get(child.aws_resource_type)
        if not rule:
            continue
        attr_name, required_parent_type = rule
        if parent.aws_resource_type != required_parent_type:
            continue  # diagram nesting doesn't match what this attribute expects

        parent_module = resource_module.get(parent.id)
        child_module = resource_module.get(child.id)

        if parent_module == child_module:
            child.properties[attr_name] = f"{parent.aws_resource_type}.{parent.logical_name}.id"
        else:
            wire = _CrossModuleWire(
                parent_module=parent_module,
                parent_resource_type=parent.aws_resource_type,
                parent_logical_name=parent.logical_name,
                child_module=child_module,
            )
            cross_module_wires.add(wire)
            child.properties[attr_name] = f"var.{wire.variable_name}"
            warnings[child.id] = (
                f"{attr_name} is wired from '{parent.label}' ({parent.aws_resource_type}) "
                f"in the '{parent_module}' module via var.{wire.variable_name} — see the "
                f"'{child_module}' module's variables.tf and the root module's "
                f'module "{child_module}" block for the full cross-module wiring'
            )

    return warnings, cross_module_wires


def _wire_role_attachments(
    module_resources: dict[str, list[ParsedResource]],
    attachment_specs: list[dict],
    cross_module_wires: set[_CrossModuleWire],
) -> None:
    """
    Mutates each attached resource's `properties` in place (same pattern as
    _wire_containment_attrs): sets the real generated IAM role/instance
    profile onto the compute resource that actually needs it -
    `iam_instance_profile` for an EC2 instance, `role` for a Lambda function,
    etc. (see security_bridge.ATTACHMENT_ATTR_BY_RESOURCE_TYPE) - and,
    since the security engine's IAM output always lands in the 'security'
    module while the compute resource lives elsewhere (compute/serverless/
    containers), adds a `_CrossModuleWire` to `cross_module_wires` (mutated
    in place) for the same output/variable/module-argument plumbing
    containment wiring already uses, generalized via `_CrossModuleWire`'s
    `parent_attr` field (here "name" for an instance profile, "arn" for a
    role) rather than the "id" every containment wire uses.

    This exists specifically so the security engine's own attachment
    templates (which render a second, incomplete standalone resource block)
    are never used in this integration — see security_bridge.py's
    run_security_engine docstring for why that would be a real
    terraform-validate-breaking bug, not just a style concern.
    """
    by_id: dict[str, ParsedResource] = {
        r.id: r for rs in module_resources.values() for r in rs
    }
    resource_module: dict[str, str] = {
        r.id: mod for mod, rs in module_resources.items() for r in rs
    }

    for spec in attachment_specs:
        child = by_id.get(spec["resource_id"])
        if not child:
            continue

        rule = ATTACHMENT_ATTR_BY_RESOURCE_TYPE.get(child.aws_resource_type)
        if not rule:
            continue
        child_attr, parent_resource_type, parent_attr, name_suffix = rule

        parent_module = "security"
        parent_logical_name = f"{spec['role_tf_id']}{name_suffix}"
        child_module = resource_module.get(child.id)

        if parent_module == child_module:
            child.properties[child_attr] = f"{parent_resource_type}.{parent_logical_name}.{parent_attr}"
        else:
            wire = _CrossModuleWire(
                parent_module=parent_module,
                parent_resource_type=parent_resource_type,
                parent_logical_name=parent_logical_name,
                child_module=child_module,
                parent_attr=parent_attr,
            )
            cross_module_wires.add(wire)
            child.properties[child_attr] = f"var.{wire.variable_name}"


def _wire_sg_attachments(
    module_resources: dict[str, list[ParsedResource]],
    sg_attachment_specs: list[dict],
    cross_module_wires: set[_CrossModuleWire],
) -> None:
    """
    Mutates each attached resource's `properties` in place (same pattern as
    _wire_role_attachments just above): sets the generated security group's
    id onto the real resource's security-group attribute
    (vpc_security_group_ids / security_groups / security_group_ids — see
    security_bridge.SG_ATTACHMENT_ATTR_BY_RESOURCE_TYPE).

    Added 2026-07-24: checkov's CKV2_AWS_5 ("Security Groups are attached to
    another resource") was failing on every generated SG for a real reason —
    the security engine created it, but nothing in the compute/database
    resource's own block ever referenced its id. Wraps the reference in a
    one-element list since all three of those attributes are list-typed,
    unlike the IAM attachment attributes _wire_role_attachments handles
    (iam_instance_profile etc. are single scalar references).
    """
    by_id: dict[str, ParsedResource] = {
        r.id: r for rs in module_resources.values() for r in rs
    }
    resource_module: dict[str, str] = {
        r.id: mod for mod, rs in module_resources.items() for r in rs
    }

    for spec in sg_attachment_specs:
        child = by_id.get(spec["resource_id"])
        if not child:
            continue

        sg_attr = spec["sg_attr"]
        sg_terraform_name = spec["sg_terraform_name"]
        parent_module = "networking"  # generated security groups always land here
        child_module = resource_module.get(child.id)

        if parent_module == child_module:
            child.properties[sg_attr] = [f"aws_security_group.{sg_terraform_name}.id"]
        else:
            wire = _CrossModuleWire(
                parent_module=parent_module,
                parent_resource_type="aws_security_group",
                parent_logical_name=sg_terraform_name,
                child_module=child_module,
                parent_attr="id",
            )
            cross_module_wires.add(wire)
            child.properties[sg_attr] = [f"var.{wire.variable_name}"]


# ── Required-field variable-ization ──────────────────────────────────────────
# Added 2026-07-08 per her explicit request, after a real `terraform apply`
# against a generated package failed twice: an EC2 instance's AMI had a stray
# trailing space baked in from a clarification answer (InvalidAMIID.Malformed
# — separately fixed in missing_info_detector.apply_clarification_answers via
# .strip()), and a since-fixed gap where aws_db_instance had no username at
# all. Both were symptoms of the same underlying design issue: every field
# MANDATORY_FIELDS asks the user about gets baked into the resource block as
# a literal string, so there's no way to change one afterwards without
# hand-editing generated .tf files. This turns every one of those fields into
# a real Terraform `variable` instead, so they're overridable via a plain
# terraform.tfvars / -var without ever touching generated code again.

def _hcl_type_for(value) -> str:
    """Maps a Python value's type to the Terraform `variable` block's `type`
    argument. Order matters: bool must be checked before (int, float) since
    `isinstance(True, int)` is True in Python."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _hcl_default_literal(value, hcl_type: str) -> str:
    """Formats `value` as the RHS of a variable's `default = ...` line,
    matching the same conventions arch2terraform's hcl_format.hcl_value()
    uses for resource attributes (bare true/false, bare numbers, quoted +
    escaped strings) — kept as a small local helper rather than importing
    hcl_value() since variable defaults are a Phase 2-only concern (see this
    module's docstring) and don't need nested-block/list handling."""
    if hcl_type == "bool":
        return "true" if value else "false"
    if hcl_type == "number":
        return str(value)
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


def _variableize_mandatory_fields(
    module_resources: dict[str, list[ParsedResource]],
) -> dict[str, dict[str, dict]]:
    """
    Mutates each ParsedResource's `properties` in place: every field
    MANDATORY_FIELDS lists for that resource type (the same source of truth
    the clarification UI already asks about — ami, instance_type, engine,
    instance_class, allocated_storage, multi_az, cidr_block, etc.) gets
    pulled OUT of the resource block as a literal and replaced with a
    `var.<name>` reference — rendered unquoted by arch2terraform's
    hcl_value() the same way cross-module wiring's `var.*` references
    already are (see _wire_containment_attrs above), so no changes were
    needed in arch2terraform itself.

    Naming (flat/shared, her explicit choice over per-resource or
    per-module): one variable per field_key, e.g. `var.ami`, so overriding
    "the AMI" means setting one obvious `ami = "..."` in tfvars. The real
    tradeoff of that choice is a genuine naming collision, not just a style
    question — two resources of the SAME type in the SAME module (two EC2
    instances, or a VPC + subnet both needing `cidr_block`) can each need a
    DIFFERENT value for the same field_key, and a Terraform variable can
    only be declared once per module. Handled here by tracking which value
    each variable name has already been assigned within a module: a second
    resource wanting a different value for a name already taken gets a
    disambiguated name (`<field_key>_<logical_name>`, with a numeric suffix
    in the rare case even THAT collides) instead of silently overwriting
    the first resource's default. Two resources that happen to want the
    SAME value share one variable, which is exactly the point of "flat".

    Returns {module_name: {var_name: {"value", "type", "field_key", "labels"}}}
    for _generate_module_hcl to render as `variable` blocks.
    """
    result: dict[str, dict[str, dict]] = {}

    for module_name, resources in module_resources.items():
        assigned_values: dict[str, object] = {}
        var_defs: dict[str, dict] = {}

        for resource in resources:
            # Union of the static MANDATORY_FIELDS list (the original ~12
            # hand-covered resource types) and whatever missing_info_detector
            # actually flagged for THIS resource (resource.variableize_keys —
            # covers both those same ~12 types AND the ~25 more caught only
            # by its generic placeholder fallback, e.g. aws_ecr_repository's
            # "name"). Falls back to MANDATORY_FIELDS alone when
            # variableize_keys is empty (e.g. synthetic ParsedResource
            # objects built directly in tests, which never went through
            # detect_missing_info) — preserves the pre-existing behavior for
            # every caller that doesn't populate it.
            static_keys = {fkey for (fkey, *_rest) in MANDATORY_FIELDS.get(resource.aws_resource_type, [])}
            field_keys = sorted(static_keys | set(resource.variableize_keys))
            for field_key in field_keys:
                if field_key not in resource.properties:
                    continue
                value = resource.properties[field_key]
                # Already a raw reference (e.g. containment wiring claimed
                # this same attribute name) — MANDATORY_FIELDS and
                # _CONTAINMENT_WIRING_RULES cover disjoint attribute names
                # in practice, so this shouldn't fire, but skip defensively
                # rather than double-wrapping a var.* string in another var.*.
                if isinstance(value, str) and value.startswith("var."):
                    continue

                var_name = field_key
                if var_name in assigned_values and assigned_values[var_name] != value:
                    base_name = f"{field_key}_{resource.logical_name}"
                    var_name = base_name
                    suffix = 2
                    while var_name in assigned_values and assigned_values[var_name] != value:
                        var_name = f"{base_name}_{suffix}"
                        suffix += 1

                assigned_values[var_name] = value
                if var_name not in var_defs:
                    var_defs[var_name] = {
                        "value": value,
                        "type": _hcl_type_for(value),
                        "field_key": field_key,
                        "labels": [],
                    }
                label = resource.label or resource.logical_name
                if label not in var_defs[var_name]["labels"]:
                    var_defs[var_name]["labels"].append(label)

                resource.properties[field_key] = f"var.{var_name}"

        result[module_name] = var_defs

    return result


def _get_output_attrs(resource_type: str) -> list[str]:
    OUTPUT_MAP = {
        "aws_vpc": ["id", "cidr_block"],
        "aws_subnet": ["id"],
        "aws_internet_gateway": ["id"],
        "aws_nat_gateway": ["id", "public_ip"],
        "aws_security_group": ["id"],
        "aws_instance": ["id", "private_ip", "public_ip"],
        "aws_autoscaling_group": ["id", "arn"],
        "aws_lb": ["arn", "dns_name"],
        "aws_lb_target_group": ["arn"],
        "aws_ecs_cluster": ["id", "arn"],
        "aws_eks_cluster": ["id", "endpoint"],
        "aws_lambda_function": ["arn", "invoke_arn"],
        "aws_api_gateway_rest_api": ["id", "execution_arn"],
        "aws_db_instance": ["id", "endpoint", "address"],
        "aws_rds_cluster": ["id", "endpoint", "reader_endpoint"],
        "aws_dynamodb_table": ["id", "arn"],
        "aws_elasticache_cluster": ["id"],
        "aws_s3_bucket": ["id", "arn", "bucket_regional_domain_name"],
        "aws_efs_file_system": ["id", "arn", "dns_name"],
        "aws_sqs_queue": ["id", "arn", "url"],
        "aws_sns_topic": ["id", "arn"],
        "aws_kms_key": ["id", "arn"],
        "aws_secretsmanager_secret": ["id", "arn"],
        "aws_cloudfront_distribution": ["id", "domain_name"],
        "aws_route53_zone": ["id", "name_servers"],
        "aws_ecr_repository": ["id", "repository_url"],
        "aws_iam_role": ["id", "arn"],
        "aws_acm_certificate": ["id", "arn"],
    }
    return OUTPUT_MAP.get(resource_type, [])


def _generate_root_module(
    modules: list[TerraformModule],
    aws_region: str,
    environment: str,
    project_name: str,
    wires_by_child: dict[str, set[_CrossModuleWire]],
    module_std_vars: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    files: dict[str, str] = {}

    # ── versions.tf ──────────────────────────────────────────────────────────
    files["versions.tf"] = _versions_tf()

    # ── ci_state_access_policy.tf ───────────────────────────────────────────
    # Only emitted once a real state backend is configured (see _backend_tf) —
    # nothing to scope a CI policy to otherwise.
    ci_policy_tf = _ci_state_access_policy_tf(project_name, environment)
    if ci_policy_tf:
        files["ci_state_access_policy.tf"] = ci_policy_tf

    # ── locals.tf ────────────────────────────────────────────────────────────
    files["locals.tf"] = "\n".join([
        "locals {",
        "  common_tags = {",
        '    Project     = var.project',
        '    Environment = var.environment',
        '    ManagedBy   = "Terraform"',
        '    GeneratedBy = "arch2terraform"',
        "  }",
        "}",
        "",
    ])

    # ── variables.tf ─────────────────────────────────────────────────────────
    files["variables.tf"] = "\n".join([
        'variable "aws_region" {',
        '  description = "AWS region"',
        '  type        = string',
        f'  default     = "{aws_region}"',
        "}",
        "",
        'variable "environment" {',
        '  description = "Environment: dev, staging, prod"',
        '  type        = string',
        f'  default     = "{environment}"',
        "}",
        "",
        'variable "project" {',
        '  description = "Project name"',
        '  type        = string',
        f'  default     = "{project_name}"',
        "}",
        "",
    ])

    # ── main.tf — root module calling all child modules ──────────────────────
    main_lines = [
        "# Root module — generated by arch2terraform",
        "# This file wires all child modules together.",
        "",
        'provider "aws" {',
        '  region = var.aws_region',
        "}",
        "",
    ]

    # Module call blocks — each module gets outputs from its dependencies
    mod_order = ["networking", "security", "storage", "database",
                 "messaging", "serverless", "compute", "containers",
                 "cdn", "observability", "cicd", "misc"]

    ordered_modules = [
        m for name in mod_order for m in modules if m.name == name
    ] + [m for m in modules if m.name not in mod_order]

    std_vars_by_module = module_std_vars or {}
    for mod in ordered_modules:
        main_lines += [
            f'module "{mod.name}" {{',
            f'  source = "./modules/{mod.name}"',
            "",
        ]
        # Only pass the standard vars this module actually still declares
        # (see _prune_unused_standard_variables) — kept set defaults to "all
        # three" when a module wasn't run through pruning (e.g. tests
        # calling this function directly), preserving prior behavior.
        kept = std_vars_by_module.get(mod.name, set(_STANDARD_MODULE_VARS))
        if "aws_region" in kept:
            main_lines.append('  aws_region  = var.aws_region')
        if "environment" in kept:
            main_lines.append('  environment = var.environment')
        if "project" in kept:
            main_lines.append('  project     = var.project')
        main_lines.append('  common_tags = local.common_tags')

        # Cross-module containment wiring: pass each parent module's real
        # output into this module's corresponding variable (see
        # _CrossModuleWire, _wire_containment_attrs, and _generate_module_hcl's
        # child_wires/parent_wires handling for the variable/output halves of
        # this plumbing). Sorted for deterministic, diffable output.
        child_wires = wires_by_child.get(mod.name, set())
        if child_wires:
            main_lines.append("")
            main_lines.append("  # Cross-module containment wiring")
            for wire in sorted(child_wires, key=lambda w: w.variable_name):
                main_lines.append(
                    f"  {wire.variable_name} = module.{wire.parent_module}.{wire.output_name}"
                )

        main_lines += ["}", ""]

    files["main.tf"] = "\n".join(main_lines)

    # ── outputs.tf ───────────────────────────────────────────────────────────
    out_lines = ["# Root outputs — expose key values from all modules", ""]
    for mod in ordered_modules:
        out_lines += [
            f'# Outputs from module "{mod.name}"',
            f'# output "{mod.name}_outputs" {{',
            f'#   value = module.{mod.name}',
            "#}",
            "",
        ]
    files["outputs.tf"] = "\n".join(out_lines)

    # ── backend.tf ───────────────────────────────────────────────────────────
    files["backend.tf"] = _backend_tf(project_name, environment)

    return files


import re as _re3

_WORKSPACE_INVALID_CHARS_RE = _re3.compile(r"[^A-Za-z0-9_-]+")
_WORKSPACE_COLLAPSE_HYPHENS_RE = _re3.compile(r"-{2,}")


def _tfc_workspace_name(project_name: str, environment: str) -> str:
    """Terraform Cloud workspace names only allow letters, numbers, `-`,
    `_` — sanitize project_name/environment (diagram-derived, arbitrary
    text) the same way the S3 backend's key path doesn't need to (S3 keys
    tolerate slashes/spaces fine), so this needs its own helper. Runs of
    invalid characters collapse to one `-`; a SECOND pass then also
    collapses runs of hyphens themselves (e.g. "My Cool Project!-dev" would
    otherwise land on "My-Cool-Project--dev" — the "!" sanitizes to its own
    "-" immediately next to the literal "-" already in the f-string
    separator, and those two aren't a contiguous invalid-char run the first
    regex would ever see together)."""
    raw = f"{project_name}-{environment}"
    cleaned = _WORKSPACE_INVALID_CHARS_RE.sub("-", raw)
    cleaned = _WORKSPACE_COLLAPSE_HYPHENS_RE.sub("-", cleaned).strip("-")
    return cleaned or "arch2tf-workspace"


def _backend_tf(project_name: str, environment: str) -> str:
    """Dispatches to whichever remote-state backend is configured
    (settings.tf_backend_type) — "s3" (default, pre-existing behavior) or
    "cloud" (Terraform Cloud / HCP Terraform, added 2026-07-29). See each
    branch's own docstring for the real-vs-placeholder gating logic, which
    is identical in shape for both: a real block only once the one-time
    infra decision (bucket, or org) is actually configured, otherwise a
    commented-out template — this tool never invents which state store to
    use."""
    _s = get_settings()
    backend_type = (_s.tf_backend_type or "s3").strip().lower()
    if backend_type == "cloud":
        return _cloud_backend_tf(project_name, environment)
    return _s3_backend_tf(project_name, environment)


def _cloud_backend_tf(project_name: str, environment: str) -> str:
    """
    Terraform Cloud / HCP Terraform backend — real `cloud { }` block, or the
    commented-out placeholder, mirroring _s3_backend_tf's gating exactly
    (real block only once TF_CLOUD_ORGANIZATION is actually set). One
    workspace per project+environment (see _tfc_workspace_name), same
    dev/staging/prod isolation the S3 backend's one-key-per-environment
    convention already gives.

    Deliberately does NOT configure TFC's own remote/managed run mode —
    this stays a CLI-driven workflow (terraform plan/apply run locally,
    same as everything else in apply_runner.py), with TFC used purely as
    the state store + lock. That means the TFC workspace itself must have
    its execution mode set to "Local" (one-time setup in the TFC UI), or
    `terraform apply` from apply_runner.py will be rejected by TFC in favor
    of a remote run it never triggers.
    """
    _s = get_settings()
    workspace = _tfc_workspace_name(project_name, environment)
    if not _s.tf_cloud_organization:
        return "\n".join([
            "# Terraform Cloud / HCP Terraform state (uncomment and configure for team use,",
            "# or set TF_BACKEND_TYPE=cloud and TF_CLOUD_ORGANIZATION in the backend's .env",
            "# to have this generated for you automatically on every future job).",
            "#",
            "# Auth: run `terraform login` once on whichever machine runs plan/apply (or",
            "# set TF_TOKEN_<hostname_with_dots_as_underscores>) — never read, stored, or",
            "# transmitted by this backend itself, same convention as AWS credentials.",
            "# The workspace's execution mode must be \"Local\" in Terraform Cloud for this",
            "# CLI-driven plan/apply flow to work (Terraform Cloud is used purely for state",
            "# storage + locking here, not managed/remote runs).",
            "#",
            "# terraform {",
            "#   cloud {",
            '#     organization = "your-tfc-organization"',
            "#     workspaces {",
            f'#       name = "{workspace}"',
            "#     }",
            "#   }",
            "# }",
            "",
        ])

    hostname_line = []
    if _s.tf_cloud_hostname and _s.tf_cloud_hostname != "app.terraform.io":
        hostname_line = [f'    hostname     = "{_s.tf_cloud_hostname}"']

    return "\n".join([
        "# Remote state — real Terraform Cloud / HCP Terraform backend, generated",
        "# from this deployment's TF_CLOUD_ORGANIZATION setting. One workspace per",
        "# project+environment, so dev/staging/prod are always isolated from each",
        "# other, same convention as the S3 backend's one-state-file-per-environment",
        "# key. Auth comes from `terraform login` / TF_TOKEN_* already active on",
        "# whichever machine runs plan/apply (see apply_runner.py's module docstring —",
        "# same 'never read/store/transmit credentials ourselves' model as AWS).",
        "# `terraform init` must be run against THIS SAME backend on every plan/apply",
        "# for this environment for drift detection to mean anything real.",
        "terraform {",
        "  cloud {",
        f'    organization = "{_s.tf_cloud_organization}"',
        *hostname_line,
        "    workspaces {",
        f'      name = "{workspace}"',
        "    }",
        "  }",
        "}",
        "",
    ])


def _s3_backend_tf(project_name: str, environment: str) -> str:
    """
    Real remote state, or the pre-existing commented-out placeholder — never
    both, and never a half-real block. Terraform `backend` blocks are
    resolved BEFORE any variable is available (init happens before the rest
    of the config is even parsed), so `key`/`bucket`/`region` can only ever
    be literal strings here, not `var.project`/`var.environment` — the old
    placeholder's `key = f"{var.project}/{var.environment}/terraform.tfstate"`
    line was Python f-string syntax that slipped into a comment (harmless
    while commented out, but would have been invalid HCL — and invalid
    Terraform, since backend blocks can't reference variables at all — the
    moment anyone uncommented it as-is). project_name/environment ARE known
    here at generation time, so they're baked in directly instead.

    One state file per environment (`key = "{project}/{environment}/terraform.tfstate"`)
    so dev/staging/prod each get their own state and can never clobber one
    another — same convention already used for the GitHub push paths
    (terraform/<environment>/, diagrams/<environment>/).
    """
    _s = get_settings()
    if not _s.tf_state_bucket:
        # No state bucket configured — keep the previous behavior exactly:
        # a commented-out template the user fills in themselves, since
        # WHICH bucket holds state is a one-time infra decision this tool
        # can't safely assume, not something to silently invent.
        return "\n".join([
            "# Remote state configuration (uncomment and configure for team use,",
            "# or set TF_STATE_BUCKET in the backend's .env to have this generated",
            "# for you automatically on every future job)",
            '#',
            '# terraform {',
            '#   backend "s3" {',
            '#     bucket         = "your-terraform-state-bucket"',
            f'#     key            = "{project_name}/{environment}/terraform.tfstate"',
            '#     region         = "us-east-1"',
            '#     encrypt        = true',
            '#     dynamodb_table = "terraform-state-lock"',
            '#   }',
            '# }',
            "",
        ])

    state_region = _s.tf_state_region or _s.aws_region
    return "\n".join([
        "# Remote state — real backend, generated from this deployment's",
        "# TF_STATE_BUCKET/TF_STATE_LOCK_TABLE settings. One state file per",
        "# environment, so dev/staging/prod are always isolated from each",
        "# other. `terraform init` must be run against THIS SAME backend on",
        "# every plan/apply for this environment (e.g. in CI) for drift",
        "# detection to mean anything real — a fresh/ephemeral backend on",
        "# every run has no memory of what's actually been applied before.",
        "terraform {",
        '  backend "s3" {',
        f'    bucket         = "{_s.tf_state_bucket}"',
        f'    key            = "{project_name}/{environment}/terraform.tfstate"',
        f'    region         = "{state_region}"',
        "    encrypt        = true",
        f'    dynamodb_table = "{_s.tf_state_lock_table}"',
        "  }",
        "}",
        "",
    ])


def _ci_state_access_policy_tf(project_name: str, environment: str) -> str | None:
    """
    Least-privilege IAM policy for whatever identity runs CI/CD against this
    project's remote state (S3 bucket + DynamoDB lock table) — NOT the
    policy the deployed infrastructure itself uses, and NOT auto-attached to
    anything: this tool has no way to know which IAM role/user actually runs
    the pipeline, so it generates a real `aws_iam_policy` resource the user
    attaches themselves (console, `aws_iam_role_policy_attachment`, etc.).

    Returns None when no state bucket is configured (same gate as
    _backend_tf's real-backend branch — a scoped access policy only makes
    sense once there's a real backend to scope it to; a bucket-less setup
    has no state-access surface to narrow in the first place).

    Scope, deliberately narrow:
      - s3:ListBucket on the bucket itself, but only for keys under this
        project's own prefix (a StringLike condition on s3:prefix) — so this
        identity can't even list, let alone read, other projects' state in
        the same shared bucket.
      - s3:GetObject/PutObject/DeleteObject scoped to
        "{bucket}/{project_name}/*" — covers every environment for this
        project (dev/staging/prod each get their own key under that prefix,
        see _backend_tf), nothing else in the bucket.
      - dynamodb:GetItem/PutItem/DeleteItem scoped to the one lock table
        (state locking needs all three: acquire, refresh, release).
    """
    _s = get_settings()
    backend_type = (_s.tf_backend_type or "s3").strip().lower()
    if backend_type != "s3" or not _s.tf_state_bucket:
        # S3-specific: Terraform Cloud governs state access via TFC teams,
        # not an AWS IAM policy — nothing to scope when "cloud" is active,
        # same as the pre-existing no-bucket-configured gate.
        return None

    state_region = _s.tf_state_region or _s.aws_region
    policy_name = f"{project_name}-{environment}-terraform-state-access"

    return "\n".join([
        "# Least-privilege IAM policy for CI/CD access to this project's",
        "# remote Terraform state. This resource is NOT attached to anything",
        "# automatically — attach it to whichever IAM role/user your CI",
        "# pipeline actually runs as (e.g. via aws_iam_role_policy_attachment,",
        "# or through your identity provider's own OIDC-role setup).",
        "",
        'data "aws_caller_identity" "ci_state_access" {}',
        "",
        f'resource "aws_iam_policy" "terraform_state_access" {{',
        f'  name        = "{policy_name}"',
        f'  description = "Least-privilege access to {project_name}/{environment} Terraform state ({_s.tf_state_bucket} + {_s.tf_state_lock_table})"',
        "",
        "  policy = jsonencode({",
        '    Version = "2012-10-17"',
        "    Statement = [",
        "      {",
        '        Sid    = "TerraformStateBucketList"',
        '        Effect = "Allow"',
        '        Action = ["s3:ListBucket"]',
        f'        Resource = "arn:aws:s3:::{_s.tf_state_bucket}"',
        "        Condition = {",
        '          StringLike = {',
        f'            "s3:prefix" = ["{project_name}/*"]',
        "          }",
        "        }",
        "      },",
        "      {",
        '        Sid    = "TerraformStateObjectAccess"',
        '        Effect = "Allow"',
        '        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]',
        f'        Resource = "arn:aws:s3:::{_s.tf_state_bucket}/{project_name}/*"',
        "      },",
        "      {",
        '        Sid    = "TerraformStateLock"',
        '        Effect = "Allow"',
        '        Action = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]',
        f'        Resource = "arn:aws:dynamodb:{state_region}:${{data.aws_caller_identity.ci_state_access.account_id}}:table/{_s.tf_state_lock_table}"',
        "      }",
        "    ]",
        "  })",
        "}",
        "",
    ])


def _versions_tf(needs_random_provider: bool = False) -> str:
    lines = [
        "terraform {",
        '  required_version = ">= 1.5.0"',
        "",
        "  required_providers {",
        "    aws = {",
        '      source  = "hashicorp/aws"',
        '      version = "~> 5.0"',
        "    }",
    ]
    if needs_random_provider:
        # A resource in this module has companion_blocks referencing
        # random_password.* (currently only aws_mq_broker — see
        # arch2terraform's classifier.py's _build_mq_broker_companion_blocks()).
        lines += [
            "    random = {",
            '      source  = "hashicorp/random"',
            '      version = "~> 3.6"',
            "    }",
        ]
    lines += [
        "  }",
        "}",
        "",
    ]
    return "\n".join(lines)
