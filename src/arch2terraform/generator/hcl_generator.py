"""
HCL generator: walks a ResourceGraph and produces five Terraform files,
matching the original Phase 1 scope:

  provider.tf    - terraform/provider block
  variables.tf   - input variables (region, environment tag, etc.)
  main.tf        - all resource blocks, ordered so containers come before
                   the resources nested in them (readability, not a
                   strict requirement since Terraform resolves its own
                   dependency graph regardless of file order)
  outputs.tf      - one output per resource that plausibly has a useful
                   attribute to expose (id, arn, endpoint, etc.)
  README.md      - human-readable summary: what was generated, any
                   unclassified nodes/low-confidence guesses to review

main.tf attaches containment via Terraform-native referencing where the
relationship implies it (e.g. subnet -> vpc_id = aws_vpc.x.id) rather than
just commenting it, so the generated code is actually wireable, not just
descriptive.
"""

from __future__ import annotations

from arch2terraform.generator.hcl_format import resource_block
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph

# Attribute name Terraform expects for "what container am I in", keyed by
# the *child's* resource type. Used to wire containment into real HCL
# references instead of leaving it as a comment.
# Maps (child_resource_type, attribute_name) -> the parent resource_type that attribute
# is actually allowed to reference. Wiring only fires when the *actual* resolved parent
# matches this — e.g. an EC2 instance's "subnet_id" must point at an aws_subnet, not
# whatever container happens to be nearest (a VPC containing it directly, for instance).
# Without this check, a diagram where EC2 sits directly inside a VPC box (no explicit
# subnet drawn) would otherwise generate `subnet_id = aws_vpc.x.id`, which is wrong.
_CONTAINMENT_WIRING_RULES: dict[str, tuple[str, str]] = {
    # child_resource_type: (attribute_name, required_parent_resource_type)
    "aws_subnet": ("vpc_id", "aws_vpc"),
    "aws_internet_gateway": ("vpc_id", "aws_vpc"),
    "aws_security_group": ("vpc_id", "aws_vpc"),
    "aws_instance": ("subnet_id", "aws_subnet"),
    "aws_route_table": ("vpc_id", "aws_vpc"),
    "aws_network_acl": ("vpc_id", "aws_vpc"),
    "aws_lb_target_group": ("vpc_id", "aws_vpc"),
    "aws_nat_gateway": ("subnet_id", "aws_subnet"),
    # aws_db_instance uses db_subnet_group_name (a separate resource), and aws_lb uses
    # a `subnets` list — neither maps to a single direct reference, so deliberately
    # left unwired here rather than generating an incorrect single-value attribute.
}

_OUTPUT_ATTR_BY_TYPE = {
    "aws_vpc": "id",
    "aws_subnet": "id",
    "aws_instance": "id",
    "aws_s3_bucket": "bucket",
    "aws_db_instance": "endpoint",
    "aws_lb": "dns_name",
    "aws_lambda_function": "arn",
    "aws_eks_cluster": "endpoint",
    "aws_cloudfront_distribution": "domain_name",
    "aws_dynamodb_table": "arn",
    "aws_sqs_queue": "url",
    "aws_sns_topic": "arn",
    "aws_ecr_repository": "repository_url",
}


def generate_provider_tf(aws_region_var: str = "var.aws_region") -> str:
    return (
        'terraform {\n'
        '  required_version = ">= 1.5.0"\n'
        '  required_providers {\n'
        '    aws = {\n'
        '      source  = "hashicorp/aws"\n'
        '      version = "~> 5.0"\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "provider \"aws\" {\n"
        f"  region = {aws_region_var}\n"
        "}\n"
    )


def generate_variables_tf() -> str:
    return (
        'variable "aws_region" {\n'
        '  description = "AWS region to deploy into"\n'
        '  type        = string\n'
        '  default     = "us-east-1"\n'
        "}\n\n"
        'variable "environment" {\n'
        '  description = "Deployment environment tag (e.g. dev, staging, prod)"\n'
        '  type        = string\n'
        '  default     = "dev"\n'
        "}\n"
    )


def generate_main_tf(graph: ResourceGraph) -> str:
    blocks: list[str] = []

    ordered = _containers_first(graph)
    for resource in ordered:
        attrs = dict(resource.attributes)
        wiring_warning = _wire_containment(resource, graph, attrs)

        comment = f"From diagram node: {resource.display_label}"
        if resource.confidence < 0.95:
            comment += f" (low-confidence match, confidence={resource.confidence:.2f} — please review)"
        if wiring_warning:
            comment += f" — NOTE: {wiring_warning}"

        blocks.append(resource_block(resource.resource_type, resource.terraform_name, attrs, comment))

    if not blocks:
        return "# No classifiable resources were found in the supplied diagram.\n"

    return "\n\n".join(blocks) + "\n"


def generate_outputs_tf(graph: ResourceGraph) -> str:
    lines: list[str] = []
    for resource in graph.resources:
        attr = _OUTPUT_ATTR_BY_TYPE.get(resource.resource_type)
        if not attr:
            continue
        lines.append(
            f'output "{resource.terraform_name}_{attr}" {{\n'
            f'  value = {resource.resource_type}.{resource.terraform_name}.{attr}\n'
            f"}}"
        )
    if not lines:
        return "# No outputs generated — no resources with a standard exposed attribute.\n"
    return "\n\n".join(lines) + "\n"


def generate_readme_md(graph: ResourceGraph, source_file: str, warnings: list[str]) -> str:
    lines = [
        "# Generated Terraform module",
        "",
        f"Source diagram: `{source_file}`",
        "",
        f"Resources generated: {len(graph.resources)}",
        f"Relationships resolved: {len(graph.relationships)}",
        "",
    ]

    low_confidence = [r for r in graph.resources if r.confidence < 0.95]
    if low_confidence:
        lines.append("## Low-confidence matches — please review")
        lines.append("")
        for r in low_confidence:
            lines.append(f"- `{r.terraform_name}` ({r.resource_type}) — matched from label "
                         f"\"{r.display_label}\" at confidence {r.confidence:.2f}")
        lines.append("")

    if graph.unclassified_nodes:
        lines.append("## Unclassified diagram nodes")
        lines.append("")
        lines.append("These shapes could not be matched to any known AWS resource and were skipped:")
        for node_id in graph.unclassified_nodes:
            lines.append(f"- `{node_id}`")
        lines.append("")

    if warnings:
        lines.append("## Parser warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append("1. Review low-confidence matches and unclassified nodes above.")
    lines.append("2. Run `terraform init && terraform validate` (or use the Phase 2 sandbox validator).")
    lines.append("3. Adjust default attribute values (instance sizes, CIDR blocks, engine versions) to fit your environment.")
    lines.append("")

    return "\n".join(lines)


# -- internals --------------------------------------------------------------

def _containers_first(graph: ResourceGraph) -> list[ClassifiedResource]:
    containers = [r for r in graph.resources if r.is_container]
    others = [r for r in graph.resources if not r.is_container]
    return containers + others


def _wire_containment(resource: ClassifiedResource, graph: ResourceGraph, attrs: dict) -> str | None:
    """Wires a containment relationship into a real HCL reference when the resolved
    parent's type matches what the attribute expects. Returns a warning string to
    surface in the resource's comment when containment exists but couldn't be safely
    wired (so the user knows to wire it by hand), or None when nothing needs flagging."""
    rule = _CONTAINMENT_WIRING_RULES.get(resource.resource_type)

    parent_rel = next(
        (rel for rel in graph.relationships
         if rel.relationship_type == "containment" and rel.target_node_id == resource.node_id),
        None,
    )
    if not parent_rel:
        return None

    parent = graph.resource_by_node_id(parent_rel.source_node_id)
    if not parent:
        return None

    if not rule:
        return None  # no wiring rule defined for this child type; nothing to validate or flag

    attr_name, required_parent_type = rule
    if parent.resource_type != required_parent_type:
        return (
            f"diagram shows this inside '{parent.display_label}' ({parent.resource_type}), "
            f"but {resource.resource_type} expects {attr_name} to reference a "
            f"{required_parent_type} — wire {attr_name} manually"
        )

    attrs[attr_name] = f"{parent.resource_type}.{parent.terraform_name}.id"
    return None
