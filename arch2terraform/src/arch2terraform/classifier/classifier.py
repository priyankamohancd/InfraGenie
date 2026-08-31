"""
Classifier: turns each DiagramNode into a ClassifiedResource by matching
against the AWS resource catalog.

Matching priority:
  1. icon_ref match (image_ref from draw.io stencils / Lucidchart shape library)
     — high confidence, this is metadata the diagram tool itself attached.
  2. label keyword match — lower confidence, used for Excalidraw and as
     fallback when icon match fails or is absent.
  3. No match -> node goes into unclassified_nodes, never silently dropped.

Confidence scores feed Phase 2's clarification-question logic: low-confidence
matches get flagged in needs_clarification so the user can confirm/correct
before Terraform is generated.
"""

from __future__ import annotations

import copy
import re

from arch2terraform.classifier.catalog import CATALOG, ResourceDefinition
from arch2terraform.schemas.diagram import DiagramNode, NodeShape, ParsedDiagram
from arch2terraform.schemas.resources import ClassifiedResource

_ICON_MATCH_CONFIDENCE = 0.95
_LABEL_MATCH_CONFIDENCE = 0.65
_CONTAINER_SHAPE_FALLBACK_CONFIDENCE = 0.5

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")

# Resource types needing extra companion resources beyond their own catalog
# nested_blocks/default_attributes. Currently just aws_mq_broker: unlike
# RDS/Aurora's manage_master_user_password (a flag AWS itself honors, no
# extra resources needed), Terraform's aws_mq_broker has no AWS-managed-
# secret equivalent for its required `user { password }` block — the catalog
# placeholder ("REPLACE_WITH_STRONG_PASSWORD_12CHARS") would otherwise be a
# real plaintext password baked into generated code. See
# _build_mq_broker_companion_blocks().
_MQ_BROKER_TYPE = "aws_mq_broker"

# Real bug found 2026-08-24 per an explicit user question ("why is this
# asked as a fill-in-the-blank value when it should be wired inside the
# module and resolved by Terraform at apply time?") — she's right: a NAT
# gateway's allocation_id (its Elastic IP) is a value Terraform can
# allocate and resolve entirely within the SAME apply, the same way
# aws_subnet's vpc_id or aws_lb's subnets already are, so it should never
# have needed a user-supplied answer at all. The catalog's
# "REPLACE_WITH_EIP_ALLOCATION_ID" placeholder (see catalog.py) had no
# companion-resource generation behind it — unlike aws_mq_broker/aws_s3_bucket
# above, which already use this exact companion-block mechanism for their
# own no-sensible-static-default fields. See
# _build_nat_gateway_companion_blocks().
_NAT_GATEWAY_TYPE = "aws_nat_gateway"


def _build_mq_broker_companion_blocks(terraform_name: str) -> tuple[list[str], str]:
    """
    Returns (companion HCL resource blocks, password reference to use in
    place of a literal in the broker's nested `user` block).

    Generates a random_password + a Secrets Manager secret storing it, so
    the actual password value is never written into generated .tf source —
    only discoverable via Secrets Manager (or Terraform state, same
    unavoidable exposure every other secret in this catalog has). Named off
    the broker's own terraform_name so multiple brokers in one diagram don't
    collide.
    """
    pw_name = f"{terraform_name}_admin_password"
    secret_name = f"{terraform_name}_admin_password"

    random_password_block = (
        f'# Auto-generated: {_MQ_BROKER_TYPE} has no AWS-managed-secret option\n'
        f'# (unlike RDS/Aurora\'s manage_master_user_password), so a real\n'
        f'# password is generated here and stored in Secrets Manager rather\n'
        f'# than written as a literal into this file.\n'
        f'resource "random_password" "{pw_name}" {{\n'
        f'  length  = 16\n'
        f'  special = true\n'
        f'}}'
    )

    secret_block = (
        f'resource "aws_secretsmanager_secret" "{secret_name}" {{\n'
        f'  name = "{terraform_name}-mq-admin-password"\n'
        f'}}'
    )

    secret_version_block = (
        f'resource "aws_secretsmanager_secret_version" "{secret_name}" {{\n'
        f'  secret_id     = aws_secretsmanager_secret.{secret_name}.id\n'
        f'  secret_string = random_password.{pw_name}.result\n'
        f'}}'
    )

    password_ref = f"random_password.{pw_name}.result"
    return [random_password_block, secret_block, secret_version_block], password_ref


# Added 2026-07-24: every generated aws_s3_bucket was failing four checkov
# checks by default (CKV_AWS_21 versioning, CKV_AWS_145 KMS encryption,
# CKV2_AWS_6 public access block, CKV2_AWS_61 lifecycle configuration) —
# real, addressable security-posture gaps, not stylistic nitpicks, and each
# has one unambiguous correct default with no user input needed. Modern AWS
# provider versions (4.x+) split these off aws_s3_bucket into separate
# resource types, so — same mechanism as the MQ broker's password above —
# they're emitted as companion blocks rather than inline attributes.
_S3_BUCKET_TYPE = "aws_s3_bucket"

# NOTE 2026-07-31: an earlier version of this file generated a companion
# aws_iam_role here, per-node, specifically for aws_eks_cluster. Superseded
# same day per her explicit follow-up — "this shouldn't happen just in case
# of eks, but whenever any resource requires a role then they should just be
# created by analysing the resource's connections with other resources":
# role generation now goes through the SAME edge-driven engine every other
# compute resource's role already goes through
# (app/services/security_engine/ — see complete_security_orchestrator.py's
# MANDATORY_ROLE_TYPES and dynamic_iam_generator.py's
# _MANDATORY_MANAGED_POLICY_ARNS in arch2tf-product/backend) instead of a
# one-off, per-type special case living here in the classifier. Doing it
# there instead of here also means it runs against the WHOLE diagram graph
# (all nodes + edges), not just this one node in isolation — required for
# "analyse the resource's connections with other resources" to mean
# anything at all.


# Wires each of _MULTI_SUBNET_RESOURCE_ATTRS' attributes to every
# aws_subnet resource actually present in the diagram, instead of leaving
# the catalog's single-placeholder-subnet default. Same motivating request
# as the EKS role above — "not possible to pass the id values beforehand
# ... these should be carried out in outputs" — neither of these is a
# containment relationship (an ALB doesn't sit "inside" one subnet the way
# an EC2 instance does, and an Auto Scaling group doesn't either), so this
# is a post-classification sibling-reference pass over the whole diagram
# rather than a per-node containment rule. Runs once classify_diagram has
# finished classifying every node, since e.g. an aws_lb node earlier in
# diagram order may reference aws_subnet nodes classified later.
#
# aws_autoscaling_group.vpc_zone_identifier added 2026-08-24: found via a
# real `terraform init` failure trail in arch2tf-product (Phase 2) — it's
# the identical "list of every subnet in the diagram" shape as aws_lb's
# subnets, but had no wiring pass at all, so it stayed the catalog's raw
# placeholder list. With no wiring, arch2tf-product's generic
# clarification fallback ended up offering it as a free-text field and
# stringifying the Python list into literal garbage
# (`"['subnet-00000000000000000']"`) — see that repo's
# missing_info_detector.py/terraform_planner.py for the two-sided
# defensive fix there. Generalized this single function (was
# `_wire_lb_subnets`, aws_lb-only) rather than writing a near-duplicate
# function, since the logic — "every subnet in the diagram, as a list" —
# is identical for both resource types.
_MULTI_SUBNET_RESOURCE_ATTRS: dict[str, str] = {
    "aws_lb": "subnets",
    "aws_autoscaling_group": "vpc_zone_identifier",
}

# Real bug found 2026-08-24 via an actual `terraform apply` (not just
# validate/plan — CIDR uniqueness is an AWS API-level rule, so neither
# `terraform validate` nor `terraform plan` ever catch this; it only
# surfaces once AWS itself rejects the CreateSubnet call): catalog.py's
# aws_subnet default is a single static literal, `cidr_block: "10.0.1.0/24"`
# — every subnet in a diagram that isn't itself labeled with its own bare
# CIDR text (see `_match_by_cidr_label`) gets that exact same default, so
# any diagram with 2+ generically-labeled subnets ("Public Subnet",
# "Private Subnet 2", etc.) fails at apply with "CIDR ... conflicts with
# another subnet." Fixed the same way the NAT gateway's allocation_id and
# the multi-subnet id wiring above were: assign each affected subnet its
# own deterministic, non-overlapping /24 instead of leaving them all on the
# identical placeholder. Known limitation, documented rather than silently
# guessed around: an auto-assigned range (10.0.<n>.0/24) is not checked
# against subnets that WERE given a real CIDR via `_match_by_cidr_label` —
# a diagram mixing explicit CIDR labels on some subnets with generic labels
# on others could still collide in the rare case their ranges overlap.
_DEFAULT_SUBNET_CIDR = "10.0.1.0/24"  # must match catalog.py's aws_subnet default_attributes


def _assign_unique_subnet_cidrs(classified: list[ClassifiedResource]) -> None:
    subnets = [r for r in classified if r.resource_type == "aws_subnet"]
    # Only subnets still holding the untouched catalog default need
    # reassignment — a subnet whose box was itself labeled with a real bare
    # CIDR already has its own real value (see _match_by_cidr_label) and is
    # left exactly as the diagram specified.
    on_default = [r for r in subnets if r.attributes.get("cidr_block") == _DEFAULT_SUBNET_CIDR]
    if len(on_default) <= 1:
        return  # 0 or 1 subnet on the default — no collision possible
    for i, r in enumerate(on_default, start=1):
        r.attributes["cidr_block"] = f"10.0.{i}.0/24"


def _wire_lb_subnets(classified: list[ClassifiedResource]) -> None:
    subnet_refs = [
        f"aws_subnet.{r.terraform_name}.id" for r in classified if r.resource_type == "aws_subnet"
    ]
    if not subnet_refs:
        return  # nothing in this diagram to wire to — leave the catalog placeholder
    for r in classified:
        attr_name = _MULTI_SUBNET_RESOURCE_ATTRS.get(r.resource_type)
        if attr_name:
            r.attributes[attr_name] = subnet_refs


# Same sibling-reference idea as _wire_lb_subnets, for aws_eks_node_group's
# cluster_name and subnet_ids — neither is a containment relationship (a
# node group isn't drawn "inside" its cluster). Only wires cluster_name when
# there's exactly ONE aws_eks_cluster in the diagram: with zero, there's
# nothing to reference; with more than one, which cluster a given node group
# belongs to is genuinely ambiguous from diagram structure alone (no edge
# convention exists yet to disambiguate), so the catalog placeholder is left
# in place for the user to wire by hand rather than guessing wrong.
def _wire_eks_node_group_refs(classified: list[ClassifiedResource]) -> None:
    subnet_refs = [
        f"aws_subnet.{r.terraform_name}.id" for r in classified if r.resource_type == "aws_subnet"
    ]
    clusters = [r for r in classified if r.resource_type == "aws_eks_cluster"]

    for r in classified:
        if r.resource_type != "aws_eks_node_group":
            continue
        if subnet_refs:
            r.attributes["subnet_ids"] = subnet_refs
        if len(clusters) == 1:
            r.attributes["cluster_name"] = f"aws_eks_cluster.{clusters[0].terraform_name}.name"


# Real diagram pattern found 2026-08-24, confirmed directly by the user
# ("eks cluster which has 2 control plane and 2 nodes" — i.e. ONE logical
# cluster). The standard AWS reference architecture for EKS draws the
# single, AWS-managed control plane as one box PER AZ purely as a
# redundancy visual — "Amazon EKS" + "Control plane" + "Control plane" (one
# per AZ) — alongside separate "Node"/node-group boxes per AZ for the actual
# worker capacity. Every "Control plane" box carries the same EKS icon as
# the overall "Amazon EKS" box, so before this fix each was independently
# classified as its OWN aws_eks_cluster — 3 clusters for what is really 1.
# That's exactly why _wire_eks_node_group_refs' "only wire cluster_name when
# there's exactly one cluster" safety valve above was backing off and
# leaving cluster_name as a placeholder: not a bug in the wiring itself, but
# in there being 3 candidate clusters instead of 1. This runs once
# classify_diagram has classified every node (same "runs after the main
# loop" reasoning as _wire_lb_subnets) since it needs to see every
# aws_eks_cluster candidate at once to tell a duplicate from a genuine
# second cluster.
_EKS_CONTROL_PLANE_LABEL_RE = re.compile(r"^\s*control\s+plane\b", re.IGNORECASE)


def _merge_eks_control_plane_duplicates(
    classified: list[ClassifiedResource],
) -> list[ClassifiedResource]:
    clusters = [r for r in classified if r.resource_type == "aws_eks_cluster"]
    if len(clusters) <= 1:
        return classified  # nothing to merge

    control_plane_dupes = [
        r for r in clusters if _EKS_CONTROL_PLANE_LABEL_RE.match(r.display_label or "")
    ]
    control_plane_dupe_ids = {id(r) for r in control_plane_dupes}
    non_control_plane = [r for r in clusters if id(r) not in control_plane_dupe_ids]

    if non_control_plane:
        # A genuine standalone cluster box exists (e.g. "Amazon EKS") — the
        # "Control plane" boxes are pure redundancy visuals for THAT cluster,
        # so drop them entirely rather than emitting duplicate resources.
        to_drop = {id(r) for r in control_plane_dupes}
    elif control_plane_dupes:
        # No standalone box at all — every eks_cluster match IS itself a
        # "Control plane" box (e.g. two control-plane-only boxes, no
        # separate overall icon). Keep the first in diagram order as the
        # one real cluster and drop the rest, so exactly one aws_eks_cluster
        # survives instead of zero or several.
        to_drop = {id(r) for r in control_plane_dupes[1:]}
    else:
        # More than one aws_eks_cluster and none look like the "Control
        # plane" redundancy-visual pattern — a genuine multi-cluster diagram
        # (or an unrecognized label variant). Leave as-is: still ambiguous,
        # still safer left to the existing placeholder + user review than
        # guessed away.
        return classified

    return [r for r in classified if id(r) not in to_drop]


def _build_s3_bucket_companion_blocks(terraform_name: str) -> list[str]:
    """
    Returns companion HCL resource blocks wiring a KMS-encrypted, versioned,
    fully-public-access-blocked S3 bucket with a baseline lifecycle rule.

    Deliberately NOT addressed here: CKV_AWS_18 (access logging) and
    CKV_AWS_144 (cross-region replication) — both require provisioning a
    SECOND bucket (a log-target bucket, or a same-shape bucket in another
    region) that this generator has no principled basis for sizing, naming,
    or placing on the user's behalf. Left as a documented known limitation
    rather than a guessed default that could itself become a footgun
    (e.g. an auto-created log bucket nobody asked for, in a region nobody
    chose).
    """
    kms_name = f"{terraform_name}_key"
    caller_identity_name = f"{terraform_name}_caller"

    # A key with no explicit policy relies on IAM alone to control access —
    # checkov (CKV2_AWS_64) flags that as undefined access control on the
    # key itself, so an explicit policy granting the account root full
    # management rights (the same baseline AWS applies by default, just
    # made explicit) is included. The caller-identity data source is named
    # off this bucket's own terraform_name — not a shared "current" name —
    # so multiple S3 buckets in one diagram each get their own data block
    # instead of colliding on a duplicate resource name.
    caller_identity_block = f'data "aws_caller_identity" "{caller_identity_name}" {{}}'

    kms_key_block = (
        f'resource "aws_kms_key" "{kms_name}" {{\n'
        f'  description         = "SSE-KMS key for the {terraform_name} S3 bucket"\n'
        f'  enable_key_rotation = true\n'
        f'\n'
        f'  policy = jsonencode({{\n'
        f'    Version = "2012-10-17"\n'
        f'    Statement = [\n'
        f'      {{\n'
        f'        Sid       = "EnableRootAccountAccess"\n'
        f'        Effect    = "Allow"\n'
        f'        Principal = {{ AWS = "arn:aws:iam::${{data.aws_caller_identity.{caller_identity_name}.account_id}}:root" }}\n'
        f'        Action    = "kms:*"\n'
        f'        Resource  = "*"\n'
        f'      }}\n'
        f'    ]\n'
        f'  }})\n'
        f'}}'
    )

    versioning_block = (
        f'resource "aws_s3_bucket_versioning" "{terraform_name}" {{\n'
        f'  bucket = aws_s3_bucket.{terraform_name}.id\n'
        f'\n'
        f'  versioning_configuration {{\n'
        f'    status = "Enabled"\n'
        f'  }}\n'
        f'}}'
    )

    encryption_block = (
        f'resource "aws_s3_bucket_server_side_encryption_configuration" "{terraform_name}" {{\n'
        f'  bucket = aws_s3_bucket.{terraform_name}.id\n'
        f'\n'
        f'  rule {{\n'
        f'    apply_server_side_encryption_by_default {{\n'
        f'      sse_algorithm     = "aws:kms"\n'
        f'      kms_master_key_id = aws_kms_key.{kms_name}.arn\n'
        f'    }}\n'
        f'    bucket_key_enabled = true\n'
        f'  }}\n'
        f'}}'
    )

    public_access_block = (
        f'resource "aws_s3_bucket_public_access_block" "{terraform_name}" {{\n'
        f'  bucket = aws_s3_bucket.{terraform_name}.id\n'
        f'\n'
        f'  block_public_acls       = true\n'
        f'  block_public_policy     = true\n'
        f'  ignore_public_acls      = true\n'
        f'  restrict_public_buckets = true\n'
        f'}}'
    )

    lifecycle_block = (
        f'resource "aws_s3_bucket_lifecycle_configuration" "{terraform_name}" {{\n'
        f'  bucket = aws_s3_bucket.{terraform_name}.id\n'
        f'\n'
        f'  rule {{\n'
        f'    id     = "abort-incomplete-multipart-uploads"\n'
        f'    status = "Enabled"\n'
        f'\n'
        f'    filter {{}}\n'
        f'\n'
        f'    abort_incomplete_multipart_upload {{\n'
        f'      days_after_initiation = 7\n'
        f'    }}\n'
        f'  }}\n'
        f'}}'
    )

    return [
        caller_identity_block, kms_key_block, versioning_block,
        encryption_block, public_access_block, lifecycle_block,
    ]


def _build_nat_gateway_companion_blocks(terraform_name: str) -> tuple[list[str], str]:
    """
    Returns (companion HCL resource blocks, allocation_id reference to use in
    place of the catalog's static placeholder).

    A NAT gateway's allocation_id is just the id of an Elastic IP it owns —
    unlike aws_mq_broker's password or aws_s3_bucket's KMS/versioning/
    encryption settings, there isn't even a design choice to make here: the
    EIP is a same-module, same-apply resource with one obviously correct
    shape (domain = "vpc", nothing else configurable that matters). Named
    off the NAT gateway's own terraform_name so multiple NAT gateways in one
    diagram (one per AZ is the common case) each get their own EIP instead
    of colliding on a duplicate resource name.
    """
    eip_name = f"{terraform_name}_eip"

    eip_block = (
        f'# Auto-generated: {_NAT_GATEWAY_TYPE} needs an Elastic IP allocation_id.\n'
        f'# Provisioned here in the same module/apply instead of asking for a\n'
        f'# pre-existing allocation id up front.\n'
        f'resource "aws_eip" "{eip_name}" {{\n'
        f'  domain = "vpc"\n'
        f'}}'
    )

    allocation_id_ref = f"aws_eip.{eip_name}.id"
    return [eip_block], allocation_id_ref


# Real bug found 2026-08-24, same `terraform apply` as the subnet CIDR
# collision above: her diagram (the standard EKS reference layout) has no
# Internet Gateway icon drawn at all — arch2terraform only ever classifies
# aws_internet_gateway/aws_route_table from an explicit icon/label match,
# there is no companion generation for either. A NAT gateway REQUIRES an
# Internet Gateway attached to its VPC to leave the "pending"/"failed"
# state (AWS checks this at the API level before the NAT gateway can
# become available), which is exactly the "Gateway.NotAttached" error she
# hit — plan/validate never catch it since it's an AWS business rule, not
# an HCL schema rule. An Internet Gateway (and the route tables wiring
# public subnets to it / private subnets to the NAT gateway) is exactly as
# implicit and non-optional as the NAT gateway's own EIP above, so it gets
# the same treatment: auto-provisioned rather than left for the user to
# draw a box for.
#
# Known, documented scope limits (consistent with this file's other
# companion-block functions calling out what they don't handle):
#   - Assumes a single VPC per diagram, same as the existing
#     _wire_lb_subnets/_wire_eks_node_group_refs sibling-reference passes —
#     classifier.py has no per-VPC containment resolution at this stage
#     (that's resolver.py's job, and it runs AFTER classification).
#   - "Public" vs "private" is decided by substring match on the subnet's
#     own terraform_name/display_label (the diagram's actual convention in
#     every case seen so far — "Public Subnet"/"Private Subnet"). A subnet
#     named without either word defaults to the private route table (no
#     accidental internet exposure) rather than being guessed public.
#   - Multiple NAT gateways (one per AZ, the common HA pattern — her
#     diagram has 2) are paired to private subnets by diagram order/index
#     rather than true AZ matching, since AZ-level containment isn't
#     resolved yet at this stage either. An exact 1:1 count still produces
#     the intended per-AZ pairing; a mismatched count falls back to modulo
#     wraparound so nothing is left unrouted.
_PUBLIC_SUBNET_LABEL_RE = re.compile(r"\bpublic\b", re.IGNORECASE)
_PRIVATE_SUBNET_LABEL_RE = re.compile(r"\bprivate\b", re.IGNORECASE)


def _wire_implicit_internet_gateway(classified: list[ClassifiedResource]) -> None:
    vpcs = [r for r in classified if r.resource_type == "aws_vpc"]
    subnets = [r for r in classified if r.resource_type == "aws_subnet"]
    nat_gateways = [r for r in classified if r.resource_type == _NAT_GATEWAY_TYPE]
    already_has_igw = any(r.resource_type == "aws_internet_gateway" for r in classified)

    if not vpcs or not subnets or already_has_igw:
        return  # nothing to attach to, nothing needing routing, or already drawn explicitly

    vpc = vpcs[0]
    igw_name = f"{vpc.terraform_name}_igw"
    igw_block = (
        f'# Auto-generated: this VPC has subnets but the diagram had no\n'
        f'# Internet Gateway of its own — required for a NAT gateway to\n'
        f'# leave "pending" and for public subnets to reach the internet.\n'
        f'resource "aws_internet_gateway" "{igw_name}" {{\n'
        f'  vpc_id = aws_vpc.{vpc.terraform_name}.id\n'
        f'}}'
    )
    companion_blocks = [igw_block]

    def label_of(r: ClassifiedResource) -> str:
        return f"{r.display_label} {r.terraform_name}"

    public_subnets = [r for r in subnets if _PUBLIC_SUBNET_LABEL_RE.search(label_of(r))]
    public_subnet_ids = {id(r) for r in public_subnets}
    private_subnets = [r for r in subnets if id(r) not in public_subnet_ids]

    if public_subnets:
        public_rt_name = f"{vpc.terraform_name}_public_rt"
        companion_blocks.append(
            f'resource "aws_route_table" "{public_rt_name}" {{\n'
            f'  vpc_id = aws_vpc.{vpc.terraform_name}.id\n'
            f'  route {{\n'
            f'    cidr_block = "0.0.0.0/0"\n'
            f'    gateway_id = aws_internet_gateway.{igw_name}.id\n'
            f'  }}\n'
            f'}}'
        )
        for r in public_subnets:
            companion_blocks.append(
                f'resource "aws_route_table_association" "{r.terraform_name}_assoc" {{\n'
                f'  subnet_id      = aws_subnet.{r.terraform_name}.id\n'
                f'  route_table_id = aws_route_table.{public_rt_name}.id\n'
                f'}}'
            )

    if private_subnets and nat_gateways:
        for i, r in enumerate(private_subnets):
            nat = nat_gateways[i % len(nat_gateways)]
            private_rt_name = f"{r.terraform_name}_private_rt"
            companion_blocks.append(
                f'resource "aws_route_table" "{private_rt_name}" {{\n'
                f'  vpc_id = aws_vpc.{vpc.terraform_name}.id\n'
                f'  route {{\n'
                f'    cidr_block     = "0.0.0.0/0"\n'
                f'    nat_gateway_id = aws_nat_gateway.{nat.terraform_name}.id\n'
                f'  }}\n'
                f'}}'
            )
            companion_blocks.append(
                f'resource "aws_route_table_association" "{r.terraform_name}_assoc" {{\n'
                f'  subnet_id      = aws_subnet.{r.terraform_name}.id\n'
                f'  route_table_id = aws_route_table.{private_rt_name}.id\n'
                f'}}'
            )

    vpc.companion_blocks = list(vpc.companion_blocks) + companion_blocks


# Container types that are AWS's implicit structural boundaries rather than
# provisionable resources:
#   - AWS Cloud is just the outer partition boundary every diagram has — there
#     is no corresponding Terraform resource at all.
#   - An Availability Zone is never created as its own resource; it's an
#     attribute (`availability_zone = "..."`) set on the zonal resources
#     inside it (subnets, EBS volumes, etc.).
# These come from the image adapter's layout_detector (see
# adapters/image/layout_detector.py::_CONTAINER_IMAGE_REF). Before this set
# existed, both fell through to the generic "unmatched container -> aws_vpc"
# fallback below, silently emitting phantom VPC resources that don't belong
# in the diagram's actual infrastructure — a correctness bug for a tool whose
# whole point is trustworthy IaC from the diagram. They're skipped here
# (not classified, not flagged as unclassified/needs-review) since they are
# expected, recognized structure, not something the user needs to fix.
_STRUCTURAL_ONLY_IMAGE_REFS = {"aws-cloud", "availability-zone"}

# Same idea as _STRUCTURAL_ONLY_IMAGE_REFS, but keyed off the OCR label text
# instead of image_ref — needed once Stage 1's color-agnostic layout_detector
# started reliably finding container regions whose color doesn't match any
# known AWS profile (image_ref="Unknown-Container"). Real bug found
# 2026-07-28: on a diagram using a non-standard border color, containers
# labeled "Region" and "AZ-1b" (real OCR reads, not noise) were both falling
# through to the generic container fallback and coming back as low-confidence
# aws_vpc — a "Region" boundary and an Availability Zone label are exactly
# the same kind of implicit AWS structure as AWS-Cloud/Availability-Zone
# above, just identified by OCR text instead of a recognized icon/color.
_STRUCTURAL_ONLY_LABEL_RE = re.compile(
    r"^\s*(?:[a-z]{2}-[a-z]+-\d\s+)?region\s*$"      # "Region", "us-east-1 Region"
    r"|^\s*az[\s-]?\d?[a-z]?\s*$"                     # "AZ-1b", "AZ 2a", "az1"
    r"|^\s*availability\s*zone(?:\s*[a-z0-9-]*)?\s*$",  # "Availability Zone", "Availability Zone A"
    re.IGNORECASE,
)

# Matches a bare CIDR block label with nothing else around it (e.g.
# "10.0.3.0/24"), which real AWS diagrams routinely use as a VPC's or
# subnet's ONLY visible label instead of the word "VPC"/"Subnet". Previously
# ignored entirely by _match_by_label (whose keywords are "vpc"/"subnet"
# literal words), so every CIDR-only container fell through to the same
# generic aws_vpc fallback regardless of whether it was actually the VPC or
# a subnet nested inside it.
_CIDR_LABEL_RE = re.compile(r"^\s*(\d{1,3}\.){3}\d{1,3}/\d{1,2}\s*$")


def _normalize_ref(ref: str) -> str:
    """Lowercase and strip separators so 'Security-Group' == 'securitygroup'.

    Different adapters format image_ref differently (draw.io stencil names
    have no separators, e.g. 'mxgraph.aws4.securityGroup'; the image adapter's
    layout_detector emits hyphenated names, e.g. 'Security-Group'). Catalog
    icon_keys are written without separators, so without this normalization,
    any hyphenated image_ref with the key split across a hyphen (e.g.
    'security-group' vs. key 'securitygroup') silently fails to match and
    falls through to the container fallback -> phantom aws_vpc bug above.
    """
    return _NON_ALNUM_RE.sub("", ref.lower())


def classify_diagram(diagram: ParsedDiagram) -> tuple[list[ClassifiedResource], list[str]]:
    """Returns (classified resources, ids of nodes that couldn't be classified)."""
    classified: list[ClassifiedResource] = []
    unclassified: list[str] = []
    used_names: set[str] = set()

    for node in diagram.nodes:
        if node.shape == NodeShape.CONTAINER and (node.image_ref or "").lower() in _STRUCTURAL_ONLY_IMAGE_REFS:
            continue  # recognized structural boundary, not a resource — see comment above
        if node.shape == NodeShape.CONTAINER and _STRUCTURAL_ONLY_LABEL_RE.match(node.raw_label or ""):
            continue  # "Region"/"AZ-*" label — same as above, just OCR-identified — see comment above

        result = _classify_node(node)
        if result is None:
            unclassified.append(node.id)
            continue

        definition, confidence, attribute_overrides = result
        tf_name = _unique_terraform_name(node, used_names)
        needs_clarification = [] if confidence >= _ICON_MATCH_CONFIDENCE else ["resource_type"]

        # deepcopy (not a shallow dict()) since nested_blocks holds lists of
        # dicts/Block instances — a shallow copy would let multiple diagram
        # instances of the same catalog type share (and risk mutating) the
        # same inner list/dict objects.
        nested_blocks = copy.deepcopy(definition.nested_blocks)
        companion_blocks: list[str] = []

        if definition.terraform_type == _MQ_BROKER_TYPE and nested_blocks.get("user"):
            companion_blocks, password_ref = _build_mq_broker_companion_blocks(tf_name)
            for user_block in nested_blocks["user"]:
                if "password" in user_block:
                    user_block["password"] = password_ref
        elif definition.terraform_type == _S3_BUCKET_TYPE:
            companion_blocks = _build_s3_bucket_companion_blocks(tf_name)
        elif definition.terraform_type == _NAT_GATEWAY_TYPE:
            companion_blocks, nat_allocation_id_ref = _build_nat_gateway_companion_blocks(tf_name)

        attributes = dict(definition.default_attributes)
        attributes.update(attribute_overrides)

        # Override AFTER attribute_overrides so the auto-provisioned EIP wins
        # over the catalog's "REPLACE_WITH_EIP_ALLOCATION_ID" placeholder
        # even if some upstream stage already surfaced that placeholder as
        # an override.
        if definition.terraform_type == _NAT_GATEWAY_TYPE:
            attributes["allocation_id"] = nat_allocation_id_ref

        classified.append(
            ClassifiedResource(
                node_id=node.id,
                resource_type=definition.terraform_type,
                terraform_name=tf_name,
                display_label=node.raw_label or definition.terraform_type,
                confidence=confidence,
                attributes=attributes,
                nested_blocks=nested_blocks,
                is_container=definition.is_container,
                needs_clarification=needs_clarification,
                tags=dict(node.tags),
                companion_blocks=companion_blocks,
            )
        )

    classified = _merge_eks_control_plane_duplicates(classified)
    _assign_unique_subnet_cidrs(classified)
    _wire_lb_subnets(classified)
    _wire_eks_node_group_refs(classified)
    _wire_implicit_internet_gateway(classified)
    return classified, unclassified


def _classify_node(node: DiagramNode) -> tuple[ResourceDefinition, float, dict] | None:
    # Added 2026-07-31: when the Vision-LLM adapter ran (see
    # vision_llm_detector.py), it has already read this node's label AND its
    # surrounding diagram context (what it's nested in, what's around it) —
    # real semantic disambiguation that plain icon/label substring matching
    # below can't do (the exact class of bug this fixes: "EKS Node 1 (t2
    # medium)" matching aws_eks_cluster purely because "eks" is a substring
    # of both). Only trusted when it names a REAL catalog type — a
    # hallucinated or invalid hint falls straight through to the icon/label
    # matching below rather than being trusted blindly.
    vision_hint = (node.extra or {}).get("terraform_type_hint")
    if vision_hint:
        hint_match = next((d for d in CATALOG if d.terraform_type == vision_hint), None)
        if hint_match:
            return hint_match, _ICON_MATCH_CONFIDENCE, {}

    icon_match = _match_by_icon(node)
    if icon_match:
        return icon_match, _ICON_MATCH_CONFIDENCE, {}

    label_match = _match_by_label(node)
    if label_match:
        return label_match, _LABEL_MATCH_CONFIDENCE, {}

    cidr_match = _match_by_cidr_label(node)
    if cidr_match:
        return cidr_match

    # Generic container shapes with no specific match (e.g. an unlabeled box
    # drawn as a container) still get treated as a container with low
    # confidence, since dropping them would break containment relationships
    # downstream.
    #
    # Real bug found 2026-07-28: this used to unconditionally return
    # aws_vpc regardless of the container's actual position in the
    # hierarchy. Once Stage 1's color-agnostic layout_detector started
    # reliably finding many more container regions (including ones whose
    # color doesn't match a known AWS profile), a real diagram could
    # produce a dozen+ genuinely different nested containers that all
    # landed here — and all of them came back labeled "aws_vpc", which is
    # actively misleading (a VPC nested inside another VPC essentially
    # never happens in a real AWS diagram) even though they were correctly
    # flagged for review.
    #
    # A TOP-LEVEL (no parent) unlabeled container still defaults to
    # aws_vpc: an unlabeled outermost box is plausibly the diagram's VPC
    # boundary, and this is covered by
    # test_container_shape_without_label_defaults_to_vpc — real signal, not
    # a pure guess.
    #
    # A NESTED container reaching this point, by contrast, has already
    # failed icon match, label-keyword match, AND CIDR-label match — i.e.
    # there is NO textual signal it's a subnet at all. Real bug found
    # 2026-07-28 investigating a dense real-world diagram: Stage 1's
    # structural detector also finds plenty of real, visibly-bordered boxes
    # that aren't AWS resources at all — organizational grouping boxes a
    # diagram author drew around a cluster of unrelated icons (e.g. a box
    # around "Frontend/Backend/Worker" pod services, or around a row of
    # "Secrets Manager/KMS/ECR/CloudWatch" ops icons). Guessing aws_subnet
    # for these produced a wall of identically-mislabeled 50%-confidence
    # cards that all looked like real (if uncertain) subnets, when they
    # were something else entirely. With zero textual signal, guessing is
    # worse than admitting uncertainty: these go to `unclassified` instead,
    # surfaced in the README's "Unclassified diagram nodes" section for
    # manual review. Any REAL children nested under a now-unclassified
    # grouping box still get correctly re-parented to the next enclosing
    # classified container by the resolver's geometric containment fallback
    # (resolver.py::_find_container_id) — they are never orphaned.
    if node.shape == NodeShape.CONTAINER:
        if node.parent_id:
            return None
        vpc_def = next(d for d in CATALOG if d.terraform_type == "aws_vpc")
        return vpc_def, _CONTAINER_SHAPE_FALLBACK_CONFIDENCE, {}

    return None


def _match_by_cidr_label(node: DiagramNode) -> tuple[ResourceDefinition, float, dict] | None:
    """
    A container labeled with nothing but a bare CIDR block (e.g.
    "10.0.3.0/24") is real, meaningful OCR signal that ``_match_by_label``
    ignores (its keywords are the literal words "vpc"/"subnet"). Real AWS
    diagrams routinely label VPC/subnet boxes with just their CIDR and no
    other text. Whether it's the VPC or a subnet is determined by nesting:
    a top-level (no parent) CIDR-labeled container is the VPC; a nested one
    is a subnet. The CIDR text itself becomes the resource's cidr_block,
    replacing the catalog's generic placeholder default.
    """
    label = (node.raw_label or "").strip()
    if node.shape != NodeShape.CONTAINER or not _CIDR_LABEL_RE.match(label):
        return None

    terraform_type = "aws_subnet" if node.parent_id else "aws_vpc"
    definition = next(d for d in CATALOG if d.terraform_type == terraform_type)
    return definition, _LABEL_MATCH_CONFIDENCE, {"cidr_block": label}


def _match_by_icon(node: DiagramNode) -> ResourceDefinition | None:
    if not node.image_ref:
        return None
    ref = _normalize_ref(node.image_ref)

    # Collect every (definition, matched_key) pair across the whole catalog rather
    # than stopping at the first list entry that matches anything. Generic keys like
    # "instance" (EC2) are substrings of more specific labels like "RDS Instance", so
    # without this, catalog order alone would decide the result. Picking the longest
    # matched key makes the more specific catalog entry win regardless of list position.
    candidates: list[tuple[ResourceDefinition, str]] = []
    for definition in CATALOG:
        for key in definition.icon_keys:
            if key in ref:
                candidates.append((definition, key))

    if not candidates:
        return None

    best_definition, _ = max(candidates, key=lambda pair: len(pair[1]))
    return best_definition


def _match_by_label(node: DiagramNode) -> ResourceDefinition | None:
    label = (node.raw_label or "").lower().strip()
    if not label:
        return None

    candidates: list[tuple[ResourceDefinition, str]] = []
    for definition in CATALOG:
        for kw in definition.label_keywords:
            if kw in label:
                candidates.append((definition, kw))

    if not candidates:
        return None

    best_definition, _ = max(candidates, key=lambda pair: len(pair[1]))
    return best_definition


def _unique_terraform_name(node: DiagramNode, used_names: set[str]) -> str:
    base = node.raw_label or node.id
    slug = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_") or "resource"
    if not re.match(r"^[a-z_]", slug):
        slug = f"r_{slug}"

    name = slug
    suffix = 1
    while name in used_names:
        suffix += 1
        name = f"{slug}_{suffix}"
    used_names.add(name)
    return name
