"""
Missing Info Detector
----------------------
Inspects ParsedResources for low-confidence matches and placeholder properties.
Builds ClarificationField questions for the user to fill in the gaps.

Rules:
- confidence < threshold → ask "what service is this?"
- property value looks like a placeholder → ask for the real value
- specific resource types always need certain fields (AMI, DB password, etc.)

As of 2026-07-08, resource properties come from arch2terraform's catalog
(via arch2terraform_bridge.py) rather than this backend's old local
icon_resource_map.py. arch2terraform's placeholders are intentionally
schema-shaped so `terraform validate` passes (e.g. "ami-00000000000000000",
"replace-with-globally-unique-name", "arn:aws:iam::000000000000:role/REPLACE_WITH_ROLE_NAME")
rather than the old "# TODO: ..." convention — see _looks_like_placeholder().

**Update (2026-07-08): vars.yaml support — her explicit "from scratch vs.
existing config" distinction.** Two cases now:
1. Building from scratch (`Job.input_vars` is None, no vars.yaml uploaded):
   every field this detector knows is a real, catalog-default-backed config
   knob (the ~12 hand-covered MANDATORY_FIELDS types + the ~25
   generic-fallback-covered types, see `_CATALOG_DEFAULT_KEYS`) gets asked
   about — NOT just the ones whose current value happens to look like a
   placeholder. This is what actually closes the instance_type/"t3.micro"
   sizing gap she found: t3.micro is a perfectly valid, non-placeholder
   default, so the old placeholder-only gate never asked about it.
2. Re-running against an already-configured diagram (`Job.input_vars` set,
   from a previously-uploaded or previously-generated vars.yaml): any
   (resource, field) pair it covers is silently pre-filled — not asked —
   and only genuinely uncovered fields go through the same catalog-default
   question logic as case 1.
Both cases route through the SAME code path below (`covered` just starts
empty in case 1) rather than two separate implementations.
"""
from __future__ import annotations
import sys
from pathlib import Path

import yaml

from app._pathboot import ensure_paths
ensure_paths()
from shared.schemas.models import (
    ParsedDiagram, ParsedResource, ClarificationField, ClarificationRequest,
    ClarificationAnswer,
)
from app.core.config import get_settings

# arch2terraform's CATALOG (same package arch2terraform_bridge.py already
# routes classification through) — used here only to build a per-resource-type
# index of which property keys are genuine catalog `default_attributes`
# ("real config knobs") vs. incidental properties (wiring-set references,
# tags, etc.) that shouldn't be surfaced as questions even under "ask
# everything."
from arch2terraform.classifier.catalog import CATALOG as _A2TF_CATALOG

_CATALOG_DEFAULT_KEYS: dict[str, set[str]] = {
    rd.terraform_type: set(rd.default_attributes.keys()) for rd in _A2TF_CATALOG
}

_s = get_settings()


def _resource_key(resource: ParsedResource) -> str:
    """Stable-across-re-uploads key for vars.yaml — "<type>.<logical_name>",
    e.g. "aws_instance.web_server". Deliberately NOT resource.id: ids like
    "r1"/"r2" are assigned by parse order and aren't guaranteed stable when
    the diagram is re-uploaded after an edit, whereas logical_name is
    derived from the diagram label. Matches the actual Terraform resource
    address (`aws_instance.web_server`) for readability."""
    return f"{resource.aws_resource_type}.{resource.logical_name}"

# ── Per-resource-type mandatory clarification fields ─────────────────────────
# (field_key, question, input_type, options, default)
MANDATORY_FIELDS: dict[str, list[tuple]] = {
    "aws_instance": [
        ("ami", "AMI ID for {label}? (e.g. ami-0abcdef1234567890)", "text", [], ""),
        ("instance_type", "Instance type for {label}?", "select",
         ["t3.micro","t3.small","t3.medium","t3.large","m5.large","m5.xlarge","c5.large"], "t3.micro"),
    ],
    "aws_db_instance": [
        ("engine", "Database engine for {label}?", "select",
         ["postgres","mysql","mariadb","oracle-se2","sqlserver-ex"], "postgres"),
        ("engine_version", "Engine version for {label}?", "text", [], "15"),
        ("instance_class", "DB instance class for {label}?", "select",
         ["db.t3.micro","db.t3.small","db.t3.medium","db.r5.large","db.r5.xlarge"], "db.t3.micro"),
        ("allocated_storage", "Storage (GB) for {label}?", "number", [], "20"),
        ("multi_az", "Enable Multi-AZ for {label}?", "select", ["true","false"], "false"),
    ],
    "aws_rds_cluster": [
        ("engine", "Aurora engine for {label}?", "select",
         ["aurora-postgresql","aurora-mysql"], "aurora-postgresql"),
        ("engine_version", "Engine version?", "text", [], "15.3"),
    ],
    "aws_elasticache_cluster": [
        ("engine", "Cache engine for {label}?", "select", ["redis","memcached"], "redis"),
        ("node_type", "Cache node type for {label}?", "select",
         ["cache.t3.micro","cache.t3.small","cache.r5.large"], "cache.t3.micro"),
        ("num_cache_nodes", "Number of cache nodes?", "number", [], "1"),
    ],
    # Real bug found 2026-08-24 via `terraform validate` ("An argument named
    # 'kubernetes_version' is not expected here"): the field_key here isn't
    # just an internal label — terraform_planner.py's
    # _variableize_mandatory_fields bakes it straight in as the literal HCL
    # attribute name on the generated resource block (see
    # _resource_block_lines: resource.properties' keys become the block's
    # `key = value` lines directly). aws_eks_cluster's real AWS-provider
    # argument for this is `version`, not `kubernetes_version` — using the
    # wrong key here meant every generated EKS cluster failed
    # `terraform validate` outright. The question text stays
    # human-readable; only the field_key (which doubles as the emitted
    # attribute name and the `var.<name>` variable name) changed.
    "aws_eks_cluster": [
        ("version", "Kubernetes version for {label}?", "select",
         ["1.28","1.29","1.30"], "1.29"),
    ],
    "aws_lambda_function": [
        ("runtime", "Lambda runtime for {label}?", "select",
         ["python3.12","python3.11","nodejs20.x","nodejs18.x","java21","go1.x"], "python3.12"),
        ("memory_size", "Memory (MB) for {label}?", "select",
         ["128","256","512","1024","2048","3008"], "256"),
    ],
    "aws_lb": [
        ("load_balancer_type", "Load balancer type for {label}?", "select",
         ["application","network","gateway"], "application"),
        ("internal", "Internal (private) load balancer?", "select", ["true","false"], "false"),
    ],
    "aws_vpc": [
        ("cidr_block", "CIDR block for {label}?", "text", [], "10.0.0.0/16"),
    ],
    "aws_subnet": [
        ("cidr_block", "CIDR block for {label}?", "text", [], "10.0.1.0/24"),
        ("availability_zone", "Availability zone for {label}?", "select",
         ["us-east-1a","us-east-1b","us-east-1c",
          "us-west-2a","us-west-2b","us-west-2c",
          "eu-west-1a","eu-west-1b","eu-west-1c"], "us-east-1a"),
    ],
    # `versioning_enabled` was intentionally removed from here (2026-07-08):
    # a real `terraform validate` run caught that it's not a valid
    # aws_s3_bucket argument on AWS provider v4+ (versioning was split out
    # into a separate `aws_s3_bucket_versioning` resource with a nested
    # `versioning_configuration { status = ... }` block) — asking this
    # question and applying the answer produced HCL that failed validate
    # outright ("Unsupported argument"). Re-adding bucket versioning support
    # properly means emitting that second resource type, which
    # terraform_planner.py doesn't do yet — a real feature, not a one-line fix.
    "aws_s3_bucket": [
        ("bucket", "S3 bucket name for {label}? (globally unique)", "text", [], ""),
    ],
    "aws_kinesis_stream": [
        ("shard_count", "Kinesis shard count for {label}?", "number", [], "1"),
    ],
    "aws_msk_cluster": [
        ("kafka_version", "Kafka version for {label}?", "select",
         ["3.5.1","3.4.0","3.3.1","2.8.1"], "3.5.1"),
        ("number_of_broker_nodes", "Number of broker nodes?", "number", [], "3"),
    ],
}

# Substrings that show up in arch2terraform's catalog placeholder values —
# see catalog.py's default_attributes for the actual conventions (fake IDs
# padded with zeros, "replace-with-*"/"REPLACE_WITH_*" strings, and the fake
# account ID "000000000000" used in placeholder ARNs). Deliberately
# case-insensitive substring matching rather than an exact-value allowlist,
# since new catalog entries can introduce new placeholder strings that follow
# the same conventions without this detector needing an update every time.
_PLACEHOLDER_MARKERS = (
    "replace-with", "replace_with", "replace with",
    "000000000000",            # fake account ID (used in placeholder ARNs)
    "00000000000000000",       # fake 17-char resource ID suffix (subnet-, vpc-, ami-, etc.)
    "0000000000000000000",     # fake IAM role name suffix ("...ROLE_NAME")
)

# Catalog default_attributes keys that must NEVER be surfaced as an
# ask-every-catalog-default question (2026-07-08's "ask all" change),
# even though they're genuine entries in _CATALOG_DEFAULT_KEYS. Both are
# handled by a secure-by-design mechanism elsewhere, not meant to be
# free-text user input: `manage_master_user_password`/`manage_master_password`
# is the AWS-Secrets-Manager-backed password mechanism (terraform_planner.py /
# arch2terraform's hcl_format.py) — turning it into a yes/no clarification
# question risks someone answering "false" and reintroducing a literal
# plaintext password. `username` is a stable RDS/Aurora convention ("admin")
# that was already deliberately left un-asked before this feature existed —
# same call preserved, not something this change should start surfacing.
_NEVER_ASK_CATALOG_DEFAULTS = frozenset({
    "manage_master_user_password", "manage_master_password", "username",
})

# Subset of security_bridge.ATTACHMENT_ATTR_BY_RESOURCE_TYPE covering only
# complete_security_orchestrator.MANDATORY_ROLE_TYPES (not imported here —
# importing security_bridge would pull in the whole security_engine/
# sys.path bootstrap for one small dict; this module already duplicates
# arch2terraform's containment-wiring rules the same way for the same
# reason, see terraform_planner.py's _CONTAINMENT_WIRING_RULES comment).
#
# Deliberately NOT the full attachment map: build_terraform_plan's
# _wire_role_attachments() (terraform_planner.py) overwrites a resource's
# role attribute ONLY IF the security engine actually generated a role for
# it, which for most eligible types (aws_lambda_function,
# aws_ecs_task_definition, aws_sfn_state_machine) depends on the diagram
# having qualifying outbound edges — unknowable at this Clarify-screen
# step, so asking about those still gives the user a real fallback value if
# no edges end up producing a role. MANDATORY_ROLE_TYPES is different: those
# get a role unconditionally regardless of edges (see
# complete_security_orchestrator.py), so the attribute is ALWAYS overwritten
# later — asking here was always pure noise, which is exactly what
# triggered her "what value is this asking for" / this whole follow-up.
_MANDATORY_ROLE_WIRED_FIELD_BY_RESOURCE_TYPE = {
    "aws_eks_cluster": "role_arn",
    "aws_eks_node_group": "node_role_arn",
    "aws_mwaa_environment": "execution_role_arn",
    "aws_codepipeline": "role_arn",
    "aws_codebuild_project": "service_role",
    "aws_glue_job": "role_arn",
}


def _looks_like_wired_reference(value) -> bool:
    """True if `value` is already a real Terraform reference wired by the
    classifier/planner (e.g. an auto-generated IAM role's arn for an EKS
    cluster's role_arn — see arch2terraform's classifier.py
    _build_eks_cluster_companion_blocks() — or a list of sibling subnet ids
    for an ALB's subnets — see _wire_lb_subnets()) rather than a literal
    value or catalog placeholder.

    Added 2026-07-31 per her explicit request: "for these kind of
    dependencies, its not possible to pass the id values beforehand, these
    should be carried out in outputs ... the model should be smart enough to
    ... create a new role" — once a field is auto-wired to a real resource
    reference, asking the user for it on the Clarify screen would just let a
    literal placeholder answer overwrite a value that's already correct.
    Mirrors arch2terraform's hcl_format.hcl_value() reference-detection rule
    (var./data./aws_*.NAME/random_*.NAME) so this stays in lockstep with
    whatever the generator itself treats as an unquoted reference, rather
    than maintaining a second, driftable definition of "looks like a wire."
    """
    def _is_ref_str(s) -> bool:
        return isinstance(s, str) and (
            s.startswith("var.")
            or s.startswith("data.")
            or (s.startswith(("aws_", "random_")) and "." in s)
        )

    if isinstance(value, list):
        return bool(value) and all(_is_ref_str(v) for v in value)
    return _is_ref_str(value)


def _looks_like_placeholder(value) -> bool:
    """True if `value` is empty, an old-style '# TODO' comment, or matches
    one of arch2terraform's catalog placeholder conventions — i.e. a value
    that will pass `terraform validate` but should still be replaced with a
    real one before `apply`."""
    if value is None:
        return True
    s = str(value)
    if not s or s.startswith("#"):
        return True
    lowered = s.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


# ── Global questions (always asked once, not per-resource) ───────────────────
GLOBAL_FIELDS: list[tuple] = [
    ("aws_region", "target_global", "Target AWS region?", "select",
     ["us-east-1","us-west-2","eu-west-1","eu-central-1","ap-southeast-1","ap-northeast-1"], "us-east-1"),
    ("environment", "target_global", "Deployment environment?", "select",
     ["dev","staging","prod"], "dev"),
    ("project_name", "target_global", "Project/product name? (used for tagging)", "text", [], "my-project"),
]


def detect_missing_info(
    parsed: ParsedDiagram, job_id: str, input_vars: dict | None = None,
) -> tuple[ClarificationRequest | None, list[ClarificationAnswer]]:
    """
    Inspect parsed diagram and build a ClarificationRequest, PLUS any answers
    that can be silently auto-filled from `input_vars` (an already-uploaded
    vars.yaml — see Job.input_vars's docstring). Returns
    (None-or-ClarificationRequest, auto_answers) — callers must merge
    auto_answers into job.clarification_answers themselves (this function
    stays a pure ParsedDiagram → outputs function, no Job access).
    """
    fields: list[ClarificationField] = []
    auto_answers: list[ClarificationAnswer] = []
    seen_keys: set[str] = set()

    vars_globals: dict = (input_vars or {}).get("globals") or {}
    vars_resources: dict = (input_vars or {}).get("resources") or {}

    # 1. Global fields (always ask once, unless vars.yaml already covers it)
    for (fkey, res_id, question, itype, opts, default) in GLOBAL_FIELDS:
        uid = f"{res_id}::{fkey}"
        if uid in seen_keys:
            continue
        seen_keys.add(uid)
        if fkey in vars_globals:
            auto_answers.append(ClarificationAnswer(
                field_key=fkey, resource_id=res_id, value=str(vars_globals[fkey])
            ))
            continue
        fields.append(ClarificationField(
            field_key=fkey,
            resource_id=res_id,
            resource_label="Global settings",
            question=question,
            input_type=itype,
            options=opts,
            default=default,
            required=True,
        ))

    # 2. Low-confidence classifications → ask what it is (never auto-filled
    # from vars.yaml — reclassification isn't a config VALUE, it's a
    # different resource type entirely, out of scope for a values file)
    for resource in parsed.resources:
        if resource.confidence < _s.confidence_threshold:
            fkey = f"reclassify_{resource.id}"
            uid = f"{resource.id}::{fkey}"
            if uid not in seen_keys:
                seen_keys.add(uid)
                fields.append(ClarificationField(
                    field_key=fkey,
                    resource_id=resource.id,
                    resource_label=resource.label or resource.id,
                    question=f"Could not confidently identify '{resource.label}'. What AWS service is this?",
                    input_type="text",
                    options=[],
                    default=resource.aws_resource_type,
                    required=False,
                ))

    # 3. Per-resource mandatory fields. As of 2026-07-08 (her explicit
    # request), these are asked UNCONDITIONALLY when building from scratch —
    # not gated on "does the current value look like a placeholder" anymore.
    # That old gate is exactly why instance_type's real, non-placeholder
    # catalog default ("t3.micro") was never asked about: a valid default
    # isn't placeholder-shaped, so it silently sailed through regardless of
    # environment (dev/staging/prod). The only thing that now suppresses a
    # question is vars.yaml already covering that (resource, field) pair.
    for resource in parsed.resources:
        mandatory = MANDATORY_FIELDS.get(resource.aws_resource_type, [])
        rkey = _resource_key(resource)
        covered = vars_resources.get(rkey) or {}
        for (fkey, question_tmpl, itype, opts, default) in mandatory:
            # Record every MANDATORY_FIELDS-listed key present on this
            # resource as variable-worthy, regardless of whether it actually
            # gets asked below — terraform_planner.py's variable-ization has
            # always covered the full MANDATORY_FIELDS set unconditionally.
            if fkey in resource.properties and fkey not in resource.variableize_keys:
                resource.variableize_keys.append(fkey)

            uid = f"{resource.id}::{fkey}"
            if uid in seen_keys:
                continue
            if fkey in covered:
                seen_keys.add(uid)
                auto_answers.append(ClarificationAnswer(
                    field_key=fkey, resource_id=resource.id, value=str(covered[fkey])
                ))
                continue
            seen_keys.add(uid)
            question = question_tmpl.format(label=resource.label or resource.logical_name)
            fields.append(ClarificationField(
                field_key=fkey,
                resource_id=resource.id,
                resource_label=resource.label or resource.logical_name,
                question=question,
                input_type=itype,
                options=opts,
                default=default,
                required=(not default),
            ))

    # 4. Generic fallback, extended 2026-07-08: previously only asked about
    # properties that both (a) had no bespoke MANDATORY_FIELDS entry and
    # (b) currently looked like a catalog placeholder. Now also asks about
    # any property matching a KNOWN catalog default_attributes key for that
    # resource type (_CATALOG_DEFAULT_KEYS) even when its current value is a
    # perfectly valid non-placeholder default — same "ask real config knobs
    # unconditionally when building from scratch" change as section 3,
    # extended to the ~25 resource types with no bespoke question set. The
    # placeholder-shape check is kept as a defensive fallback for values
    # that don't match a known catalog key (e.g. something set dynamically
    # elsewhere in the pipeline) — still variable-worthy, just not
    # guaranteed to be a "real" catalog config knob.
    for resource in parsed.resources:
        already_asked = {fkey for (fkey, *_rest) in MANDATORY_FIELDS.get(resource.aws_resource_type, [])}
        catalog_default_keys = _CATALOG_DEFAULT_KEYS.get(resource.aws_resource_type, set())
        rkey = _resource_key(resource)
        covered = vars_resources.get(rkey) or {}
        for prop_key, prop_val in resource.properties.items():
            if prop_key in already_asked or prop_key.startswith("_"):
                continue  # covered by the bespoke question above already
            if _looks_like_wired_reference(prop_val):
                continue  # already auto-wired to a real resource reference — nothing to ask
            if _MANDATORY_ROLE_WIRED_FIELD_BY_RESOURCE_TYPE.get(resource.aws_resource_type) == prop_key:
                continue  # always overwritten later by the security engine's generated role — nothing to ask
            if isinstance(prop_val, list):
                # Real bug found 2026-08-24 via `terraform init` ("Invalid
                # variable name"/type-mismatch trail that traced back to
                # this exact spot): a list-valued catalog default (e.g.
                # aws_autoscaling_group's vpc_zone_identifier, a list of
                # subnet ids) reaching this generic fallback got offered as
                # an `input_type="text"` field with `default=str(prop_val)`
                # a few lines below — str()'ing a Python list produces
                # `"['subnet-...']"`, which is neither a valid single
                # subnet id nor anything a one-line text box can sensibly
                # edit. A resource type with a real diagram-driven wiring
                # pass for this attribute (see arch2terraform's
                # _wire_lb_subnets/_wire_eks_node_group_refs) never reaches
                # here at all — _looks_like_wired_reference already skips
                # those above once wired. This is the fallback for when no
                # such wiring pass exists yet (or the diagram genuinely has
                # nothing to wire it to): leave the catalog placeholder list
                # as a literal rather than asking a question that can only
                # produce a worse value than what's already there.
                continue
            is_catalog_default = (
                prop_key in catalog_default_keys
                and prop_key not in _NEVER_ASK_CATALOG_DEFAULTS
            )
            if not is_catalog_default and not _looks_like_placeholder(prop_val):
                continue  # neither a known config knob nor placeholder-shaped

            # Fed into terraform_planner.py's _variableize_mandatory_fields()
            # the same way as MANDATORY_FIELDS keys, so these resource types
            # get real tfvars-overridable variables, not baked literals.
            if prop_key not in resource.variableize_keys:
                resource.variableize_keys.append(prop_key)

            uid = f"{resource.id}::{prop_key}"
            if uid in seen_keys:
                continue
            if prop_key in covered:
                seen_keys.add(uid)
                auto_answers.append(ClarificationAnswer(
                    field_key=prop_key, resource_id=resource.id, value=str(covered[prop_key])
                ))
                continue
            seen_keys.add(uid)
            label = resource.label or resource.logical_name
            fields.append(ClarificationField(
                field_key=prop_key,
                resource_id=resource.id,
                resource_label=label,
                question=f"Real value needed for '{prop_key}' on {label}?",
                input_type="text",
                options=[],
                default=(str(prop_val) if is_catalog_default else None),
                required=(not is_catalog_default),
            ))

    request = ClarificationRequest(job_id=job_id, fields=fields) if fields else None
    return request, auto_answers


def generate_vars_yaml(parsed: ParsedDiagram, answers: list[ClarificationAnswer]) -> str:
    """
    Build a vars.yaml artifact from a job's FINALIZED clarification answers
    (whether collected fully from scratch, or gap-filled on top of an
    uploaded vars.yaml) — the write-side counterpart to detect_missing_info()'s
    `input_vars` read-side parameter. Deliberately keyed by
    "<aws_resource_type>.<logical_name>" rather than the final Terraform
    variable name: those names are only settled during planning
    (terraform_planner.py's _variableize_mandatory_fields() disambiguates
    same-type-different-value collisions like cidr_block /
    cidr_block_public_subnet AFTER clarification is done, not before), so
    resource+field is the only key available at this point that both (a)
    unambiguously identifies which value belongs to which resource and (b)
    round-trips across re-uploads of an updated diagram — logical_name is
    derived from the diagram label, not parse-order-dependent like
    resource.id ("r1"/"r2"), which can shift between parses.
    """
    by_id = {r.id: r for r in parsed.resources}
    resources_out: dict[str, dict[str, str]] = {}
    globals_out: dict[str, str] = {}

    for ans in answers:
        if ans.field_key.startswith("reclassify_"):
            continue  # a resource-type correction, not a Terraform variable
        if ans.resource_id == "target_global":
            globals_out[ans.field_key] = ans.value
            continue
        resource = by_id.get(ans.resource_id)
        if not resource:
            continue  # stale answer for a resource no longer in the diagram
        rkey = _resource_key(resource)
        resources_out.setdefault(rkey, {})[ans.field_key] = ans.value

    payload: dict = {}
    if globals_out:
        payload["globals"] = globals_out
    if resources_out:
        payload["resources"] = resources_out

    return yaml.dump(payload, sort_keys=True, default_flow_style=False)


def _coerce_answer_value(raw: str, original_value):
    """
    Clarification answers always arrive as `str` (ClarificationAnswer.value's
    type — every field in the UI, including untouched ones that just carry
    the catalog default rendered via `str(prop_val)`, round-trips as text).
    Found for real 2026-07-28: `aws_instance.ebs_optimized` (a Python `bool`
    catalog default, added 2026-07-24 for checkov compliance) got silently
    turned into the string `"True"` by this function's old unconditional
    `resource.properties[fkey] = value` assignment, which then made
    terraform_planner.py's `_hcl_type_for()` — itself correct — misdetect it
    as a plain string and wrap it in quotes, producing invalid HCL
    (`ebs_optimized = "True"`, a bool required). Coerces the raw string back
    to `original_value`'s type (bool/int/float) before it overwrites
    `resource.properties`, so type information survives the
    property → clarification-answer → property round-trip intact. Falls back
    to the raw string unchanged for anything that isn't bool/int/float (AMI
    IDs, engine names, etc. — the overwhelming majority of fields), and for
    values that don't actually parse as the expected type (never silently
    drops the user's answer).
    """
    if isinstance(original_value, bool):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return raw  # unparseable — leave as the raw string rather than guess
    if isinstance(original_value, int):
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    if isinstance(original_value, float):
        try:
            return float(raw.strip())
        except ValueError:
            return raw
    return raw


def apply_clarification_answers(
    parsed: ParsedDiagram,
    answers: list,
) -> ParsedDiagram:
    """
    Apply user-provided answers back into ParsedResources.
    Mutates and returns the diagram.
    """
    # Build lookup: (resource_id, field_key) → value
    ans_map: dict[tuple[str, str], str] = {
        (a.resource_id, a.field_key): a.value
        for a in answers
    }

    for resource in parsed.resources:
        for (res_id, fkey), value in ans_map.items():
            if res_id == resource.id:
                # Strip stray whitespace before it ever reaches generated HCL —
                # found for real 2026-07-08: a copy-pasted AMI ID with a
                # trailing space passed straight through into the resource
                # block ("ami-002192a70217ac181 "), which AWS's API rejected
                # as InvalidAMIID.Malformed at real `terraform apply` time.
                # terraform validate doesn't catch this (a string with
                # trailing whitespace is still syntactically a valid string).
                if isinstance(value, str):
                    value = value.strip()
                if fkey.startswith("reclassify_"):
                    # User corrected the resource type
                    resource.aws_resource_type = value
                    resource.confidence = 1.0
                else:
                    # Coerce back to the original property's type (bool/
                    # number) — see _coerce_answer_value's docstring for the
                    # real ebs_optimized bug this fixes.
                    original = resource.properties.get(fkey)
                    resource.properties[fkey] = _coerce_answer_value(value, original)

    return parsed
