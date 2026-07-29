"""
Missing Info Detector — unit tests
------------------------------------
Covers two real fixes made 2026-07-08, both found via her actual local
`terraform apply` run against a generated package (see git history /
memory for the full incident):

1. A clarification answer with a trailing space ("ami-002192a70217ac181 ")
   passed straight through into generated HCL and made AWS reject the AMI
   as InvalidAMIID.Malformed at real apply time — `terraform validate`
   doesn't catch stray whitespace in a string, so this only ever surfaced
   against a real AWS account. Fixed by stripping string answers in
   apply_clarification_answers() before they're stored.

2. MANDATORY_FIELDS only hand-covered ~12 of the catalog's ~59 resource
   types, so most placeholder default_attributes (aws_ecr_repository's
   "replace-with-repository-name", aws_dynamodb_table's
   "replace-with-table-name", etc.) were never surfaced to the user for
   input at all. Fixed with a generic catalog-wide placeholder-detection
   fallback in detect_missing_info() (step 4) — see that function's
   docstring/comments for the full reasoning.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import ParsedDiagram, ParsedResource, ClarificationAnswer, DiagramFormat
from app.services.parser.missing_info_detector import (
    detect_missing_info, apply_clarification_answers, generate_vars_yaml,
)


def _diagram(resources) -> ParsedDiagram:
    return ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=resources, connections=[])


def test_trailing_whitespace_in_answer_is_stripped_before_storage():
    """The exact real-world repro: a copy-pasted AMI ID with a trailing
    space must not survive into resource.properties."""
    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-00000000000000000", "instance_type": "t3.micro"},
    )
    diagram = _diagram([r])
    answers = [ClarificationAnswer(field_key="ami", resource_id="ec2-1", value="ami-002192a70217ac181 ")]
    result = apply_clarification_answers(diagram, answers)

    stored = result.resources[0].properties["ami"]
    assert stored == "ami-002192a70217ac181"
    assert not stored.endswith(" ")


def test_numeric_answer_coerced_back_to_original_int_type():
    """Fixed 2026-07-28 alongside the ebs_optimized bool bug: an answer's
    raw string is coerced back to the original property's type (here int)
    before overwriting, not left as a string — otherwise
    terraform_planner.py's _hcl_type_for() misdetects it as a plain string
    and quotes it in generated HCL (`allocated_storage = "50"`, a number
    required, mirroring the ebs_optimized failure for numeric fields)."""
    r = ParsedResource(
        id="db-1", aws_resource_type="aws_db_instance", logical_name="db",
        label="DB", properties={"allocated_storage": 20},
    )
    diagram = _diagram([r])
    answers = [ClarificationAnswer(field_key="allocated_storage", resource_id="db-1", value="50")]
    result = apply_clarification_answers(diagram, answers)
    assert result.resources[0].properties["allocated_storage"] == 50
    assert isinstance(result.resources[0].properties["allocated_storage"], int)


def test_bool_answer_coerced_back_to_python_bool_not_string():
    """The real bug reported 2026-07-28: a real `terraform plan` against a
    generated package failed with `Inappropriate value for attribute
    "ebs_optimized": a bool is required` because the resource's catalog
    default (Python `True`) was being overwritten with the string "True"
    (the untouched clarification field's rendered default, `str(prop_val)`
    from missing_info_detector's generic fallback) — which
    terraform_planner.py's `_hcl_type_for()` then misdetected as a string
    and wrapped in quotes. Also covers the "false" spelling and case
    variants, since real answers may come from a select/checkbox UI."""
    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server",
        properties={"ami": "ami-00000000000000000", "ebs_optimized": True, "monitoring": True},
    )
    diagram = _diagram([r])
    answers = [
        ClarificationAnswer(field_key="ebs_optimized", resource_id="ec2-1", value="True"),
        ClarificationAnswer(field_key="monitoring", resource_id="ec2-1", value="false"),
    ]
    result = apply_clarification_answers(diagram, answers)

    ebs = result.resources[0].properties["ebs_optimized"]
    mon = result.resources[0].properties["monitoring"]
    assert ebs is True and isinstance(ebs, bool)
    assert mon is False and isinstance(mon, bool)


def test_generic_fallback_asks_about_placeholder_on_uncovered_resource_type():
    """aws_ecr_repository has no MANDATORY_FIELDS entry at all — the
    generic fallback (step 4) must still catch its placeholder 'name'."""
    r = ParsedResource(
        id="ecr-1", aws_resource_type="aws_ecr_repository", logical_name="my_repo",
        label="My ECR Repo", properties={"name": "replace-with-repository-name"},
    )
    clar, _ = detect_missing_info(_diagram([r]), "job1")
    assert clar is not None
    matching = [f for f in clar.fields if f.resource_id == "ecr-1" and f.field_key == "name"]
    assert len(matching) == 1
    # Wording dropped "(currently a generated placeholder)" 2026-07-08 — the
    # generic fallback now also asks about genuine non-placeholder catalog
    # defaults (the "ask all" change), so that phrasing would be misleading
    # for those cases. Just confirm the question references the real field.
    assert "name" in matching[0].question.lower()


def test_generic_fallback_does_not_duplicate_a_mandatory_fields_question():
    """aws_instance's 'ami' IS in MANDATORY_FIELDS — the generic fallback
    (step 4) must not ask about it a second time."""
    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-00000000000000000", "instance_type": "t3.micro"},
    )
    clar, _ = detect_missing_info(_diagram([r]), "job1")
    ami_questions = [f for f in clar.fields if f.resource_id == "ec2-1" and f.field_key == "ami"]
    assert len(ami_questions) == 1


def test_generic_fallback_does_not_fire_on_real_non_placeholder_values():
    """aws_db_instance's 'username': 'admin' is a real, valid, non-placeholder
    default (and deliberately never asked — see catalog.py) — the generic
    fallback must not start asking about every single property, only ones
    that actually look like placeholders."""
    r = ParsedResource(
        id="db-1", aws_resource_type="aws_db_instance", logical_name="db",
        label="Postgres DB",
        properties={
            "engine": "postgres", "instance_class": "db.t3.micro",
            "allocated_storage": 20, "username": "admin",
            "manage_master_user_password": True,
        },
    )
    clar, _ = detect_missing_info(_diagram([r]), "job1")
    usernames_asked = [f for f in (clar.fields if clar else []) if f.field_key == "username"]
    assert usernames_asked == []


# ─────────────────────────────────────────────────────────────────────────────
# vars.yaml: "ask everything from scratch" vs. "reuse an existing vars.yaml"
# (her explicit distinction, 2026-07-08 — this is what actually fixes the
# EC2/RDS sizing gap: instance_type's real catalog default "t3.micro" isn't
# placeholder-shaped, so the old placeholder-only gate never asked about it,
# meaning sizing was silently always t3.micro regardless of environment).
# ─────────────────────────────────────────────────────────────────────────────

def test_from_scratch_asks_about_instance_type_despite_valid_default():
    """The exact real bug she found: instance_type="t3.micro" is a real,
    valid, non-placeholder catalog default — under the OLD placeholder-only
    gate this was never asked about. With no vars.yaml (building from
    scratch), it must now be asked unconditionally."""
    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-00000000000000000", "instance_type": "t3.micro"},
    )
    clar, auto_answers = detect_missing_info(_diagram([r]), "job1")
    assert auto_answers == []
    instance_type_q = next((f for f in clar.fields if f.resource_id == "ec2-1" and f.field_key == "instance_type"), None)
    assert instance_type_q is not None
    assert instance_type_q.default == "t3.micro"


def test_from_scratch_asks_about_non_placeholder_generic_fallback_catalog_default():
    """Same 'ask real config knobs unconditionally' extension applied to the
    ~25 generic-fallback-covered resource types: aws_kinesis_stream's
    shard_count=1 is a real catalog default_attributes entry, not
    placeholder-shaped — must now be asked about too when building from
    scratch, with its current value surfaced as the suggested default."""
    r = ParsedResource(
        id="kinesis-1", aws_resource_type="aws_kinesis_stream", logical_name="events",
        label="Events Stream", properties={"shard_count": 1},
    )
    clar, _ = detect_missing_info(_diagram([r]), "job1")
    shard_q = next((f for f in clar.fields if f.resource_id == "kinesis-1" and f.field_key == "shard_count"), None)
    assert shard_q is not None
    assert shard_q.default == "1"


def test_input_vars_prefills_covered_field_and_skips_asking_it():
    """When vars.yaml covers a (resource, field) pair, it must come back as
    an auto_answer, NOT a question — this is the 'reuse an existing
    vars.yaml' case."""
    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-00000000000000000", "instance_type": "t3.micro"},
    )
    input_vars = {"resources": {"aws_instance.web_server": {"instance_type": "m5.large"}}}
    clar, auto_answers = detect_missing_info(_diagram([r]), "job1", input_vars=input_vars)

    instance_type_answer = next((a for a in auto_answers if a.resource_id == "ec2-1" and a.field_key == "instance_type"), None)
    assert instance_type_answer is not None
    assert instance_type_answer.value == "m5.large"

    instance_type_q = next((f for f in clar.fields if f.resource_id == "ec2-1" and f.field_key == "instance_type"), None)
    assert instance_type_q is None  # not asked — vars.yaml already covers it

    # "ami" isn't covered by this vars.yaml — must still be a real gap question.
    ami_q = next((f for f in clar.fields if f.resource_id == "ec2-1" and f.field_key == "ami"), None)
    assert ami_q is not None


def test_input_vars_globals_prefill_skips_global_questions():
    r = ParsedResource(
        id="kms-1", aws_resource_type="aws_kms_key", logical_name="kms",
        label="KMS Key", properties={},
    )
    input_vars = {"globals": {"aws_region": "eu-west-1", "environment": "prod", "project_name": "infra-genie"}}
    clar, auto_answers = detect_missing_info(_diagram([r]), "job1", input_vars=input_vars)

    global_answers = {a.field_key: a.value for a in auto_answers if a.resource_id == "target_global"}
    assert global_answers == {"aws_region": "eu-west-1", "environment": "prod", "project_name": "infra-genie"}
    assert clar is None  # aws_kms_key has no catalog defaults/mandatory fields — nothing left to ask


def test_generate_vars_yaml_round_trips_through_input_vars():
    """The write side (generate_vars_yaml) and read side (detect_missing_info's
    input_vars) must agree on the same key format — build one job's answers
    into a vars.yaml, parse it back, and confirm feeding it in as input_vars
    correctly pre-fills the same fields without asking again."""
    import yaml

    r = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-00000000000000000", "instance_type": "t3.micro"},
    )
    diagram = _diagram([r])
    answers = [
        ClarificationAnswer(field_key="ami", resource_id="ec2-1", value="ami-0123456789abcdef0"),
        ClarificationAnswer(field_key="instance_type", resource_id="ec2-1", value="m5.xlarge"),
        ClarificationAnswer(field_key="aws_region", resource_id="target_global", value="us-west-2"),
    ]
    yaml_text = generate_vars_yaml(diagram, answers)
    parsed_back = yaml.safe_load(yaml_text)

    assert parsed_back["resources"]["aws_instance.web_server"]["ami"] == "ami-0123456789abcdef0"
    assert parsed_back["resources"]["aws_instance.web_server"]["instance_type"] == "m5.xlarge"
    assert parsed_back["globals"]["aws_region"] == "us-west-2"

    # Feed it back in — this second "run" should ask about neither field.
    clar2, auto_answers2 = detect_missing_info(_diagram([r]), "job2", input_vars=parsed_back)
    asked_keys = {f.field_key for f in (clar2.fields if clar2 else [])}
    assert "ami" not in asked_keys
    assert "instance_type" not in asked_keys
    auto_values = {a.field_key: a.value for a in auto_answers2}
    assert auto_values["ami"] == "ami-0123456789abcdef0"
    assert auto_values["instance_type"] == "m5.xlarge"


def test_generate_vars_yaml_skips_reclassify_answers():
    """A reclassify_ answer corrects the resource TYPE, not a config value —
    it isn't a real Terraform variable and shouldn't end up in vars.yaml."""
    r = ParsedResource(
        id="r1", aws_resource_type="aws_null_resource", logical_name="unknown",
        label="Unknown Box", confidence=0.3,
    )
    diagram = _diagram([r])
    answers = [ClarificationAnswer(field_key="reclassify_r1", resource_id="r1", value="aws_instance")]
    yaml_text = generate_vars_yaml(diagram, answers)
    assert yaml_text.strip() == "{}"
