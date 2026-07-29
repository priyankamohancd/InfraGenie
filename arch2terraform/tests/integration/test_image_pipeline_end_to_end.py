"""
End-to-end integration test: real AWS-icon-style PNG fixture -> full
5-stage image cascade (layout -> phash -> NCC -> OCR -> edges) -> classifier
-> resolver -> 5 Terraform files on disk.

This fixture (aws_icon_diagram.png) is a real diagram: 1 AWS Cloud boundary,
1 VPC, 2 Availability Zones, 3 subnet bands, and 8 service icons (Route 53,
2x Elastic Load Balancing, 2x EC2, Lambda, RDS, S3). It exists to catch three
regressions found while wiring the image adapter into the production
pipeline (see registry.py / classifier.py / ocr_extractor.py comments):

1. AWS-Cloud and Availability-Zone containers (which have no Terraform
   resource equivalent) must NOT produce phantom `aws_vpc` resources via the
   classifier's generic "unmatched container -> vpc" fallback.
2. Icon matching must tolerate hyphenated image_ref values (e.g.
   "Security-Group") the same way it tolerates unseparated ones (e.g.
   draw.io's "mxgraph.aws4.securityGroup").
3. Tesseract reliably hallucinates short "words" out of the dashed
   AZ/Subnet border lines these diagrams always have; those false positives
   must not leak into resource names/labels.

Icon identification for the 2 Elastic Load Balancing icons in this fixture
requires Stage 3 (NCC), which needs the real aws-icons asset pack
(set via ARCH2TERRAFORM_ICONS_DIR — see registry.py). That pack isn't
bundled with the repo, so assertions here only rely on phash-level
(Stage 2) matches, which work with just the committed reference_hashes.pkl
and hold regardless of whether ARCH2TERRAFORM_ICONS_DIR is configured in
the environment running the tests.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from arch2terraform.pipeline import run_pipeline

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "images", "aws_icon_diagram.png"
)

EXPECTED_FILES = {"provider.tf", "variables.tf", "main.tf", "outputs.tf", "README.md"}

# Provider-required arguments (no valid default) that must be present for
# `terraform validate` to pass on the current catalog defaults.
_REQUIRED_ARGS_BY_TYPE = {
    "aws_instance": {"ami", "instance_type"},
    "aws_lambda_function": {"function_name", "role", "filename"},
    "aws_route53_zone": {"name"},
    "aws_db_instance": {"allocated_storage", "username"},
}

# S3 bucket names: lowercase letters, digits, dots, hyphens only.
_S3_BUCKET_NAME_RE = re.compile(r"^[a-z0-9.-]{3,63}$")


@pytest.fixture
def output_dir():
    d = tempfile.mkdtemp(prefix="arch2tf_image_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pipeline_produces_exactly_five_expected_files(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    produced = {os.path.basename(p) for p in result.files_written}
    assert produced == EXPECTED_FILES


def test_exactly_one_vpc_no_phantom_containers(output_dir):
    """Regression test: AWS-Cloud + 2 Availability Zones must not become
    3 extra low-confidence aws_vpc resources alongside the real VPC."""
    result = run_pipeline(FIXTURE, output_dir)
    vpcs = [r for r in result.graph.resources if r.resource_type == "aws_vpc"]
    assert len(vpcs) == 1
    assert vpcs[0].confidence >= 0.9  # the real VPC, matched by icon, not the fallback


def test_all_three_subnet_bands_classified(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    subnets = [r for r in result.graph.resources if r.resource_type == "aws_subnet"]
    assert len(subnets) == 3
    for s in subnets:
        assert s.confidence >= 0.9  # matched via image_ref="Subnet", not label guessing


def test_subnets_wired_to_the_single_vpc(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    vpc = next(r for r in result.graph.resources if r.resource_type == "aws_vpc")
    containment = [
        rel for rel in result.graph.relationships
        if rel.relationship_type == "containment" and rel.source_node_id == vpc.node_id
    ]
    subnet_ids = {r.node_id for r in result.graph.resources if r.resource_type == "aws_subnet"}
    contained_subnets = {rel.target_node_id for rel in containment} & subnet_ids
    assert contained_subnets == subnet_ids


def test_phash_identifiable_services_classified(output_dir):
    """These 6 icons are identifiable from the committed reference_hashes.pkl
    alone (no aws-icons asset pack / Stage 3 needed), so this holds in any
    environment, including CI without ARCH2TERRAFORM_ICONS_DIR set."""
    result = run_pipeline(FIXTURE, output_dir)
    resource_types = {r.resource_type for r in result.graph.resources}
    assert "aws_route53_zone" in resource_types
    assert "aws_db_instance" in resource_types
    assert "aws_s3_bucket" in resource_types
    assert "aws_instance" in resource_types
    assert "aws_lambda_function" in resource_types


def test_no_ocr_noise_in_resource_names(output_dir):
    """Regression test: Tesseract hallucinates short words from dashed
    container borders (observed: 'SS', 'Log'). None of those should end up
    as a resource's terraform_name — subnet bands have no real text label in
    this fixture, so their names should stay UUID-derived, not word-like."""
    result = run_pipeline(FIXTURE, output_dir)
    subnets = [r for r in result.graph.resources if r.resource_type == "aws_subnet"]
    for s in subnets:
        assert not re.fullmatch(r"[a-z]{2,4}", s.terraform_name), (
            f"subnet {s.terraform_name!r} looks like OCR noise, not a UUID-derived name"
        )


def test_required_provider_arguments_present(output_dir):
    """Every resource type with a known required-argument set must actually
    emit those arguments — otherwise the generated HCL is syntactically
    valid but fails real `terraform validate`."""
    result = run_pipeline(FIXTURE, output_dir)
    for resource in result.graph.resources:
        required = _REQUIRED_ARGS_BY_TYPE.get(resource.resource_type)
        if not required:
            continue
        missing = required - set(resource.attributes)
        assert not missing, f"{resource.resource_type} missing required args: {missing}"


def test_s3_bucket_name_placeholder_is_valid_format(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    buckets = [r for r in result.graph.resources if r.resource_type == "aws_s3_bucket"]
    assert buckets
    for b in buckets:
        assert _S3_BUCKET_NAME_RE.match(b.attributes["bucket"]), (
            f"bucket name {b.attributes['bucket']!r} violates S3 naming rules "
            "and would fail terraform validate"
        )


def test_generated_main_tf_has_no_unresolved_placeholders(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    with open(os.path.join(output_dir, "main.tf")) as f:
        content = f.read()
    assert "TODO" not in content
    assert "{{" not in content


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform binary not installed in this environment")
def test_generated_terraform_passes_real_validate(output_dir):
    """Real sandbox validation: terraform init (backend=false) + validate."""
    run_pipeline(FIXTURE, output_dir)

    init = subprocess.run(
        ["terraform", "init", "-backend=false", "-input=false"],
        cwd=output_dir, capture_output=True, text=True, timeout=120,
    )
    assert init.returncode == 0, f"terraform init failed:\n{init.stdout}\n{init.stderr}"

    validate = subprocess.run(
        ["terraform", "validate"],
        cwd=output_dir, capture_output=True, text=True, timeout=60,
    )
    assert validate.returncode == 0, f"terraform validate failed:\n{validate.stdout}\n{validate.stderr}"
