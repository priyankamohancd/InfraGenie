"""
End-to-end integration test: real draw.io fixture -> full pipeline -> 5
Terraform files on disk -> (if the `terraform` binary is available)
a real `terraform init && terraform validate` run against the output.

The `terraform validate` step is skipped with a clear reason, not faked,
when the binary isn't installed (e.g. in network-restricted sandboxes).
On a normal dev machine with Terraform installed, this test actually
proves the generated HCL is syntactically and referentially valid.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from arch2terraform.pipeline import run_pipeline

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "drawio", "sample_architecture.drawio"
)

EXPECTED_FILES = {"provider.tf", "variables.tf", "main.tf", "outputs.tf", "README.md"}


@pytest.fixture
def output_dir():
    d = tempfile.mkdtemp(prefix="arch2tf_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_pipeline_produces_exactly_five_expected_files(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    produced = {os.path.basename(p) for p in result.files_written}
    assert produced == EXPECTED_FILES


def test_pipeline_classifies_expected_resources(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    resource_types = {r.resource_type for r in result.graph.resources}
    assert "aws_vpc" in resource_types
    assert "aws_subnet" in resource_types
    assert "aws_instance" in resource_types
    assert "aws_internet_gateway" in resource_types
    assert "aws_s3_bucket" in resource_types


def test_pipeline_resolves_containment_and_edges(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    rel_types = {r.relationship_type for r in result.graph.relationships}
    assert "containment" in rel_types
    # ec2 -> s3 and ec2 -> igw edges should resolve to *some* non-containment type
    assert rel_types - {"containment"}


def test_generated_main_tf_has_no_unresolved_placeholders(output_dir):
    result = run_pipeline(FIXTURE, output_dir)
    main_tf_path = os.path.join(output_dir, "main.tf")
    with open(main_tf_path) as f:
        content = f.read()
    assert "TODO" not in content
    assert "{{" not in content  # no leftover template syntax


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform binary not installed in this environment")
def test_generated_terraform_passes_real_validate(output_dir):
    """Real sandbox validation: terraform init (backend=false, no provider download
    blocked by network) + terraform validate against the generated HCL."""
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
