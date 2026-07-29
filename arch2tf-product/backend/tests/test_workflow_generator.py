"""
GitHub Actions Terraform workflow generator — unit tests.

Covers workflow_generator.py (added 2026-07-22, closing the "GitHub Actions
workflow YAML" gap flagged in earlier sessions — OIDC auth and
auto-apply-on-merge-gated-by-required-reviewers were decided back then but
the workflow file itself was never built). No real GitHub Actions runner is
available in this sandbox, so these tests confirm: the output is valid YAML
(via PyYAML), the OIDC/environment-gating design elements are actually
present, and the matrix covers the expected environments — not that the
workflow runs correctly on real GitHub, which would need a real repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.services.github.workflow_generator import (
    DEFAULT_ENVIRONMENTS,
    generate_terraform_workflow,
)


def test_output_is_valid_yaml():
    text = generate_terraform_workflow(default_branch="main")
    parsed = yaml.safe_load(text)
    assert parsed["name"] == "Terraform"
    # PyYAML parses the `on:` key as boolean True (YAML 1.1 quirk) - this is
    # a real property of the generated file, not a bug in the test.
    assert True in parsed or "on" in parsed


def test_default_matrix_covers_dev_staging_prod():
    text = generate_terraform_workflow(default_branch="main")
    for env in DEFAULT_ENVIRONMENTS:
        assert f"- {env}" in text
    assert DEFAULT_ENVIRONMENTS == ["dev", "staging", "prod"]


def test_custom_environments_replace_default_matrix():
    text = generate_terraform_workflow(default_branch="main", environments=["qa", "sandbox"])
    assert "- qa" in text
    assert "- sandbox" in text
    assert "- dev" not in text
    assert "- prod" not in text


def test_uses_oidc_not_static_credentials():
    text = generate_terraform_workflow(default_branch="main")
    assert "id-token: write" in text
    assert "role-to-assume: ${{ vars.AWS_OIDC_ROLE_ARN }}" in text
    # No static-key input names anywhere - would be a real regression back
    # to long-lived credentials if these ever appeared.
    assert "aws-access-key-id" not in text
    assert "aws-secret-access-key" not in text


def test_apply_job_gated_by_matching_github_environment():
    """The actual required-reviewers gate: each apply matrix entry's
    `environment:` key must reference the matrix variable, not a hardcoded
    name - so dev/staging/prod are each gated by their OWN Environment's
    protection rules, not a shared/incorrect one."""
    text = generate_terraform_workflow(default_branch="main")
    parsed = yaml.safe_load(text)
    apply_job = parsed["jobs"]["apply"]
    assert apply_job["environment"] == "${{ matrix.environment }}"
    assert apply_job["strategy"]["matrix"]["environment"] == DEFAULT_ENVIRONMENTS


def test_plan_runs_on_pull_request_apply_runs_on_push_to_default_branch():
    text = generate_terraform_workflow(default_branch="release")
    parsed = yaml.safe_load(text)
    assert parsed["jobs"]["plan"]["if"] == "github.event_name == 'pull_request'"
    assert (
        parsed["jobs"]["apply"]["if"]
        == "github.event_name == 'push' && github.ref == 'refs/heads/release'"
    )
    on_key = parsed.get("on", parsed.get(True))
    assert on_key["push"]["branches"] == ["release"]


def test_working_directory_scoped_to_tf_subdir_and_environment():
    text = generate_terraform_workflow(default_branch="main", tf_subdir="infra")
    parsed = yaml.safe_load(text)
    assert parsed["jobs"]["plan"]["defaults"]["run"]["working-directory"] == (
        "infra/${{ matrix.environment }}"
    )
    assert parsed["jobs"]["apply"]["defaults"]["run"]["working-directory"] == (
        "infra/${{ matrix.environment }}"
    )
    # Path trigger filters must match the same tf_subdir, or a real push
    # would never actually trigger this workflow.
    on_key = parsed.get("on", parsed.get(True))
    assert on_key["pull_request"]["paths"] == ["infra/**"]
    assert on_key["push"]["paths"] == ["infra/**"]


def test_aws_region_threaded_into_both_jobs():
    text = generate_terraform_workflow(default_branch="main", aws_region="eu-west-1")
    parsed = yaml.safe_load(text)
    for job_name in ("plan", "apply"):
        steps = parsed["jobs"][job_name]["steps"]
        aws_step = next(s for s in steps if s["uses"].startswith("aws-actions/configure-aws-credentials"))
        assert aws_step["with"]["aws-region"] == "eu-west-1"


def test_header_comment_documents_one_time_setup():
    """The workflow will fail with a confusing error until the user does
    the AWS OIDC role + GitHub Environment setup - the file's own header
    must say so, not fail silently/mysteriously on first run."""
    text = generate_terraform_workflow(default_branch="main")
    assert "AWS_OIDC_ROLE_ARN" in text
    assert "GitHub Environment" in text or "Environments" in text
