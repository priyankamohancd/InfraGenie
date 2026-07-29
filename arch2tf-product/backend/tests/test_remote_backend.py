"""
Remote State Backend — unit tests
-------------------------------------
Covers _backend_tf() in terraform_planner.py (added 2026-07-08, per her
explicit request to close the "drift detection needs real persisted state"
gap): when no TF_STATE_BUCKET is configured, the generated backend.tf must
stay exactly the pre-existing commented-out placeholder (backward
compatible — nothing breaks for anyone who hasn't set this up). When one
IS configured, it must emit a real, uncommented S3 backend block with
literal (not var.*) bucket/key/region values — Terraform backend blocks
are resolved before any variable exists, so a real backend block can never
reference `var.project`/`var.environment` the way the old placeholder
comment incorrectly showed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hcl2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import ParsedDiagram, ParsedResource, DiagramFormat
from app.services.planner.terraform_planner import build_terraform_plan
from app.core.config import get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings() is @lru_cache()'d — must clear before AND after every
    test in this file so env var changes actually take effect and don't
    leak into other test files' Settings()."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _one_resource_diagram() -> ParsedDiagram:
    instance = ParsedResource(
        id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
        label="Web Server", properties={"ami": "ami-0realvalue1234567", "instance_type": "t3.micro"},
    )
    return ParsedDiagram(source_format=DiagramFormat.DRAWIO, resources=[instance], connections=[])


@pytest.mark.anyio
async def test_no_state_bucket_configured_keeps_commented_placeholder(monkeypatch):
    monkeypatch.delenv("TF_STATE_BUCKET", raising=False)
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="test")
    backend_tf = plan.root_module_files["backend.tf"]

    assert 'backend "s3"' in backend_tf
    # Every real line must be commented out — no live `terraform {` block.
    for line in backend_tf.splitlines():
        stripped = line.strip()
        if stripped:
            assert stripped.startswith("#"), f"expected a comment, got: {line!r}"


@pytest.mark.anyio
async def test_state_bucket_configured_generates_real_backend_block(monkeypatch):
    monkeypatch.setenv("TF_STATE_BUCKET", "my-org-tfstate")
    monkeypatch.setenv("TF_STATE_LOCK_TABLE", "my-org-tf-locks")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="staging", project_name="myproj")
    backend_tf = plan.root_module_files["backend.tf"]

    assert "terraform {" in backend_tf
    assert 'backend "s3" {' in backend_tf
    assert 'bucket         = "my-org-tfstate"' in backend_tf
    assert 'key            = "myproj/staging/terraform.tfstate"' in backend_tf
    assert 'dynamodb_table = "my-org-tf-locks"' in backend_tf
    assert "encrypt        = true" in backend_tf
    # No var.* reference anywhere — backend blocks can't use variables.
    assert "var." not in backend_tf

    parsed = hcl2.loads(backend_tf)
    assert parsed["terraform"][0]["backend"][0]["s3"]["bucket"] == ["my-org-tfstate"]


@pytest.mark.anyio
async def test_different_environments_get_different_state_keys(monkeypatch):
    """The whole point: dev/staging/prod must never share (or clobber)
    each other's state file."""
    monkeypatch.setenv("TF_STATE_BUCKET", "my-org-tfstate")
    get_settings.cache_clear()

    dev_plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="myproj")
    prod_plan = await build_terraform_plan(_one_resource_diagram(), environment="prod", project_name="myproj")

    assert 'key            = "myproj/dev/terraform.tfstate"' in dev_plan.root_module_files["backend.tf"]
    assert 'key            = "myproj/prod/terraform.tfstate"' in prod_plan.root_module_files["backend.tf"]


@pytest.mark.anyio
async def test_state_region_falls_back_to_aws_region_when_unset(monkeypatch):
    monkeypatch.setenv("TF_STATE_BUCKET", "my-org-tfstate")
    monkeypatch.delenv("TF_STATE_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="myproj")
    assert 'region         = "eu-west-1"' in plan.root_module_files["backend.tf"]


# ── Terraform Cloud / HCP Terraform backend (2026-07-29) ────────────────────
# Same real-block-vs-commented-placeholder gating as the S3 tests above, just
# keyed off TF_BACKEND_TYPE=cloud + TF_CLOUD_ORGANIZATION instead of
# TF_STATE_BUCKET.

@pytest.mark.anyio
async def test_cloud_backend_type_without_org_keeps_commented_placeholder(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_TYPE", "cloud")
    monkeypatch.delenv("TF_CLOUD_ORGANIZATION", raising=False)
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="test")
    backend_tf = plan.root_module_files["backend.tf"]

    assert "cloud {" in backend_tf
    for line in backend_tf.splitlines():
        stripped = line.strip()
        if stripped:
            assert stripped.startswith("#"), f"expected a comment, got: {line!r}"


@pytest.mark.anyio
async def test_cloud_backend_configured_generates_real_cloud_block(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_TYPE", "cloud")
    monkeypatch.setenv("TF_CLOUD_ORGANIZATION", "my-tfc-org")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="staging", project_name="myproj")
    backend_tf = plan.root_module_files["backend.tf"]

    assert "terraform {" in backend_tf
    assert "cloud {" in backend_tf
    assert 'organization = "my-tfc-org"' in backend_tf
    assert 'name = "myproj-staging"' in backend_tf
    # No S3-specific block emitted alongside it, and no var.* reference —
    # backend/cloud blocks can't reference variables either.
    assert 'backend "s3"' not in backend_tf
    assert "var." not in backend_tf

    parsed = hcl2.loads(backend_tf)
    assert parsed["terraform"][0]["cloud"][0]["organization"] == ["my-tfc-org"]


@pytest.mark.anyio
async def test_cloud_backend_workspace_name_sanitized(monkeypatch):
    """Diagram-derived project names can contain characters TFC workspace
    names don't allow (spaces, slashes, etc.) — must be sanitized to
    letters/numbers/-/_ only."""
    monkeypatch.setenv("TF_BACKEND_TYPE", "cloud")
    monkeypatch.setenv("TF_CLOUD_ORGANIZATION", "my-tfc-org")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="My Cool Project!")
    backend_tf = plan.root_module_files["backend.tf"]

    assert 'name = "My-Cool-Project-dev"' in backend_tf


@pytest.mark.anyio
async def test_cloud_backend_custom_hostname_included(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_TYPE", "cloud")
    monkeypatch.setenv("TF_CLOUD_ORGANIZATION", "my-tfc-org")
    monkeypatch.setenv("TF_CLOUD_HOSTNAME", "tfe.internal.example.com")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="myproj")
    backend_tf = plan.root_module_files["backend.tf"]

    assert 'hostname     = "tfe.internal.example.com"' in backend_tf


@pytest.mark.anyio
async def test_cloud_backend_default_hostname_omitted(monkeypatch):
    """app.terraform.io is TFC's default — no need to spell it out."""
    monkeypatch.setenv("TF_BACKEND_TYPE", "cloud")
    monkeypatch.setenv("TF_CLOUD_ORGANIZATION", "my-tfc-org")
    monkeypatch.delenv("TF_CLOUD_HOSTNAME", raising=False)
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="myproj")
    backend_tf = plan.root_module_files["backend.tf"]

    assert "hostname" not in backend_tf


@pytest.mark.anyio
async def test_cloud_backend_omits_ci_state_access_policy():
    """The S3-specific IAM policy makes no sense for Terraform Cloud state
    (TFC governs access via teams, not AWS IAM) — must not be generated."""
    get_settings.cache_clear()
    import os
    os.environ["TF_BACKEND_TYPE"] = "cloud"
    os.environ["TF_CLOUD_ORGANIZATION"] = "my-tfc-org"
    os.environ["TF_STATE_BUCKET"] = "leftover-bucket-from-other-config"
    get_settings.cache_clear()
    try:
        plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="test")
        assert "ci_state_access_policy.tf" not in plan.root_module_files
    finally:
        os.environ.pop("TF_BACKEND_TYPE", None)
        os.environ.pop("TF_CLOUD_ORGANIZATION", None)
        os.environ.pop("TF_STATE_BUCKET", None)
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_no_state_bucket_configured_omits_ci_state_access_policy(monkeypatch):
    """2026-07-21: the least-privilege CI IAM policy only makes sense once
    there's a real backend to scope it to — no bucket means nothing to
    generate, same gate as _backend_tf's real-backend branch."""
    monkeypatch.delenv("TF_STATE_BUCKET", raising=False)
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="test")
    assert "ci_state_access_policy.tf" not in plan.root_module_files


@pytest.mark.anyio
async def test_state_bucket_configured_generates_scoped_ci_iam_policy(monkeypatch):
    monkeypatch.setenv("TF_STATE_BUCKET", "my-org-tfstate")
    monkeypatch.setenv("TF_STATE_LOCK_TABLE", "my-org-tf-locks")
    get_settings.cache_clear()

    plan = await build_terraform_plan(_one_resource_diagram(), environment="staging", project_name="myproj")
    policy_tf = plan.root_module_files.get("ci_state_access_policy.tf", "")
    assert policy_tf, "expected ci_state_access_policy.tf to be generated"

    parsed = hcl2.loads(policy_tf)
    assert 'resource "aws_iam_policy" "terraform_state_access"' in policy_tf
    assert 'data "aws_caller_identity" "ci_state_access"' in policy_tf

    # Scoped to exactly this project's prefix / objects / lock table — not
    # the whole bucket, not other projects sharing it.
    assert '"myproj/*"' in policy_tf
    assert "arn:aws:s3:::my-org-tfstate/myproj/*" in policy_tf
    assert "arn:aws:s3:::my-org-tfstate\"" in policy_tf  # bare bucket ARN for ListBucket
    assert "table/my-org-tf-locks" in policy_tf
    # Account id resolved dynamically, never hardcoded/guessed.
    assert "${data.aws_caller_identity.ci_state_access.account_id}" in policy_tf

    # Never broader than this one bucket/table - no wildcard resource.
    assert '"Resource": "*"' not in policy_tf.replace(" ", "")


@pytest.mark.anyio
async def test_ci_state_access_policy_different_projects_get_disjoint_prefixes(monkeypatch):
    monkeypatch.setenv("TF_STATE_BUCKET", "shared-tfstate")
    get_settings.cache_clear()

    plan_a = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="project-a")
    plan_b = await build_terraform_plan(_one_resource_diagram(), environment="dev", project_name="project-b")

    policy_a = plan_a.root_module_files["ci_state_access_policy.tf"]
    policy_b = plan_b.root_module_files["ci_state_access_policy.tf"]
    assert "project-a/*" in policy_a and "project-b/*" not in policy_a
    assert "project-b/*" in policy_b and "project-a/*" not in policy_b
