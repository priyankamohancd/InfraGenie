"""
Apply-to-Sandbox Runner — unit tests
---------------------------------------
Focused tests for app/services/apply/apply_runner.py's orchestration logic:
preflight placeholder blocking, the plan -> confirm-token -> apply gate,
single-use/expiring tokens, and destroy + auto-destroy scheduling.

No real `terraform` binary or AWS account is used or required — every test
monkeypatches `apply_runner._run_subprocess` with a fake async function that
returns canned (returncode, stdout, stderr) per command, so these pin down
the STATE MACHINE and SAFETY GATING behavior independent of what terraform
itself actually does. That real-world behavior can only be verified by
actually running `terraform apply` against a live AWS account on her
machine (see the README this session also produced for those exact
commands) — these tests exist to make sure the orchestration code around
that real call is correct and safe.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import ApplyStatus, DriftStatus, Job, TerraformPlan, TerraformModule
from app.core.job_store import save_job, get_job
from app.services.apply import apply_runner


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _clean_plan() -> TerraformPlan:
    """A plan with no placeholder values left — should sail through preflight."""
    return TerraformPlan(
        root_module_files={
            "main.tf": 'provider "aws" {\n  region = var.aws_region\n}\n',
            "variables.tf": (
                'variable "aws_region" {\n'
                '  type    = string\n'
                '  default = "us-east-1"\n'
                "}\n"
            ),
        },
        modules=[
            TerraformModule(
                name="compute",
                source_resources=["ec2-1"],
                description="EC2 instances",
                files={
                    "main.tf": 'resource "aws_instance" "web" {\n  ami = var.ami\n}\n',
                    "variables.tf": (
                        'variable "ami" {\n'
                        '  type    = string\n'
                        '  default = "ami-0a1b2c3d4e5f6g7h8"\n'
                        "}\n"
                    ),
                },
            )
        ],
        resource_count=1,
    )


def _placeholder_plan() -> TerraformPlan:
    """A plan that still has an unresolved catalog placeholder — must be blocked."""
    return TerraformPlan(
        root_module_files={"main.tf": "provider \"aws\" {}\n"},
        modules=[
            TerraformModule(
                name="compute",
                source_resources=["ec2-1"],
                description="EC2 instances",
                files={
                    "main.tf": 'resource "aws_instance" "web" {\n  ami = var.ami\n}\n',
                    "variables.tf": (
                        'variable "ami" {\n'
                        '  type    = string\n'
                        '  default = "ami-00000000000000000"\n'
                        "}\n"
                    ),
                },
            )
        ],
        resource_count=1,
    )


async def _make_job(plan: TerraformPlan) -> Job:
    job = Job(terraform_plan=plan)
    await save_job(job)
    return job


def _fake_subprocess(responses: dict[str, tuple[int, str, str]]):
    """Returns a fake async _run_subprocess matching on the terraform
    subcommand (cmd[1]) — e.g. responses={"init": (0, "ok", ""), ...}."""
    async def _fake(cmd, cwd, timeout):
        subcmd = cmd[1] if len(cmd) > 1 else cmd[0]
        return responses.get(subcmd, (0, "", ""))
    return _fake


@pytest.mark.anyio
async def test_preflight_blocks_unresolved_placeholder_values(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = await _make_job(_placeholder_plan())

    result = await apply_runner.plan_apply(job.job_id)

    assert result.apply_status == ApplyStatus.NOT_STARTED
    assert result.apply_confirm_token is None
    # Structured, UI-renderable — not a filesystem instruction (2026-07-27
    # correction: the person filling this in is using the UI, not the
    # backend's terminal).
    assert len(result.apply_blocked_variables) == 1
    blocked = result.apply_blocked_variables[0]
    assert blocked.id == "compute/variables.tf::ami"
    assert blocked.variable_name == "ami"
    assert blocked.current_value == "ami-00000000000000000"


@pytest.mark.anyio
async def test_clean_plan_reaches_awaiting_confirm_with_token(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({"init": (0, "Terraform initialized", ""), "plan": (0, "Plan: 1 to add", "")}),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.plan_apply(job.job_id)

    assert result.apply_status == ApplyStatus.AWAITING_CONFIRM
    assert result.apply_confirm_token is not None
    assert result.apply_confirm_token_expires_at > datetime.utcnow()
    assert "Plan: 1 to add" in result.apply_plan_output


@pytest.mark.anyio
async def test_init_failure_marks_job_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({"init": (1, "", "Error: no credentials found")}),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.plan_apply(job.job_id)

    assert result.apply_status == ApplyStatus.FAILED
    assert "no credentials found" in result.apply_error


@pytest.mark.anyio
async def test_confirm_rejects_wrong_token(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({"init": (0, "ok", ""), "plan": (0, "Plan: 1 to add", "")}),
    )
    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)
    assert planned.apply_status == ApplyStatus.AWAITING_CONFIRM

    with pytest.raises(ValueError, match="Invalid confirm token"):
        await apply_runner.confirm_apply(job.job_id, "totally-wrong-token")


@pytest.mark.anyio
async def test_confirm_rejects_expired_token(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({"init": (0, "ok", ""), "plan": (0, "Plan: 1 to add", "")}),
    )
    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)

    # Simulate the 15-minute TTL already having elapsed.
    fresh = await get_job(job.job_id)
    fresh.apply_confirm_token_expires_at = datetime.utcnow() - timedelta(minutes=1)
    await save_job(fresh)

    with pytest.raises(ValueError, match="expired"):
        await apply_runner.confirm_apply(job.job_id, planned.apply_confirm_token)


@pytest.mark.anyio
async def test_confirm_without_a_prior_plan_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = await _make_job(_clean_plan())

    with pytest.raises(ValueError, match="No plan awaiting confirmation"):
        await apply_runner.confirm_apply(job.job_id, "some-token")


@pytest.mark.anyio
async def test_full_plan_confirm_apply_succeeds_and_schedules_destroy(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(apply_runner._s, "apply_auto_destroy_hours", 2.0)
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({
            "init": (0, "Terraform initialized", ""),
            "plan": (0, "Plan: 1 to add, 0 to change, 0 to destroy", ""),
            "apply": (0, "Apply complete! Resources: 1 added", ""),
        }),
    )
    # Auto-destroy scheduling shells out to asyncio.create_task + sleep(7200s)
    # — stub it so the test doesn't actually wait 2 hours or touch the
    # scheduling machinery, which is covered separately below.
    scheduled = {}
    def _fake_schedule(job_id, delay_seconds):
        scheduled["job_id"] = job_id
        scheduled["delay_seconds"] = delay_seconds
    monkeypatch.setattr(apply_runner, "_schedule_destroy", _fake_schedule)

    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)
    assert planned.apply_status == ApplyStatus.AWAITING_CONFIRM

    # The tfplan file confirm_apply checks for is a real terraform artifact
    # normally written by `terraform plan -out=tfplan` — since terraform
    # itself is stubbed out here, create the file it would have produced.
    workdir = Path(planned.apply_workdir)
    (workdir / "tfplan").write_text("fake-plan-binary")

    applied = await apply_runner.confirm_apply(job.job_id, planned.apply_confirm_token)

    assert applied.apply_status == ApplyStatus.APPLIED
    assert applied.apply_destroy_at is not None
    assert applied.apply_confirm_token is None, "token must be consumed (single-use)"
    assert scheduled["job_id"] == job.job_id
    assert scheduled["delay_seconds"] == pytest.approx(7200, rel=0.01)


@pytest.mark.anyio
async def test_confirm_twice_is_rejected_token_already_consumed(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(apply_runner, "_schedule_destroy", lambda *a, **k: None)
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({
            "init": (0, "ok", ""), "plan": (0, "Plan: 1 to add", ""),
            "apply": (0, "Apply complete!", ""),
        }),
    )
    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)
    (Path(planned.apply_workdir) / "tfplan").write_text("fake")
    token = planned.apply_confirm_token

    await apply_runner.confirm_apply(job.job_id, token)

    with pytest.raises(ValueError, match="No plan awaiting confirmation"):
        await apply_runner.confirm_apply(job.job_id, token)


@pytest.mark.anyio
async def test_destroy_after_apply_succeeds_and_clears_destroy_at(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(apply_runner, "_schedule_destroy", lambda *a, **k: None)
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({
            "init": (0, "ok", ""), "plan": (0, "Plan: 1 to add", ""),
            "apply": (0, "Apply complete!", ""), "destroy": (0, "Destroy complete!", ""),
        }),
    )
    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)
    (Path(planned.apply_workdir) / "tfplan").write_text("fake")
    await apply_runner.confirm_apply(job.job_id, planned.apply_confirm_token)

    destroyed = await apply_runner.destroy_apply(job.job_id)

    assert destroyed.apply_status == ApplyStatus.DESTROYED
    assert destroyed.apply_destroy_at is None


@pytest.mark.anyio
async def test_destroy_failure_is_surfaced_not_silently_marked_destroyed(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(apply_runner, "_schedule_destroy", lambda *a, **k: None)
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({
            "init": (0, "ok", ""), "plan": (0, "Plan: 1 to add", ""),
            "apply": (0, "Apply complete!", ""),
            "destroy": (1, "", "Error: DependencyViolation"),
        }),
    )
    job = await _make_job(_clean_plan())
    planned = await apply_runner.plan_apply(job.job_id)
    (Path(planned.apply_workdir) / "tfplan").write_text("fake")
    await apply_runner.confirm_apply(job.job_id, planned.apply_confirm_token)

    result = await apply_runner.destroy_apply(job.job_id)

    assert result.apply_status == ApplyStatus.FAILED
    assert "DependencyViolation" in result.apply_error


@pytest.mark.anyio
async def test_destroy_with_nothing_applied_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = await _make_job(_clean_plan())

    with pytest.raises(ValueError, match="Nothing to destroy"):
        await apply_runner.destroy_apply(job.job_id)


@pytest.mark.anyio
async def test_reconcile_destroys_overdue_job_after_simulated_restart(monkeypatch, tmp_path):
    """Simulates the crash-safety case: a job is APPLIED with a destroy_at
    already in the past (as if the backend restarted and lost the
    in-process asyncio task), and no live task is tracking it."""
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    destroyed_ids = []

    async def _fake_destroy(job_id, reason="auto-destroy"):
        destroyed_ids.append((job_id, reason))
        job = await get_job(job_id)
        job.apply_status = ApplyStatus.DESTROYED
        job.apply_destroy_at = None
        await save_job(job)
        return job

    monkeypatch.setattr(apply_runner, "destroy_apply", _fake_destroy)
    apply_runner._scheduled_destroys.clear()

    job = Job(
        terraform_plan=_clean_plan(),
        apply_status=ApplyStatus.APPLIED,
        apply_destroy_at=datetime.utcnow() - timedelta(minutes=5),
    )
    await save_job(job)

    await apply_runner.reconcile_overdue_destroys()

    assert len(destroyed_ids) == 1
    assert destroyed_ids[0][0] == job.job_id
    assert "overdue" in destroyed_ids[0][1]


# ── UI-driven placeholder resolution (2026-07-27) ────────────────────────────
# Her explicit correction: the original design had the backend tell her to
# hand-edit a terraform.tfvars on the apply_workdir's filesystem. That's
# wrong on two counts — (1) whoever's using the UI should supply real values
# through the UI, not need terminal access to the backend's disk, and (2) it
# wouldn't even have worked correctly: a root-level terraform.tfvars can only
# override ROOT module variables, and vpc_id (the actual blocking case) lives
# in a CHILD module with no passthrough wiring, so Terraform would have
# silently ignored it. These tests cover the real fix: resolve_placeholders()
# patches the variable's default value directly in the apply-workdir copy of
# whichever file it actually lives in.

def _plan_with_two_placeholders() -> TerraformPlan:
    """Two unresolved placeholders in DIFFERENT files with the SAME
    variable name ('cidr_block') — the exact collision case BlockedVariable
    IDs (file::variable_name, not just variable_name) exist to disambiguate."""
    return TerraformPlan(
        root_module_files={"main.tf": 'provider "aws" {}\n'},
        modules=[
            TerraformModule(
                name="networking", source_resources=["vpc-1"], description="VPC",
                files={
                    "main.tf": 'resource "aws_vpc" "main" {\n  cidr_block = var.cidr_block\n}\n',
                    "variables.tf": (
                        'variable "cidr_block" {\n  type = string\n'
                        '  default = "replace-with-real-cidr-block"\n}\n'
                    ),
                },
            ),
            TerraformModule(
                name="compute", source_resources=["ec2-1"], description="EC2",
                files={
                    "main.tf": 'resource "aws_instance" "web" {\n  ami = var.ami\n}\n',
                    "variables.tf": (
                        'variable "ami" {\n  type = string\n'
                        '  default = "ami-00000000000000000"\n}\n'
                    ),
                },
            ),
        ],
        resource_count=2,
    )


@pytest.mark.anyio
async def test_resolve_patches_only_the_targeted_file_and_variable(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = await _make_job(_plan_with_two_placeholders())

    first = await apply_runner.plan_apply(job.job_id)
    assert first.apply_status == ApplyStatus.NOT_STARTED
    ids = {b.id for b in first.apply_blocked_variables}
    assert ids == {"networking/variables.tf::cidr_block", "compute/variables.tf::ami"}

    # Resolve only the VPC's cidr_block — the EC2 ami placeholder must stay
    # blocked, proving overrides are scoped per (file, variable), not by
    # variable name alone.
    second = await apply_runner.resolve_placeholders(
        job.job_id, {"networking/variables.tf::cidr_block": "10.0.0.0/16"}
    )

    assert second.apply_status == ApplyStatus.NOT_STARTED
    remaining_ids = {b.id for b in second.apply_blocked_variables}
    assert remaining_ids == {"compute/variables.tf::ami"}

    # And the resolved file, as written to the persistent apply_workdir,
    # actually has the real value patched in — not just accepted and
    # discarded.
    networking_vars = Path(second.apply_workdir, "modules", "networking", "variables.tf").read_text()
    assert 'default = "10.0.0.0/16"' in networking_vars
    assert "replace-with-real-cidr-block" not in networking_vars


@pytest.mark.anyio
async def test_resolve_all_placeholders_proceeds_to_real_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess({"init": (0, "ok", ""), "plan": (0, "Plan: 2 to add", "")}),
    )
    job = await _make_job(_plan_with_two_placeholders())
    await apply_runner.plan_apply(job.job_id)

    result = await apply_runner.resolve_placeholders(job.job_id, {
        "networking/variables.tf::cidr_block": "10.0.0.0/16",
        "compute/variables.tf::ami": "ami-0real0000000000",
    })

    assert result.apply_status == ApplyStatus.AWAITING_CONFIRM
    assert result.apply_blocked_variables == []
    assert result.apply_confirm_token is not None


@pytest.mark.anyio
async def test_overrides_survive_a_second_unrelated_plan_rerun(monkeypatch, tmp_path):
    """plan_apply() rewrites the whole workdir from job.terraform_plan on
    every call — a previously-submitted override MUST be re-applied every
    time, or clicking 'Run terraform plan' again would silently regress an
    already-fixed value back to the fake placeholder."""
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = await _make_job(_plan_with_two_placeholders())
    await apply_runner.plan_apply(job.job_id)
    await apply_runner.resolve_placeholders(
        job.job_id, {"networking/variables.tf::cidr_block": "10.0.0.0/16"}
    )

    # Simulate her clicking "Run terraform plan" again directly (not via
    # resolve) — the earlier fix must still hold.
    again = await apply_runner.plan_apply(job.job_id)

    remaining_ids = {b.id for b in again.apply_blocked_variables}
    assert remaining_ids == {"compute/variables.tf::ami"}
    networking_vars = Path(again.apply_workdir, "modules", "networking", "variables.tf").read_text()
    assert 'default = "10.0.0.0/16"' in networking_vars


@pytest.mark.anyio
async def test_resolve_without_a_terraform_plan_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = Job()
    await save_job(job)

    with pytest.raises(ValueError, match="No Terraform plan generated"):
        await apply_runner.resolve_placeholders(job.job_id, {"x::y": "z"})


# ── Drift detection (2026-07-29) ─────────────────────────────────────────────
# check_drift() runs `terraform plan -refresh-only -detailed-exitcode` — a
# read-only comparison of state vs. real infra that never proposes or
# applies config changes. -detailed-exitcode turns the result into an
# unambiguous exit code: 0 = no drift, 2 = drift found (NOT an error), 1 =
# a real failure. These tests fake that exact contract directly (rather
# than the subcmd-only _fake_subprocess helper above, which can't
# distinguish flag variations of the same "plan" subcommand).

_DRIFT_PLAN_OUTPUT = """\
Note: Objects have changed outside of Terraform

Terraform detected the following changes made outside of Terraform since the
last "terraform apply" which may have affected this plan:

  # aws_instance.web_server has changed
  ~ resource "aws_instance" "web_server" {
        id            = "i-0123456789abcdef0"
      ~ instance_type = "t3.micro" -> "t3.small"
    }

  # aws_security_group.web_sg has changed
  ~ resource "aws_security_group" "web_sg" {
        id = "sg-0123456789abcdef0"
    }
"""


def _fake_subprocess_by_flags(init_result, plan_results: dict[str, tuple[int, str, str]]):
    """plan_results keyed by whether -refresh-only is present in the
    command, so drift's refresh-only plan and a normal plan can return
    different canned results within the same test."""
    async def _fake(cmd, cwd, timeout):
        if len(cmd) > 1 and cmd[1] == "init":
            return init_result
        if len(cmd) > 1 and cmd[1] == "plan":
            key = "refresh_only" if "-refresh-only" in cmd else "normal"
            return plan_results.get(key, (0, "", ""))
        return (0, "", "")
    return _fake


@pytest.mark.anyio
async def test_check_drift_clean_reports_no_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess_by_flags(
            (0, "Terraform initialized", ""),
            {"refresh_only": (0, "No changes. Your infrastructure matches the configuration.", "")},
        ),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.check_drift(job.job_id)

    assert result.drift_status == DriftStatus.CLEAN
    assert result.drift_resources == []
    assert result.drift_checked_at is not None


@pytest.mark.anyio
async def test_check_drift_detects_and_parses_changed_resources(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess_by_flags(
            (0, "Terraform initialized", ""),
            {"refresh_only": (2, _DRIFT_PLAN_OUTPUT, "")},
        ),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.check_drift(job.job_id)

    assert result.drift_status == DriftStatus.DRIFT_DETECTED
    assert result.drift_resources == ["aws_instance.web_server", "aws_security_group.web_sg"]
    assert "instance_type" in result.drift_output


@pytest.mark.anyio
async def test_check_drift_real_plan_error_marks_failed_not_drifted(monkeypatch, tmp_path):
    """Exit code 1 from -detailed-exitcode is a genuine failure (e.g. a
    provider error) — must never be confused with exit code 2 (drift)."""
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess_by_flags(
            (0, "Terraform initialized", ""),
            {"refresh_only": (1, "", "Error: no credentials found")},
        ),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.check_drift(job.job_id)

    assert result.drift_status == DriftStatus.FAILED
    assert result.drift_resources == []
    assert "no credentials found" in result.drift_output


@pytest.mark.anyio
async def test_check_drift_init_failure_marks_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess_by_flags((1, "", "Error: backend init failed"), {}),
    )
    job = await _make_job(_clean_plan())

    result = await apply_runner.check_drift(job.job_id)

    assert result.drift_status == DriftStatus.FAILED
    assert "backend init failed" in result.drift_output


@pytest.mark.anyio
async def test_check_drift_without_a_terraform_plan_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    job = Job()
    await save_job(job)

    with pytest.raises(ValueError, match="No Terraform plan generated"):
        await apply_runner.check_drift(job.job_id)


@pytest.mark.anyio
async def test_check_drift_reuses_existing_variable_overrides(monkeypatch, tmp_path):
    """The whole point of the earlier design discussion: a drift check must
    write files with whatever variable overrides are ALREADY set (from a
    prior resolve_placeholders round), not silently drop back to fake
    placeholder defaults."""
    monkeypatch.setattr(apply_runner._s, "local_output_dir", str(tmp_path))
    monkeypatch.setattr(
        apply_runner, "_run_subprocess",
        _fake_subprocess_by_flags(
            (0, "ok", ""),
            {"refresh_only": (0, "No changes.", "")},
        ),
    )
    job = await _make_job(_plan_with_two_placeholders())
    await apply_runner.plan_apply(job.job_id)
    await apply_runner.resolve_placeholders(job.job_id, {
        "networking/variables.tf::cidr_block": "10.0.0.0/16",
        "compute/variables.tf::ami": "ami-0real0000000000",
    })

    result = await apply_runner.check_drift(job.job_id)

    assert result.drift_status == DriftStatus.CLEAN
    networking_vars = Path(result.apply_workdir, "modules", "networking", "variables.tf").read_text()
    assert 'default = "10.0.0.0/16"' in networking_vars
