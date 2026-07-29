"""
Apply-to-Sandbox Runner
-------------------------
Completes the pipeline all the way through a REAL `terraform apply` against
her own AWS sandbox account (2026-07-24, her explicit request). This is
deliberately a separate, manually-triggered stage from the main
upload -> parse -> plan -> generate -> validate -> package pipeline
(pipeline_worker.py): applying real infrastructure should never happen
automatically just because a job reached JobStatus.DONE.

Credentials: this module NEVER reads, stores, or transmits AWS credentials
itself. `_run_subprocess()` inherits the backend process's own environment
(`os.environ`) unchanged, so `terraform` picks up whatever AWS credential
chain is already active wherever uvicorn is running — her local
~/.aws/credentials [default] profile, or AWS_PROFILE/AWS_ACCESS_KEY_ID if
she's exported them first. This is why the "Apply" trigger has to run on
HER machine (see frontend), not from any sandboxed/remote environment that
doesn't have her credentials.

Flow (see api/routes/apply.py):
  1. plan_apply(job_id)     -> writes Terraform to a PERSISTENT working dir,
                               runs a placeholder-value preflight check,
                               then `terraform init` + `terraform plan -out=tfplan`.
                               Returns a one-time confirm_token (15 min TTL).
  2. confirm_apply(job_id, token) -> validates + consumes the token, runs
                               `terraform apply tfplan`, schedules an
                               auto-destroy safety-net task.
  3. destroy_apply(job_id)  -> `terraform destroy -auto-approve`, cancels
                               any pending scheduled auto-destroy.

Safety model: apply can NEVER run without a fresh, just-reviewed plan
immediately before it (the confirm_token ties them together and expires
quickly) — no "blind apply" path exists in this module.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

import sys
# services/apply/apply_runner.py -> apply(0)/services(1)/app(2)/backend(3)/
# arch2tf-product(4) — same sys.path convention as tf_validator.py/packager.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.schemas.models import ApplyStatus, BlockedVariable, DriftStatus, Job, TerraformPlan
from app.core.config import get_settings
from app.core.job_store import get_job, save_job

_s = get_settings()

# Confirm tokens are single-use and short-lived — see Job.apply_confirm_token
# docstring in models.py for why.
_CONFIRM_TOKEN_TTL_MINUTES = 15

# Scheduled auto-destroy asyncio tasks, keyed by job_id, so a manual destroy
# (or a second `plan` re-run) can cancel a pending one instead of racing it.
# In-process only — see main.py's lifespan reconciliation loop for the
# crash/restart-safe fallback (re-reads apply_destroy_at from the persisted
# Job, not this dict).
_scheduled_destroys: dict[str, asyncio.Task] = {}

# Reuse arch2terraform's own placeholder-value conventions rather than
# re-inventing the pattern list — see missing_info_detector.py's docstring
# for what these look like ("ami-00000000000000000",
# "replace-with-globally-unique-name", "arn:aws:iam::000000000000:role/...").
from app.services.parser.missing_info_detector import _PLACEHOLDER_MARKERS  # noqa: E402


def _apply_workdir(job_id: str) -> Path:
    """
    Persistent (NOT tempfile.TemporaryDirectory) working directory for a
    job's real apply lifecycle — has to survive across separate
    plan -> confirm -> apply -> destroy HTTP requests and hold the real
    terraform.tfstate in between, unlike tf_validator.py's/packager.py's
    ephemeral dirs which are each discarded within a single function call.
    Keyed off job_id under local_output_dir, same convention as
    packager.py's zip path and storage.py's upload/output paths.
    """
    return Path(_s.local_output_dir) / job_id / "apply_workdir"


import re as _re

_VAR_BLOCK_RE = _re.compile(r'variable\s+"([^"]+)"\s*\{([^}]*)\}', _re.DOTALL)
_DEFAULT_LINE_RE = _re.compile(r'(default\s*=\s*")([^"]*)(")')
_DESCRIPTION_RE = _re.compile(r'description\s*=\s*"([^"]*)"')


def _all_files(plan: TerraformPlan) -> dict[str, str]:
    """Root + module files as one dict, keyed the same way
    BlockedVariable.id's file portion is — plain filename for root
    ("variables.tf"), "<module>/<filename>" for a child module
    ("networking/generated_security_groups_variables.tf")."""
    files = dict(plan.root_module_files)
    for module in plan.modules:
        for filename, content in module.files.items():
            files[f"{module.name}/{filename}"] = content
    return files


def _apply_overrides_to_content(file_key: str, content: str, overrides: dict[str, str]) -> str:
    """
    Patches `default = "<old>"` -> `default = "<new>"` in place for every
    variable in this file that has a matching override (keyed by
    "<file_key>::<variable_name>"). Real fix, 2026-07-27, replacing the
    original terraform.tfvars-on-disk design: a plain root-level
    terraform.tfvars can only override ROOT module variables — it has no
    effect on a CHILD module's variable default (e.g. vpc_id lives in
    modules/networking/generated_security_groups_variables.tf with no
    root-level passthrough wiring), so a tfvars sitting next to it would be
    silently ignored by Terraform. Patching the default in-place inside the
    module's own file, in this apply-only working-directory copy, works
    uniformly for root and child-module variables alike and never touches
    the original job.terraform_plan (the downloadable ZIP is unaffected).
    """
    if not overrides:
        return content

    def _replace_block(match: _re.Match) -> str:
        var_name, body = match.group(1), match.group(2)
        override_key = f"{file_key}::{var_name}"
        if override_key not in overrides:
            return match.group(0)
        new_value = overrides[override_key].replace('"', '\\"')
        new_body, n = _DEFAULT_LINE_RE.subn(rf'\g<1>{new_value}\g<3>', body, count=1)
        if n == 0:
            return match.group(0)  # no `default = "..."` line to patch — leave untouched
        return f'variable "{var_name}" {{{new_body}}}'

    return _VAR_BLOCK_RE.sub(_replace_block, content)


def _write_tf_files(plan: TerraformPlan, root_dir: Path, overrides: dict[str, str]) -> None:
    """Same shape as tf_validator.py's _write_tf_files, plus applying any
    stored variable-default overrides (see Job.apply_variable_overrides)
    before writing each file — duplicated from tf_validator.py's version
    rather than imported since that one is paired with an ephemeral tempdir
    and has no override concept at all."""
    root_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in plan.root_module_files.items():
        content = _apply_overrides_to_content(filename, content, overrides)
        (root_dir / filename).write_text(content, encoding="utf-8")
    for module in plan.modules:
        mod_dir = root_dir / "modules" / module.name
        mod_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in module.files.items():
            file_key = f"{module.name}/{filename}"
            content = _apply_overrides_to_content(file_key, content, overrides)
            (mod_dir / filename).write_text(content, encoding="utf-8")


def _find_unresolved_placeholders(plan: TerraformPlan, overrides: dict[str, str]) -> list[dict]:
    """
    Preflight safety check, 2026-07-24 (redesigned 2026-07-27 to feed a UI
    form instead of a terraform.tfvars instruction): arch2terraform's
    catalog deliberately fills any field it can't infer with a
    schema-valid FAKE value (e.g. "vpc-00000000000000000") so
    `terraform validate`/`plan` always succeed even before clarification
    answers are in. Those fakes are fine for validate/plan (which never
    touch real AWS resource IDs), but a real `apply` against a fake
    "vpc-00000000000000000" or "ami-00000000000000000" would either fail
    outright or, worse, might coincidentally collide with something —
    either way, this must never be silently applied. Scans every generated
    variables.tf `default = "..."` line (root + all modules, WITH any
    already-submitted `overrides` applied first) and returns one dict per
    still-unresolved placeholder — exactly the shape api/routes/apply.py
    turns into `BlockedVariable` entries for the UI to render real input
    fields against.
    """
    findings: list[dict] = []
    for file_key, content in _all_files(plan).items():
        if not file_key.endswith("variables.tf"):
            continue
        patched = _apply_overrides_to_content(file_key, content, overrides)
        for var_match in _VAR_BLOCK_RE.finditer(patched):
            var_name, body = var_match.group(1), var_match.group(2)
            default_match = _DEFAULT_LINE_RE.search(body)
            if not default_match:
                continue
            value = default_match.group(2)
            lowered = value.lower()
            if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
                desc_match = _DESCRIPTION_RE.search(body)
                findings.append({
                    "id": f"{file_key}::{var_name}",
                    "file": file_key,
                    "variable_name": var_name,
                    "current_value": value,
                    "description": desc_match.group(1) if desc_match else "",
                })

    return findings


async def _run_subprocess(
    cmd: list[str], cwd: str, timeout: int
) -> tuple[int, str, str]:
    """Same pattern as tf_validator.py's _run_subprocess — deliberately does
    NOT pass an `env` override, so the subprocess inherits this backend
    process's os.environ unchanged (her real AWS credential chain), per
    this module's docstring."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return -1, "", f"Binary not found: {cmd[0]}. Install: brew install hashicorp/tap/terraform"


async def plan_apply(job_id: str) -> Job:
    """
    Stage 1: write Terraform to a persistent working dir, preflight-check
    for unresolved placeholder values, then `terraform init` + `plan -out`.
    Returns the updated Job (apply_status is AWAITING_CONFIRM on success
    with a fresh confirm_token, or FAILED with apply_error set).
    """
    job = await get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if not job.terraform_plan:
        raise ValueError("No Terraform plan generated yet for this job")

    job.apply_status = ApplyStatus.PLANNING
    job.apply_error = ""
    job.log_apply("terraform apply: preflight + init + plan started")
    await save_job(job)

    workdir = _apply_workdir(job_id)
    job.apply_workdir = str(workdir)

    # Write files first, WITH any already-submitted overrides applied —
    # job.apply_variable_overrides (from previous POST /apply/resolve
    # rounds) has to be re-applied on every call since this rewrites the
    # whole workdir from job.terraform_plan fresh each time, nothing on
    # disk from a prior run is trusted to still be there. Writing before
    # the placeholder check (rather than only writing once everything's
    # resolved) means a partial fix is visible on disk immediately, and the
    # remaining-placeholder scan below reads the exact same content that
    # would otherwise reach `terraform init`.
    _write_tf_files(job.terraform_plan, workdir, job.apply_variable_overrides)

    # ── Preflight: block on unresolved placeholder values ────────────────
    unresolved = _find_unresolved_placeholders(job.terraform_plan, job.apply_variable_overrides)
    if unresolved:
        job.apply_status = ApplyStatus.NOT_STARTED
        job.apply_blocked_variables = [BlockedVariable(**f) for f in unresolved]
        job.apply_error = (
            "Blocked: these generated variables still hold fake placeholder "
            "values and would fail (or worse, silently misbehave) against a "
            "real AWS account. Fill in real values in the UI before retrying."
        )
        job.log_apply(
            f"terraform apply blocked: {len(unresolved)} unresolved placeholder value(s) — "
            "awaiting input via the UI"
        )
        await save_job(job)
        return job

    job.apply_blocked_variables = []

    # ── terraform init (real backend this time — no -backend=false; a
    # real apply needs real local/remote state, not a disposable check) ──
    rc_init, out_init, err_init = await _run_subprocess(
        [_s.terraform_binary, "init", "-input=false", "-no-color"],
        cwd=str(workdir),
        timeout=_s.terraform_init_timeout_seconds,
    )
    if rc_init != 0:
        job.apply_status = ApplyStatus.FAILED
        job.apply_error = f"terraform init failed:\n{err_init or out_init}"
        job.log_apply("terraform init failed")
        await save_job(job)
        return job

    # ── terraform plan -out=tfplan (real AWS API calls happen here) ──────
    rc_plan, out_plan, err_plan = await _run_subprocess(
        [_s.terraform_binary, "plan", "-input=false", "-no-color", "-out=tfplan"],
        cwd=str(workdir),
        timeout=_s.terraform_timeout_seconds,
    )
    plan_output = (out_plan + "\n" + err_plan).strip()
    job.apply_plan_output = plan_output

    if rc_plan != 0:
        job.apply_status = ApplyStatus.FAILED
        job.apply_error = f"terraform plan failed:\n{plan_output}"
        job.log_apply("terraform plan failed")
        await save_job(job)
        return job

    job.apply_status = ApplyStatus.AWAITING_CONFIRM
    job.apply_confirm_token = secrets.token_urlsafe(24)
    job.apply_confirm_token_expires_at = datetime.utcnow() + timedelta(
        minutes=_CONFIRM_TOKEN_TTL_MINUTES
    )
    job.log_apply("terraform plan succeeded — awaiting explicit confirm")
    await save_job(job)
    return job


_DRIFT_RESOURCE_RE = _re.compile(r'^\s*#\s+(\S+)\s+has changed', _re.MULTILINE)


def _parse_drift_resources(output: str) -> list[str]:
    """Extracts resource addresses from a `-refresh-only` plan's
    "# <address> has changed" headers — the same header Terraform prints
    whether or not -detailed-exitcode is used, stable across recent
    Terraform versions. Best-effort: an empty list here doesn't by itself
    mean no drift, only that this particular text pattern wasn't matched —
    check_drift() decides drift_status from the exit code, not from
    whether this list is non-empty."""
    return _DRIFT_RESOURCE_RE.findall(output)


async def check_drift(job_id: str) -> Job:
    """
    Read-only drift check: refreshes state against real infrastructure and
    reports what's changed OUTSIDE Terraform since the last apply, without
    proposing or applying any config changes. A deliberately separate
    primitive from plan_apply(), her explicit request 2026-07-29: a normal
    `terraform plan` conflates two different questions — "has real infra
    drifted from state" and "does my code differ from what's applied" —
    into one diff. `-refresh-only` isolates just the first question, and
    `-detailed-exitcode` turns the answer into an unambiguous exit code
    (0 = no drift, 2 = drift found, 1 = real error) instead of having to
    string-match plan text to tell "no changes" apart from "some changes".

    Reuses the exact same apply_workdir, generated files, and variable
    overrides plan_apply() uses (rewritten fresh here too, for the same
    reason plan_apply() rewrites on every call — nothing on disk from a
    prior run is trusted to still be there), so this always compares
    against the same backend.tf-configured state — local, S3, or Terraform
    Cloud, whichever terraform_planner.py generated — that plan/apply would
    use, never a different or stale one.
    """
    job = await get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if not job.terraform_plan:
        raise ValueError("No Terraform plan generated yet for this job")

    job.drift_status = DriftStatus.CHECKING
    job.log_apply("drift check: init + refresh-only plan started")
    await save_job(job)

    workdir = _apply_workdir(job_id)
    job.apply_workdir = str(workdir)
    _write_tf_files(job.terraform_plan, workdir, job.apply_variable_overrides)

    rc_init, out_init, err_init = await _run_subprocess(
        [_s.terraform_binary, "init", "-input=false", "-no-color"],
        cwd=str(workdir),
        timeout=_s.terraform_init_timeout_seconds,
    )
    if rc_init != 0:
        job.drift_status = DriftStatus.FAILED
        job.drift_output = f"terraform init failed:\n{err_init or out_init}"
        job.drift_checked_at = datetime.utcnow()
        job.log_apply("drift check: terraform init failed")
        await save_job(job)
        return job

    # -detailed-exitcode: 0 = no drift, 2 = drift found (NOT an error, even
    # though it's a non-zero exit code), 1 = a real failure. Must be
    # checked before treating any non-zero return as failure.
    rc_plan, out_plan, err_plan = await _run_subprocess(
        [_s.terraform_binary, "plan", "-refresh-only", "-detailed-exitcode", "-input=false", "-no-color"],
        cwd=str(workdir),
        timeout=_s.terraform_timeout_seconds,
    )
    plan_output = (out_plan + "\n" + err_plan).strip()
    job.drift_output = plan_output
    job.drift_checked_at = datetime.utcnow()

    if rc_plan == 0:
        job.drift_status = DriftStatus.CLEAN
        job.drift_resources = []
        job.log_apply("drift check: no drift detected — real infra matches last-known state")
    elif rc_plan == 2:
        job.drift_status = DriftStatus.DRIFT_DETECTED
        job.drift_resources = _parse_drift_resources(plan_output)
        summary = ", ".join(job.drift_resources) if job.drift_resources else "see drift_output for details"
        job.log_apply(f"drift check: drift detected in {len(job.drift_resources)} resource(s) — {summary}")
    else:
        job.drift_status = DriftStatus.FAILED
        job.drift_resources = []
        job.log_apply("drift check: terraform plan -refresh-only failed")

    await save_job(job)
    return job


async def resolve_placeholders(job_id: str, overrides: dict[str, str]) -> Job:
    """
    Takes the real values she typed into the UI's blocked-variables form
    (keyed by BlockedVariable.id — see models.py), merges them into
    job.apply_variable_overrides (merge, not replace, so fixes from an
    earlier partial submission aren't lost), then immediately re-runs
    plan_apply() so she gets a fresh result in one round trip: either
    still-blocked (with whatever's left) or straight through to a real
    `terraform plan` if that resolved everything.
    """
    job = await get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if not job.terraform_plan:
        raise ValueError("No Terraform plan generated yet for this job")

    job.apply_variable_overrides.update(overrides)
    job.log_apply(f"received {len(overrides)} real value(s) from the UI for previously-blocked variables")
    await save_job(job)

    return await plan_apply(job_id)


async def confirm_apply(job_id: str, confirm_token: str) -> Job:
    """
    Stage 2: validate the (single-use, time-limited) confirm token against
    the plan just produced by plan_apply(), then run the real
    `terraform apply tfplan`. On success, schedules the auto-destroy
    safety net.
    """
    job = await get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    if job.apply_status != ApplyStatus.AWAITING_CONFIRM:
        raise ValueError(
            f"No plan awaiting confirmation (apply_status={job.apply_status.value}). "
            "Run plan_apply again first."
        )
    if not job.apply_confirm_token or confirm_token != job.apply_confirm_token:
        raise ValueError("Invalid confirm token.")
    if not job.apply_confirm_token_expires_at or datetime.utcnow() > job.apply_confirm_token_expires_at:
        raise ValueError("Confirm token expired — re-run plan to review a fresh plan before applying.")

    # Consume the token immediately — single use, prevents a duplicate
    # request (e.g. a double-click, or a retried network call) from
    # triggering a second real apply against the same plan.
    job.apply_confirm_token = None
    job.apply_confirm_token_expires_at = None
    job.apply_status = ApplyStatus.APPLYING
    job.log_apply("terraform apply confirmed — applying to real AWS account now")
    await save_job(job)

    workdir = Path(job.apply_workdir) if job.apply_workdir else _apply_workdir(job_id)
    tfplan_path = workdir / "tfplan"
    if not tfplan_path.exists():
        job.apply_status = ApplyStatus.FAILED
        job.apply_error = "Saved plan file (tfplan) not found — run plan_apply again."
        job.log_apply("apply aborted: tfplan file missing")
        await save_job(job)
        return job

    rc, out, err = await _run_subprocess(
        [_s.terraform_binary, "apply", "-input=false", "-no-color", "tfplan"],
        cwd=str(workdir),
        timeout=_s.terraform_apply_timeout_seconds,
    )
    output = (out + "\n" + err).strip()

    if rc != 0:
        job.apply_status = ApplyStatus.FAILED
        job.apply_error = f"terraform apply failed:\n{output}"
        job.log_apply("terraform apply FAILED")
        await save_job(job)
        return job

    job.apply_status = ApplyStatus.APPLIED
    job.apply_destroy_at = datetime.utcnow() + timedelta(hours=_s.apply_auto_destroy_hours)
    job.log_apply(
        f"terraform apply SUCCEEDED — live in your AWS account. "
        f"Auto-destroy scheduled for {job.apply_destroy_at.isoformat()}Z "
        f"unless destroyed manually first."
    )
    await save_job(job)

    _schedule_destroy(job_id, _s.apply_auto_destroy_hours * 3600)
    return job


async def destroy_apply(job_id: str, reason: str = "manual") -> Job:
    """Stage 3: `terraform destroy -auto-approve`. Cancels any pending
    scheduled auto-destroy task for this job (whether this call itself IS
    that scheduled task firing, or a manual early destroy pre-empting it)."""
    job = await get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    _cancel_scheduled_destroy(job_id)

    if job.apply_status not in (ApplyStatus.APPLIED, ApplyStatus.FAILED):
        raise ValueError(
            f"Nothing to destroy (apply_status={job.apply_status.value})."
        )

    workdir = Path(job.apply_workdir) if job.apply_workdir else _apply_workdir(job_id)
    if not workdir.exists():
        job.apply_status = ApplyStatus.DESTROYED
        job.apply_destroy_at = None
        job.log_apply(f"destroy skipped ({reason}): no working directory found — nothing was ever applied")
        await save_job(job)
        return job

    job.apply_status = ApplyStatus.DESTROYING
    job.log_apply(f"terraform destroy started ({reason})")
    await save_job(job)

    rc, out, err = await _run_subprocess(
        [_s.terraform_binary, "destroy", "-auto-approve", "-input=false", "-no-color"],
        cwd=str(workdir),
        timeout=_s.terraform_destroy_timeout_seconds,
    )
    output = (out + "\n" + err).strip()

    if rc != 0:
        job.apply_status = ApplyStatus.FAILED
        job.apply_error = f"terraform destroy failed:\n{output}"
        job.log_apply("terraform destroy FAILED — resources may still be live in your AWS account, check manually")
        await save_job(job)
        return job

    job.apply_status = ApplyStatus.DESTROYED
    job.apply_destroy_at = None
    job.log_apply("terraform destroy succeeded — sandbox resources removed")
    await save_job(job)
    return job


def _schedule_destroy(job_id: str, delay_seconds: float) -> None:
    _cancel_scheduled_destroy(job_id)

    async def _fire():
        try:
            await asyncio.sleep(delay_seconds)
            log.info("Auto-destroy firing for job %s after %.0fs", job_id, delay_seconds)
            await destroy_apply(job_id, reason="auto-destroy (2h safety net)")
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Auto-destroy failed for job %s", job_id)

    task = asyncio.create_task(_fire())
    _scheduled_destroys[job_id] = task


def _cancel_scheduled_destroy(job_id: str) -> None:
    task = _scheduled_destroys.pop(job_id, None)
    if task and not task.done():
        task.cancel()


async def reconcile_overdue_destroys() -> None:
    """
    Crash-safe fallback for the auto-destroy safety net: `_scheduled_destroys`
    is an in-process dict, so a backend restart mid-window loses the
    scheduled asyncio task entirely, even though `job.apply_destroy_at` is
    persisted. Called once at startup (main.py's lifespan) and then on a
    recurring poll (apply_reconcile_poll_seconds) so a job that's already
    past its deadline gets destroyed immediately, and one that isn't yet
    gets a fresh in-process task scheduled for its remaining time.
    """
    from app.core.job_store import list_job_ids

    now = datetime.utcnow()
    for job_id in await list_job_ids():
        try:
            job = await get_job(job_id)
        except Exception:
            continue
        if not job or job.apply_status != ApplyStatus.APPLIED or not job.apply_destroy_at:
            continue
        if job_id in _scheduled_destroys and not _scheduled_destroys[job_id].done():
            continue  # already has a live in-process task covering it
        remaining = (job.apply_destroy_at - now).total_seconds()
        if remaining <= 0:
            log.warning("Job %s has an overdue auto-destroy (was due %s) — destroying now", job_id, job.apply_destroy_at)
            await destroy_apply(job_id, reason="auto-destroy (overdue after restart)")
        else:
            log.info("Job %s auto-destroy re-scheduled for %.0fs from now after restart", job_id, remaining)
            _schedule_destroy(job_id, remaining)


async def reconcile_loop() -> None:
    """Recurring background loop — see reconcile_overdue_destroys()."""
    while True:
        try:
            await reconcile_overdue_destroys()
        except Exception:
            log.exception("apply auto-destroy reconciliation loop failed")
        await asyncio.sleep(_s.apply_reconcile_poll_seconds)
