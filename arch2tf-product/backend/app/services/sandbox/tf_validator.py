"""
Sandbox Validation Service
---------------------------
Runs the generated Terraform against a real AWS sandbox account:
  1. terraform init      (with mocked provider if no AWS creds)
  2. terraform validate  (always runs — pure syntax check)
  3. tflint              (linting, if binary present)
  4. checkov             (security checks, if binary present)
  5. terraform plan      (only when sandbox_enabled=True and AWS creds set)

Designed to be safe: validate+lint always run, plan only with explicit opt-in.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

import sys
# services/sandbox/tf_validator.py -> sandbox(0)/services(1)/app(2)/backend(3)/
# arch2tf-product(4). Was parents[5] (one level too far, lands on "thesis") —
# pre-existing bug, same class as missing_info_detector.py's.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.schemas.models import (
    TerraformPlan, ValidationResult, ValidationCheck, ValidationStatus
)
from app.core.config import get_settings

_s = get_settings()


async def validate_terraform(
    plan: TerraformPlan, existing_state_bytes: bytes | None = None
) -> ValidationResult:
    """
    Write plan files to a temp dir and run all validation tools.
    Returns a ValidationResult.

    `existing_state_bytes` (added 2026-07-08, per her request to see real
    drift without needing a live remote backend wired into this sandbox
    check): when provided, seeded into the temp dir as `terraform.tfstate`
    BEFORE `terraform init`/`plan` run. Since this whole flow always inits
    with `-backend=false` (see _run_tf_validate — deliberate, keeps this
    check side-effect-free, no real AWS state lock ever touched from a
    casual UI preview), Terraform falls back to reading/writing LOCAL state
    from whatever `terraform.tfstate` file already exists in the working
    directory. Seeding one here means `terraform plan` diffs against that
    real snapshot instead of an empty slate — "0 to add, 1 to change"
    instead of always "everything is new" — without ever needing live AWS
    backend credentials or touching a real DynamoDB lock from this sandbox
    path. No state is ever written back anywhere; the temp dir (and
    whatever Terraform updates in that local file) is discarded when this
    function returns, same as every other file in this flow.
    """
    checks: list[ValidationCheck] = []
    errors: list[str] = []
    warnings: list[str] = []
    tf_plan_output = ""

    # ── Ensure provider cache dir exists ─────────────────────────────────────
    try:
        Path(_s.tf_plugin_cache_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log.warning("Could not create TF plugin cache dir: %s", e)

    with tempfile.TemporaryDirectory(prefix="arch2tf_sandbox_") as tmpdir:
        # Write all Terraform files to temp dir
        root_dir = Path(tmpdir)
        _write_tf_files(plan, root_dir)

        if existing_state_bytes:
            (root_dir / "terraform.tfstate").write_bytes(existing_state_bytes)

        # ── 1. terraform validate ────────────────────────────────────────────
        if not shutil.which(_s.terraform_binary):
            checks.append(ValidationCheck(
                name="terraform_validate",
                status=ValidationStatus.WARNING,
                tool="terraform",
                message=(
                    f"terraform binary not found at '{_s.terraform_binary}'. "
                    "Install: brew install hashicorp/tap/terraform"
                ),
                severity="warning",
            ))
        else:
            validate_check = await _run_tf_validate(root_dir)
            checks.append(validate_check)
            if validate_check.status == ValidationStatus.FAILED:
                errors.extend(validate_check.message.splitlines())

        # ── 2. tflint ────────────────────────────────────────────────────────
        if shutil.which(_s.tflint_binary):
            lint_checks = await _run_tflint(root_dir)
            checks.extend(lint_checks)
            warnings.extend(
                c.message for c in lint_checks if c.severity == "warning"
            )
            errors.extend(
                c.message for c in lint_checks if c.severity == "error"
            )
        else:
            checks.append(ValidationCheck(
                name="tflint",
                status=ValidationStatus.WARNING,
                tool="tflint",
                message="tflint not installed — skipped. Install: https://github.com/terraform-linters/tflint",
                severity="warning",
            ))

        # ── 3. checkov ──────────────────────────────────────────────────────
        if shutil.which(_s.checkov_binary):
            sec_checks = await _run_checkov(root_dir)
            checks.extend(sec_checks)
            warnings.extend(
                c.message for c in sec_checks if c.severity == "warning"
            )
        else:
            checks.append(ValidationCheck(
                name="checkov",
                status=ValidationStatus.WARNING,
                tool="checkov",
                message="checkov not installed — security scan skipped. pip install checkov",
                severity="warning",
            ))

        # ── 4. terraform plan (only with real AWS creds) ─────────────────────
        if _s.sandbox_enabled and _s.aws_access_key_id:
            plan_check, tf_plan_output = await _run_tf_plan(root_dir)
            checks.append(plan_check)
            if plan_check.status == ValidationStatus.FAILED:
                errors.extend(plan_check.message.splitlines())
        else:
            checks.append(ValidationCheck(
                name="terraform_plan",
                status=ValidationStatus.WARNING,
                tool="terraform",
                message=(
                    "sandbox_enabled=False or AWS credentials not set. "
                    "Set SANDBOX_ENABLED=true and AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                    "to run terraform plan against a real AWS account."
                ),
                severity="warning",
            ))

    # Determine overall status
    has_error = any(c.status == ValidationStatus.FAILED for c in checks)
    has_warning = any(c.status == ValidationStatus.WARNING for c in checks)

    overall = (
        ValidationStatus.FAILED if has_error
        else ValidationStatus.WARNING if has_warning
        else ValidationStatus.PASSED
    )

    return ValidationResult(
        overall_status=overall,
        checks=checks,
        terraform_plan_output=tf_plan_output,
        errors=errors,
        warnings=warnings,
        passed_count=sum(1 for c in checks if c.status == ValidationStatus.PASSED),
        failed_count=sum(1 for c in checks if c.status == ValidationStatus.FAILED),
        warning_count=sum(1 for c in checks if c.status == ValidationStatus.WARNING),
    )


def _write_tf_files(plan: TerraformPlan, root_dir: Path) -> None:
    """Write all Terraform files into the temp directory structure."""
    # Root files
    for filename, content in plan.root_module_files.items():
        (root_dir / filename).write_text(content, encoding="utf-8")

    # Module files
    for module in plan.modules:
        mod_dir = root_dir / "modules" / module.name
        mod_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in module.files.items():
            (mod_dir / filename).write_text(content, encoding="utf-8")


async def _run_subprocess(
    cmd: list[str], cwd: str, env: dict | None = None, timeout: int = 60
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously and return (returncode, stdout, stderr)."""
    merged_env = {**os.environ, **(env or {})}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return proc.returncode, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return -1, "", f"Binary not found: {cmd[0]}"


async def _run_tf_validate(root_dir: Path) -> ValidationCheck:
    """Run terraform init + validate. Returns a single ValidationCheck."""
    tf = _s.terraform_binary
    cwd = str(root_dir)

    tf_env = {"TF_PLUGIN_CACHE_DIR": _s.tf_plugin_cache_dir, "TF_INPUT": "false"}

    # terraform init with -backend=false (no real backend needed for validate)
    rc_init, stdout_init, stderr_init = await _run_subprocess(
        [tf, "init", "-backend=false", "-input=false", "-no-color"],
        cwd=cwd,
        env=tf_env,
        timeout=_s.terraform_init_timeout_seconds,
    )

    if rc_init != 0:
        # Try with upgrade flag in case provider cache issues
        rc_init, stdout_init, stderr_init = await _run_subprocess(
            [tf, "init", "-backend=false", "-input=false", "-no-color", "-upgrade"],
            cwd=cwd,
            env=tf_env,
            timeout=90,
        )

    if rc_init != 0:
        return ValidationCheck(
            name="terraform_init",
            status=ValidationStatus.FAILED,
            tool="terraform",
            message=f"terraform init failed:\n{stderr_init or stdout_init}",
            severity="error",
        )

    # terraform validate
    rc_val, stdout_val, stderr_val = await _run_subprocess(
        [tf, "validate", "-json", "-no-color"],
        cwd=cwd,
        timeout=30,
    )

    if rc_val == 0:
        return ValidationCheck(
            name="terraform_validate",
            status=ValidationStatus.PASSED,
            tool="terraform",
            message="All Terraform configuration files are valid.",
            severity="info",
        )
    else:
        # Parse JSON output for better error messages
        try:
            val_out = json.loads(stdout_val)
            diag_msgs = []
            for diag in val_out.get("diagnostics", []):
                loc = ""
                if r := diag.get("range"):
                    loc = f" [{r.get('filename','')}:{r.get('start',{}).get('line','')}]"
                diag_msgs.append(f"{diag.get('severity','').upper()}: {diag.get('summary','')}{loc}")
            msg = "\n".join(diag_msgs) or stderr_val
        except Exception:
            msg = stderr_val or stdout_val

        return ValidationCheck(
            name="terraform_validate",
            status=ValidationStatus.FAILED,
            tool="terraform",
            message=msg,
            severity="error",
        )


async def _run_tflint(root_dir: Path) -> list[ValidationCheck]:
    """Run tflint and parse its JSON output."""
    checks = []
    rc, stdout, stderr = await _run_subprocess(
        [_s.tflint_binary, "--format=json", "--recursive"],
        cwd=str(root_dir),
        timeout=60,
    )

    try:
        result = json.loads(stdout)
        for issue in result.get("issues", []):
            severity = issue.get("rule", {}).get("severity", "warning")
            checks.append(ValidationCheck(
                name=f"tflint_{issue.get('rule', {}).get('name', 'unknown')}",
                status=ValidationStatus.WARNING if severity == "warning" else ValidationStatus.FAILED,
                tool="tflint",
                message=issue.get("message", ""),
                severity=severity,
                file=issue.get("range", {}).get("filename"),
                line=issue.get("range", {}).get("start", {}).get("line"),
            ))
    except Exception:
        if rc == 0:
            checks.append(ValidationCheck(
                name="tflint",
                status=ValidationStatus.PASSED,
                tool="tflint",
                message="No linting issues found.",
                severity="info",
            ))
        else:
            checks.append(ValidationCheck(
                name="tflint",
                status=ValidationStatus.WARNING,
                tool="tflint",
                message=stderr or stdout or "tflint returned non-zero",
                severity="warning",
            ))

    if not checks:
        checks.append(ValidationCheck(
            name="tflint",
            status=ValidationStatus.PASSED,
            tool="tflint",
            message="No linting issues found.",
            severity="info",
        ))

    return checks


async def _run_checkov(root_dir: Path) -> list[ValidationCheck]:
    """Run checkov security scan and parse JSON output."""
    checks = []
    rc, stdout, stderr = await _run_subprocess(
        [_s.checkov_binary, "-d", str(root_dir), "-o", "json", "--quiet"],
        cwd=str(root_dir),
        timeout=120,
    )

    try:
        result = json.loads(stdout) if stdout.strip() else {}
        # checkov returns a single dict when it only runs one check framework
        # (e.g. just "terraform"), but a LIST of one dict per framework when
        # it detects more than one applicable framework in the scanned
        # directory — most commonly "terraform" + "secrets" (checkov's
        # secrets scanner runs over all files by default and can trigger on
        # any string that pattern-matches a credential, e.g. a clarification
        # answer or a future placeholder value that happens to look like an
        # AWS key). Verified 2026-07-08: `checkov -d . -o json` on this
        # project's own generated output returned a plain dict normally, but
        # returned a 2-item list (`[{"check_type": "terraform", ...},
        # {"check_type": "secrets", ...}]`) as soon as one file contained
        # something secret-shaped. The old code called `.get()` straight on
        # `result`, which raises AttributeError on a list and silently
        # degrades to the generic "could not parse JSON output" warning below
        # — meaning every real finding (including any actual leaked secret)
        # would vanish without a trace. Normalizing to a list of framework
        # reports up front so both shapes are handled the same way.
        framework_reports = result if isinstance(result, list) else [result]

        failed: list[dict] = []
        passed: list[dict] = []
        for report in framework_reports:
            failed.extend(report.get("results", {}).get("failed_checks", []))
            passed.extend(report.get("results", {}).get("passed_checks", []))

        if failed:
            for check in failed[:20]:  # Cap at 20 findings
                checks.append(ValidationCheck(
                    name=f"checkov_{check.get('check_id', 'unknown')}",
                    status=ValidationStatus.WARNING,
                    tool="checkov",
                    # `check_type` isn't a real field in checkov's JSON output
                    # (verified 2026-07-08 against a real `checkov -o json` run —
                    # every failed check came back with an empty description,
                    # e.g. "CKV_AWS_79:  — module.compute...", because
                    # check.get('check_type', '') always resolved to '').
                    # checkov's actual human-readable description field is
                    # `check_name` (e.g. "Ensure Instance Metadata Service
                    # Version 1 is not enabled").
                    message=f"{check.get('check_id')}: {check.get('check_name', '')} — {check.get('resource', '')}",
                    severity="warning",
                    resource_id=check.get("resource"),
                    file=check.get("file_path"),
                    line=check.get("file_line_range", [None])[0],
                ))
        if passed:
            checks.append(ValidationCheck(
                name="checkov_passed",
                status=ValidationStatus.PASSED,
                tool="checkov",
                message=f"{len(passed)} security checks passed.",
                severity="info",
            ))
    except Exception:
        checks.append(ValidationCheck(
            name="checkov",
            status=ValidationStatus.WARNING,
            tool="checkov",
            message="checkov scan completed (could not parse JSON output).",
            severity="warning",
        ))

    return checks


async def _run_tf_plan(root_dir: Path) -> tuple[ValidationCheck, str]:
    """Run terraform plan against real AWS sandbox account."""
    env = {
        "AWS_ACCESS_KEY_ID": _s.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": _s.aws_secret_access_key,
        "AWS_DEFAULT_REGION": _s.aws_region,
    }
    if _s.aws_session_token:
        env["AWS_SESSION_TOKEN"] = _s.aws_session_token

    rc, stdout, stderr = await _run_subprocess(
        [_s.terraform_binary, "plan", "-no-color", "-input=false"],
        cwd=str(root_dir),
        env=env,
        timeout=_s.terraform_timeout_seconds,
    )

    output = stdout + stderr

    if rc == 0:
        # Extract plan summary line
        summary = ""
        for line in output.splitlines():
            if "Plan:" in line or "No changes" in line:
                summary = line.strip()
                break

        return ValidationCheck(
            name="terraform_plan",
            status=ValidationStatus.PASSED,
            tool="terraform",
            message=summary or "terraform plan succeeded",
            severity="info",
        ), output
    else:
        return ValidationCheck(
            name="terraform_plan",
            status=ValidationStatus.FAILED,
            tool="terraform",
            message=_extract_tf_error(output),
            severity="error",
        ), output


def _extract_tf_error(output: str) -> str:
    """Extract the most relevant error lines from terraform output."""
    error_lines = []
    for line in output.splitlines():
        if re.search(r"Error:|error:|│", line, re.IGNORECASE):
            clean = re.sub(r"[│╷╵]", "", line).strip()
            if clean:
                error_lines.append(clean)
    return "\n".join(error_lines[:15]) if error_lines else output[:500]
