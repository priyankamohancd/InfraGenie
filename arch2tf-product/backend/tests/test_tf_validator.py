"""
Sandbox Validator — unit tests
---------------------------------
Focused tests for app/services/sandbox/tf_validator.py's checkov/tflint JSON
parsing, using mocked subprocess output pinned to *real* tool output shapes
(captured 2026-07-08 against actual `checkov==3.3.7` runs against this
project's own generated Terraform — see the audit that found these bugs).
No terraform/tflint/checkov binaries are required to run these.

Two real bugs were found and fixed here, both silent — the module never
raised or logged an error, it just produced worse-than-useless output:

1. `_run_checkov` read `check.get('check_type', ...)` for the human-readable
   finding description, but checkov's real field is `check_name` —
   `check_type` doesn't exist on a failed-check entry at all, so every
   message rendered as e.g. "CKV_AWS_79:  — module.compute..." with the
   description silently blank.
2. `_run_checkov` called `.get()` directly on the parsed JSON, assuming it's
   always a dict — but checkov returns a LIST of one dict per framework
   (e.g. `[{"check_type": "terraform", ...}, {"check_type": "secrets", ...}]`)
   whenever it detects more than one applicable framework in the scanned
   directory, which happens whenever checkov's secrets scanner (on by
   default, scans every file) matches anything credential-shaped. On a list,
   `.get()` raised `AttributeError`, caught by the broad `except Exception`,
   silently discarding every finding — including any real leaked secret —
   in favor of a generic "could not parse JSON output" message.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.services.sandbox import tf_validator
from shared.schemas.models import ValidationStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


# Trimmed but structurally real single-framework checkov output (captured
# from `checkov -d . -o json --quiet` against this project's own generated
# `compute` module — only the fields _run_checkov actually reads are kept).
_CHECKOV_SINGLE_FRAMEWORK = {
    "check_type": "terraform",
    "results": {
        "passed_checks": [{"check_id": "CKV_AWS_88", "resource": "module.compute.aws_instance.web_server"}],
        "failed_checks": [
            {
                "check_id": "CKV_AWS_79",
                "check_name": "Ensure Instance Metadata Service Version 1 is not enabled",
                "resource": "module.compute.aws_instance.web_server",
                "file_path": "/modules/compute/main.tf",
                "file_line_range": [6, 15],
            }
        ],
    },
}

# Real shape when checkov's secrets scanner also triggers (e.g. a
# clarification answer or placeholder that happens to look credential-shaped)
# — a list of per-framework reports instead of a single dict.
_CHECKOV_MULTI_FRAMEWORK = [
    _CHECKOV_SINGLE_FRAMEWORK,
    {
        "check_type": "secrets",
        "results": {
            "passed_checks": [],
            "failed_checks": [
                {
                    "check_id": "CKV_SECRET_2",
                    "check_name": "AWS Access Key",
                    "resource": "fake_secret.tf:1-1",
                    "file_path": "/fake_secret.tf",
                    "file_line_range": [1, 1],
                }
            ],
        },
    },
]


async def _run_checkov_with_mocked_stdout(monkeypatch, stdout_obj) -> list:
    async def fake_run_subprocess(cmd, cwd, env=None, timeout=60):
        return 1, json.dumps(stdout_obj), ""

    monkeypatch.setattr(tf_validator, "_run_subprocess", fake_run_subprocess)
    return await tf_validator._run_checkov(Path("/tmp/irrelevant"))


@pytest.mark.anyio
async def test_checkov_finding_message_includes_real_description(monkeypatch):
    checks = await _run_checkov_with_mocked_stdout(monkeypatch, _CHECKOV_SINGLE_FRAMEWORK)
    finding = next(c for c in checks if c.name == "checkov_CKV_AWS_79")
    assert "Ensure Instance Metadata Service Version 1 is not enabled" in finding.message
    assert finding.status == ValidationStatus.WARNING


@pytest.mark.anyio
async def test_checkov_handles_multi_framework_list_output(monkeypatch):
    checks = await _run_checkov_with_mocked_stdout(monkeypatch, _CHECKOV_MULTI_FRAMEWORK)

    # The generic "could not parse" fallback must NOT fire — this is exactly
    # the silent-failure mode that was the real bug.
    assert not any(c.name == "checkov" for c in checks)

    # Findings from BOTH frameworks must be present.
    terraform_finding = next((c for c in checks if c.name == "checkov_CKV_AWS_79"), None)
    secrets_finding = next((c for c in checks if c.name == "checkov_CKV_SECRET_2"), None)
    assert terraform_finding is not None
    assert secrets_finding is not None
    assert "AWS Access Key" in secrets_finding.message


@pytest.mark.anyio
async def test_checkov_still_reports_passed_count_across_frameworks(monkeypatch):
    checks = await _run_checkov_with_mocked_stdout(monkeypatch, _CHECKOV_MULTI_FRAMEWORK)
    passed_check = next(c for c in checks if c.name == "checkov_passed")
    assert "1 security checks passed" in passed_check.message


@pytest.mark.anyio
async def test_checkov_falls_back_gracefully_on_actually_unparseable_output(monkeypatch):
    async def fake_run_subprocess(cmd, cwd, env=None, timeout=60):
        return 1, "not json at all", ""

    monkeypatch.setattr(tf_validator, "_run_subprocess", fake_run_subprocess)
    checks = await tf_validator._run_checkov(Path("/tmp/irrelevant"))
    assert len(checks) == 1
    assert checks[0].name == "checkov"
    assert checks[0].status == ValidationStatus.WARNING


@pytest.mark.anyio
async def test_existing_state_bytes_exist_at_validate_time(monkeypatch):
    """Same intent as above, checking file existence WHILE the temp dir is
    still alive (inside the mocked validate call itself) rather than after —
    the temp directory is deleted the moment validate_terraform() returns."""
    from shared.schemas.models import TerraformPlan, TerraformModule

    seen = {}

    async def fake_run_tf_validate(root_dir):
        state_path = root_dir / "terraform.tfstate"
        seen["exists"] = state_path.exists()
        seen["content"] = state_path.read_bytes() if state_path.exists() else None
        from app.services.sandbox.tf_validator import ValidationCheck
        return ValidationCheck(name="terraform_validate", status=ValidationStatus.PASSED, tool="terraform", message="ok")

    monkeypatch.setattr(tf_validator, "_run_tf_validate", fake_run_tf_validate)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/fake" if name == tf_validator._s.terraform_binary else None)

    plan = TerraformPlan(
        modules=[TerraformModule(name="compute", source_resources=["r1"], description="",
                                  files={"main.tf": 'resource "aws_instance" "web" {}\n'})],
        root_module_files={"main.tf": "# root\n"},
        resource_count=1,
    )
    fake_state = b'{"version": 4, "resources": []}'
    await tf_validator.validate_terraform(plan, existing_state_bytes=fake_state)

    assert seen["exists"] is True
    assert seen["content"] == fake_state


@pytest.mark.anyio
async def test_no_state_file_written_when_none_provided(monkeypatch):
    """The common case (no upload yet) must NOT create an empty/stray
    terraform.tfstate — that would make `-backend=false` treat it as real
    (empty) state rather than genuinely "nothing to compare against"."""
    from shared.schemas.models import TerraformPlan, TerraformModule

    seen = {}

    async def fake_run_tf_validate(root_dir):
        seen["exists"] = (root_dir / "terraform.tfstate").exists()
        from app.services.sandbox.tf_validator import ValidationCheck
        return ValidationCheck(name="terraform_validate", status=ValidationStatus.PASSED, tool="terraform", message="ok")

    monkeypatch.setattr(tf_validator, "_run_tf_validate", fake_run_tf_validate)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/fake" if name == tf_validator._s.terraform_binary else None)

    plan = TerraformPlan(
        modules=[TerraformModule(name="compute", source_resources=["r1"], description="",
                                  files={"main.tf": 'resource "aws_instance" "web" {}\n'})],
        root_module_files={"main.tf": "# root\n"},
        resource_count=1,
    )
    await tf_validator.validate_terraform(plan)  # no existing_state_bytes

    assert seen["exists"] is False


@pytest.mark.anyio
@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform binary not installed in this environment")
async def test_validate_terraform_passes_real_terraform_json_diagnostics_path():
    """
    The only other test exercising real terraform against generated output
    (test_terraform_validate_e2e.py) shells out to `terraform validate`
    directly and never goes through this module's `-json` diagnostics
    parsing in `_run_tf_validate` — so that parsing path had never actually
    been run against a real terraform binary before this test. Builds a real
    plan from the drawio fixture and runs it through the actual
    `validate_terraform()` entry point, confirming the `terraform_validate`
    check comes back PASSED (not just that some check with that name exists).
    """
    from app.services.parser.diagram_parser import parse_diagram
    from app.services.parser.missing_info_detector import apply_clarification_answers, detect_missing_info
    from app.services.planner.terraform_planner import build_terraform_plan
    from shared.schemas.models import ClarificationAnswer, DiagramFormat

    fixture = REPO_ROOT / "arch2terraform/tests/fixtures/drawio/sample_architecture.drawio"
    if not fixture.exists():
        pytest.skip(f"fixture not found: {fixture}")

    parsed = await parse_diagram(str(fixture), fixture.name)
    clarification, _ = detect_missing_info(parsed, "tf-validator-test-job")
    if clarification:
        answers = [
            ClarificationAnswer(field_key=f.field_key, resource_id=f.resource_id, value=(f.default or "test-value"))
            for f in clarification.fields
        ]
        parsed = apply_clarification_answers(parsed, answers)

    plan = await build_terraform_plan(parsed, aws_region="us-east-1", environment="dev", project_name="tf-validator-test")
    result = await tf_validator.validate_terraform(plan)

    tf_validate_check = next(c for c in result.checks if c.name == "terraform_validate")
    assert tf_validate_check.status == ValidationStatus.PASSED, tf_validate_check.message
