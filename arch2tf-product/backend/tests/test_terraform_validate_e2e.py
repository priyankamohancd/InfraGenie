"""
Real terraform validate — end-to-end
--------------------------------------
Runs the full Phase 2 pipeline (parse -> classify -> resolve -> detect
missing info -> apply default answers -> plan -> generate -> package) against
a real diagram fixture, unzips the packaged output, and runs a real
`terraform init` + `terraform validate` against the ROOT module (which pulls
in every child module via relative `source = "./modules/<name>"` paths, so
validating at the root is the only way to actually exercise the module
wiring, not just each module file in isolation).

This is the gold-standard check for the 2026-07-08 rewiring of Phase 2 to
reuse arch2terraform's classifier/catalog/generator (see
arch2terraform_bridge.py and terraform_planner.py) — `python-hcl2` syntax
parsing was already confirmed clean for these fixtures, but arch2terraform's
own history this session (see its test_nested_block_terraform_validate.py)
showed syntax-clean output can still fail real terraform validate on
client-side format validators and cross-field constraints. Same
`skipif(shutil.which("terraform") is None)` pattern as
arch2terraform/tests/integration/test_pipeline_end_to_end.py.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.services.parser.diagram_parser import parse_diagram
from app.services.parser.missing_info_detector import apply_clarification_answers, detect_missing_info
from app.services.planner.terraform_planner import build_terraform_plan
from app.services.packager.packager import package_output
from shared.schemas.models import ClarificationAnswer, DiagramFormat, Job

FIXTURES = [
    ("drawio", REPO_ROOT / "arch2terraform/tests/fixtures/drawio/sample_architecture.drawio", DiagramFormat.DRAWIO),
    ("excalidraw", REPO_ROOT / "arch2terraform/tests/fixtures/excalidraw/sample_architecture.excalidraw", DiagramFormat.EXCALIDRAW),
]


@pytest.fixture
def anyio_backend():
    # No conftest.py in this test package, so (as in test_api.py) each file
    # using @pytest.mark.anyio needs its own copy of this fixture.
    return "asyncio"


async def _build_package(fixture_path: Path, fmt: DiagramFormat) -> tuple[str, str]:
    """Runs the full pipeline for one fixture, returns (zip_path, readme)."""
    parsed = await parse_diagram(str(fixture_path), fixture_path.name)

    clarification, _ = detect_missing_info(parsed, "e2e-test-job")
    if clarification:
        # Answer every question with its default (or a safe fallback) so the
        # pipeline can proceed unattended, the same way a real user accepting
        # every suggested default would.
        answers = [
            ClarificationAnswer(field_key=f.field_key, resource_id=f.resource_id, value=(f.default or "e2e-test-value"))
            for f in clarification.fields
        ]
        parsed = apply_clarification_answers(parsed, answers)

    plan = await build_terraform_plan(parsed, aws_region="us-east-1", environment="dev", project_name="e2e-test")

    job = Job()
    job.original_filename = fixture_path.name
    job.diagram_format = fmt
    job.parsed_diagram = parsed
    job.terraform_plan = plan

    return await package_output(job)


@pytest.mark.anyio
@pytest.mark.parametrize("name,fixture_path,fmt", FIXTURES, ids=[f[0] for f in FIXTURES])
async def test_pipeline_produces_a_package(name, fixture_path, fmt):
    """Structural check that always runs (no terraform binary needed):
    confirms the pipeline completes and produces a non-empty ZIP with the
    expected module layout, so a missing terraform binary doesn't hide a
    real pipeline break."""
    if not fixture_path.exists():
        pytest.skip(f"fixture not found: {fixture_path}")

    zip_path, readme = await _build_package(fixture_path, fmt)
    assert Path(zip_path).exists()
    assert Path(zip_path).stat().st_size > 0
    assert "Known limitations" in readme

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert "terraform/main.tf" in names
        assert "terraform/versions.tf" in names
        assert any(n.startswith("terraform/modules/") for n in names)


@pytest.mark.anyio
@pytest.mark.parametrize("name,fixture_path,fmt", FIXTURES, ids=[f[0] for f in FIXTURES])
@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform binary not installed in this environment")
async def test_generated_package_passes_real_terraform_validate(name, fixture_path, fmt):
    """The gold-standard check: unzip a real generated package and run real
    `terraform init` + `terraform validate` against the root module."""
    if not fixture_path.exists():
        pytest.skip(f"fixture not found: {fixture_path}")

    zip_path, _ = await _build_package(fixture_path, fmt)

    with tempfile.TemporaryDirectory(prefix="arch2tf_e2e_validate_") as tmpdir:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)

        root_dir = Path(tmpdir) / "terraform"
        assert root_dir.exists(), f"expected 'terraform/' inside the package, found: {list(Path(tmpdir).iterdir())}"

        init = subprocess.run(
            ["terraform", "init", "-backend=false", "-input=false"],
            cwd=root_dir, capture_output=True, text=True, timeout=180,
        )
        assert init.returncode == 0, f"[{name}] terraform init failed:\n{init.stdout}\n{init.stderr}"

        validate = subprocess.run(
            ["terraform", "validate"],
            cwd=root_dir, capture_output=True, text=True, timeout=60,
        )
        assert validate.returncode == 0, f"[{name}] terraform validate failed:\n{validate.stdout}\n{validate.stderr}"
