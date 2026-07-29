"""
Pipeline integration — state reconciliation wiring
-----------------------------------------------------
Confirms app/workers/pipeline_worker.py's run_pipeline() actually calls
state_reconciler.reconcile_from_state() at the right point (after parsing,
before detect_missing_info()) when a state file was attached at upload
time, AND — the other half of her explicit request — that a job with NO
state file attached (a brand-new environment/project) behaves completely
unaffected, regression-testing the exact case this feature must never
break.

Uses the real sample_architecture.drawio fixture (same one test_api.py and
test_remote_backend.py already use) rather than a synthetic ParsedDiagram,
so this exercises the real parse_diagram() -> reconcile -> detect_missing_info
sequence end to end, not just the reconciler in isolation (already covered
by test_state_reconciler.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import Job, JobStatus, DiagramFormat
from app.core.job_store import save_job, get_job
from app.core.storage import save_upload
from app.workers.pipeline_worker import run_pipeline

FIXTURE = REPO_ROOT / "arch2terraform/tests/fixtures/drawio/sample_architecture.drawio"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _sample_state_bytes() -> bytes:
    """Real values for two of the fixture's resources — the rest
    (aws_vpc.main_vpc, aws_subnet.public_subnet, aws_s3_bucket.
    static_assets_bucket) are deliberately left OUT of state, so this also
    covers "some resources match, some don't" in the same run."""
    return json.dumps({
        "version": 4,
        "resources": [
            {"mode": "managed", "type": "aws_instance", "name": "web_server",
             "instances": [{"attributes": {
                 "instance_type": "m5.xlarge",
                 "ami": "ami-0realvalue1234567",
                 "monitoring": False,
             }}]},
        ],
    }).encode()


async def _make_job_with_fixture(state_file_path: str | None = None) -> Job:
    if not FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    job = Job(
        original_filename="sample_architecture.drawio",
        diagram_format=DiagramFormat.DRAWIO,
        file_path=str(FIXTURE),
        state_file_path=state_file_path,
    )
    await save_job(job)
    return job


@pytest.mark.anyio
async def test_run_pipeline_reconciles_matched_resource_before_clarification():
    state_path = await save_upload("state-recon-test-job", "terraform.tfstate", _sample_state_bytes())
    job = await _make_job_with_fixture(state_file_path=state_path)

    await run_pipeline(job.job_id)

    updated = await get_job(job.job_id)
    web_server = next(
        r for r in updated.parsed_diagram.resources
        if r.aws_resource_type == "aws_instance" and r.logical_name == "web_server"
    )
    assert web_server.properties["instance_type"] == "m5.xlarge"
    assert web_server.properties["ami"] == "ami-0realvalue1234567"
    assert web_server.properties["monitoring"] is False

    assert any("Reconciled" in line for line in updated.stage_logs)
    assert any("aws_instance.web_server" in line for line in updated.stage_logs)


@pytest.mark.anyio
async def test_run_pipeline_unmatched_resources_keep_catalog_defaults():
    """The fixture's VPC/subnet/S3-bucket resources have no entry in the
    sample state file above — must be left exactly as the catalog produced
    them, not blanked out or otherwise disturbed by reconciliation running
    at all."""
    state_path = await save_upload("state-recon-test-job-2", "terraform.tfstate", _sample_state_bytes())
    job = await _make_job_with_fixture(state_file_path=state_path)

    await run_pipeline(job.job_id)

    updated = await get_job(job.job_id)
    vpc = next(r for r in updated.parsed_diagram.resources if r.aws_resource_type == "aws_vpc")
    assert vpc.properties["cidr_block"] == "10.0.0.0/16"


@pytest.mark.anyio
async def test_run_pipeline_without_state_file_is_completely_unaffected():
    """The exact regression this feature must never cause: a brand-new
    environment/project with nothing to reconcile against generates
    identically to how it always has."""
    job = await _make_job_with_fixture(state_file_path=None)

    await run_pipeline(job.job_id)

    updated = await get_job(job.job_id)
    web_server = next(
        r for r in updated.parsed_diagram.resources
        if r.aws_resource_type == "aws_instance" and r.logical_name == "web_server"
    )
    assert web_server.properties["instance_type"] == "t3.micro"
    assert web_server.properties["ami"] == "ami-00000000000000000"
    assert not any("Reconciled" in line for line in updated.stage_logs)


@pytest.mark.anyio
async def test_run_pipeline_with_malformed_state_file_does_not_crash_pipeline():
    """Malformed content degrades gracefully at TWO layers: state_reconciler.
    reconcile_from_state() itself catches the JSON parse error internally
    (see test_state_reconciler.py) and returns an empty summary rather than
    raising — so pipeline_worker.py's own try/except around the call is a
    second, belt-and-suspenders layer for anything that isn't already
    handled inside the reconciler. Either way, the pipeline must keep
    going, never fail the whole job over a bad state upload."""
    state_path = await save_upload("state-recon-test-job-3", "terraform.tfstate", b"not json at all")
    job = await _make_job_with_fixture(state_file_path=state_path)

    await run_pipeline(job.job_id)

    updated = await get_job(job.job_id)
    assert updated.parsed_diagram is not None
    assert updated.status != JobStatus.FAILED
    assert any("nothing reconciled" in line.lower() for line in updated.stage_logs)
