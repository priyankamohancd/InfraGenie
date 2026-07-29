"""
API + Pipeline Integration Tests
----------------------------------
Tests the full HTTP API surface and pipeline state machine.
Run with: pytest tests/test_api.py -v
"""
import sys
import os
import json
import asyncio
from pathlib import Path
import pytest

# Patch sys.path before any app imports.
# REPO_ROOT must be the parent of BOTH arch2terraform/ and arch2tf-product/
# (i.e. ~/work/thesis) — this was previously parents[2], which only reaches
# arch2tf-product/ itself (backend/tests/test_api.py -> tests(0)/backend(1)/
# arch2tf-product(2)), one level too shallow to find the arch2terraform
# sibling at all. Also: the importable arch2terraform package lives under
# arch2terraform/src/, not arch2terraform/ itself.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.job_store import save_job, get_job
from shared.schemas.models import (
    Job, JobStatus, ParsedDiagram, ParsedResource, DiagramFormat,
    TerraformPlan, TerraformModule, ValidationResult, ValidationStatus,
    ValidationCheck, ClarificationRequest, ClarificationField,
)

FIXTURES = Path(__file__).parent / "fixtures"
# These filenames previously referenced fixtures that don't exist in
# arch2terraform/tests/fixtures/ ("three_tier_aws.drawio",
# "serverless_arch.excalidraw") — pointed at the real ones so these tests
# actually exercise arch2terraform_bridge.py end to end instead of always
# skipping.
DRAWIO_FIXTURE = REPO_ROOT / "arch2terraform/tests/fixtures/drawio/sample_architecture.drawio"
EXCALIDRAW_FIXTURE = REPO_ROOT / "arch2terraform/tests/fixtures/excalidraw/sample_architecture.excalidraw"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _make_job(status=JobStatus.DONE, **kwargs) -> Job:
    job = Job(**kwargs)
    job.status = status
    return job


def _make_parsed_diagram() -> ParsedDiagram:
    return ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(id="r1", aws_resource_type="aws_vpc",
                           logical_name="main_vpc", label="Main VPC", confidence=0.95),
            ParsedResource(id="r2", aws_resource_type="aws_instance",
                           logical_name="web_server", label="Web Server",
                           properties={"instance_type": "t3.micro", "ami": "ami-0abc"},
                           confidence=0.9),
            ParsedResource(id="r3", aws_resource_type="aws_db_instance",
                           logical_name="postgres_db", label="Postgres DB",
                           properties={"engine": "postgres"}, confidence=0.85),
        ],
        connections=[],
        total_resources=3,
        total_connections=0,
        resource_type_summary={"aws_vpc": 1, "aws_instance": 1, "aws_db_instance": 1},
    )


def _make_plan() -> TerraformPlan:
    return TerraformPlan(
        modules=[
            TerraformModule(
                name="networking",
                source_resources=["r1"],
                description="VPC and networking",
                files={
                    "main.tf": 'resource "aws_vpc" "main_vpc" {\n  cidr_block = "10.0.0.0/16"\n}\n',
                    "variables.tf": 'variable "aws_region" { type = string }\n',
                    "outputs.tf": 'output "main_vpc_id" { value = aws_vpc.main_vpc.id }\n',
                    "versions.tf": 'terraform { required_version = ">= 1.5.0" }\n',
                },
            ),
            TerraformModule(
                name="compute",
                source_resources=["r2"],
                description="EC2 compute",
                files={
                    "main.tf": 'resource "aws_instance" "web_server" {\n  instance_type = "t3.micro"\n}\n',
                    "variables.tf": 'variable "aws_region" { type = string }\n',
                    "outputs.tf": 'output "web_server_id" { value = aws_instance.web_server.id }\n',
                    "versions.tf": 'terraform { required_version = ">= 1.5.0" }\n',
                },
            ),
        ],
        root_module_files={
            "main.tf": 'provider "aws" { region = var.aws_region }\nmodule "networking" { source = "./modules/networking" }\n',
            "variables.tf": 'variable "aws_region" { type = string\n  default = "us-east-1" }\n',
            "outputs.tf": "# outputs\n",
            "versions.tf": 'terraform { required_version = ">= 1.5.0" }\n',
            "locals.tf": 'locals { common_tags = {} }\n',
            "backend.tf": "# backend config\n",
        },
        resource_count=2,
    )


def _make_validation() -> ValidationResult:
    return ValidationResult(
        overall_status=ValidationStatus.PASSED,
        checks=[
            ValidationCheck(
                name="terraform_validate", status=ValidationStatus.PASSED,
                tool="terraform", message="All configuration files are valid.",
            ),
        ],
        passed_count=1,
        failed_count=0,
        warning_count=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.anyio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "arch2terraform" in resp.json()["service"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_upload_drawio_accepted(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("test.drawio", content, "text/xml")},
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "uploaded"


@pytest.mark.anyio
async def test_upload_excalidraw_accepted(client):
    if not EXCALIDRAW_FIXTURE.exists():
        pytest.skip("excalidraw fixture not found")
    content = EXCALIDRAW_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("arch.excalidraw", content, "application/json")},
        data={"aws_region": "eu-west-1", "environment": "staging"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data


@pytest.mark.anyio
async def test_upload_invalid_extension_rejected(client):
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("document.pdf", b"fake content", "application/pdf")},
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_creates_job_in_store(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("test.drawio", content, "text/xml")},
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    job_id = resp.json()["job_id"]
    job = await get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert job.original_filename == "test.drawio"


@pytest.mark.anyio
async def test_upload_with_valid_vars_yaml_stores_input_vars(client):
    """The optional `vars_file` param (2026-07-08's "reuse an existing
    vars.yaml" case) must be parsed and stashed on job.input_vars so
    missing_info_detector.detect_missing_info() can read it once the
    pipeline reaches clarification."""
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    vars_yaml = b"globals:\n  project_name: infra-genie\nresources:\n  aws_instance.web_server:\n    instance_type: m5.large\n"
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "vars_file": ("vars.yaml", vars_yaml, "application/x-yaml"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    job = await get_job(job_id)
    assert job.input_vars is not None
    assert job.input_vars["globals"]["project_name"] == "infra-genie"
    assert job.input_vars["resources"]["aws_instance.web_server"]["instance_type"] == "m5.large"
    # vars.yaml's globals should win over the upload form's own aws_region/
    # environment fields when both are present — project_name isn't a form
    # field at all, so it can only have come from vars.yaml here.
    project_answer = next(a for a in job.clarification_answers if a.field_key == "project_name")
    assert project_answer.value == "infra-genie"


@pytest.mark.anyio
async def test_upload_with_invalid_yaml_vars_file_returns_400(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "vars_file": ("vars.yaml", b"resources: [unclosed", "application/x-yaml"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 400
    assert "not valid YAML" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_with_non_dict_vars_yaml_returns_400(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "vars_file": ("vars.yaml", b"- just\n- a\n- list\n", "application/x-yaml"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 400
    assert "mapping" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_without_vars_file_leaves_input_vars_none(client):
    """The common case — building from scratch — must not silently set
    job.input_vars to some falsy-but-not-None value that would confuse
    detect_missing_info()'s None-check."""
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("test.drawio", content, "text/xml")},
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    job_id = resp.json()["job_id"]
    job = await get_job(job_id)
    assert job.input_vars is None


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Upload endpoint — optional state_file (2026-07-29, state reconciliation)
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_STATE = json.dumps({
    "version": 4,
    "resources": [
        {"mode": "managed", "type": "aws_instance", "name": "web_server",
         "instances": [{"attributes": {"instance_type": "m5.large"}}]},
    ],
}).encode()


@pytest.mark.anyio
async def test_upload_with_valid_state_file_stores_state_file_path(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "state_file": ("terraform.tfstate", _SAMPLE_STATE, "application/json"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    job = await get_job(job_id)
    assert job.state_file_path is not None


@pytest.mark.anyio
async def test_upload_with_invalid_json_state_file_returns_400(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "state_file": ("terraform.tfstate", b"not json at all", "application/json"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_with_non_state_json_returns_400(client):
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={
            "file": ("test.drawio", content, "text/xml"),
            "state_file": ("terraform.tfstate", b'{"hello": "world"}', "application/json"),
        },
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    assert resp.status_code == 400
    assert "resources" in resp.json()["detail"]


@pytest.mark.anyio
async def test_upload_without_state_file_leaves_state_file_path_none(client):
    """The new-environment/new-project case — must be completely
    unaffected by this feature existing at all."""
    if not DRAWIO_FIXTURE.exists():
        pytest.skip("draw.io fixture not found")
    content = DRAWIO_FIXTURE.read_bytes()
    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("test.drawio", content, "text/xml")},
        data={"aws_region": "us-east-1", "environment": "dev"},
    )
    job_id = resp.json()["job_id"]
    job = await get_job(job_id)
    assert job.state_file_path is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Job status endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_get_job_status_not_found(client):
    resp = await client.get("/api/v1/jobs/nonexistent-job-id")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_job_status_returns_correct_fields(client):
    job = _make_job(status=JobStatus.PARSING)
    job.original_filename = "test.drawio"
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job.job_id
    assert data["status"] == "parsing"
    assert "progress_percent" in data
    assert "current_stage" in data
    assert "stage_logs" in data


@pytest.mark.anyio
async def test_job_progress_percent_parsing(client):
    job = _make_job(status=JobStatus.PARSING)
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.json()["progress_percent"] == 15


@pytest.mark.anyio
async def test_job_progress_percent_done(client):
    job = _make_job(status=JobStatus.DONE)
    job.zip_path = "/tmp/fake.zip"
    # Create a fake zip file so the download check works
    Path("/tmp/fake.zip").touch()
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}")
    data = resp.json()
    assert data["progress_percent"] == 100
    assert data["zip_ready"] is True


@pytest.mark.anyio
async def test_job_shows_clarification_when_needed(client):
    job = _make_job(status=JobStatus.NEEDS_CLARIFY)
    job.clarification_request = ClarificationRequest(
        job_id=job.job_id,
        fields=[
            ClarificationField(
                field_key="ami",
                resource_id="r1",
                resource_label="Web Server",
                question="AMI ID for Web Server?",
                input_type="text",
            )
        ]
    )
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}")
    data = resp.json()
    assert data["clarification_request"] is not None
    assert len(data["clarification_request"]["fields"]) == 1


@pytest.mark.anyio
async def test_job_hides_clarification_when_not_needed(client):
    job = _make_job(status=JobStatus.PLANNING)
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}")
    assert resp.json()["clarification_request"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Clarification endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_clarify_wrong_state_returns_409(client):
    job = _make_job(status=JobStatus.PLANNING)
    await save_job(job)
    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/clarify",
        json={"job_id": job.job_id, "answers": []},
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_clarify_not_found_returns_404(client):
    resp = await client.post(
        "/api/v1/jobs/ghost-job/clarify",
        json={"job_id": "ghost-job", "answers": []},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_clarify_accepts_answers_and_resumes(client):
    job = _make_job(status=JobStatus.NEEDS_CLARIFY)
    job.parsed_diagram = _make_parsed_diagram()
    job.clarification_request = ClarificationRequest(
        job_id=job.job_id,
        fields=[
            ClarificationField(
                field_key="ami", resource_id="r2",
                resource_label="Web Server", question="AMI?",
                input_type="text", default="ami-0abc",
            )
        ]
    )
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/clarify",
        json={
            "job_id": job.job_id,
            "answers": [{"field_key": "ami", "resource_id": "r2", "value": "ami-0abc123"}]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answers_received"] == 1
    assert data["status"] == "resuming"


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Resource review corrections (Phase B — 2026-07-24)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_clarify_deletes_resource_and_its_connections(client):
    job = _make_job(status=JobStatus.NEEDS_CLARIFY)
    pd = _make_parsed_diagram()
    # r1 -> r2 containment edge so we can assert it's dropped along with r1
    from shared.schemas.models import ParsedConnection
    pd.connections = [ParsedConnection(source_id="r1", target_id="r2", connection_type="containment")]
    pd.total_connections = 1
    job.parsed_diagram = pd
    job.clarification_request = ClarificationRequest(job_id=job.job_id, fields=[])
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/clarify",
        json={
            "job_id": job.job_id,
            "answers": [],
            "resource_corrections": {"deleted_ids": ["r1"], "relabeled": {}, "retyped": {}},
        },
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    ids = {r.id for r in updated.parsed_diagram.resources}
    assert "r1" not in ids
    assert ids == {"r2", "r3"}
    assert updated.parsed_diagram.total_resources == 2
    assert updated.parsed_diagram.connections == []  # edge referencing deleted r1 dropped too
    assert updated.parsed_diagram.total_connections == 0
    assert "aws_vpc" not in updated.parsed_diagram.resource_type_summary


@pytest.mark.anyio
async def test_clarify_relabels_and_retypes_resource(client):
    job = _make_job(status=JobStatus.NEEDS_CLARIFY)
    job.parsed_diagram = _make_parsed_diagram()
    job.clarification_request = ClarificationRequest(job_id=job.job_id, fields=[])
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/clarify",
        json={
            "job_id": job.job_id,
            "answers": [],
            "resource_corrections": {
                "deleted_ids": [],
                "relabeled": {"r1": "Corrected VPC Name"},
                "retyped": {"r3": "aws_rds_cluster"},
            },
        },
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    by_id = {r.id: r for r in updated.parsed_diagram.resources}
    assert by_id["r1"].label == "Corrected VPC Name"
    assert by_id["r3"].aws_resource_type == "aws_rds_cluster"
    assert by_id["r3"].confidence == 1.0  # user-confirmed retype is no longer a guess


@pytest.mark.anyio
async def test_clarify_drops_stale_answers_for_deleted_resource(client):
    """An answer submitted for a resource that was also just deleted in the
    same request must not be merged into job.clarification_answers — it
    would otherwise sit there orphaned forever (apply_clarification_answers
    would simply never match it, but there's no reason to keep it)."""
    job = _make_job(status=JobStatus.NEEDS_CLARIFY)
    job.parsed_diagram = _make_parsed_diagram()
    job.clarification_request = ClarificationRequest(job_id=job.job_id, fields=[])
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/clarify",
        json={
            "job_id": job.job_id,
            "answers": [{"field_key": "ami", "resource_id": "r2", "value": "ami-0abc123"}],
            "resource_corrections": {"deleted_ids": ["r2"], "relabeled": {}, "retyped": {}},
        },
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    assert all(a.resource_id != "r2" for a in updated.clarification_answers)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Preview endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_preview_returns_files(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    job.readme_content = "# Test README"
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    assert resp.status_code == 200
    data = resp.json()
    assert "files" in data
    assert "main.tf" in data["files"]
    assert data["file_count"] > 0


@pytest.mark.anyio
async def test_preview_includes_module_files(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    data = resp.json()
    files = data["files"]
    # Should contain module files with path prefix
    assert any("modules/" in k for k in files.keys())


@pytest.mark.anyio
async def test_preview_no_plan_returns_409(client):
    job = _make_job(status=JobStatus.PARSING)
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_preview_includes_vars_yaml_when_present(client):
    """Found 2026-07-29: vars.yaml was already bundled into the downloadable
    ZIP by packager.py, but /preview's `files` dict never included it, so it
    never showed up in the frontend's file tree or edit mode at all — same
    root-level treatment as README.md now."""
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    job.generated_vars_yaml = "globals:\n  aws_region: us-east-1\n"
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    data = resp.json()
    assert "vars.yaml" in data["files"]
    assert data["files"]["vars.yaml"] == "globals:\n  aws_region: us-east-1\n"


@pytest.mark.anyio
async def test_preview_omits_vars_yaml_when_not_generated(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    assert "vars.yaml" not in resp.json()["files"]


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Edit-file endpoint (2026-07-29, in-browser Review-screen edit mode)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_edit_root_file_persists_and_is_reflected_in_preview(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    new_content = 'provider "aws" { region = "eu-west-1" }\n'
    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/main.tf", json={"content": new_content}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["file_key"] == "main.tf"
    assert "re-validated" in body["message"]

    preview = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    assert preview.json()["files"]["main.tf"] == new_content


@pytest.mark.anyio
async def test_edit_module_file_uses_modules_prefix_key(client):
    """The edit endpoint must key module files the SAME way /preview
    already does ("modules/<name>/<file>") — NOT apply_runner.py's
    _all_files() convention ("<name>/<file>", no "modules/" prefix), which
    is a different key scheme used elsewhere in this codebase for a
    different purpose (BlockedVariable ids)."""
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    new_content = 'resource "aws_vpc" "main_vpc" {\n  cidr_block = "10.1.0.0/16"\n}\n'
    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/modules/networking/main.tf",
        json={"content": new_content},
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    assert updated.terraform_plan.modules[0].files["main.tf"] == new_content


@pytest.mark.anyio
async def test_edit_readme_updates_job_readme_content(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    job.readme_content = "# Old README"
    await save_job(job)

    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/README.md", json={"content": "# New README"}
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    assert updated.readme_content == "# New README"


@pytest.mark.anyio
async def test_edit_vars_yaml_updates_job_generated_vars_yaml(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    job.generated_vars_yaml = "globals:\n  aws_region: us-east-1\n"
    await save_job(job)

    new_yaml = "globals:\n  aws_region: eu-west-1\n"
    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/vars.yaml", json={"content": new_yaml}
    )
    assert resp.status_code == 200

    updated = await get_job(job.job_id)
    assert updated.generated_vars_yaml == new_yaml

    preview = await client.get(f"/api/v1/jobs/{job.job_id}/preview")
    assert preview.json()["files"]["vars.yaml"] == new_yaml


@pytest.mark.anyio
async def test_edit_unknown_root_file_returns_404(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/nonexistent.tf", json={"content": "x"}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_edit_unknown_module_returns_404(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/modules/nonexistent/main.tf",
        json={"content": "x"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_edit_without_a_terraform_plan_returns_409(client):
    job = _make_job(status=JobStatus.PARSING)
    await save_job(job)

    resp = await client.put(
        f"/api/v1/jobs/{job.job_id}/files/main.tf", json={"content": "x"}
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_edit_nonexistent_job_returns_404(client):
    resp = await client.put(
        "/api/v1/jobs/does-not-exist/files/main.tf", json={"content": "x"}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_edit_is_logged_on_the_job(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    await client.put(f"/api/v1/jobs/{job.job_id}/files/main.tf", json={"content": "x"})

    updated = await get_job(job.job_id)
    assert any("Edited main.tf" in line for line in updated.stage_logs)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Download endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_download_not_ready_returns_409(client):
    job = _make_job(status=JobStatus.VALIDATING)
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}/download")
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_download_no_zip_returns_500(client):
    job = _make_job(status=JobStatus.DONE)
    job.zip_path = "/nonexistent/path/file.zip"
    await save_job(job)
    resp = await client.get(f"/api/v1/jobs/{job.job_id}/download")
    assert resp.status_code == 500


@pytest.mark.anyio
async def test_download_sends_exactly_one_content_disposition_header(client, tmp_path):
    """2026-07-24: FileResponse's `filename=` already sets Content-Disposition
    itself — also passing it explicitly via `headers=` sent the header
    TWICE, which Chrome refuses outright with
    ERR_RESPONSE_HEADERS_MULTIPLE_CONTENT_DISPOSITION (found via a real
    browser download attempt against a completed job)."""
    zip_path = tmp_path / "output.zip"
    zip_path.write_bytes(b"PK\x03\x04fake zip content")

    job = _make_job(status=JobStatus.DONE, original_filename="diagram.drawio")
    job.zip_path = str(zip_path)
    await save_job(job)

    resp = await client.get(f"/api/v1/jobs/{job.job_id}/download")
    assert resp.status_code == 200

    cd_headers = [v for k, v in resp.headers.raw if k.lower() == b"content-disposition"]
    assert len(cd_headers) == 1, f"Expected exactly one Content-Disposition header, got {cd_headers}"
    assert b"attachment" in cd_headers[0]
    assert job.job_id[:8].encode() in cd_headers[0]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Delete endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_delete_job(client):
    job = _make_job(status=JobStatus.DONE)
    await save_job(job)
    resp = await client.delete(f"/api/v1/jobs/{job.job_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == job.job_id
    assert await get_job(job.job_id) is None


@pytest.mark.anyio
async def test_delete_nonexistent_returns_404(client):
    resp = await client.delete("/api/v1/jobs/ghost-job-delete")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 8. Push to GitHub endpoint
# ─────────────────────────────────────────────────────────────────────────────
# The underlying GitHub API call sequence (repo create -> git data -> PR) is
# already thoroughly covered against a MockTransport in
# test_github_pusher.py. These tests only cover the route's own wiring:
# request/response shapes, status codes for the missing-job/no-plan/
# GitHub-error cases, and — importantly — that the submitted token is never
# echoed back or otherwise leaked in any response.

@pytest.mark.anyio
async def test_push_to_github_success(client, monkeypatch):
    from app.api.routes import pipeline as pipeline_routes

    async def fake_push(job, github_token, repo_full_name):
        assert github_token == "ghp_faketoken123"
        assert repo_full_name == "priyankamohan/infra-repo"
        return {
            "repo_url": "https://github.com/priyankamohan/infra-repo",
            "pr_url": "https://github.com/priyankamohan/infra-repo/pull/1",
            "repo_full_name": "priyankamohan/infra-repo",
            "environment": "dev",
        }

    monkeypatch.setattr(pipeline_routes, "push_job_to_existing_github_repo", fake_push)

    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/push-to-github",
        json={"github_token": "ghp_faketoken123", "repo_full_name": "priyankamohan/infra-repo"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_url"] == "https://github.com/priyankamohan/infra-repo"
    assert data["pr_url"] == "https://github.com/priyankamohan/infra-repo/pull/1"
    assert data["environment"] == "dev"
    # The token must never appear anywhere in the response.
    assert "ghp_faketoken123" not in resp.text


@pytest.mark.anyio
async def test_push_to_github_job_not_found(client):
    resp = await client.post(
        "/api/v1/jobs/ghost-job/push-to-github",
        json={"github_token": "ghp_faketoken123", "repo_full_name": "priyankamohan/infra-repo"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_push_to_github_no_plan_yet(client):
    job = _make_job(status=JobStatus.PARSING)  # no terraform_plan set
    await save_job(job)
    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/push-to-github",
        json={"github_token": "ghp_faketoken123", "repo_full_name": "priyankamohan/infra-repo"},
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_push_to_github_missing_repo_full_name_returns_422(client):
    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)
    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/push-to-github",
        json={"github_token": "ghp_faketoken123"},  # repo_full_name omitted
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_push_to_github_propagates_github_error_as_502(client, monkeypatch):
    from app.api.routes import pipeline as pipeline_routes
    from app.services.github.github_pusher import GithubPushError

    async def fake_push(job, github_token, repo_full_name):
        raise GithubPushError("GitHub rejected that token (401 Unauthorized) — check it's valid.")

    monkeypatch.setattr(pipeline_routes, "push_job_to_existing_github_repo", fake_push)

    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/push-to-github",
        json={"github_token": "bad-token", "repo_full_name": "priyankamohan/infra-repo"},
    )
    assert resp.status_code == 502
    assert "401" in resp.json()["detail"]
    assert "bad-token" not in resp.text


# ─────────────────────────────────────────────────────────────────────────────
# 9. Shared models unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_job_log_appends():
    job = Job()
    job.log("Step 1")
    job.log("Step 2")
    assert len(job.stage_logs) == 2
    assert "Step 1" in job.stage_logs[0]
    assert "Step 2" in job.stage_logs[1]


def test_job_progress_map_covers_all_statuses():
    from shared.schemas.models import JOB_PROGRESS
    for status in JobStatus:
        assert status in JOB_PROGRESS, f"Missing progress for {status}"


def test_validation_result_counts():
    vr = ValidationResult(
        overall_status=ValidationStatus.WARNING,
        checks=[
            ValidationCheck(name="a", status=ValidationStatus.PASSED, tool="tf", message="ok"),
            ValidationCheck(name="b", status=ValidationStatus.WARNING, tool="tflint", message="warn"),
            ValidationCheck(name="c", status=ValidationStatus.FAILED, tool="checkov", message="fail"),
        ],
        passed_count=1, failed_count=1, warning_count=1,
    )
    assert vr.passed_count == 1
    assert vr.failed_count == 1
    assert vr.warning_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 9. Missing info detector
# ─────────────────────────────────────────────────────────────────────────────

def test_detector_asks_for_ec2_ami():
    from app.services.parser.missing_info_detector import detect_missing_info
    parsed = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(
                id="r1", aws_resource_type="aws_instance",
                logical_name="web", label="Web Server",
                properties={"ami": "# TODO"},
                confidence=0.9,
            )
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    req, _ = detect_missing_info(parsed, "test-job")
    assert req is not None
    ami_field = next((f for f in req.fields if f.field_key == "ami"), None)
    assert ami_field is not None
    assert "AMI" in ami_field.question


def test_detector_asks_for_rds_engine():
    from app.services.parser.missing_info_detector import detect_missing_info
    parsed = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(
                id="r1", aws_resource_type="aws_db_instance",
                logical_name="db", label="Database",
                properties={},
                confidence=0.9,
            )
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    req, _ = detect_missing_info(parsed, "test-job")
    assert req is not None
    assert any(f.field_key == "engine" for f in req.fields)


def test_detector_no_clarification_for_known_resources():
    from app.services.parser.missing_info_detector import detect_missing_info
    # A resource with no mandatory TODO fields and high confidence
    # Note: global fields (region, env) will always be asked
    parsed = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(
                id="r1", aws_resource_type="aws_kms_key",
                logical_name="kms", label="KMS Key",
                properties={"enable_key_rotation": "true"},
                confidence=0.95,
            )
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    req, _ = detect_missing_info(parsed, "test-job")
    # Global fields will still be there — that's expected
    if req:
        non_global = [f for f in req.fields if f.resource_id != "target_global"]
        assert len(non_global) == 0  # no resource-specific questions


def test_low_confidence_triggers_reclassify():
    from app.services.parser.missing_info_detector import detect_missing_info
    parsed = ParsedDiagram(
        source_format=DiagramFormat.IMAGE,
        resources=[
            ParsedResource(
                id="r1", aws_resource_type="aws_null_resource",
                logical_name="unknown", label="Unknown Box",
                confidence=0.3,  # below threshold
            )
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    req, _ = detect_missing_info(parsed, "test-job")
    assert req is not None
    reclassify = next((f for f in req.fields if f.field_key.startswith("reclassify_")), None)
    assert reclassify is not None


def test_apply_answers_updates_resource():
    from app.services.parser.missing_info_detector import apply_clarification_answers
    from shared.schemas.models import ClarificationAnswer
    parsed = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(
                id="r1", aws_resource_type="aws_instance",
                logical_name="web", label="Web Server",
                properties={"ami": "# TODO"},
            )
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    answers = [ClarificationAnswer(field_key="ami", resource_id="r1", value="ami-0abc123")]
    result = apply_clarification_answers(parsed, answers)
    assert result.resources[0].properties["ami"] == "ami-0abc123"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Terraform Planner
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_planner_creates_modules():
    from app.services.planner.terraform_planner import build_terraform_plan
    parsed = _make_parsed_diagram()
    plan = await build_terraform_plan(parsed, "us-east-1", "dev", "test-project")
    assert plan.resource_count > 0
    assert len(plan.modules) > 0
    assert plan.root_module_files.get("main.tf") is not None


@pytest.mark.anyio
async def test_planner_vpc_goes_to_networking_module():
    from app.services.planner.terraform_planner import build_terraform_plan
    parsed = ParsedDiagram(
        source_format=DiagramFormat.DRAWIO,
        resources=[
            ParsedResource(id="vpc1", aws_resource_type="aws_vpc",
                           logical_name="main_vpc", label="Main VPC")
        ],
        connections=[], total_resources=1, total_connections=0,
    )
    plan = await build_terraform_plan(parsed)
    mod_names = [m.name for m in plan.modules]
    assert "networking" in mod_names


@pytest.mark.anyio
async def test_planner_generates_main_tf():
    from app.services.planner.terraform_planner import build_terraform_plan
    parsed = _make_parsed_diagram()
    plan = await build_terraform_plan(parsed)
    for mod in plan.modules:
        assert "main.tf" in mod.files
        assert "variables.tf" in mod.files
        assert "outputs.tf" in mod.files
        assert "versions.tf" in mod.files


@pytest.mark.anyio
async def test_planner_root_has_backend_tf():
    from app.services.planner.terraform_planner import build_terraform_plan
    parsed = _make_parsed_diagram()
    plan = await build_terraform_plan(parsed)
    assert "backend.tf" in plan.root_module_files
    assert "backend" in plan.root_module_files["backend.tf"]


# ─────────────────────────────────────────────────────────────────────────────
# 11. Packager
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_packager_creates_zip():
    import os
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "test.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()

    zip_path, readme = await package_output(job)
    assert os.path.exists(zip_path)
    assert zip_path.endswith(".zip")
    assert os.path.getsize(zip_path) > 0


@pytest.mark.anyio
async def test_packager_readme_contains_resource_table():
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()

    _, readme = await package_output(job)
    assert "aws_vpc" in readme
    assert "aws_instance" in readme


@pytest.mark.anyio
async def test_packager_readme_contains_usage():
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()

    _, readme = await package_output(job)
    assert "terraform init" in readme
    assert "terraform plan" in readme
    assert "terraform apply" in readme


@pytest.mark.anyio
async def test_packager_bundles_state_plan_output_and_diagram():
    """Tasks 41-44: her explicit request ("I want the state file, terraform
    plan, the modules and the architecture diagram intact") together in the
    download ZIP. Verifies all three land inside terraform/ at the paths
    packager.py writes them to, with the exact bytes/content preserved."""
    import zipfile
    from app.services.packager.packager import package_output
    from app.core.storage import save_upload

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.validation_result.terraform_plan_output = "Plan: 2 to add, 0 to change, 0 to destroy."
    job.parsed_diagram = _make_parsed_diagram()

    fake_state_bytes = b'{"version": 4, "terraform_version": "1.6.0", "resources": []}'
    fake_diagram_bytes = b"<mxfile><mxGraphModel/></mxfile>"

    job.state_file_path = await save_upload(job.job_id, "uploaded_terraform.tfstate", fake_state_bytes)
    job.file_path = await save_upload(job.job_id, job.original_filename, fake_diagram_bytes)

    zip_path, readme = await package_output(job)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "terraform/terraform.tfstate" in names
        assert "terraform/terraform-plan-output.txt" in names
        assert f"terraform/diagram/{job.original_filename}" in names

        assert zf.read("terraform/terraform.tfstate") == fake_state_bytes
        assert zf.read("terraform/terraform-plan-output.txt").decode("utf-8") == job.validation_result.terraform_plan_output
        assert zf.read(f"terraform/diagram/{job.original_filename}") == fake_diagram_bytes

    # README should call out the state file with the sensitivity warning.
    assert "terraform.tfstate" in readme
    assert "sensitive = true" in readme


@pytest.mark.anyio
async def test_packager_omits_state_file_when_none_uploaded():
    """The common case (no state uploaded yet) must not create a stray/empty
    terraform.tfstate in the ZIP, and the README's state-handling warning
    should only appear when there actually is one."""
    import zipfile
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()
    # No state_file_path, no file_path — neither state nor diagram uploaded.

    zip_path, readme = await package_output(job)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "terraform/terraform.tfstate" not in names
        assert not any(n.startswith("terraform/diagram/") for n in names)
        # terraform_plan_output defaults to "" in _make_validation(), so no plan-output file either.
        assert "terraform/terraform-plan-output.txt" not in names

    assert "the existing state file you uploaded" not in readme


@pytest.mark.anyio
async def test_packager_bundles_vars_yaml_when_generated():
    """vars.yaml (2026-07-08's from-scratch-vs-reuse feature) must land in
    the ZIP right alongside the other bundled artifacts, with its exact
    content preserved so it can be re-uploaded as-is next time."""
    import zipfile
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()
    job.generated_vars_yaml = (
        "globals:\n  aws_region: us-east-1\nresources:\n  aws_instance.web_server:\n    instance_type: m5.large\n"
    )

    zip_path, readme = await package_output(job)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "terraform/vars.yaml" in names
        assert zf.read("terraform/vars.yaml").decode("utf-8") == job.generated_vars_yaml

    assert "vars.yaml" in readme


@pytest.mark.anyio
async def test_packager_omits_vars_yaml_when_not_generated():
    import zipfile
    from app.services.packager.packager import package_output

    job = Job()
    job.original_filename = "arch.drawio"
    job.diagram_format = DiagramFormat.DRAWIO
    job.terraform_plan = _make_plan()
    job.validation_result = _make_validation()
    job.parsed_diagram = _make_parsed_diagram()
    assert job.generated_vars_yaml is None

    zip_path, readme = await package_output(job)

    with zipfile.ZipFile(zip_path) as zf:
        assert "terraform/vars.yaml" not in zf.namelist()


# ─────────────────────────────────────────────────────────────────────────────
# 12. State file upload route
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_upload_state_success_queues_revalidation(client, monkeypatch):
    from app.workers import pipeline_worker

    revalidate_calls = []

    async def fake_revalidate(job_id):
        revalidate_calls.append(job_id)

    # Patched where pipeline.py imported it, not where it's defined —
    # matches the push-to-github tests' pattern for monkeypatching a
    # background-task target.
    from app.api.routes import pipeline as pipeline_routes
    monkeypatch.setattr(pipeline_routes, "revalidate_with_state", fake_revalidate)

    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/upload-state",
        files={"file": ("terraform.tfstate", b'{"version": 4}', "application/json")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job.job_id

    updated = await get_job(job.job_id)
    assert updated.state_file_path is not None

    # Background task runs within the same ASGI call for httpx+ASGITransport.
    assert revalidate_calls == [job.job_id]


@pytest.mark.anyio
async def test_upload_state_job_not_found(client):
    resp = await client.post(
        "/api/v1/jobs/ghost-job/upload-state",
        files={"file": ("terraform.tfstate", b'{"version": 4}', "application/json")},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_upload_state_no_plan_yet_returns_409(client):
    job = _make_job(status=JobStatus.PARSING)  # no terraform_plan set
    await save_job(job)
    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/upload-state",
        files={"file": ("terraform.tfstate", b'{"version": 4}', "application/json")},
    )
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_upload_state_empty_file_returns_400(client, monkeypatch):
    from app.api.routes import pipeline as pipeline_routes

    async def fake_revalidate(job_id):
        pass

    monkeypatch.setattr(pipeline_routes, "revalidate_with_state", fake_revalidate)

    job = _make_job(status=JobStatus.DONE)
    job.terraform_plan = _make_plan()
    await save_job(job)

    resp = await client.post(
        f"/api/v1/jobs/{job.job_id}/upload-state",
        files={"file": ("terraform.tfstate", b"", "application/json")},
    )
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
