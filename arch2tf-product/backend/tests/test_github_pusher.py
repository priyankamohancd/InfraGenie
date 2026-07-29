"""
GitHub Pusher — unit tests
-----------------------------
Uses httpx.MockTransport to simulate GitHub's real REST API response shapes
without needing a real token or touching real GitHub. Pins the exact call
sequence github_pusher.py depends on for pushing into an EXISTING repo (see
that module's docstring for why this doesn't create repos): GET repo info
-> GET default branch ref -> one blob per file (generated Terraform +
README + .gitignore + the source diagram) -> POST tree -> POST commit ->
POST branch ref -> POST pull request.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.services.github import github_pusher
from app.services.github.github_pusher import push_job_to_existing_github_repo, GithubPushError
from shared.schemas.models import Job, TerraformPlan, TerraformModule, ClarificationAnswer


@pytest.fixture
def anyio_backend():
    return "asyncio"


DIAGRAM_BYTES = b"<mxGraphModel>fake drawio content</mxGraphModel>"


@pytest.fixture(autouse=True)
def _fake_read_upload(monkeypatch):
    """Stand in for app.core.storage.read_upload — avoids touching real
    disk/S3 for the source-diagram file this module reads and pushes."""
    async def fake_read_upload(path: str) -> bytes:
        return DIAGRAM_BYTES
    monkeypatch.setattr(github_pusher, "read_upload", fake_read_upload)


def _make_job(environment: str | None = "staging") -> Job:
    job = Job(original_filename="sample_architecture.drawio")
    job.file_path = "/tmp/whatever/sample_architecture.drawio"
    job.terraform_plan = TerraformPlan(
        modules=[
            TerraformModule(
                name="networking",
                source_resources=["r1"],
                description="VPC etc",
                files={"main.tf": 'resource "aws_vpc" "main_vpc" {\n  cidr_block = "10.0.0.0/16"\n}\n'},
            ),
        ],
        root_module_files={"main.tf": '# root\nmodule "networking" {\n  source = "./modules/networking"\n}\n'},
        resource_count=1,
    )
    job.readme_content = "# Terraform Infrastructure\n"
    if environment is not None:
        job.clarification_answers = [
            ClarificationAnswer(field_key="environment", resource_id="target_global", value=environment),
        ]
    return job


class _FakeGithub:
    """Mock GitHub server for an EXISTING repo: records every request and
    returns canned responses shaped like real GitHub API responses."""

    def __init__(self, repo_exists=True, default_branch="main", existing_diagram_bytes=None, contents_status=None):
        self.repo_exists = repo_exists
        self.default_branch = default_branch
        self.requests: list[httpx.Request] = []
        self._blob_counter = 0
        # None -> 404 (no previous diagram, the common first-push case).
        # Set to real bytes to simulate a previous version already committed.
        self.existing_diagram_bytes = existing_diagram_bytes
        # Override for simulating a broken/unexpected Contents API response
        # (should never fail the push itself — see _diff_against_existing_diagram).
        self.contents_status = contents_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        method = request.method

        if path == "/repos/priyankamohan/infra-repo" and method == "GET":
            if not self.repo_exists:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={
                "full_name": "priyankamohan/infra-repo",
                "html_url": "https://github.com/priyankamohan/infra-repo",
                "default_branch": self.default_branch,
            })

        if path.endswith(f"/git/ref/heads/{self.default_branch}") and method == "GET":
            return httpx.Response(200, json={"object": {"sha": "base-sha-0000"}})

        if "/contents/" in path and method == "GET":
            if self.contents_status is not None:
                return httpx.Response(self.contents_status, json={"message": "simulated error"})
            if self.existing_diagram_bytes is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={
                "content": base64.b64encode(self.existing_diagram_bytes).decode("ascii"),
                "encoding": "base64",
            })

        if path.endswith("/git/blobs") and method == "POST":
            self._blob_counter += 1
            return httpx.Response(201, json={"sha": f"blob-sha-{self._blob_counter}"})

        if path.endswith("/git/trees") and method == "POST":
            return httpx.Response(201, json={"sha": "tree-sha-0000"})

        if path.endswith("/git/commits") and method == "POST":
            return httpx.Response(201, json={"sha": "commit-sha-0000"})

        if path.endswith("/git/refs") and method == "POST":
            body = json.loads(request.content)
            return httpx.Response(201, json={"ref": body["ref"]})

        if path.endswith("/pulls") and method == "POST":
            body = json.loads(request.content)
            return httpx.Response(201, json={
                "html_url": "https://github.com/priyankamohan/infra-repo/pull/7",
                "head": body["head"],
                "base": body["base"],
            })

        return httpx.Response(404, json={"message": f"unhandled mock path: {method} {path}"})


def _client_for(fake: _FakeGithub, token: str = "fake-token-123") -> httpx.AsyncClient:
    transport = httpx.MockTransport(fake.handler)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return httpx.AsyncClient(base_url="https://api.github.com", transport=transport, headers=headers, timeout=30)


@pytest.mark.anyio
async def test_successful_push_returns_repo_and_pr_urls():
    job = _make_job()
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        result = await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    assert result["repo_url"] == "https://github.com/priyankamohan/infra-repo"
    assert result["pr_url"] == "https://github.com/priyankamohan/infra-repo/pull/7"
    assert result["repo_full_name"] == "priyankamohan/infra-repo"
    assert result["environment"] == "staging"

    for r in fake.requests:
        assert r.headers.get("authorization") == "Bearer fake-token-123"


@pytest.mark.anyio
async def test_never_creates_a_repo():
    """The whole point of this rework: no /user/repos POST should ever
    happen — only GET on the existing repo."""
    job = _make_job()
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    assert not any(r.url.path == "/user/repos" for r in fake.requests)


@pytest.mark.anyio
async def test_pr_opens_against_the_repos_real_default_branch_not_hardcoded_main():
    job = _make_job()
    fake = _FakeGithub(default_branch="develop")
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    pr_requests = [r for r in fake.requests if r.url.path.endswith("/pulls")]
    assert len(pr_requests) == 1
    body = json.loads(pr_requests[0].content)
    assert body["base"] == "develop"


@pytest.mark.anyio
async def test_terraform_files_are_nested_under_environment_and_diagram_under_diagrams():
    job = _make_job(environment="staging")
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    assert len(tree_requests) == 1
    body = json.loads(tree_requests[0].content)
    paths = {entry["path"] for entry in body["tree"]}

    assert "terraform/staging/main.tf" in paths
    assert "terraform/staging/modules/networking/main.tf" in paths
    assert "terraform/staging/README.md" in paths
    assert "terraform/staging/.gitignore" in paths
    assert "diagrams/staging/sample_architecture.drawio" in paths
    # Nothing should be written at repo root, or at a flat terraform/
    # without an environment segment (would risk colliding with whatever an
    # existing repo already has there, or with a different environment's push).
    assert "main.tf" not in paths
    assert "README.md" not in paths
    assert "terraform/main.tf" not in paths
    assert "diagrams/sample_architecture.drawio" not in paths

    # base_tree must be set so the new tree MERGES with (doesn't replace)
    # whatever else is already in the repo's default branch.
    assert body["base_tree"] == "base-sha-0000"


@pytest.mark.anyio
async def test_vars_yaml_included_alongside_terraform_when_generated():
    """2026-07-08: vars.yaml is config, not secrets (unlike terraform.tfstate,
    which is deliberately local-ZIP-only, see packager.py) — her explicit
    call to also push it to GitHub, at the same terraform/<env>/ path as the
    rest of the generated Terraform, so a future push can pull it back down
    and reuse it via the /upload `vars_file` param."""
    job = _make_job(environment="staging")
    job.generated_vars_yaml = "resources:\n  aws_instance.web_server:\n    instance_type: m5.large\n"
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    body = json.loads(tree_requests[0].content)
    paths = {entry["path"] for entry in body["tree"]}
    assert "terraform/staging/vars.yaml" in paths

    blob_requests = [r for r in fake.requests if r.url.path.endswith("/git/blobs")]
    blob_contents = [base64.b64decode(json.loads(r.content)["content"]) for r in blob_requests]
    assert job.generated_vars_yaml.encode("utf-8") in blob_contents


@pytest.mark.anyio
async def test_vars_yaml_omitted_when_not_generated():
    """The common case for a job that never went through clarification
    (synthetic test jobs, or a job whose vars.yaml generation failed) must
    not write an empty/stray vars.yaml into the repo."""
    job = _make_job(environment="staging")
    assert job.generated_vars_yaml is None
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    body = json.loads(tree_requests[0].content)
    paths = {entry["path"] for entry in body["tree"]}
    assert "terraform/staging/vars.yaml" not in paths


@pytest.mark.anyio
async def test_different_environments_never_collide_in_path():
    """The whole point of this feature: pushing 'dev' and pushing 'prod'
    for the same repo must never write to the same paths."""
    fake = _FakeGithub()

    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(_make_job(environment="dev"), "fake-token-123", "priyankamohan/infra-repo", client=client)
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(_make_job(environment="prod"), "fake-token-123", "priyankamohan/infra-repo", client=client)

    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    assert len(tree_requests) == 2
    dev_paths = {entry["path"] for entry in json.loads(tree_requests[0].content)["tree"]}
    prod_paths = {entry["path"] for entry in json.loads(tree_requests[1].content)["tree"]}

    # .github/workflows/terraform.yml is deliberately the ONE exception: a
    # single repo-root workflow covers every environment via its own
    # internal matrix (see workflow_generator.py), so it's legitimately
    # identical/shared across every environment's push - not a collision.
    shared_paths = {".github/workflows/terraform.yml"}
    assert (dev_paths - shared_paths).isdisjoint(prod_paths - shared_paths)
    assert dev_paths & prod_paths == shared_paths
    assert any(p.startswith("terraform/dev/") for p in dev_paths)
    assert any(p.startswith("terraform/prod/") for p in prod_paths)
    assert any(p.startswith("diagrams/dev/") for p in dev_paths)
    assert any(p.startswith("diagrams/prod/") for p in prod_paths)


@pytest.mark.anyio
async def test_environment_defaults_to_dev_when_not_set():
    job = _make_job(environment=None)  # no "environment" clarification answer at all
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        result = await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    assert result["environment"] == "dev"
    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    paths = {entry["path"] for entry in json.loads(tree_requests[0].content)["tree"]}
    assert "terraform/dev/main.tf" in paths


@pytest.mark.anyio
async def test_pr_title_and_branch_name_reference_the_environment():
    job = _make_job(environment="prod")
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    pr_requests = [r for r in fake.requests if r.url.path.endswith("/pulls")]
    pr_body = json.loads(pr_requests[0].content)
    assert "prod" in pr_body["title"]
    assert pr_body["head"].startswith("arch2terraform/prod/")


@pytest.mark.anyio
async def test_diagram_file_content_is_uploaded_correctly():
    job = _make_job()
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    blob_requests = [r for r in fake.requests if r.url.path.endswith("/git/blobs")]
    contents = [base64.b64decode(json.loads(r.content)["content"]) for r in blob_requests]
    assert DIAGRAM_BYTES in contents


@pytest.mark.anyio
async def test_github_actions_workflow_pushed_at_repo_root():
    """2026-07-22: every push must also (re)write .github/workflows/terraform.yml
    at repo root - GitHub only reads workflows from that exact path, so it
    can never live under terraform/<environment>/ alongside everything else."""
    job = _make_job(environment="staging")
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    tree_requests = [r for r in fake.requests if r.url.path.endswith("/git/trees")]
    paths = {entry["path"] for entry in json.loads(tree_requests[0].content)["tree"]}
    assert ".github/workflows/terraform.yml" in paths

    blob_requests = [r for r in fake.requests if r.url.path.endswith("/git/blobs")]
    contents = [base64.b64decode(json.loads(r.content)["content"]).decode("utf-8") for r in blob_requests]
    workflow_content = next(c for c in contents if c.startswith("# Auto-generated by arch2terraform"))
    assert "name: Terraform" in workflow_content
    # The repo's REAL default branch (not a hardcoded "main") must drive the
    # apply trigger, same principle already established for PR base branches.
    assert "refs/heads/main" in workflow_content  # _FakeGithub's default branch is "main"


@pytest.mark.anyio
async def test_repo_not_found_raises_clear_error():
    job = _make_job()
    fake = _FakeGithub(repo_exists=False)
    async with _client_for(fake) as client:
        with pytest.raises(GithubPushError, match="not found"):
            await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)


@pytest.mark.anyio
async def test_missing_repo_full_name_raises_before_any_api_call():
    job = _make_job()
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        with pytest.raises(GithubPushError, match="owner/repo"):
            await push_job_to_existing_github_repo(job, "fake-token-123", "not-a-valid-repo-name", client=client)

    assert len(fake.requests) == 0


@pytest.mark.anyio
async def test_missing_terraform_plan_raises_before_any_api_call():
    job = Job(original_filename="test.drawio")  # no terraform_plan set
    fake = _FakeGithub()
    async with _client_for(fake) as client:
        with pytest.raises(GithubPushError, match="No Terraform plan"):
            await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)


# ── Architecture diff in the PR body (added 2026-07-08) ─────────────────────
# Real fixture files (not synthetic bytes) so diff_diagram_files() actually
# has something meaningful to parse/classify/diff — DIAGRAM_BYTES above is
# fake XML that doesn't parse as a real drawio file.
_OLD_FIXTURE = REPO_ROOT / "arch2terraform" / "tests" / "fixtures" / "drawio" / "sample_architecture.drawio"
_NEW_FIXTURE = REPO_ROOT / "arch2tf-product" / "manual_test_diagrams" / "sample_architecture.drawio"


@pytest.mark.anyio
async def test_pr_body_includes_diff_when_a_previous_diagram_version_exists(monkeypatch):
    async def fake_read_upload(path: str) -> bytes:
        return _NEW_FIXTURE.read_bytes()
    monkeypatch.setattr(github_pusher, "read_upload", fake_read_upload)

    job = _make_job()
    job.original_filename = "sample_architecture.drawio"
    fake = _FakeGithub(existing_diagram_bytes=_OLD_FIXTURE.read_bytes())
    async with _client_for(fake) as client:
        result = await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    assert result["pr_url"]  # push itself still succeeds
    pr_requests = [r for r in fake.requests if r.url.path.endswith("/pulls")]
    body = json.loads(pr_requests[0].content)["body"]
    assert "Architecture diff" in body
    assert "Added: 1" in body
    assert "aws_db_instance" in body


@pytest.mark.anyio
async def test_pr_body_notes_first_push_when_no_previous_diagram_exists():
    job = _make_job()
    job.original_filename = "sample_architecture.drawio"
    fake = _FakeGithub(existing_diagram_bytes=None)  # 404 — first push for this environment
    async with _client_for(fake) as client:
        await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    pr_requests = [r for r in fake.requests if r.url.path.endswith("/pulls")]
    body = json.loads(pr_requests[0].content)["body"]
    assert "first push" in body.lower()


@pytest.mark.anyio
async def test_push_still_succeeds_when_diff_fetch_errors():
    """The whole point of making this best-effort: a broken/unexpected
    Contents API response must never take down the actual push."""
    job = _make_job()
    job.original_filename = "sample_architecture.drawio"
    fake = _FakeGithub(contents_status=500)
    async with _client_for(fake) as client:
        result = await push_job_to_existing_github_repo(job, "fake-token-123", "priyankamohan/infra-repo", client=client)

    assert result["pr_url"]
    pr_requests = [r for r in fake.requests if r.url.path.endswith("/pulls")]
    body = json.loads(pr_requests[0].content)["body"]
    assert "first push" in body.lower()  # falls back gracefully, same as no-previous-version
