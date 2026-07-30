"""
GitHub Pusher Service
----------------------
Pushes a completed job's generated Terraform — AND the original source
architecture diagram — into an EXISTING GitHub repo the caller specifies,
via a feature branch + Pull Request against that repo's real default
branch. Never creates a repo.

Design note (reworked 2026-07-08, superseding an earlier "create a new repo
per job" version): the whole point of this product is "the architecture
diagram behaves as source of truth." A new throwaway repo per job works
against that — infrastructure should live in ONE repo whose history tells
the story of how it evolved, alongside the diagram that drove each change.
So this pushes into a repo the user already has, at stable, per-environment
paths:

    terraform/<environment>/            <- generated Terraform (main.tf, modules/, README, .gitignore, vars.yaml)
    diagrams/<environment>/<filename>   <- the exact diagram file that was uploaded for this job

Because the commit tree is built with `base_tree` set to the target
branch's current tree, only these specific paths are added/updated — every
other file already in the repo is left completely untouched (GitHub's Git
Data API only replaces tree entries you explicitly include).

Directory-per-environment (added 2026-07-08, decided over Terraform
workspaces): each job already regenerates a complete, independent set of
Terraform straight from whatever's drawn in one diagram — there's no
parameterized single codebase for workspaces to switch between, and
different environments' diagrams may be structurally different (prod with
a load balancer + multi-AZ RDS, dev with one instance), not just different
variable values. Giving each environment its own folder needs no generator
changes at all; every environment's push already targets the SAME repo's
default branch (this is directory-per-environment, not
branch-per-environment) — only the path prefix differs, so `terraform/dev/`
and `terraform/prod/` are both just regular folders reviewed via normal PRs
against the same base branch.

Diagram versioning: committing the diagram to the SAME stable per-environment
path on every push means git's own history for that path *is* the diagram's
version history for that environment — `git log --follow -- diagrams/prod/<filename>`
(or GitHub's per-file "History" view) shows every version of PROD's
architecture that was ever pushed, independent of dev/staging's history,
each one paired in the same commit with the Terraform it produced. No
bespoke versioning system needed; git already does this well.

Auth: the caller supplies a GitHub Personal Access Token per request (never
persisted server-side — see GithubPushRequest in shared/schemas/models.py).
Since this no longer creates repos, the token only needs Contents + Pull
requests write access to the ONE target repo — a fine-grained PAT scoped to
just that repo works, no account-wide 'repo' scope required anymore.
"""
from __future__ import annotations

import base64
import logging
import sys
import tempfile
from pathlib import Path

import httpx

from app.core.storage import read_upload

log = logging.getLogger(__name__)

# arch2terraform package (see arch2terraform_bridge.py for the parsing-side
# integration point)
from app._pathboot import ensure_paths
ensure_paths()
from arch2terraform.differ.diagram_differ import DiagramDiff, diff_diagram_files
from app.services.github.workflow_generator import generate_terraform_workflow

GITHUB_API = "https://api.github.com"


class GithubPushError(Exception):
    """Raised for any GitHub API failure. Message is always safe to show
    the user directly — never includes the token."""


def _project_name_from_job(job) -> str:
    for a in job.clarification_answers:
        if a.field_key == "project_name":
            return a.value
    return (job.original_filename or "arch2tf").rsplit(".", 1)[0]


def _environment_from_job(job) -> str:
    """The environment picked on the upload screen (dev/staging/prod) —
    always present in practice: pipeline.py's upload endpoint stashes it as
    a clarification answer for every job unconditionally, before parsing
    even starts. Defaulting to "dev" here anyway matches
    pipeline_worker.py's own fallback, purely as a last-resort safety net."""
    for a in job.clarification_answers:
        if a.field_key == "environment":
            return a.value
    return "dev"


def _aws_region_from_job(job) -> str:
    """Same pattern as _environment_from_job/_project_name_from_job — reads
    the "aws_region" clarification answer (see missing_info_detector.py's
    MANDATORY_FIELDS entry), defaulting to "us-east-1" to match
    pipeline_worker.py's own fallback."""
    for a in job.clarification_answers:
        if a.field_key == "aws_region":
            return a.value
    return "us-east-1"


async def _collect_files(
    job, tf_subdir: str, diagram_subdir: str, environment: str, default_branch: str,
) -> dict[str, bytes]:
    """Same file set as packager.py's ZIP (generated Terraform + README +
    a sensible .gitignore), nested under `tf_subdir/environment` so this
    can't collide with anything already in an existing repo AND so
    different environments' pushes never overwrite each other, PLUS the
    original uploaded diagram file at `diagram_subdir/environment/<original
    filename>` — see this module's docstring for why that path is the
    actual versioning mechanism, per environment. Returns raw bytes per
    path (blobs API needs bytes to base64-encode regardless of whether the
    underlying file is text or binary, e.g. drawio/excalidraw XML/JSON vs a
    PNG screenshot)."""
    plan = job.terraform_plan
    files: dict[str, bytes] = {}
    env_tf_dir = f"{tf_subdir}/{environment}"

    for filename, content in plan.root_module_files.items():
        files[f"{env_tf_dir}/{filename}"] = content.encode("utf-8")

    for mod in plan.modules:
        for filename, content in mod.files.items():
            files[f"{env_tf_dir}/modules/{mod.name}/{filename}"] = content.encode("utf-8")

    if job.readme_content:
        files[f"{env_tf_dir}/README.md"] = job.readme_content.encode("utf-8")

    # vars.yaml — the full set of config values (sizes, engines, CIDRs, etc.)
    # this job was built with, see missing_info_detector.generate_vars_yaml().
    # Unlike terraform.tfstate (never pushed — see packager.py's docstring on
    # that), this holds config, not secrets, so it's fine to version alongside
    # the Terraform it configures — her explicit call, 2026-07-08. Committing
    # it to this same stable per-environment path means the next push against
    # an updated diagram can pull it back down and reuse it (via the
    # `vars_file` upload param) instead of re-answering everything.
    if job.generated_vars_yaml:
        files[f"{env_tf_dir}/vars.yaml"] = job.generated_vars_yaml.encode("utf-8")

    files[f"{env_tf_dir}/.gitignore"] = (
        "# Terraform\n"
        ".terraform/\n"
        "*.tfstate\n"
        "*.tfstate.*\n"
        "crash.log\n"
        "crash.*.log\n"
        "*.tfvars\n"
        "*.tfvars.json\n"
        "override.tf\n"
        "override.tf.json\n"
        "*_override.tf\n"
        "*_override.tf.json\n"
        ".terraformrc\n"
        "terraform.rc\n"
    ).encode("utf-8")

    # The diagram itself — the actual "source of truth" artifact. job.file_path
    # is the storage-backend path (local disk or S3, storage.py abstracts
    # which), so this works regardless of STORAGE_BACKEND.
    if job.file_path and job.original_filename:
        diagram_bytes = await read_upload(job.file_path)
        files[f"{diagram_subdir}/{environment}/{job.original_filename}"] = diagram_bytes

    # CI/CD workflow — repo-root path (GitHub only ever reads workflows from
    # .github/workflows/, not from tf_subdir), regenerated fresh on every
    # push so it always reflects the current default branch. See
    # workflow_generator.py for the OIDC-auth / required-reviewers design
    # and the one-time AWS/GitHub setup this needs before it'll actually run.
    files[".github/workflows/terraform.yml"] = generate_terraform_workflow(
        default_branch=default_branch,
        aws_region=_aws_region_from_job(job),
        tf_subdir=tf_subdir,
    ).encode("utf-8")

    return files


def _pr_body(job, diagram_path: str, environment: str, diagram_diff: DiagramDiff | None) -> str:
    v = job.validation_result
    lines = [
        f"Generated by **arch2terraform** from `{job.original_filename}` for the **{environment}** environment.",
        "",
        f"**Source diagram:** `{diagram_path}` (this PR updates it — see that file's",
        f'GitHub "History" for every previous version of {environment}\'s architecture)',
        "",
        f"**Resources:** {job.terraform_plan.resource_count}  ",
        f"**Modules:** {', '.join(m.name for m in job.terraform_plan.modules)}",
        "",
    ]
    if v:
        lines.append(
            f"**Validation:** {v.passed_count} passed, {v.warning_count} warnings, {v.failed_count} errors"
        )
    lines.append("")
    if diagram_diff is not None:
        lines.append(diagram_diff.summary())
    else:
        lines.append(
            "### Architecture diff\n\nNo previous version of this diagram found at this path — "
            "this is the first push for this environment."
        )
    lines += [
        "",
        "---",
        "",
        "This push also updated `.github/workflows/terraform.yml` (plan on PRs, "
        "apply on merge). If this is the first push to this repo, it won't run "
        "successfully until you complete the one-time AWS OIDC role + GitHub "
        "Environment setup described at the top of that file.",
    ]
    return "\n".join(lines)


async def _diff_against_existing_diagram(
    http: httpx.AsyncClient,
    repo_full_name: str,
    diagram_path: str,
    default_branch: str,
    new_diagram_bytes: bytes,
    original_filename: str,
) -> DiagramDiff | None:
    """
    Best-effort: fetches whatever's currently committed at `diagram_path` on
    the repo's default branch (via the Contents API, which base64-encodes
    small files inline — GitHub's own docs cap this at 1MB, comfortably
    above any drawio/excalidraw/lucidchart diagram file in practice) and
    diffs it against the newly uploaded diagram using arch2terraform's
    diff_diagram_files(). Returns None whenever a diff genuinely can't be
    produced — no previous version exists yet (404, the common first-push
    case), the existing file is too large for this API to inline, or
    anything else goes wrong parsing either version — since a diff is a
    nice-to-have addition to the PR description, not something that should
    ever block the actual push of Terraform + diagram, which is this
    module's real job.
    """
    try:
        resp = await http.get(f"/repos/{repo_full_name}/contents/{diagram_path}", params={"ref": default_branch})
    except Exception:
        log.warning("Could not reach GitHub to fetch existing diagram at %s for diffing", diagram_path, exc_info=True)
        return None

    if resp.status_code == 404:
        return None  # first push for this environment — nothing to diff against
    if resp.status_code != 200:
        log.warning("Unexpected status %s fetching existing diagram at %s for diffing", resp.status_code, diagram_path)
        return None

    data = resp.json()
    if data.get("encoding") != "base64" or not data.get("content"):
        log.warning("Existing diagram at %s has no inline content (too large?) — skipping diff", diagram_path)
        return None

    try:
        old_bytes = base64.b64decode(data["content"])
    except Exception:
        log.warning("Could not decode existing diagram content at %s — skipping diff", diagram_path, exc_info=True)
        return None

    suffix = Path(original_filename).suffix or ".drawio"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as old_f, tempfile.NamedTemporaryFile(suffix=suffix) as new_f:
            old_f.write(old_bytes)
            old_f.flush()
            new_f.write(new_diagram_bytes)
            new_f.flush()
            return diff_diagram_files(old_f.name, new_f.name)
    except Exception:
        log.warning("Could not diff diagram versions at %s — skipping diff", diagram_path, exc_info=True)
        return None


def _api_error_message(resp: httpx.Response, context: str) -> str:
    try:
        detail = resp.json().get("message", resp.text)
    except Exception:
        detail = resp.text
    return f"GitHub API error while {context} ({resp.status_code}): {detail}"


async def push_job_to_existing_github_repo(
    job,
    github_token: str,
    repo_full_name: str,
    *,
    tf_subdir: str = "terraform",
    diagram_subdir: str = "diagrams",
    client: httpx.AsyncClient | None = None,
) -> dict:
    """
    Pushes this job's generated Terraform + source diagram into an EXISTING
    repo (`repo_full_name`, e.g. "yourname/infra-repo") via a feature
    branch + PR against that repo's real default branch.

    `client` is exposed only so tests can inject an httpx.AsyncClient bound
    to a MockTransport instead of hitting real GitHub.

    Returns {"repo_url", "pr_url", "repo_full_name"}.
    Raises GithubPushError with a user-safe message on any failure.
    """
    if not job.terraform_plan:
        raise GithubPushError("No Terraform plan has been generated for this job yet.")
    if not github_token or not github_token.strip():
        raise GithubPushError("A GitHub personal access token is required.")
    if not repo_full_name or "/" not in repo_full_name:
        raise GithubPushError('Repository must be in "owner/repo" form, e.g. "yourname/infra-repo".')

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    owns_client = client is None
    http = client or httpx.AsyncClient(base_url=GITHUB_API, headers=headers, timeout=30)
    try:
        # 1. Confirm the repo exists and this token can see it.
        repo_resp = await http.get(f"/repos/{repo_full_name}")
        if repo_resp.status_code == 404:
            raise GithubPushError(
                f"Repo '{repo_full_name}' not found, or this token doesn't have access to it."
            )
        if repo_resp.status_code != 200:
            raise GithubPushError(_api_error_message(repo_resp, "looking up the repo"))
        repo = repo_resp.json()
        default_branch = repo["default_branch"]

        # 2. Current tip of the default branch — this is both the base_tree
        #    for our new tree (so unrelated existing files are preserved)
        #    and the parent commit for our new commit.
        ref_resp = await http.get(f"/repos/{repo_full_name}/git/ref/heads/{default_branch}")
        if ref_resp.status_code != 200:
            raise GithubPushError(_api_error_message(ref_resp, "reading the default branch"))
        base_sha = ref_resp.json()["object"]["sha"]

        # 3. One blob per file (generated Terraform + README + .gitignore +
        #    the source diagram), one tree merged on top of the existing
        #    default branch's tree, one commit. Everything lands under this
        #    job's environment folder — see _collect_files' docstring.
        environment = _environment_from_job(job)
        files = await _collect_files(job, tf_subdir, diagram_subdir, environment, default_branch)

        # 3b. Best-effort architecture diff: compare the diagram we're about
        # to push against whatever's already at this same path on the
        # default branch (if anything — first push for an environment has
        # nothing to diff against). See _diff_against_existing_diagram's
        # docstring for why this can never fail the push itself.
        diagram_path = f"{diagram_subdir}/{environment}/{job.original_filename}"
        diagram_diff = await _diff_against_existing_diagram(
            http, repo_full_name, diagram_path, default_branch,
            files[diagram_path], job.original_filename,
        )

        tree_entries = []
        for path, content_bytes in files.items():
            blob_resp = await http.post(f"/repos/{repo_full_name}/git/blobs", json={
                "content": base64.b64encode(content_bytes).decode("ascii"),
                "encoding": "base64",
            })
            if blob_resp.status_code not in (200, 201):
                raise GithubPushError(_api_error_message(blob_resp, f"uploading {path}"))
            tree_entries.append({
                "path": path, "mode": "100644", "type": "blob",
                "sha": blob_resp.json()["sha"],
            })

        tree_resp = await http.post(f"/repos/{repo_full_name}/git/trees", json={
            "base_tree": base_sha,
            "tree": tree_entries,
        })
        if tree_resp.status_code not in (200, 201):
            raise GithubPushError(_api_error_message(tree_resp, "building the commit tree"))
        tree_sha = tree_resp.json()["sha"]

        commit_resp = await http.post(f"/repos/{repo_full_name}/git/commits", json={
            "message": f"Update {environment} infrastructure from {job.original_filename} (arch2terraform)",
            "tree": tree_sha,
            "parents": [base_sha],
        })
        if commit_resp.status_code not in (200, 201):
            raise GithubPushError(_api_error_message(commit_resp, "creating the commit"))
        commit_sha = commit_resp.json()["sha"]

        # 4. Branch pointing at that commit, then a PR against the repo's
        #    actual default branch (never main hardcoded — always whatever
        #    this specific repo's default branch really is). Branch name is
        #    prefixed with the environment purely so branches group
        #    sensibly when browsing the repo — this is still
        #    directory-per-environment, not branch-per-environment: every
        #    environment's PR targets the SAME default branch.
        branch_name = f"arch2terraform/{environment}/{job.job_id[:8]}"
        branch_resp = await http.post(f"/repos/{repo_full_name}/git/refs", json={
            "ref": f"refs/heads/{branch_name}",
            "sha": commit_sha,
        })
        if branch_resp.status_code not in (200, 201):
            raise GithubPushError(_api_error_message(branch_resp, "creating the branch"))

        pr_resp = await http.post(f"/repos/{repo_full_name}/pulls", json={
            "title": f"Update {environment} infrastructure from {job.original_filename}",
            "head": branch_name,
            "base": default_branch,
            "body": _pr_body(job, diagram_path, environment, diagram_diff),
        })
        if pr_resp.status_code not in (200, 201):
            raise GithubPushError(_api_error_message(pr_resp, "opening the pull request"))
        pr = pr_resp.json()

        log.info("Pushed job %s (%s) to %s, PR %s", job.job_id, environment, repo_full_name, pr["html_url"])
        return {
            "repo_url": repo["html_url"],
            "pr_url": pr["html_url"],
            "repo_full_name": repo_full_name,
            "environment": environment,
        }
    finally:
        if owns_client:
            await http.aclose()
