"""
API Routes
-----------
POST /api/v1/upload              — upload diagram (+ optional vars.yaml, + optional
                                    state file for reconciling generated properties with
                                    real values — see state_reconciler.py), start pipeline
GET  /api/v1/jobs/{job_id}       — poll job status + progress
POST /api/v1/jobs/{job_id}/upload-state — upload existing terraform.tfstate for real drift checking
                                    (post-hoc, plan-diff only — see state_reconciler.py's
                                    module docstring for how this differs from /upload's state_file)
POST /api/v1/jobs/{job_id}/clarify — submit clarification answers
GET  /api/v1/jobs/{job_id}/download — stream ZIP file
GET  /api/v1/jobs/{job_id}/preview  — preview generated files (for UI code viewer)
PUT  /api/v1/jobs/{job_id}/files/{file_key} — edit a generated file's content (UI edit mode)
DELETE /api/v1/jobs/{job_id}     — cleanup
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import sys

import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

# api/routes/pipeline.py -> routes(0)/api(1)/app(2)/backend(3)/
# arch2tf-product(4). Was parents[5] (one level too far, lands on "thesis") —
# pre-existing bug, same class as missing_info_detector.py's.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.schemas.models import (
    Job, JobStatus, ClarificationResponse, JobStatusResponse,
    UploadResponse, JOB_PROGRESS, STAGE_LABELS, DiagramFormat,
    GithubPushRequest, GithubPushResponse, StateUploadResponse,
    FileEditRequest, FileEditResponse,
)

from app.core.job_store import save_job, get_job, delete_job
from app.core.storage import save_upload
from app.workers.pipeline_worker import run_pipeline, resume_pipeline, revalidate_with_state
from app.services.github.github_pusher import push_job_to_existing_github_repo, GithubPushError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["pipeline"])

# Allowed diagram extensions
ALLOWED_EXTENSIONS = {
    ".drawio", ".xml", ".svg", ".excalidraw", ".json",
    ".png", ".jpg", ".jpeg", ".webp"
}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", response_model=UploadResponse)
async def upload_diagram(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    aws_region: str = Form(default="us-east-1"),
    environment: str = Form(default="dev"),
    vars_file: UploadFile | None = File(default=None),
    state_file: UploadFile | None = File(default=None),
):
    """
    Upload an architecture diagram and start the pipeline.
    Returns a job_id to poll for status.

    `vars_file` is an OPTIONAL previously-generated vars.yaml (see
    missing_info_detector.generate_vars_yaml()) — her explicit distinction,
    2026-07-08, between building "from scratch" (no vars.yaml: every
    catalog-default-covered field gets asked about, not just the
    placeholder-shaped ones) vs. re-running against an already-configured
    diagram (vars.yaml present: covered fields are silently reused, only
    genuinely new/uncovered ones get asked). Parsed here and stashed on
    `job.input_vars` — missing_info_detector.detect_missing_info() reads it
    when the pipeline reaches the clarification stage.

    `state_file` is an OPTIONAL existing terraform.tfstate for the SAME
    environment this diagram describes — 2026-07-29, her explicit
    follow-up request. Distinct from the pre-existing POST
    /jobs/{job_id}/upload-state (which only ever re-runs validation on an
    ALREADY-DONE job — a plan diff against that state, never touching the
    generated module's actual values): attaching it HERE, at upload time,
    lets state_reconciler.reconcile_from_state() run before planning even
    starts, so matched resources' generated properties reflect the real
    values already in state instead of catalog defaults/placeholders. A
    brand-new environment/project with no prior state simply omits this —
    generation proceeds exactly as it always has, entirely unaffected.
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read and size-check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"
        )

    # Create job
    job = Job(
        original_filename=file.filename or "diagram",
        diagram_format=_detect_format(file.filename or "", content),
    )

    # Save uploaded file
    file_path = await save_upload(job.job_id, file.filename or "diagram", content)
    job.file_path = file_path
    job.log(f"Uploaded {file.filename} ({size_mb:.2f} MB)")

    # Optional vars.yaml — parse and validate its shape up front so a
    # malformed upload fails loudly now rather than silently doing nothing
    # useful once the pipeline reaches clarification.
    if vars_file is not None:
        vars_content = await vars_file.read()
        if vars_content:
            try:
                parsed_vars = yaml.safe_load(vars_content)
            except yaml.YAMLError as e:
                raise HTTPException(status_code=400, detail=f"vars.yaml is not valid YAML: {e}")
            if parsed_vars is not None and not isinstance(parsed_vars, dict):
                raise HTTPException(
                    status_code=400,
                    detail="vars.yaml must be a mapping with top-level 'resources'/'globals' keys",
                )
            if parsed_vars:
                job.input_vars = parsed_vars
                job.log(f"Uploaded vars.yaml ({len(vars_content) / 1024:.1f} KB) — reusing its values where present")

    # Optional existing terraform.tfstate — validated up front for the same
    # reason vars.yaml is (fail loudly now, not silently later): a state
    # file this malformed would otherwise just make state_reconciler.py's
    # own defensive parsing quietly skip reconciliation deep inside a
    # background task, with nothing telling her why the generated values
    # didn't change the way she expected.
    if state_file is not None:
        state_content = await state_file.read()
        if state_content:
            try:
                parsed_state = json.loads(state_content)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise HTTPException(status_code=400, detail=f"State file is not valid JSON: {e}")
            if not isinstance(parsed_state, dict) or "resources" not in parsed_state:
                raise HTTPException(
                    status_code=400,
                    detail="State file doesn't look like a Terraform state file (no top-level 'resources' key)",
                )
            state_path = await save_upload(job.job_id, "uploaded_terraform.tfstate", state_content)
            job.state_file_path = state_path
            job.log(
                f"Uploaded existing state file ({len(state_content) / 1024:.1f} KB) — "
                "will reconcile matched resources' properties with real values before planning"
            )

    # Stash region/env preferences for the pipeline to pick up. vars.yaml's
    # own "globals" section (if present) takes precedence — an explicit
    # config file should win over the upload form's defaults.
    from shared.schemas.models import ClarificationAnswer
    vars_globals = (job.input_vars or {}).get("globals") or {}
    job.clarification_answers = [
        ClarificationAnswer(field_key="aws_region",    resource_id="target_global",
                             value=str(vars_globals.get("aws_region", aws_region))),
        ClarificationAnswer(field_key="environment",   resource_id="target_global",
                             value=str(vars_globals.get("environment", environment))),
        ClarificationAnswer(field_key="project_name",  resource_id="target_global",
                             value=str(vars_globals.get("project_name", "arch2terraform"))),
    ]

    await save_job(job)

    # Start pipeline in background
    background_tasks.add_task(run_pipeline, job.job_id)

    log.info("Job created: %s for file %s", job.job_id, file.filename)
    return UploadResponse(
        job_id=job.job_id,
        status=job.status,
        message=f"Diagram uploaded successfully. Pipeline started for job {job.job_id}.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll pipeline status. Frontend calls this every 2s."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percent=JOB_PROGRESS.get(job.status, 0),
        current_stage=STAGE_LABELS.get(job.status, job.status.value),
        stage_logs=job.stage_logs[-20:],         # last 20 log lines
        parsed_diagram=job.parsed_diagram,
        clarification_request=(
            job.clarification_request
            if job.status == JobStatus.NEEDS_CLARIFY else None
        ),
        validation_result=job.validation_result,
        zip_ready=job.status == JobStatus.DONE and bool(job.zip_path),
        error_message=job.error_message,
    )


@router.post("/jobs/{job_id}/upload-state", response_model=StateUploadResponse)
async def upload_state_file(
    job_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload an existing terraform.tfstate for this job so the sandbox
    validator can show REAL drift ("0 to add, 1 to change") against that
    snapshot instead of always planning against an empty slate. Never
    forwarded to GitHub (her explicit call, 2026-07-08 — state can carry
    sensitive resource attributes that shouldn't enter git history); only
    ever used locally in the ephemeral sandbox check and bundled into the
    downloadable ZIP (see packager.py). Re-runs validation + packaging in
    the background so the Review screen reflects it without a full re-parse.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet for this job")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded state file is empty")

    state_path = await save_upload(job.job_id, "uploaded_terraform.tfstate", content)
    job.state_file_path = state_path
    job.log(f"Uploaded existing state file ({len(content) / 1024:.1f} KB) for drift checking")
    await save_job(job)

    background_tasks.add_task(revalidate_with_state, job.job_id)

    log.info("State file uploaded for job %s, revalidation queued", job_id)
    return StateUploadResponse(
        job_id=job.job_id,
        message="State file uploaded — re-running validation to show drift against it.",
    )


@router.post("/jobs/{job_id}/clarify")
async def submit_clarification(
    job_id: str,
    payload: ClarificationResponse,
    background_tasks: BackgroundTasks,
):
    """
    Submit user answers to clarification questions.
    Resumes the pipeline from the planning stage.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.NEEDS_CLARIFY:
        raise HTTPException(
            status_code=409,
            detail=f"Job is in state '{job.status}', not awaiting clarification"
        )

    # Apply resource-level corrections from the Review step BEFORE merging
    # clarification answers — deleting a resource here should also drop any
    # answer that referenced it (harmless either way, since
    # apply_clarification_answers() already skips answers whose resource_id
    # no longer exists, but doing it here keeps job.clarification_answers
    # itself clean rather than accumulating orphaned entries across
    # multiple review/resume cycles).
    rc = payload.resource_corrections
    if rc and job.parsed_diagram:
        pd = job.parsed_diagram
        if rc.deleted_ids:
            deleted = set(rc.deleted_ids)
            pd.resources = [r for r in pd.resources if r.id not in deleted]
            pd.connections = [
                c for c in pd.connections
                if c.source_id not in deleted and c.target_id not in deleted
            ]
            pd.total_resources = len(pd.resources)
            pd.total_connections = len(pd.connections)
            pd.resource_type_summary = {}
            for r in pd.resources:
                pd.resource_type_summary[r.aws_resource_type] = (
                    pd.resource_type_summary.get(r.aws_resource_type, 0) + 1
                )
            job.clarification_answers = [
                a for a in job.clarification_answers if a.resource_id not in deleted
            ]

        for r in pd.resources:
            if r.id in rc.relabeled:
                r.label = rc.relabeled[r.id]
            if r.id in rc.retyped:
                r.aws_resource_type = rc.retyped[r.id]
                r.confidence = 1.0  # user-confirmed, no longer a guess

        if rc.deleted_ids or rc.relabeled or rc.retyped:
            job.log(
                f"Resource review: {len(rc.deleted_ids)} deleted, "
                f"{len(rc.relabeled)} relabeled, {len(rc.retyped)} retyped"
            )

    # Merge new answers with any pre-existing ones (region/env from upload).
    # Skip answers for resources just deleted in the review step above —
    # the frontend shouldn't send these (deleted resources' fields are
    # removed from the form), but guard against it regardless.
    deleted_ids = set(rc.deleted_ids) if rc else set()
    existing_keys = {(a.resource_id, a.field_key) for a in job.clarification_answers}
    for ans in payload.answers:
        if ans.resource_id in deleted_ids:
            continue
        key = (ans.resource_id, ans.field_key)
        if key not in existing_keys:
            job.clarification_answers.append(ans)
        else:
            # Update existing answer
            for existing in job.clarification_answers:
                if existing.resource_id == ans.resource_id and existing.field_key == ans.field_key:
                    existing.value = ans.value
                    break

    job.clarification_request = None
    job.log(f"Received {len(payload.answers)} clarification answers")
    await save_job(job)

    # Resume pipeline in background
    background_tasks.add_task(resume_pipeline, job_id)

    return {"job_id": job_id, "status": "resuming", "answers_received": len(payload.answers)}


@router.get("/jobs/{job_id}/download")
async def download_output(job_id: str):
    """Stream the generated ZIP file."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready for download (status: {job.status})"
        )

    if not job.zip_path or not Path(job.zip_path).exists():
        raise HTTPException(status_code=500, detail="ZIP file not found on disk")

    filename = f"terraform_{job.original_filename.rsplit('.', 1)[0]}_{job.job_id[:8]}.zip"
    # `filename=` already makes FileResponse set Content-Disposition itself —
    # also passing it explicitly in `headers` used to send the header TWICE
    # in the response, which Chrome rejects outright with
    # ERR_RESPONSE_HEADERS_MULTIPLE_CONTENT_DISPOSITION (found 2026-07-24,
    # real download attempt against a completed job). filename= alone is
    # sufficient — no headers= needed.
    return FileResponse(
        path=job.zip_path,
        media_type="application/zip",
        filename=filename,
    )


@router.get("/jobs/{job_id}/preview")
async def preview_files(job_id: str):
    """
    Return generated file contents for in-browser code preview.
    Used by the frontend Review screen.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet")

    plan = job.terraform_plan
    files: dict[str, str] = {}

    # Root files
    for fname, content in plan.root_module_files.items():
        files[fname] = content

    # Module files
    for mod in plan.modules:
        for fname, content in mod.files.items():
            files[f"modules/{mod.name}/{fname}"] = content

    # README
    if job.readme_content:
        files["README.md"] = job.readme_content

    # vars.yaml — every clarification answer this job ended up with (see
    # missing_info_detector.generate_vars_yaml()). Already bundled into the
    # downloadable ZIP by packager.py; was just never added to THIS dict,
    # so it never showed up in the frontend's file tree / edit mode even
    # though the backend had it the whole time. Root-level, like README.md.
    if job.generated_vars_yaml:
        files["vars.yaml"] = job.generated_vars_yaml

    return {
        "job_id": job_id,
        "files": files,
        "file_count": len(files),
        "modules": [{"name": m.name, "description": m.description, "file_count": len(m.files)} for m in plan.modules],
    }


@router.put("/jobs/{job_id}/files/{file_key:path}", response_model=FileEditResponse)
async def edit_file(job_id: str, file_key: str, payload: FileEditRequest):
    """
    In-browser edit mode for the Review screen's code preview — 2026-07-29,
    her explicit request. `file_key` uses the SAME convention
    /preview already returns keys in (not apply_runner.py's `_all_files()`
    convention, which omits the "modules/" prefix — the two evolved
    separately and this endpoint has to match whichever one the frontend
    is actually keying its file tree by): a bare filename for a root file
    ("main.tf"), "modules/<name>/<filename>" for a module file, "README.md"
    for the generated README, or "vars.yaml" for the generated clarification
    answers file — none of these last three are part of `terraform_plan`
    itself, each is stored separately on the job.

    Edits land directly in `job.terraform_plan` (or `job.readme_content` /
    `job.generated_vars_yaml`), which is the SAME data download/
    push-to-github/apply_runner.py's plan_apply() all read from — so a
    saved edit is immediately what gets zipped, pushed, or planned/applied
    next, with no separate "draft" state to reconcile. Deliberately
    whole-file replacement, no diffing.

    Does NOT re-run validation (checkov/tflint/terraform validate) or
    re-plan automatically — the response says so explicitly, since an edit
    that introduces a real syntax error would otherwise only surface much
    later at download/apply time with no obvious cause. Editing vars.yaml
    specifically also does NOT re-apply it to `terraform_plan`'s baked-in
    literals/variable defaults — it only changes the vars.yaml file itself
    (e.g. as a record to re-upload next time, or to hand-correct before a
    GitHub push) — see missing_info_detector.py if the goal is actually
    changing a resource's current values, not just this artifact.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet for this job")

    plan = job.terraform_plan

    if file_key == "README.md":
        job.readme_content = payload.content
    elif file_key == "vars.yaml":
        job.generated_vars_yaml = payload.content
    elif file_key.startswith("modules/"):
        parts = file_key.split("/", 2)
        if len(parts) != 3:
            raise HTTPException(status_code=400, detail=f"Malformed module file key: '{file_key}'")
        _, mod_name, fname = parts
        mod = next((m for m in plan.modules if m.name == mod_name), None)
        if not mod:
            raise HTTPException(status_code=404, detail=f"No module named '{mod_name}' in this job")
        if fname not in mod.files:
            raise HTTPException(status_code=404, detail=f"No file '{fname}' in module '{mod_name}'")
        mod.files[fname] = payload.content
    else:
        if file_key not in plan.root_module_files:
            raise HTTPException(status_code=404, detail=f"No root file '{file_key}' in this job")
        plan.root_module_files[file_key] = payload.content

    job.log(f"Edited {file_key} via UI code editor")
    await save_job(job)

    return FileEditResponse(
        job_id=job_id,
        file_key=file_key,
        message="Saved. Note: this file hasn't been re-validated — re-run plan/validate if the edit changed resource logic.",
    )


@router.post("/jobs/{job_id}/push-to-github", response_model=GithubPushResponse)
async def push_to_github(job_id: str, payload: GithubPushRequest):
    """
    Push this job's generated Terraform + source diagram into an existing
    GitHub repo via a feature branch + PR against its default branch. The
    token in `payload.github_token` is used only for this request (see
    github_pusher.py's module docstring) — never saved to the job, never
    logged, never written anywhere.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet for this job")

    try:
        result = await push_job_to_existing_github_repo(job, payload.github_token, payload.repo_full_name)
    except GithubPushError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return GithubPushResponse(**result)


@router.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    """Clean up job state."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    await delete_job(job_id)
    return {"deleted": job_id}


@router.get("/health")
async def health():
    return {"status": "ok", "service": "arch2terraform"}


# ── Format detection helper ───────────────────────────────────────────────────

def _detect_format(filename: str, content: bytes) -> DiagramFormat:
    ext = Path(filename).suffix.lower()
    ext_map = {
        ".drawio": DiagramFormat.DRAWIO,
        ".xml":    DiagramFormat.DRAWIO,
        ".svg":    DiagramFormat.LUCIDCHART,
        ".excalidraw": DiagramFormat.EXCALIDRAW,
        ".png": DiagramFormat.IMAGE,
        ".jpg": DiagramFormat.IMAGE,
        ".jpeg": DiagramFormat.IMAGE,
        ".webp": DiagramFormat.IMAGE,
    }
    if ext in ext_map:
        return ext_map[ext]

    # Sniff content
    try:
        head = content[:300].decode("utf-8", errors="replace")
        if "<mxGraphModel" in head or "<mxfile" in head:
            return DiagramFormat.DRAWIO
        if "<svg" in head:
            return DiagramFormat.LUCIDCHART
        if '"type":"excalidraw"' in head or '"elements"' in head:
            return DiagramFormat.EXCALIDRAW
    except Exception:
        pass

    if content[:4] in (b"\x89PNG", b"\xff\xd8\xff"):
        return DiagramFormat.IMAGE

    return DiagramFormat.UNKNOWN
