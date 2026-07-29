"""
Pipeline Worker
----------------
Async background task that drives all pipeline stages for a job:

  uploaded → parsing → parsed → [needs_clarification] →
  planning → generating → validating → packaging → done

Runs as a FastAPI BackgroundTask (no Celery needed for MVP).
"""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path
import sys

log = logging.getLogger(__name__)

# app/workers/pipeline_worker.py -> workers(0)/app(1)/backend(2)/
# arch2tf-product(3). Was parents[4] (one level too far, lands on "thesis") —
# pre-existing bug, same class as missing_info_detector.py's.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.schemas.models import Job, JobStatus

from app.core.job_store import save_job, get_job, update_status
from app.core.storage import read_upload
from app.services.parser.diagram_parser import parse_diagram
from app.services.parser.missing_info_detector import detect_missing_info
from app.services.parser.state_reconciler import reconcile_from_state
from app.services.planner.terraform_planner import build_terraform_plan
from app.services.sandbox.tf_validator import validate_terraform
from app.services.packager.packager import package_output


async def run_pipeline(job_id: str) -> None:
    """
    Main pipeline runner. Called as a background task after upload.
    Handles all stages and updates job state in Redis after each one.
    """
    log.info("Pipeline started for job %s", job_id)

    job = await get_job(job_id)
    if not job:
        log.error("Job %s not found — aborting", job_id)
        return

    try:
        # ── Stage 1: Parse diagram ───────────────────────────────────────────
        await _update(job, JobStatus.PARSING, "Starting diagram parsing...")
        parsed = await parse_diagram(job.file_path, job.original_filename)
        job.parsed_diagram = parsed
        job.log(f"Parsed {parsed.total_resources} resources, {parsed.total_connections} connections")

        if parsed.total_resources == 0:
            await _fail(job, "Parser found no recognizable resources in the diagram.")
            return

        await _update(job, JobStatus.PARSED, f"Diagram parsed: {parsed.total_resources} resources found")

        # ── Optional: reconcile with an uploaded existing state file ─────────
        # 2026-07-29, her explicit follow-up request. Only runs at all when
        # `state_file` was attached at upload time (see /upload's param) —
        # a brand-new environment/project with nothing to reconcile against
        # simply has no job.state_file_path set here, so this whole block is
        # skipped and generation proceeds exactly as it always has. When a
        # state file WAS attached, matched resources' properties are
        # overwritten with real values BEFORE detect_missing_info() runs, so
        # any clarification question still asked about a reconciled field
        # shows the REAL value as its default instead of a catalog
        # placeholder — see state_reconciler.py's module docstring for the
        # full reasoning (conservative matching, why this is safe).
        if job.state_file_path:
            try:
                state_bytes = await read_upload(job.state_file_path)
                reconciled = reconcile_from_state(parsed.resources, state_bytes)
                if reconciled:
                    job.log(
                        f"Reconciled {len(reconciled)} resource(s) with real values from uploaded state: "
                        + "; ".join(reconciled)
                    )
                else:
                    job.log("Uploaded state file didn't match any parsed resource — nothing reconciled")
            except Exception:
                log.warning("State reconciliation failed for job %s — continuing without it", job_id, exc_info=True)
                job.log("State reconciliation failed — continuing with catalog defaults")

        # ── Stage 2: Detect missing info ─────────────────────────────────────
        # `job.input_vars` (an optional uploaded vars.yaml — see /upload's
        # `vars_file` param) drives the from-scratch-vs-reuse split: covered
        # (resource, field) pairs come back as auto_answers instead of
        # questions. See missing_info_detector.py's module docstring.
        clarification, auto_answers = detect_missing_info(parsed, job_id, input_vars=job.input_vars)
        if auto_answers:
            _merge_answers(job, auto_answers)
            job.log(f"Pre-filled {len(auto_answers)} answer(s) from uploaded vars.yaml")
        if clarification and clarification.fields:
            job.clarification_request = clarification
            await _update(
                job, JobStatus.NEEDS_CLARIFY,
                f"Need answers to {len(clarification.fields)} questions before generating"
            )
            await save_job(job)
            log.info("Job %s paused for clarification (%d fields)", job_id, len(clarification.fields))
            return  # Pipeline pauses here — resume_pipeline() called after user answers

        # No clarification needed → proceed
        await _continue_pipeline(job)

    except Exception as e:
        log.exception("Unhandled exception in pipeline for job %s", job_id)
        await _fail(job, f"Internal pipeline error: {str(e)}")


async def resume_pipeline(job_id: str) -> None:
    """
    Called after user submits clarification answers.
    Picks up from the planning stage.
    """
    log.info("Resuming pipeline for job %s", job_id)
    job = await get_job(job_id)
    if not job:
        log.error("Job %s not found for resume", job_id)
        return

    if job.status != JobStatus.NEEDS_CLARIFY:
        log.warning("Job %s not in NEEDS_CLARIFY state (is %s)", job_id, job.status)
        return

    try:
        await _continue_pipeline(job)
    except Exception as e:
        log.exception("Error resuming pipeline for job %s", job_id)
        await _fail(job, f"Resume error: {str(e)}")


async def _continue_pipeline(job: Job) -> None:
    """Stages 3-6: plan → generate → validate → package."""

    parsed = job.parsed_diagram
    if not parsed:
        await _fail(job, "No parsed diagram available for planning")
        return

    # Apply clarification answers to parsed diagram
    if job.clarification_answers:
        from app.services.parser.missing_info_detector import apply_clarification_answers
        parsed = apply_clarification_answers(parsed, job.clarification_answers)
        job.parsed_diagram = parsed

    # Generate this job's vars.yaml from the now-finalized answer set (her
    # explicit request: whether built from scratch or gap-filled on top of
    # an uploaded one, the pipeline always produces an up-to-date vars.yaml
    # that could be fed back in as `input_vars` next time this diagram is
    # re-uploaded). Bundled into the ZIP and pushed to GitHub next to the
    # diagram — see packager.py / github_pusher.py.
    from app.services.parser.missing_info_detector import generate_vars_yaml
    job.generated_vars_yaml = generate_vars_yaml(parsed, job.clarification_answers)

    # ── Stage 3: Plan modules ────────────────────────────────────────────────
    await _update(job, JobStatus.PLANNING, "Building Terraform module plan...")

    # Extract global settings from clarification answers
    answers_map = {a.field_key: a.value for a in job.clarification_answers}
    aws_region   = answers_map.get("aws_region", "us-east-1")
    environment  = answers_map.get("environment", "dev")
    project_name = answers_map.get("project_name", "arch2terraform")

    tf_plan = await build_terraform_plan(
        parsed,
        aws_region=aws_region,
        environment=environment,
        project_name=project_name,
    )
    job.terraform_plan = tf_plan
    job.log(f"Planned {len(tf_plan.modules)} modules with {tf_plan.resource_count} resources")

    # ── Stage 4: Generate HCL ────────────────────────────────────────────────
    await _update(job, JobStatus.GENERATING, "Generating Terraform HCL code...")
    # Plan generation is already done inside build_terraform_plan;
    # this stage marker is for UI feedback (could be a slow step for large diagrams)
    total_files = len(tf_plan.root_module_files) + sum(len(m.files) for m in tf_plan.modules)
    job.log(f"Generated {total_files} .tf files across {len(tf_plan.modules)} modules")

    # ── Stage 5: Validate ────────────────────────────────────────────────────
    await _update(job, JobStatus.VALIDATING, "Running terraform validate, tflint, checkov...")
    existing_state_bytes = await _read_existing_state_bytes(job)
    validation = await validate_terraform(tf_plan, existing_state_bytes=existing_state_bytes)
    job.validation_result = validation
    job.log(
        f"Validation complete: {validation.passed_count} passed, "
        f"{validation.warning_count} warnings, {validation.failed_count} errors"
    )

    # ── Stage 6: Package ─────────────────────────────────────────────────────
    await _update(job, JobStatus.PACKAGING, "Packaging output ZIP...")
    zip_path, readme = await package_output(job)
    job.zip_path = zip_path
    job.readme_content = readme
    job.log(f"ZIP ready: {zip_path}")

    # ── Done ─────────────────────────────────────────────────────────────────
    await _update(job, JobStatus.DONE, "Pipeline complete — ready to download")
    await save_job(job)
    log.info("Pipeline complete for job %s", job.job_id)


async def revalidate_with_state(job_id: str) -> None:
    """
    Re-runs just the validate + package stages for an ALREADY-COMPLETED job,
    reusing its existing terraform_plan (no re-parse/re-plan needed) — this
    is what /jobs/{job_id}/upload-state kicks off after storing a new state
    file, so the drift check (and the re-packaged ZIP — see packager.py)
    reflect it without redoing the whole pipeline from scratch.
    """
    log.info("Revalidating job %s with uploaded state", job_id)
    job = await get_job(job_id)
    if not job:
        log.error("Job %s not found for revalidate", job_id)
        return
    if not job.terraform_plan:
        log.warning("Job %s has no terraform_plan yet — cannot revalidate", job_id)
        return

    try:
        await _update(job, JobStatus.VALIDATING, "Re-running validation with uploaded state...")
        existing_state_bytes = await _read_existing_state_bytes(job)
        validation = await validate_terraform(job.terraform_plan, existing_state_bytes=existing_state_bytes)
        job.validation_result = validation
        job.log(
            f"Re-validation complete: {validation.passed_count} passed, "
            f"{validation.warning_count} warnings, {validation.failed_count} errors"
        )

        await _update(job, JobStatus.PACKAGING, "Repackaging output ZIP...")
        zip_path, readme = await package_output(job)
        job.zip_path = zip_path
        job.readme_content = readme
        job.log(f"ZIP re-packaged: {zip_path}")

        await _update(job, JobStatus.DONE, "Re-validation complete — ready to download")
        await save_job(job)
        log.info("Revalidation complete for job %s", job.job_id)
    except Exception as e:
        log.exception("Error revalidating job %s", job_id)
        await _fail(job, f"Revalidate error: {str(e)}")


async def _read_existing_state_bytes(job: Job) -> bytes | None:
    """If she's uploaded an existing state file for this job (see the
    /jobs/{job_id}/upload-state route), read it so validate_terraform() can
    seed it into the sandbox check — `terraform plan` then reflects real
    drift against that snapshot instead of an empty slate. See
    validate_terraform's docstring for why this stays side-effect-free (no
    real backend/lock ever touched from this sandbox path)."""
    if not job.state_file_path:
        return None
    try:
        return await read_upload(job.state_file_path)
    except Exception:
        log.warning("Could not read uploaded state file for job %s — validating without it", job.job_id, exc_info=True)
        return None


def _merge_answers(job: Job, new_answers: list) -> None:
    """Merge answers into job.clarification_answers — same dedup-by-
    (resource_id, field_key) logic as the /clarify route's
    submit_clarification, reused here for vars.yaml auto-fills so a later
    real clarification submission (if she still edits one in the UI) can
    still override an auto-filled value rather than duplicating it."""
    existing = {(a.resource_id, a.field_key): a for a in job.clarification_answers}
    for ans in new_answers:
        key = (ans.resource_id, ans.field_key)
        if key in existing:
            existing[key].value = ans.value
        else:
            job.clarification_answers.append(ans)
            existing[key] = ans


async def _update(job: Job, status: JobStatus, msg: str) -> None:
    """Update job status and persist."""
    job.status = status
    job.log(msg)
    await save_job(job)
    log.info("[%s] %s: %s", job.job_id[:8], status.value, msg)


async def _fail(job: Job, msg: str) -> None:
    """Mark job as failed."""
    job.status = JobStatus.FAILED
    job.error_message = msg
    job.log(f"FAILED: {msg}")
    await save_job(job)
    log.error("[%s] Pipeline failed: %s", job.job_id[:8], msg)
