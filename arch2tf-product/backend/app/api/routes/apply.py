"""
API Routes — "Apply to Sandbox"
---------------------------------
POST /api/v1/jobs/{job_id}/apply/plan     — write TF, preflight check, init + plan, returns confirm_token
POST /api/v1/jobs/{job_id}/apply/resolve  — submit real values for blocked placeholder variables, re-plans
POST /api/v1/jobs/{job_id}/apply/drift    — read-only: refresh-only plan, reports drift vs real infra
POST /api/v1/jobs/{job_id}/apply/confirm  — apply the plan just reviewed (requires confirm_token)
POST /api/v1/jobs/{job_id}/apply/destroy  — terraform destroy, cancels any pending auto-destroy
GET  /api/v1/jobs/{job_id}/apply/status   — poll apply_status + log + destroy countdown

Added 2026-07-24, her explicit request to complete the pipeline through a
real `terraform apply` against her own AWS sandbox account. Runs on HER
machine (see apply_runner.py's module docstring for why) — this backend
never touches her AWS credentials itself, only inherits whatever chain is
already active in its own process environment.
"""
from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
import sys

from fastapi import APIRouter, BackgroundTasks, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from shared.schemas.models import (
    ApplyStatus, ApplyPlanResponse, ApplyConfirmRequest, ApplyResolveRequest,
    ApplyStatusResponse, DriftCheckResponse,
)

from app.core.job_store import get_job
from app.services.apply import apply_runner

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["apply"])


@router.post("/jobs/{job_id}/apply/plan", response_model=ApplyPlanResponse)
async def apply_plan(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet for this job")
    if job.apply_status in (ApplyStatus.PLANNING, ApplyStatus.APPLYING, ApplyStatus.DESTROYING):
        raise HTTPException(
            status_code=409,
            detail=f"An apply operation is already in progress (apply_status={job.apply_status.value})",
        )

    try:
        job = await apply_runner.plan_apply(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        log.exception("apply_plan failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="terraform plan crashed unexpectedly — check server logs")

    return _plan_response(job)


@router.post("/jobs/{job_id}/apply/resolve", response_model=ApplyPlanResponse)
async def apply_resolve(job_id: str, body: ApplyResolveRequest):
    """
    Submits real values for whatever apply/plan flagged as blocked_variables
    (2026-07-27, her explicit correction: the person who needs to supply
    these values is whoever's using the UI, not whoever has terminal
    access to the backend's filesystem). Re-runs plan automatically —
    the response has the exact same shape as apply/plan, so the frontend
    can reuse one render path for "still blocked, here's what's left" vs.
    "resolved, here's the real plan output."
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not body.overrides:
        raise HTTPException(status_code=400, detail="No values submitted.")

    try:
        job = await apply_runner.resolve_placeholders(job_id, body.overrides)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        log.exception("apply_resolve failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="Resolving placeholders crashed unexpectedly — check server logs")

    return _plan_response(job)


@router.post("/jobs/{job_id}/apply/drift", response_model=DriftCheckResponse)
async def apply_drift(job_id: str):
    """
    Read-only: refreshes state against real infrastructure and reports what
    changed OUTSIDE Terraform, without proposing or applying anything.
    Separate from /apply/plan on purpose (2026-07-29, her explicit
    request) — can be run any time there's a plan on the job, independent
    of the plan -> confirm -> apply gate, to answer "did anything drift"
    on its own.
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if not job.terraform_plan:
        raise HTTPException(status_code=409, detail="No Terraform plan generated yet for this job")
    if job.apply_status in (ApplyStatus.PLANNING, ApplyStatus.APPLYING, ApplyStatus.DESTROYING):
        raise HTTPException(
            status_code=409,
            detail=f"An apply operation is already in progress (apply_status={job.apply_status.value})",
        )

    try:
        job = await apply_runner.check_drift(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        log.exception("apply_drift failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="drift check crashed unexpectedly — check server logs")

    return DriftCheckResponse(
        job_id=job.job_id,
        drift_status=job.drift_status,
        drift_output=job.drift_output,
        drift_resources=job.drift_resources,
        checked_at=job.drift_checked_at,
    )


@router.post("/jobs/{job_id}/apply/confirm", response_model=ApplyStatusResponse)
async def apply_confirm(job_id: str, body: ApplyConfirmRequest):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        job = await apply_runner.confirm_apply(job_id, body.confirm_token)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        log.exception("apply_confirm failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="terraform apply crashed unexpectedly — check server logs")

    return _status_response(job)


@router.post("/jobs/{job_id}/apply/destroy", response_model=ApplyStatusResponse)
async def apply_destroy(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        job = await apply_runner.destroy_apply(job_id, reason="manual")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        log.exception("apply_destroy failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="terraform destroy crashed unexpectedly — check server logs")

    return _status_response(job)


@router.get("/jobs/{job_id}/apply/status", response_model=ApplyStatusResponse)
async def apply_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _status_response(job)


def _plan_response(job) -> ApplyPlanResponse:
    return ApplyPlanResponse(
        job_id=job.job_id,
        apply_status=job.apply_status,
        plan_output=job.apply_plan_output,
        confirm_token=job.apply_confirm_token,
        confirm_token_expires_at=job.apply_confirm_token_expires_at,
        blocked_reason=job.apply_error if job.apply_status == ApplyStatus.NOT_STARTED else "",
        blocked_variables=job.apply_blocked_variables,
    )


def _status_response(job) -> ApplyStatusResponse:
    destroy_in = None
    if job.apply_destroy_at:
        destroy_in = max(0, int((job.apply_destroy_at - datetime.utcnow()).total_seconds()))
    return ApplyStatusResponse(
        job_id=job.job_id,
        apply_status=job.apply_status,
        apply_log=job.apply_log,
        apply_error=job.apply_error,
        destroy_at=job.apply_destroy_at,
        destroy_in_seconds=destroy_in,
    )
