"""
Job Store
----------
Persists Job objects in Redis (JSON-serialised).
Falls back to an in-process dict when Redis is unavailable (dev mode).
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional

from app.core.config import get_settings

log = logging.getLogger(__name__)

# Shared models imported from the monorepo shared package.
# core/job_store.py -> core/../.. = app/../.. = backend/../.. = arch2tf-product
# (3 levels up). Was 4 levels up ("../../../.."), landing on "thesis" instead —
# pre-existing bug, same class as missing_info_detector.py's.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from shared.schemas.models import Job, JobStatus

_settings = get_settings()

# ── in-process fallback ──────────────────────────────────────────────────────
_LOCAL_STORE: dict[str, str] = {}

# ── Redis client (lazy) ──────────────────────────────────────────────────────
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis
        client = redis.from_url(_settings.redis_url, decode_responses=True)
        client.ping()
        _redis = client
        log.info("Connected to Redis at %s", _settings.redis_url)
    except Exception as e:
        log.warning("Redis unavailable (%s) — using in-process store", e)
        _redis = None
    return _redis


def _serialize(job: Job) -> str:
    return job.model_dump_json()


def _deserialize(raw: str) -> Job:
    data = json.loads(raw)
    return Job.model_validate(data)


async def save_job(job: Job) -> None:
    job.updated_at = datetime.utcnow()
    raw = _serialize(job)
    r = _get_redis()
    if r:
        try:
            r.setex(f"job:{job.job_id}", _settings.job_ttl_seconds, raw)
            return
        except Exception as e:
            # `_get_redis()`'s `ping()` only proves Redis was reachable at
            # *connection* time — a network blip, Redis restart, or eviction
            # after that leaves `_redis` holding a now-dead client, and this
            # command raises. Verified 2026-07-08: an uncaught exception here
            # propagates straight out of a FastAPI BackgroundTask (this is
            # always called from pipeline_worker.py's stage transitions),
            # silently killing the pipeline mid-run — the job's last
            # successfully-saved state just sits there forever, the frontend
            # keeps polling, and the user sees a spinner that never resolves,
            # with no error surfaced anywhere. Falling back to the in-process
            # store lets this specific call still succeed, and clearing
            # `_redis` forces a fresh reconnect attempt on the next call
            # instead of continuing to hand out a client that's actually
            # tainted.
            log.warning("Redis save failed (%s) — falling back to in-process store for this job", e)
            _reset_redis()
    _LOCAL_STORE[job.job_id] = raw


async def get_job(job_id: str) -> Optional[Job]:
    r = _get_redis()
    raw = None
    if r:
        try:
            raw = r.get(f"job:{job_id}")
        except Exception as e:
            # Same class of failure as save_job's — see that comment. Falls
            # through to the in-process store, which won't have this job's
            # data unless it was also saved there (e.g. Redis was already
            # down when it was created), but that's still a clean "not
            # found" rather than an unhandled crash.
            log.warning("Redis read failed (%s) — falling back to in-process store", e)
            _reset_redis()

    if raw is None:
        raw = _LOCAL_STORE.get(job_id)

    if not raw:
        return None
    return _deserialize(raw)


def _reset_redis() -> None:
    """Force the next _get_redis() call to attempt a fresh connection
    instead of continuing to hand out a client whose connection has gone
    bad after the initial ping() succeeded."""
    global _redis
    _redis = None


async def update_status(job_id: str, status: JobStatus, msg: str = "") -> Optional[Job]:
    job = await get_job(job_id)
    if not job:
        return None
    job.status = status
    if msg:
        job.log(msg)
    await save_job(job)
    return job


async def list_job_ids() -> list[str]:
    """
    Added 2026-07-24 for the "Apply to Sandbox" auto-destroy reconciliation
    loop (see apply_runner.py / main.py's lifespan hook) — needs to scan
    every job for an overdue apply_destroy_at after a backend restart, and
    there was previously no way to enumerate jobs at all (every other call
    site already knows its job_id up front). Best-effort only: the
    in-process fallback dict is authoritative for itself, and Redis's
    `keys()` is O(n) but this store is small/dev-scale (single sandbox
    project, not a production job queue) so that's an acceptable tradeoff
    over adding a secondary index just for this.
    """
    r = _get_redis()
    if r:
        try:
            keys = r.keys("job:*")
            return [k.split("job:", 1)[1] for k in keys]
        except Exception as e:
            log.warning("Redis keys() failed (%s) — falling back to in-process store", e)
            _reset_redis()
    return list(_LOCAL_STORE.keys())


async def delete_job(job_id: str) -> None:
    r = _get_redis()
    if r:
        try:
            r.delete(f"job:{job_id}")
            return
        except Exception as e:
            log.warning("Redis delete failed (%s) — falling back to in-process store", e)
            _reset_redis()
    _LOCAL_STORE.pop(job_id, None)
