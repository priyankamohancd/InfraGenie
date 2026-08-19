"""
Job Store — Redis-failure fallback tests
--------------------------------------------
`app/core/job_store.py` falls back to an in-process dict when Redis is
unreachable at connection time (`_get_redis()`'s `ping()` fails) — that path
was already exercised implicitly by every other test in this suite, since
this sandbox/dev environment has no Redis running.

What was NOT covered, and turned out to be a real bug (found 2026-07-08
while auditing pipeline_worker.py + job_store.py, previously only had their
sys.path bugs fixed, never a full logic review): `save_job`/`get_job`/
`delete_job` called the actual Redis command (`r.setex`/`r.get`/`r.delete`)
with no exception handling at all — only the initial `ping()` inside
`_get_redis()` was guarded. If Redis answers the ping but then the actual
command fails (network blip, Redis restart, eviction — anything after the
connection is established), the exception propagated straight out of
`save_job()`. Since `save_job` is called after every pipeline stage
transition from inside a FastAPI `BackgroundTask` (see pipeline_worker.py's
`_update()`), an uncaught exception there silently kills the pipeline
mid-run: the job's last successfully-saved state just sits there, the
frontend keeps polling forever, and the user sees a spinner that never
resolves with no error surfaced anywhere. Reproduced directly against the
real code before fixing (a fake Redis client whose `ping()` succeeds but
whose `setex()`/`get()`/`delete()` raise `ConnectionError`), confirming the
exception really did propagate uncaught.

These tests pin the fixed behavior: a mid-operation Redis failure degrades
to the in-process store instead of raising.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.core import job_store
from shared.schemas.models import Job, JobStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FlakyRedis:
    """A Redis client that answers ping() (so _get_redis() caches it as
    connected) but raises on every actual command — the exact shape of a
    connection that dies after being established."""

    def ping(self):
        return True

    def setex(self, *a, **kw):
        raise ConnectionError("Connection reset by peer")

    def get(self, *a, **kw):
        raise ConnectionError("Connection reset by peer")

    def delete(self, *a, **kw):
        raise ConnectionError("Connection reset by peer")


@pytest.fixture(autouse=True)
def _reset_job_store_state():
    """Isolate each test's view of the module-level _redis/_LOCAL_STORE
    globals so tests can't leak state into each other or into the rest of
    the suite."""
    original_redis = job_store._redis
    original_store = dict(job_store._LOCAL_STORE)
    yield
    job_store._redis = original_redis
    job_store._LOCAL_STORE.clear()
    job_store._LOCAL_STORE.update(original_store)


# NOTE (2026-08-19): this file used to carry its own `_no_real_redis_connection`
# monkeypatch fixture here. It's been superseded by a suite-wide equivalent
# in tests/conftest.py, after the same real-Redis-reachability problem this
# fixture guarded against turned out to also corrupt real job records via
# test_apply_runner.py::test_reconcile_destroys_overdue_job_after_simulated_restart
# (see that file and conftest.py's docstring for the full story). Keeping
# the isolation centralized there so every test file gets it automatically
# instead of each one needing to remember to add its own copy.


@pytest.mark.anyio
async def test_save_job_falls_back_when_redis_command_fails_after_successful_ping():
    job_store._redis = _FlakyRedis()
    job = Job(original_filename="test.drawio")

    # Must NOT raise — this is the exact bug: it used to.
    await job_store.save_job(job)

    # And the job must actually be retrievable afterwards (fell all the way
    # through to the in-process store, not just swallowed the write).
    assert job.job_id in job_store._LOCAL_STORE


@pytest.mark.anyio
async def test_save_job_failure_resets_redis_client_for_next_call():
    job_store._redis = _FlakyRedis()
    job = Job(original_filename="test.drawio")
    await job_store.save_job(job)

    # A failed command should force a fresh connection attempt next time,
    # rather than continuing to hand out the same tainted client forever.
    assert job_store._redis is None


@pytest.mark.anyio
async def test_get_job_falls_back_when_redis_command_fails():
    job_store._redis = None
    job = Job(original_filename="test.drawio", status=JobStatus.PARSING)
    await job_store.save_job(job)  # lands in the in-process store (no real redis in test env)

    job_store._redis = _FlakyRedis()
    fetched = await job_store.get_job(job.job_id)

    assert fetched is not None
    assert fetched.job_id == job.job_id
    assert fetched.status == JobStatus.PARSING


@pytest.mark.anyio
async def test_delete_job_falls_back_when_redis_command_fails():
    job_store._redis = None
    job = Job(original_filename="test.drawio")
    await job_store.save_job(job)
    assert job.job_id in job_store._LOCAL_STORE

    job_store._redis = _FlakyRedis()
    await job_store.delete_job(job.job_id)  # must not raise

    assert job.job_id not in job_store._LOCAL_STORE


@pytest.mark.anyio
async def test_full_pipeline_state_transition_survives_a_mid_run_redis_blip():
    """End-to-end simulation of the actual failure scenario: a job is
    progressing through stage updates (as pipeline_worker.py's `_update()`
    does), Redis dies mid-run, and the job's status must still end up
    correctly persisted (via fallback) instead of the update silently
    vanishing."""
    job_store._redis = None
    job = Job(original_filename="test.drawio", status=JobStatus.UPLOADED)
    await job_store.save_job(job)

    # Redis "dies" partway through the pipeline.
    job_store._redis = _FlakyRedis()

    job.status = JobStatus.PARSING
    await job_store.save_job(job)  # would have raised before the fix

    fetched = await job_store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.status == JobStatus.PARSING
