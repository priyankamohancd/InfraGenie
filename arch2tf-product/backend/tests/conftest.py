"""
Shared fixtures for the arch2tf-product backend test suite.

Global Redis isolation
-----------------------
`app.core.job_store` lazily connects to a real Redis at
`settings.redis_url` and falls back to an in-process `_LOCAL_STORE` dict
whenever that connection is unavailable. Two real, distinct problems
traced back to this on 2026-08-19 while chasing a fresh full-suite run:

1. `test_job_store.py`'s Redis-fallback tests set `job_store._redis = None`
   to simulate "not connected yet," but if a real Redis happens to be
   reachable at `settings.redis_url` on the machine running the tests
   (e.g. left running from a prior `docker compose up`, or `brew services
   start redis`), `_get_redis()` genuinely reconnects to it instead of
   staying in the fallback state those tests are asserting against --
   making pass/fail depend on incidental local machine state rather than
   the code under test. 3 of 5 tests in that file failed on a run where a
   local Redis was reachable and passed on one where it wasn't.

2. Much more serious, surfaced via `test_apply_runner.py::
   test_reconcile_destroys_overdue_job_after_simulated_restart`: with a
   real Redis reachable, `reconcile_overdue_destroys()`'s `list_job_ids()`
   scan (`r.keys("job:*")`) returns EVERY job key in that real Redis, not
   just the one job the test created -- so the test destroyed 5 jobs
   instead of the 1 it expected, and 4 of those were real leftover records
   from actual local use of the product, not test fixtures at all.
   `destroy_apply` was monkeypatched inside that specific test, so no real
   `terraform destroy` ran against real infrastructure -- but those real
   job records still got their `apply_status` field overwritten to
   DESTROYED and `apply_destroy_at` cleared. For any of those jobs that
   still had real sandbox infrastructure actually running, that silently
   defeats the auto-destroy safety net for them (nothing will ever try to
   destroy them again, since the reconciler only acts on
   `apply_status == APPLIED`), which is a real safety/cost concern, not
   just a test artifact. If you're reading this after that happened: check
   the AWS sandbox account for anything left running from job ids
   e1eb6bd2, 38e5c9e7, ebd5e31d, 9e0b3a83, or 106f06d7 and destroy it
   manually if still present.

No test file should ever be able to reach a real Redis, or leak Job
records into `_LOCAL_STORE` for other tests to trip over -- both matter
for a thesis whose central claim is reproducibility. This fixture patches
`job_store._get_redis` so it only ever reflects the in-process `_redis`/
`_LOCAL_STORE` state that a test explicitly sets, never a real connection,
and resets both to a clean slate before and after every test.
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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _isolated_job_store(monkeypatch):
    monkeypatch.setattr(job_store, "_get_redis", lambda: job_store._redis)
    original_redis = job_store._redis
    original_store = dict(job_store._LOCAL_STORE)
    job_store._redis = None
    job_store._LOCAL_STORE.clear()
    yield
    job_store._redis = original_redis
    job_store._LOCAL_STORE.clear()
    job_store._LOCAL_STORE.update(original_store)
