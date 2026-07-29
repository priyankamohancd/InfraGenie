"""
arch2terraform — FastAPI Application
"""
import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.config import get_settings
from app.api.routes.pipeline import router as pipeline_router
from app.api.routes.apply import router as apply_router
from app.services.apply.apply_runner import reconcile_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)
_s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("arch2terraform API starting up (debug=%s)", _s.debug)
    # Ensure storage dirs exist
    Path(_s.local_upload_dir).mkdir(parents=True, exist_ok=True)
    Path(_s.local_output_dir).mkdir(parents=True, exist_ok=True)

    # "Apply to Sandbox" auto-destroy safety net (2026-07-24): catches any
    # job whose 2-hour auto-destroy deadline is being tracked only in an
    # in-process asyncio task that was lost by a server restart (e.g.
    # uvicorn --reload picking up a code change mid-window). See
    # apply_runner.reconcile_overdue_destroys()'s docstring.
    reconcile_task = asyncio.create_task(reconcile_loop())

    yield

    reconcile_task.cancel()
    log.info("arch2terraform API shutting down")


app = FastAPI(
    title="arch2terraform API",
    description="Convert architecture diagrams to production-ready Terraform modules",
    version=_s.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_s.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)
app.include_router(apply_router)


@app.get("/")
async def root():
    return {
        "service": "arch2terraform",
        "version": _s.app_version,
        "docs": "/docs",
    }
