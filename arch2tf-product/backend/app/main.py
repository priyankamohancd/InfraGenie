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

# Load backend/.env into the REAL process environment (os.environ), not just
# into pydantic-settings' own Settings object. Fixed 2026-08-20: Settings
# (app.core.config) reads .env fine on its own for its declared fields, but
# arch2terraform's vision-LLM toggle (ANTHROPIC_API_KEY,
# ARCH2TERRAFORM_USE_VISION_LLM, ARCH2TERRAFORM_VISION_LLM_MODEL — see
# image_adapter.py) is deliberately NOT a Settings field (see config.py's
# Config.extra="ignore" comment) and is read directly via os.environ.get()
# instead. Under `docker compose up`, Compose's own env_file: mechanism sets
# real container env vars, so this worked in that deployment without anyone
# noticing the gap. Running the backend directly via `uvicorn app.main:app`
# for local dev (no Docker involved) never had anything that actually
# populated os.environ from .env — so setting ARCH2TERRAFORM_USE_VISION_LLM=true
# in .env silently had no effect locally: the classical CV cascade kept
# running instead of the vision-LLM path, with no error or warning anywhere.
# load_dotenv() here closes that gap for both dev flows identically. Must run
# before any other app import that might read one of these vars at import
# time.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app._pathboot import ensure_paths
ensure_paths()

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
    # Log the diagram-parsing path being used at startup, loud and
    # unmissable — added 2026-08-20 alongside the load_dotenv() fix above,
    # specifically so this toggle never again silently fails to take effect
    # (it did exactly that for weeks: ARCH2TERRAFORM_USE_VISION_LLM=true sat
    # in .env, but nothing loaded it into os.environ for local `uvicorn`
    # runs, so every diagram silently kept going through the classical CV
    # cascade instead — see image_adapter.py's docstring for the failure
    # mode this caused). Confirm this line actually says "ENABLED" before
    # trusting that a diagram upload will use the vision-LLM path.
    import os
    _vision_on = os.environ.get("ARCH2TERRAFORM_USE_VISION_LLM", "").strip().lower() in ("1", "true", "yes")
    log.info(
        "Diagram image parsing: vision-LLM path %s (model=%s)",
        "ENABLED" if _vision_on else "DISABLED — using classical CV cascade",
        os.environ.get("ARCH2TERRAFORM_VISION_LLM_MODEL") or "claude-sonnet-5 (default)",
    )
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
