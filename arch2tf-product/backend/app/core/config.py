"""
App configuration — all env vars with sane defaults for local dev.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App
    app_name: str = "arch2terraform"
    app_version: str = "0.1.0"
    debug: bool = True

    # Storage — local filesystem for dev, swap to S3 for prod
    storage_backend: str = "local"           # "local" | "s3"
    local_upload_dir: str = "/tmp/arch2tf/uploads"
    local_output_dir: str = "/tmp/arch2tf/outputs"
    s3_bucket: str = ""
    s3_prefix: str = "arch2tf"

    # Redis (job state)
    redis_url: str = "redis://localhost:6379/0"
    job_ttl_seconds: int = 3600 * 24        # 24h

    # AWS sandbox (for terraform validate/plan)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    aws_region: str = "us-east-1"
    sandbox_enabled: bool = False            # flip to True when AWS creds configured

    # Terraform runner
    terraform_binary: str = "terraform"
    terraform_timeout_seconds: int = 120
    terraform_init_timeout_seconds: int = 120
    tf_plugin_cache_dir: str = os.path.expanduser("~/.arch2tf/provider-cache")
    tflint_binary: str = "tflint"
    checkov_binary: str = "checkov"

    # "Apply to Sandbox" — 2026-07-24, her explicit request to run a real
    # `terraform apply` against her own AWS account. Deliberately does NOT
    # add its own aws_access_key_id/secret settings the way the
    # tf_validator.py sandbox-plan path does above: apply_runner.py
    # inherits whatever AWS credential chain is already active in the
    # backend process's own environment (her ~/.aws/credentials [default]
    # profile, or AWS_PROFILE/AWS_ACCESS_KEY_ID if she's exported them
    # before starting uvicorn) — this backend never reads, stores, or
    # transmits her credentials itself, it just doesn't override them.
    terraform_apply_timeout_seconds: int = 600      # real infra takes longer than a plan
    terraform_destroy_timeout_seconds: int = 600
    apply_auto_destroy_hours: float = 2.0
    apply_reconcile_poll_seconds: int = 300          # how often the startup loop checks for overdue destroys

    # Remote state backend — a one-time infra decision (which bucket holds
    # everyone's Terraform state), not something that should vary per diagram
    # upload, so it's configured here rather than asked in the clarification
    # UI. Left empty by default: terraform_planner.py's _generate_root_module
    # only emits a REAL (uncommented) S3 backend block when tf_state_bucket
    # is set — otherwise it falls back to the pre-existing commented-out
    # placeholder, so nothing breaks for anyone who hasn't set this up yet.
    # One state file per environment (key = "{project}/{environment}/terraform.tfstate")
    # so dev/staging/prod never share or clobber each other's state, matching
    # the directory-per-environment convention already used for the
    # generated code itself and the GitHub push paths.
    tf_state_bucket: str = ""
    tf_state_lock_table: str = "terraform-locks"
    tf_state_region: str = ""  # falls back to the job's aws_region if unset

    # Alternative remote state backend — Terraform Cloud / HCP Terraform,
    # added 2026-07-29 per her explicit request (used for CLI-driven
    # plan/apply against a TFC-hosted workspace, not TFC's own managed
    # remote runs — the workspace's execution mode must be set to "Local"
    # in the TFC UI for this to work, same CLI-driven flow apply_runner.py
    # already does against S3). "s3" (the pre-existing default) keeps every
    # current deployment's behavior unchanged; set to "cloud" to switch
    # _backend_tf() over to generating a `cloud { }` block instead of
    # `backend "s3" { }`. Auth is never handled by this backend itself —
    # exactly the same "inherit whatever's already active locally" model as
    # AWS credentials (see apply_runner.py's module docstring): run
    # `terraform login` once on whichever machine runs plan/apply, or set
    # TF_TOKEN_<hostname_with_dots_as_underscores> in that machine's own
    # environment.
    tf_backend_type: str = "s3"        # "s3" | "cloud"
    tf_cloud_organization: str = ""
    tf_cloud_hostname: str = "app.terraform.io"

    # Clarification
    max_clarification_rounds: int = 2
    confidence_threshold: float = 0.6       # below this → ask user

    # CORS
    # "null" is what browsers send as the Origin header for pages opened
    # directly from disk (file://...), which is exactly how this project's
    # own frontend/index.html is meant to be opened (there's no dev server
    # for it, no localhost:3000/3001 ever serves it) — found 2026-07-08 when
    # opening index.html via file:// produced "Cannot connect to backend" on
    # upload even though uvicorn was running: the request was reaching the
    # server fine, the browser was just blocking the response because
    # Origin: null wasn't in this list. Keeping localhost:3000/3001 too in
    # case a real dev server (Vite/CRA) ever fronts this frontend instead.
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001", "null"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
