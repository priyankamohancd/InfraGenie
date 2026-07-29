"""
Shared data contracts — used by backend services and surfaced to frontend via API.
All pipeline state flows through these models.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    UPLOADED       = "uploaded"
    PARSING        = "parsing"
    PARSED         = "parsed"
    NEEDS_CLARIFY  = "needs_clarification"
    PLANNING       = "planning"
    GENERATING     = "generating"
    VALIDATING     = "validating"
    PACKAGING      = "packaging"
    DONE           = "done"
    FAILED         = "failed"


class DiagramFormat(str, Enum):
    DRAWIO      = "drawio"
    LUCIDCHART  = "lucidchart"
    EXCALIDRAW  = "excalidraw"
    IMAGE       = "image"
    UNKNOWN     = "unknown"


class CloudProvider(str, Enum):
    AWS = "aws"


class ValidationStatus(str, Enum):
    PASSED  = "passed"
    FAILED  = "failed"
    WARNING = "warning"


class ApplyStatus(str, Enum):
    """
    "Apply to Sandbox" stage — added 2026-07-24, her explicit request to
    complete the pipeline all the way through a REAL `terraform apply`
    against her own AWS sandbox account (credentials picked up from her
    local ~/.aws/credentials [default] profile — never uploaded to or
    stored by this backend, see apply_runner.py). Deliberately a SEPARATE
    state machine from JobStatus: applying is a manual, post-download,
    explicitly-confirmed action the user triggers on her own machine, not
    an automatic pipeline stage every job passes through.
    """
    NOT_STARTED       = "not_started"
    PLANNING          = "planning"           # terraform init + plan running
    AWAITING_CONFIRM  = "awaiting_confirm"    # plan succeeded, waiting on her explicit confirm
    APPLYING          = "applying"
    APPLIED           = "applied"             # live in her AWS account right now
    DESTROYING        = "destroying"
    DESTROYED         = "destroyed"
    FAILED            = "failed"


class DriftStatus(str, Enum):
    """
    "Check Drift" — added 2026-07-29, her explicit request: distinguish
    "does the last-applied state still match real AWS" from the normal
    plan-before-apply flow. Technically `terraform plan` already refreshes
    state against real infra on every call (that's how Terraform always
    works), but that fact was buried in raw plan text — this makes it a
    first-class, separately-triggerable, read-only check
    (`terraform plan -refresh-only`, never proposes or applies config
    changes) with its own status so the UI can show "drift detected" as a
    distinct signal from "there's a pending plan to apply".
    """
    UNKNOWN         = "unknown"          # never checked yet for this job
    CHECKING        = "checking"
    CLEAN           = "clean"            # refresh-only plan found no differences
    DRIFT_DETECTED  = "drift_detected"   # real infra has changed outside Terraform
    FAILED          = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Parsed resource graph (output of parser service)
# ─────────────────────────────────────────────────────────────────────────────

class ParsedResource(BaseModel):
    id: str
    aws_resource_type: str         # e.g. "aws_instance"
    logical_name: str              # slugified TF name
    label: str                     # original diagram label
    properties: dict[str, Any] = {}
    # Required HCL nested blocks (e.g. aws_eks_cluster's vpc_config), carried
    # through from arch2terraform's catalog via arch2terraform_bridge.py.
    # Rendered by terraform_planner.py via arch2terraform's hcl_format.resource_block()
    # rather than a Phase 2-local reimplementation — see arch2terraform_bridge.py's
    # module docstring for why.
    nested_blocks: dict[str, list[dict]] = {}
    tags: dict[str, str] = {}
    confidence: float = 1.0        # 0–1, lower = needs clarification
    match_source: str = "style"    # "style" | "label" | "fallback"
    # Field keys that missing_info_detector.py flagged as "needs a real
    # value" for this specific resource — populated for every MANDATORY_FIELDS
    # entry (the ~12 hand-covered resource types) AND every field caught only
    # by the generic catalog-wide placeholder fallback (~25 more resource
    # types: aws_ecr_repository, aws_dynamodb_table, aws_route53_zone, etc.).
    # terraform_planner.py's _variableize_mandatory_fields() reads this to
    # decide which properties become real `variable` blocks instead of
    # literals — without it, only the original ~12 hand-covered types would
    # ever get tfvars-overridable variables, leaving the newly-discovered
    # ~25 asked-but-still-baked. Empty by default so any caller building a
    # ParsedResource directly (tests, synthetic diagrams) without going
    # through detect_missing_info() still gets the pre-existing
    # MANDATORY_FIELDS-only behavior, not silently zero variables.
    variableize_keys: list[str] = []
    # Extra, pre-rendered top-level HCL resource blocks that must land in the
    # SAME module as this resource (e.g. aws_mq_broker's random_password +
    # Secrets Manager pair backing its required password — see
    # arch2terraform's classifier.py's _build_mq_broker_companion_blocks()).
    # Carried through unchanged from ClassifiedResource.companion_blocks via
    # arch2terraform_bridge.py; empty for every other resource type.
    companion_blocks: list[str] = []


class ParsedConnection(BaseModel):
    source_id: str
    target_id: str
    # "network" | "security" | "data" | "iam" | "dependency" | "containment"
    # "containment" (source=container, target=nested resource) is handled
    # specially by terraform_planner.py's wiring pass, NOT folded into the
    # generic depends_on logic — treating a containment edge as a normal
    # "source depends on target" edge would be backwards (the VPC doesn't
    # depend on its subnet) and would create a circular reference once the
    # subnet's vpc_id attribute references the VPC.
    connection_type: str
    attribute_map: dict[str, Any] = {}


class ParsedDiagram(BaseModel):
    source_format: DiagramFormat
    resources: list[ParsedResource] = []
    connections: list[ParsedConnection] = []
    total_resources: int = 0
    total_connections: int = 0
    resource_type_summary: dict[str, int] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Clarification (missing info detector output)
# ─────────────────────────────────────────────────────────────────────────────

class ClarificationField(BaseModel):
    field_key: str                 # e.g. "instance_type", "ami_id"
    resource_id: str
    resource_label: str
    question: str                  # Human-readable question shown in UI
    input_type: str = "text"       # "text" | "select" | "boolean" | "number"
    options: list[str] = []        # For select inputs
    default: Optional[str] = None
    required: bool = True


class ClarificationRequest(BaseModel):
    job_id: str
    fields: list[ClarificationField]


class ClarificationAnswer(BaseModel):
    field_key: str
    resource_id: str
    value: str


class ResourceCorrections(BaseModel):
    """
    User-driven corrections to the detected resource list, submitted
    alongside clarification answers (added 2026-07-24, her explicit
    request: image-based parsing can't guarantee perfect detection, so the
    Clarify screen must let the user review and fix what was actually
    found before HCL generation runs, not just fill in missing values for
    resources that were already assumed correct).

    Distinct from the pre-existing "reclassify_<id>" clarification field
    (which only ever triggers below `confidence_threshold`): image-parsed
    resources routinely come back at a flat high confidence (e.g. 0.95)
    even when the label is a garbled OCR merge or an outright phantom
    duplicate, so that safety net alone doesn't catch what review needs to
    catch. This lets the user act on ANY resource, regardless of its
    reported confidence.
    """
    deleted_ids: list[str] = []            # resource ids to drop entirely
    relabeled: dict[str, str] = {}          # resource_id -> corrected label
    retyped: dict[str, str] = {}            # resource_id -> corrected aws_resource_type


class ClarificationResponse(BaseModel):
    job_id: str
    answers: list[ClarificationAnswer]
    resource_corrections: Optional[ResourceCorrections] = None


# ─────────────────────────────────────────────────────────────────────────────
# Terraform plan (planner output)
# ─────────────────────────────────────────────────────────────────────────────

class TerraformModule(BaseModel):
    name: str                      # module folder name
    source_resources: list[str]    # resource IDs included
    description: str
    files: dict[str, str] = {}     # filename → HCL content


class TerraformPlan(BaseModel):
    modules: list[TerraformModule] = []
    root_module_files: dict[str, str] = {}   # root main.tf, versions.tf, etc.
    estimated_monthly_cost_usd: Optional[float] = None
    resource_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Validation results (sandbox output)
# ─────────────────────────────────────────────────────────────────────────────

class ValidationCheck(BaseModel):
    name: str
    status: ValidationStatus
    tool: str                      # "terraform_validate" | "tflint" | "checkov"
    message: str = ""
    severity: str = "info"         # "error" | "warning" | "info"
    resource_id: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None


class ValidationResult(BaseModel):
    overall_status: ValidationStatus
    checks: list[ValidationCheck] = []
    terraform_plan_output: str = ""
    errors: list[str] = []
    warnings: list[str] = []
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0


class BlockedVariable(BaseModel):
    """
    One unresolved catalog placeholder value (e.g. a no-VPC diagram's fake
    "vpc-00000000000000000") standing between a job and a real
    `terraform apply` — see apply_runner.py's preflight check. `id` is
    "<file>::<variable_name>" (not just variable_name, since the same name
    can legitimately appear in more than one module, e.g. `cidr_block` in
    both a VPC and a subnet) and is exactly the key the UI must echo back in
    POST /apply/resolve's `overrides` dict for this value to land on the
    right variable.
    """
    id: str
    file: str
    variable_name: str
    current_value: str
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Job — the main pipeline state object (stored in Redis)
# ─────────────────────────────────────────────────────────────────────────────

class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Upload metadata
    original_filename: str = ""
    diagram_format: DiagramFormat = DiagramFormat.UNKNOWN
    file_path: str = ""           # local path or S3 key

    # Pipeline stage outputs
    parsed_diagram: Optional[ParsedDiagram] = None
    clarification_request: Optional[ClarificationRequest] = None
    clarification_answers: list[ClarificationAnswer] = []
    terraform_plan: Optional[TerraformPlan] = None
    validation_result: Optional[ValidationResult] = None

    # Optional, user-supplied existing terraform.tfstate for THIS job — two
    # distinct uses depending on WHEN it's attached:
    #   1. Attached at /upload time (state_file param): pipeline_worker.py's
    #      run_pipeline() runs state_reconciler.reconcile_from_state() on it
    #      right after parsing, BEFORE planning — matched resources'
    #      properties get overwritten with real values out of state instead
    #      of catalog defaults/placeholders. Added 2026-07-29, her explicit
    #      follow-up: the original (2026-07-08) design only ever fed a plan
    #      DIFF, never actually changed what the generated modules contain.
    #   2. Attached later via POST /jobs/{job_id}/upload-state (on an
    #      already-DONE job): tf_validator.py seeds the sandbox's ephemeral
    #      working directory with it before running `terraform plan`, so
    #      the plan reflects real drift against that snapshot — this path
    #      is UNCHANGED, still a diff only, never touches terraform_plan.
    # Storage-backend path (local disk or S3, storage.py abstracts which) —
    # same convention as file_path. Never pushed to GitHub (her explicit
    # call, 2026-07-08: state can carry sensitive resource attributes that
    # shouldn't enter git history) — only ever bundled into the local
    # download ZIP, see packager.py.
    state_file_path: Optional[str] = None

    # Optional, user-supplied vars.yaml — pre-answers to the clarification
    # questions this job would otherwise ask, keyed resource+field (not raw
    # Terraform variable names, since those are only settled after planning —
    # see generate_vars_yaml()'s docstring). Structure:
    #   {"resources": {"<aws_resource_type>.<logical_name>": {"<field>": "<value>", ...}},
    #    "globals": {"aws_region": "...", "environment": "...", "project_name": "..."}}
    # None means "building from scratch" (her explicit distinction,
    # 2026-07-08): every catalog-default-covered field gets asked about
    # regardless of whether its current value looks like a placeholder —
    # see missing_info_detector.py's `ask_all` behavior. When present,
    # covered (resource, field) pairs are silently pre-filled instead of
    # asked, and only the gaps go through clarification.
    input_vars: Optional[dict] = None

    # The vars.yaml this job actually ended up with — either her uploaded
    # one gap-filled with newly-asked answers, or, if she started from
    # scratch, a brand-new one built from every answer collected. Raw YAML
    # text (not the parsed dict — this is a generated ARTIFACT, meant to be
    # bundled into the ZIP / pushed to GitHub / re-uploaded next time as
    # input_vars for the next run against an updated diagram). Populated by
    # missing_info_detector.generate_vars_yaml() once clarification answers
    # are finalized. Unlike terraform.tfstate, this holds config values
    # (sizes, engines, CIDRs) not secrets, so — her call, 2026-07-08 — it DOES
    # get pushed to GitHub alongside the diagram, not local-ZIP-only.
    generated_vars_yaml: Optional[str] = None

    # Final output
    zip_path: Optional[str] = None
    readme_content: str = ""

    # Error tracking
    error_message: str = ""
    stage_logs: list[str] = []

    # ── "Apply to Sandbox" — 2026-07-24, her explicit request to complete
    # the pipeline through a real `terraform apply` against her own AWS
    # sandbox account. Runs from a persistent working directory (NOT the
    # ephemeral tempdir tf_validator.py/packager.py each use and discard —
    # this one has to survive across separate plan -> confirm -> apply ->
    # destroy requests, and hold the real terraform.tfstate in between), see
    # apply_runner.py for the directory layout convention (keyed off
    # job.job_id under local_output_dir, same as everything else).
    apply_status: ApplyStatus = ApplyStatus.NOT_STARTED
    apply_workdir: Optional[str] = None
    apply_log: list[str] = []
    apply_plan_output: str = ""
    # One-time token returned by POST /apply/plan, required (and consumed —
    # single use) by POST /apply/confirm. Exists so a client can never
    # trigger a real `terraform apply` without first having actually seen
    # the plan output her review is meant to be gated on — no "confirm"
    # without a matching, still-fresh "plan" immediately before it. Expires
    # after 15 minutes so a stale browser tab can't blind-apply an outdated
    # plan much later.
    apply_confirm_token: Optional[str] = None
    apply_confirm_token_expires_at: Optional[datetime] = None
    # Wall-clock deadline for the auto-destroy safety net (her explicit
    # call, 2026-07-24: default 2 hours after a successful apply) so a
    # forgotten sandbox resource can't rack up cost unnoticed overnight
    # before a presentation. Reconciled on backend startup too (see
    # main.py's lifespan hook) in case the process restarts mid-window.
    apply_destroy_at: Optional[datetime] = None
    apply_error: str = ""
    # Real fix, 2026-07-27: her explicit correction — a blocked placeholder
    # value (e.g. the fake "vpc-00000000000000000" a no-VPC diagram falls
    # back to) must be resolved by the PERSON USING THE UI, not by someone
    # SSH'd into the backend hand-editing a terraform.tfvars on disk. When
    # plan_apply() finds unresolved placeholders it populates
    # apply_blocked_variables with exactly what the UI needs to render an
    # input per variable; her submitted answers land in
    # apply_variable_overrides (keyed by the SAME "file::variable_name" id
    # apply_blocked_variables uses, since variable names can collide across
    # root/modules — e.g. a `cidr_block` in both the VPC and a subnet).
    # Persisted on the Job (not just held in a request) because plan_apply()
    # rewrites the whole apply_workdir from job.terraform_plan on every
    # re-run — without storing overrides here they'd be silently wiped the
    # next time she clicks "Run terraform plan".
    apply_blocked_variables: list["BlockedVariable"] = []
    apply_variable_overrides: dict[str, str] = {}

    # ── Drift detection — 2026-07-29, her explicit request ────────────────
    # A dedicated, read-only check separate from plan/confirm/apply: runs
    # `terraform plan -refresh-only -detailed-exitcode` against whichever
    # workdir/state plan_apply() already uses (same apply_workdir, same
    # backend — local, S3, or Terraform Cloud, whatever terraform_planner.py
    # generated into backend.tf), so it always compares against the exact
    # same state plan/apply would. Never proposes or applies config changes
    # itself — see apply_runner.check_drift()'s docstring for why
    # -refresh-only is the right primitive here instead of a normal plan.
    drift_status: DriftStatus = DriftStatus.UNKNOWN
    drift_output: str = ""
    # Resource addresses ("aws_instance.web_server") the refresh-only plan
    # flagged as changed outside Terraform — parsed from drift_output so the
    # UI can show a short list without asking her to read the full plan
    # text. Empty when drift_status is CLEAN/FAILED/UNKNOWN.
    drift_resources: list[str] = []
    drift_checked_at: Optional[datetime] = None

    def log_apply(self, msg: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.apply_log.append(f"[{ts}] {msg}")
        self.updated_at = datetime.utcnow()

    def log(self, msg: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.stage_logs.append(f"[{ts}] {msg}")
        self.updated_at = datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────
# API request/response shapes
# ─────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class StateUploadResponse(BaseModel):
    job_id: str
    message: str


class FileEditRequest(BaseModel):
    """Body for PUT /jobs/{job_id}/files/{file_key} — added 2026-07-29, her
    explicit request for an in-browser edit mode on the Review screen's
    generated-code preview. `content` replaces the WHOLE file verbatim (no
    diff/patch semantics) — same all-or-nothing granularity the preview
    endpoint already reads at, so there's no risk of a partial/corrupt merge
    on the backend."""
    content: str


class FileEditResponse(BaseModel):
    job_id: str
    file_key: str
    message: str


class GithubPushRequest(BaseModel):
    # Per-request GitHub Personal Access Token. Pushes into an EXISTING
    # repo (never creates one — see github_pusher.py's module docstring for
    # why), so this only needs Contents + Pull requests write access to
    # `repo_full_name`; a fine-grained PAT scoped to just that one repo is
    # enough. Never persisted server-side: used only for the duration of
    # the push call, then discarded.
    github_token: str
    # Required, "owner/repo" form, e.g. "yourname/infra-repo".
    repo_full_name: str


class GithubPushResponse(BaseModel):
    repo_url: str
    pr_url: str
    repo_full_name: str
    environment: str


class ApplyPlanResponse(BaseModel):
    job_id: str
    apply_status: ApplyStatus
    plan_output: str
    confirm_token: Optional[str] = None
    confirm_token_expires_at: Optional[datetime] = None
    # Structured — one entry per unresolved placeholder, for the UI to
    # render a real input per variable. `blocked_reason` is kept alongside
    # as a plain-text summary (log display, non-UI clients) but the UI
    # itself should drive off `blocked_variables`, not parse this string.
    blocked_reason: str = ""
    blocked_variables: list[BlockedVariable] = []


class ApplyConfirmRequest(BaseModel):
    confirm_token: str


class ApplyResolveRequest(BaseModel):
    # Keyed by BlockedVariable.id ("<file>::<variable_name>"), value is the
    # real value she typed in the UI for that field. Merged into
    # Job.apply_variable_overrides (not replaced — submitting one screen's
    # worth of fixes must not forget ones already resolved on an earlier
    # partial submission), then a fresh plan_apply() is run automatically.
    overrides: dict[str, str]


class DriftCheckResponse(BaseModel):
    job_id: str
    drift_status: DriftStatus
    drift_output: str
    drift_resources: list[str] = []
    checked_at: Optional[datetime] = None


class ApplyStatusResponse(BaseModel):
    job_id: str
    apply_status: ApplyStatus
    apply_log: list[str]
    apply_error: str = ""
    destroy_at: Optional[datetime] = None
    destroy_in_seconds: Optional[int] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_percent: int
    current_stage: str
    stage_logs: list[str]
    parsed_diagram: Optional[ParsedDiagram] = None
    clarification_request: Optional[ClarificationRequest] = None
    validation_result: Optional[ValidationResult] = None
    zip_ready: bool = False
    error_message: str = ""


# Progress map for UI progress bar
JOB_PROGRESS: dict[JobStatus, int] = {
    JobStatus.UPLOADED:      5,
    JobStatus.PARSING:       15,
    JobStatus.PARSED:        25,
    JobStatus.NEEDS_CLARIFY: 30,
    JobStatus.PLANNING:      45,
    JobStatus.GENERATING:    60,
    JobStatus.VALIDATING:    75,
    JobStatus.PACKAGING:     90,
    JobStatus.DONE:          100,
    JobStatus.FAILED:        0,
}

STAGE_LABELS: dict[JobStatus, str] = {
    JobStatus.UPLOADED:      "Diagram uploaded",
    JobStatus.PARSING:       "Parsing diagram...",
    JobStatus.PARSED:        "Diagram parsed",
    JobStatus.NEEDS_CLARIFY: "Waiting for clarification",
    JobStatus.PLANNING:      "Planning Terraform modules...",
    JobStatus.GENERATING:    "Generating Terraform code...",
    JobStatus.VALIDATING:    "Validating in sandbox...",
    JobStatus.PACKAGING:     "Packaging output...",
    JobStatus.DONE:          "Ready to download",
    JobStatus.FAILED:        "Pipeline failed",
}
