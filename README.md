# arch2terraform — Phases 1 & 3

Converts architecture diagrams — draw.io, Lucidchart CSV export, Excalidraw,
and raster PNG/JPG (AWS-icon-style reference diagrams) — into validated
Terraform modules. The diagram is the single source of truth — no AI
subscription is required for any stage; classification is deterministic
(icon/label matching against a 57+ resource catalog).

PNG/JPG diagrams go through a 5-stage detection cascade
(`src/arch2terraform/adapters/image/`), fully wired into `ImageAdapter`:

1. **Layout detection** (`layout_detector.py`) — classical OpenCV colour/shape
   matching finds AWS Cloud / VPC / Availability Zone / Subnet / Security
   Group boundary boxes and their containment hierarchy.
2. **Icon candidate detection + perceptual hash** (`icon_detector.py`,
   `hash_matcher.py`) — locates service-icon blobs and identifies them
   against a pre-built phash table (`data/reference_hashes.pkl`, 316 entries).
3. **NCC template matching** (`stage3_matcher.py`) — fallback for icons phash
   can't confidently place. Chosen over YOLOv8: no labelled dataset or GPU
   needed, fully offline, and empirically ≥0.85 NCC for true matches vs
   <0.35 for impostors on the AWS icon set. Needs the real aws-icons asset
   pack on disk (see `ARCH2TERRAFORM_ICONS_DIR` below) — without it, Stage 3
   is silently skipped and unmatched icons fall through to OCR/UNKNOWN.
4. **OCR** (`ocr_extractor.py`) — Tesseract, run unconditionally on every
   detected region, since labels (CIDR blocks, custom names) are unique data
   no icon classifier produces. Filters both low-confidence and
   too-short-vertically blocks — the latter specifically to reject
   Tesseract's tendency to hallucinate short "words" out of dashed
   AZ/Subnet border lines.
5. **Edge detection** (`edge_detector.py`) — Hough line + arrowhead
   connected-component detection to recover directed connections between
   services.

Set `ARCH2TERRAFORM_ICONS_DIR` to the root of the AWS Architecture Icons
asset pack (the directory containing `Architecture-Service-Icons_*` /
`Architecture-Group-Icons_*`) before running the pipeline against real
diagrams, so Stage 3 can actually run:

```bash
export ARCH2TERRAFORM_ICONS_DIR=~/work/thesis/aws-icons
```

## Setup (Python 3.12 via Homebrew)

```bash
brew install python@3.12
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,phase3]"
```

## Run the test suite

```bash
.venv/bin/python -m pytest -v
```

Two integration tests (`test_generated_terraform_passes_real_validate`, one
per drawio and image fixtures) run a real `terraform init && terraform
validate` against generated output. They're automatically skipped if the
`terraform` binary isn't on your PATH — install Terraform to get real
sandbox validation of generated HCL, not just structural checks.

`checkov` (`pip install checkov`) is a useful complementary check that
doesn't need the `terraform` binary at all — it parses the HCL directly and
runs security/best-practice policy checks:

```bash
checkov -d /path/to/generated/output
```

Expect it to report failures on things like missing encryption-at-rest,
missing monitoring, and no IAM role attached to EC2 — these are intentional
per this project's "minimal viable, not production-hardened" design
philosophy (see Known limitations below), not bugs. What it's useful for
catching is any resource checkov's schema-aware engine fails to recognize
or parse at all, which would indicate an actual generator bug.

## Run the pipeline directly

```python
from arch2terraform.pipeline import run_pipeline

result = run_pipeline("path/to/diagram.drawio", "output_dir")
print(result.files_written)   # provider.tf, variables.tf, main.tf, outputs.tf, README.md
print(result.graph.resources) # classified AWS resources
```

## Project layout

```
src/arch2terraform/
  schemas/      canonical diagram + resource intermediate representations
  adapters/     draw.io, Lucidchart (CSV), Excalidraw, image (5-stage cascade)
  classifier/   57+ AWS resource catalog + icon/label matching logic
  resolver/     relationship resolution (containment, network, IAM, routing)
  generator/    HCL generation — 5 files per run (provider/variables/main/outputs/README)
  pipeline.py   top-level orchestration: file in, Terraform module out

tests/
  unit/         one test file per module
  integration/  end-to-end pipeline tests (drawio + image fixtures) + real terraform validate
  fixtures/     sample diagrams for each supported format
```

## Known limitations (by design, flagged in generated README.md per run)

- Resource attribute defaults are minimal viable values (e.g. `t3.micro`,
  `db.t3.micro`, `10.0.0.0/16`) — meant to produce *valid*, not *production-sized*,
  infrastructure. Review before applying. A handful of required-but-inherently-
  environment-specific arguments (`ami`, Lambda's `role`/`function_name`/
  `filename`, Route 53's `name`) get an obviously-fake placeholder in the same
  spirit, so `terraform validate` passes but `apply` still forces you to fill
  in the real value.
- Containment is only wired into a real Terraform reference (e.g.
  `subnet_id = aws_subnet.x.id`) when the diagram's actual nesting matches what
  that attribute expects. If a diagram shows an EC2 instance sitting directly
  inside a VPC with no subnet drawn, the generator deliberately does NOT wire
  `subnet_id` to the VPC's id (that would be wrong) — it flags this in a comment
  for manual wiring instead.
- Low-confidence classifications (label-matched rather than icon-matched) are
  flagged both in `main.tf` comments and in the generated `README.md`.
- A real `terraform validate` run (not just the syntax-level checks a sandbox
  without network access to HashiCorp's servers can do) caught two more subtle
  issues past the initial audit: (1) AWS-provider fields named `role`/`*_arn`
  are format-validated client-side as real ARNs — a plain descriptive
  placeholder like `"REPLACE_WITH_IAM_ROLE_ARN"` fails validate outright with
  "invalid ARN: arn: invalid prefix", so every such placeholder is now shaped
  like `arn:aws:iam::000000000000:role/...`; (2) `aws_lb` requires one of
  `subnets`/`subnet_mapping` — a cross-field constraint, not a simple
  per-argument Required flag, which the per-argument audit missed. Both are
  now covered by a placeholder default and locked in by
  `test_arn_typed_placeholders_are_actually_shaped_like_arns` in
  `tests/unit/test_catalog.py`. This is exactly the class of bug that only a
  real `terraform validate` run surfaces — worth re-running the test suite
  with the real binary after any future catalog change.
- All 59 catalog entries have now been individually audited against the AWS
  provider's actually-required arguments (`tests/unit/test_catalog.py` locks
  this in — every entry is either in the audited required-args map or
  explicitly listed as needing nested-block support, so an unaudited entry
  can't be silently added again). Two entries had outright wrong Terraform
  resource type names (`aws_eventbridge_rule` → `aws_cloudwatch_event_rule`,
  `aws_step_function_state_machine` → `aws_sfn_state_machine`) that would
  have failed `terraform init`/parse immediately — worse than any missing
  argument, since the resource type itself didn't exist in the provider.
- 10 resource types have a provider-required argument that is itself a
  *nested HCL block* rather than a flat attribute (`aws_autoscaling_group`'s
  `launch_template`, `aws_eks_cluster`'s `vpc_config`, `aws_batch_job_queue`'s
  `compute_environment_order`, `aws_waf_web_acl`'s `default_action`,
  `aws_cloudfront_distribution`'s `origin`/`default_cache_behavior`/
  `restrictions`/`viewer_certificate`, `aws_dynamodb_table`'s `attribute`,
  `aws_mq_broker`'s `user`, `aws_codepipeline`'s `artifact_store`/`stage`,
  `aws_codebuild_project`'s `artifacts`/`environment`/`source`,
  `aws_glue_job`'s `command`). The generator (`hcl_format.py`) only emits
  flat `key = value` attribute lines, not nested blocks, so these resources
  are generated with every flat argument filled in but are still incomplete
  — real `terraform validate` will flag the missing block. Fixing this
  properly means teaching the generator to render nested blocks, which is a
  real feature addition, not a catalog tweak.
- Tesseract still occasionally hallucinates plausible-looking short words
  (not just obvious noise) out of dense dashed-line intersections; the
  height filter in `ocr_extractor.py` catches the common case (thin 2-4px
  strips) but not every case. Review any container/resource with an
  unexpectedly specific name.

## Next: Phase 2

FastAPI backend + async pipeline + AWS sandbox validator (`terraform validate`
+ `tflint` + `checkov` against real sandbox credentials) + ZIP packager, to be
built on top of this Phase 1 + Phase 3 package.
