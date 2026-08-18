# InfraGenie

Turn an AWS architecture diagram into a validated, modular Terraform package — and keep the diagram as the source of truth after the code exists, not just before it.

InfraGenie reads draw.io, Excalidraw, and Lucidchart (CSV export) files natively, and raster images (PNG/JPG) via either a deterministic computer-vision cascade or a Vision Language Model. Whatever the input, it normalises the diagram into one architecture graph, matches resources against an audited catalogue of AWS Terraform provider types, infers the security groups and IAM policies implied by the connections, asks the user only for values that genuinely need a human decision, and validates the generated Terraform (`terraform validate`, TFLint, Checkov — and, on request, a real `plan`/`apply` in a sandbox) before calling it done.

This is the codebase behind Priyanka Mohan's thesis project — referred to as **InfraGenie** in product-facing docs and **Terraform Accelerators** in academic ones; same system, two audiences.

## Repository layout

This is a monorepo containing two codebases that are developed and tested independently but deployed together:

```
arch2terraform/    Core library — diagram parsing, resource classification,
                    relationship resolution, and HCL generation. No web
                    framework, no job queue — a pure Python package with its
                    own test suite. See arch2terraform/README.md for the
                    5-stage image-detection cascade, setup, and test
                    instructions.

arch2tf-product/    The product layer built on top of arch2terraform: a
                    FastAPI backend (async job pipeline, clarification
                    engine, security-rule generation, Terraform planning,
                    sandbox validation, GitHub PR integration) and a browser
                    frontend. See arch2tf-product/DEPLOYMENT.md for the
                    Docker Compose / EC2 deployment runbook.

docker-compose.yml  Stands up the full stack (nginx + FastAPI backend +
                    Redis) from this root, since the backend's Docker build
                    needs both sibling repos as its build context.

docs/               Supporting assets — currently a sample reference
                    architecture diagram used to exercise the pipeline.
```

## Quick start

```bash
cp arch2tf-product/backend/.env.example arch2tf-product/backend/.env
# edit .env — at minimum set ANTHROPIC_API_KEY if you want the
# Vision Language Model path; the classical CV path needs no API key at all
docker compose up -d --build
```

Then open `http://localhost/` and upload a diagram — or try the sample one at [`docs/sample-diagrams/ecommerce-reference-architecture.png`](docs/sample-diagrams/ecommerce-reference-architecture.png).

For local development without Docker, or to run either test suite directly, see the per-package instructions in `arch2terraform/README.md` and `arch2tf-product/DEPLOYMENT.md`.

## Reproducing the thesis results

The code state referenced in the thesis is tagged [`thesis-final-eval`](https://github.com/priyankamohancd/InfraGenie/tree/thesis-final-eval). Check it out directly if you want to reproduce a specific reported result rather than running against the latest `main`:

```bash
git checkout thesis-final-eval
```

Both test suites are run independently:

```bash
cd arch2terraform && python -m pytest -v
cd arch2tf-product/backend && python -m pytest -v
```

A handful of tests are conditionally skipped when a local `terraform` binary isn't on `PATH` (they run a real `terraform init`/`validate`, not just structural checks) — install Terraform to get the full run.

## Key environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | `arch2terraform` (image adapter) | Enables the Vision Language Model diagram-understanding path |
| `ARCH2TERRAFORM_VISION_LLM_MODEL` | `arch2terraform` | Overrides the default VLM model string |
| `ARCH2TERRAFORM_ICONS_DIR` | `arch2terraform` | Path to the AWS Architecture Icons asset pack, enables Stage 3 (NCC) icon matching |
| `TF_STATE_BUCKET` / `TF_BACKEND_TYPE` | `arch2tf-product` backend | Configures the Terraform state backend (S3 or Terraform Cloud) |

Full lists with defaults are in each package's `.env.example`.

## Status

Both codebases have their own automated test suites (unit + integration), and generated Terraform output is checked with `terraform validate`, TFLint, and Checkov before being considered ready for review. See each package's README for current test counts and known limitations — these are also reported transparently in the thesis rather than only here.
