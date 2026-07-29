FROM python:3.12-slim

# ── Build context note ──────────────────────────────────────────────────────
# This Dockerfile expects the build CONTEXT to be the parent "thesis/"
# directory (the one containing both arch2tf-product/ and arch2terraform/ as
# siblings), not arch2tf-product/backend/ itself — arch2terraform is a
# separate repo this backend imports cross-repo via PYTHONPATH (see
# app/services/*.py's `sys.path.insert(...)` calls and every test file's
# REPO_ROOT convention), so its source has to be reachable at build time too.
# See ../../docker-compose.yml at the thesis/ root, which sets this context
# correctly — don't `docker build` this file directly from inside backend/.

WORKDIR /app

ARG TERRAFORM_VERSION=1.9.8
ARG TFLINT_VERSION=0.53.0

# System deps:
#  - libglib2.0-0/libsm6/libxext6/libxrender-dev/libgomp1: OpenCV (image parsing)
#  - tesseract-ocr: pytesseract (Stage 4 OCR text extraction in image_adapter.py)
#  - curl/unzip: fetching the terraform + tflint binaries below
#  - git: checkov needs it for some check sources
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    tesseract-ocr \
    curl unzip git \
    && rm -rf /var/lib/apt/lists/*

# terraform + tflint — real Go binaries, not pip-installable. Pinned exact
# versions (build args above) rather than "latest", so a rebuild months from
# now doesn't silently pick up a breaking major-version bump under everyone's
# feet. checkov, the third validation tool, IS pip-installable — see
# requirements.txt instead.
RUN ARCH="$(dpkg --print-architecture)" && \
    curl -fsSL -o /tmp/terraform.zip \
      "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${ARCH}.zip" && \
    unzip -o /tmp/terraform.zip -d /usr/local/bin terraform && \
    rm /tmp/terraform.zip && \
    curl -fsSL -o /tmp/tflint.zip \
      "https://github.com/terraform-linters/tflint/releases/download/v${TFLINT_VERSION}/tflint_linux_${ARCH}.zip" && \
    unzip -o /tmp/tflint.zip -d /usr/local/bin && \
    rm /tmp/tflint.zip && \
    chmod +x /usr/local/bin/terraform /usr/local/bin/tflint

COPY arch2tf-product/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend app code.
COPY arch2tf-product/backend/ .

# Cross-repo imports, placed directly under /app so a single PYTHONPATH=/app
# resolves everything: `from shared.schemas.models import ...` needs a
# "shared" package under /app, and `from arch2terraform...` / `import
# arch2terraform` needs an "arch2terraform" package under /app. Copying the
# actual package directories here (not the outer repo folders, which have
# their own nested src/ layout — see arch2terraform/pyproject.toml's
# `where = ["src"]`) means no extra PYTHONPATH entries are needed at all.
COPY arch2tf-product/shared/ /app/shared/
COPY arch2terraform/src/arch2terraform/ /app/arch2terraform/

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
