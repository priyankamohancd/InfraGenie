"""
Storage Service
----------------
Abstracts file I/O behind a simple interface.
Uses local filesystem in dev, S3 in prod (controlled by STORAGE_BACKEND env var).
"""
from __future__ import annotations
import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings

log = logging.getLogger(__name__)
_s = get_settings()


def _ensure_dirs():
    Path(_s.local_upload_dir).mkdir(parents=True, exist_ok=True)
    Path(_s.local_output_dir).mkdir(parents=True, exist_ok=True)


async def save_upload(job_id: str, filename: str, content: bytes) -> str:
    """Save uploaded file, return storage path."""
    _ensure_dirs()
    if _s.storage_backend == "s3":
        return await _s3_put(f"{_s.s3_prefix}/uploads/{job_id}/{filename}", content)
    dest = Path(_s.local_upload_dir) / job_id / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    log.info("Saved upload: %s", dest)
    return str(dest)


async def read_upload(path: str) -> bytes:
    """Read an uploaded file by its storage path."""
    if _s.storage_backend == "s3":
        return await _s3_get(path)
    return Path(path).read_bytes()


async def save_output(job_id: str, filename: str, content: bytes | str) -> str:
    """Save a generated output file, return storage path."""
    _ensure_dirs()
    data = content.encode() if isinstance(content, str) else content
    if _s.storage_backend == "s3":
        return await _s3_put(f"{_s.s3_prefix}/outputs/{job_id}/{filename}", data)
    dest = Path(_s.local_output_dir) / job_id / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return str(dest)


async def save_output_dir(job_id: str, source_dir: str) -> str:
    """Save an entire directory as a ZIP, return zip path."""
    _ensure_dirs()
    zip_base = Path(_s.local_output_dir) / job_id / "terraform_output"
    zip_path = shutil.make_archive(str(zip_base), "zip", source_dir)
    log.info("Created ZIP: %s", zip_path)
    return zip_path


async def read_output(path: str) -> bytes:
    if _s.storage_backend == "s3":
        return await _s3_get(path)
    return Path(path).read_bytes()


def get_output_path(job_id: str, filename: str) -> str:
    return str(Path(_s.local_output_dir) / job_id / filename)


def output_exists(path: str) -> bool:
    if _s.storage_backend == "s3":
        return False  # TODO: implement S3 head
    return Path(path).exists()


# ── S3 helpers (production path) ─────────────────────────────────────────────

async def _s3_put(key: str, content: bytes) -> str:
    import boto3
    s3 = boto3.client("s3",
        aws_access_key_id=_s.aws_access_key_id,
        aws_secret_access_key=_s.aws_secret_access_key,
        region_name=_s.aws_region,
    )
    s3.put_object(Bucket=_s.s3_bucket, Key=key, Body=content)
    return f"s3://{_s.s3_bucket}/{key}"


async def _s3_get(s3_url: str) -> bytes:
    import boto3
    # Parse s3://bucket/key
    parts = s3_url.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3 = boto3.client("s3",
        aws_access_key_id=_s.aws_access_key_id,
        aws_secret_access_key=_s.aws_secret_access_key,
        region_name=_s.aws_region,
    )
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()
