"""
Parser Service
---------------
Entry point for turning an uploaded diagram into a ParsedDiagram.

As of 2026-07-08 this routes every supported format (.drawio/.xml,
.excalidraw, lucidchart .svg/.csv, and image formats) through
arch2terraform_bridge.py, which runs arch2terraform's full pipeline
(parse -> classify -> resolve relationships) rather than a per-format,
per-service reimplementation. Previously only image uploads got real
parsing (via arch2terraform's ImageAdapter directly, with classification
still done locally in icon_resource_map.py); .drawio/.excalidraw hit a bare
stub. See arch2terraform_bridge.py's module docstring for the full
rationale. icon_resource_map.py is no longer used by this module — it's
superseded by arch2terraform's audited catalog, but kept in the repo rather
than deleted in case any of its reference data is still useful.

Falls back to `_stub_parse()` only when arch2terraform's registry can't
find an adapter for the file at all, or the pipeline raises unexpectedly —
this keeps the API usable for manual/dev testing even without a real
diagram on hand.
"""
from __future__ import annotations
import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── Path setup ────────────────────────────────────────────────────────────────
_ARCH2TF_SRC = Path(__file__).resolve().parents[5] / "arch2terraform" / "src"
if str(_ARCH2TF_SRC) not in sys.path:
    sys.path.insert(0, str(_ARCH2TF_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shared.schemas.models import DiagramFormat, ParsedConnection, ParsedDiagram, ParsedResource

from app.services.parser.arch2terraform_bridge import run_arch2terraform_pipeline

try:
    from arch2terraform.adapters.registry import UnsupportedFormatError
except ImportError:
    UnsupportedFormatError = ValueError  # arch2terraform not importable at all — treat everything as unsupported


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def parse_diagram(file_path: str, original_filename: str) -> ParsedDiagram:
    """
    Run arch2terraform's full pipeline on the uploaded diagram and return a
    ParsedDiagram. Falls back to a minimal stub only if arch2terraform has no
    adapter for this file type, or the pipeline itself raises unexpectedly.
    """
    log.info("Running arch2terraform pipeline on '%s'", original_filename)
    try:
        loop = asyncio.get_event_loop()
        # arch2terraform's pipeline (esp. the image cascade) is CPU-bound;
        # run in a thread pool so we don't block the FastAPI event loop.
        parsed = await loop.run_in_executor(
            None, run_arch2terraform_pipeline, file_path, original_filename
        )
        log.info(
            "arch2terraform parse complete: %d resources, %d connections from '%s'",
            parsed.total_resources, parsed.total_connections, original_filename,
        )
        return parsed
    except UnsupportedFormatError as exc:
        log.warning("No arch2terraform adapter for '%s' (%s) — using stub", original_filename, exc)
        return _stub_parse(original_filename)
    except Exception:
        # Real bug found 2026-07-28: this used to swallow EVERY exception
        # (parse errors, OCR crashes, classifier bugs, anything) and fall
        # back to the same hardcoded 3-resource stub (Main VPC/Web
        # Server/PostgreSQL DB) — so a real diagram that hit an unexpected
        # error looked EXACTLY like a successfully-parsed generic demo
        # diagram, with no indication anything had gone wrong. A user
        # uploading completely different diagrams and getting the identical
        # stub output every time is how this surfaced — there was no way to
        # tell "the pipeline is silently failing" from "this diagram really
        # does only have a VPC/EC2/RDS". Re-raising instead lets
        # pipeline_worker.py's existing outer try/except correctly fail the
        # job with the real error message surfaced in the UI, which is far
        # more useful than a confident-looking wrong answer. The
        # UnsupportedFormatError branch above is intentionally kept as a
        # stub — that's a genuinely expected case (unrecognized file type),
        # not a bug being masked.
        log.exception("arch2terraform pipeline failed on '%s'", original_filename)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Stub fallback
# ─────────────────────────────────────────────────────────────────────────────

def _stub_parse(filename: str) -> ParsedDiagram:
    """Minimal stub for dev/testing when arch2terraform can't handle the file."""
    log.info("Using stub parser for '%s'", filename)
    return ParsedDiagram(
        source_format=DiagramFormat.UNKNOWN,
        resources=[
            ParsedResource(
                id="stub-vpc", aws_resource_type="aws_vpc",
                logical_name="main_vpc", label="Main VPC",
                properties={"cidr_block": "10.0.0.0/16"},
                confidence=0.9,
            ),
            ParsedResource(
                id="stub-ec2", aws_resource_type="aws_instance",
                logical_name="web_server", label="Web Server",
                properties={"instance_type": "t3.micro", "ami": "ami-00000000000000000"},
                confidence=0.8,
            ),
            ParsedResource(
                id="stub-rds", aws_resource_type="aws_db_instance",
                logical_name="postgres_db", label="PostgreSQL DB",
                properties={"engine": "postgres", "instance_class": "db.t3.micro"},
                confidence=0.85,
            ),
        ],
        connections=[
            ParsedConnection(
                source_id="stub-vpc", target_id="stub-ec2",
                connection_type="containment",
            ),
        ],
        total_resources=3,
        total_connections=1,
        resource_type_summary={"aws_vpc": 1, "aws_instance": 1, "aws_db_instance": 1},
    )
