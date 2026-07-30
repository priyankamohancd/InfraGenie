"""
State Reconciler
------------------
Reads real attribute values out of an uploaded terraform.tfstate and uses
them to correct the parsed diagram's resource properties BEFORE planning —
added 2026-07-29, her explicit follow-up request after the state-upload
path turned out to only ever feed a `terraform plan` DIFF (see
tf_validator.py / pipeline_worker.py's revalidate_with_state()), never the
generated module's actual values. That's still exactly what it does for a
job that's already reached DONE and gets a state file attached afterward
(re-validation only) — this module is what makes attaching a state file at
UPLOAD time (see /upload's new `state_file` param) actually change what
gets generated, not just what gets diffed against.

Design, deliberately conservative:
  - Matches state resources to parsed diagram resources by
    (aws_resource_type, logical_name) — the SAME "<type>.<logical_name>"
    convention missing_info_detector.py's _resource_key() already uses for
    vars.yaml, so a diagram that hasn't been relabeled since the state was
    last applied matches cleanly with zero new naming scheme to learn.
  - Only OVERWRITES property keys the diagram's resource ALREADY has a
    value for (i.e. keys the classifier's catalog already populated a
    default/placeholder for) — never introduces new, previously-unknown
    property keys from the state file's much larger raw attribute set.
    This is what keeps this safe: a real `aws_instance` state entry has
    dozens of computed/read-only attributes (arn, id, primary_network_
    interface_id, ...) that would be nonsensical as literal HCL input
    args; restricting to keys the catalog already tracks as real config
    knobs (ami, instance_type, cidr_block, engine, ...) sidesteps that
    entirely without needing a second, provider-schema-aware allowlist.
  - Single-instance resources only (instances[0]) — a resource created via
    count/for_each in the ORIGINAL apply has multiple state instances with
    no single obvious diagram-node to reconcile each one against; that's a
    real but out-of-scope extension (this tool's parsed-diagram model is
    already one-node-per-resource, not one-node-per-N-instances).
  - Never raises on a malformed/non-JSON/non-state upload — logs a warning
    and returns an empty summary instead, so a bad upload degrades to
    "reconciliation skipped," never fails the whole pipeline run. The same
    graceful-degradation philosophy _read_existing_state_bytes() already
    uses for a missing/unreadable state file.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app._pathboot import ensure_paths
ensure_paths()
from shared.schemas.models import ParsedResource

log = logging.getLogger(__name__)


class StateParseError(ValueError):
    """Raised internally by _parse_state / _build_state_index — always
    caught by reconcile_from_state() itself, never propagates out of this
    module. Kept as a real exception type (not a bare ValueError) so a unit
    test can assert on it specifically without stringly-typed matching."""


def _parse_state(state_bytes: bytes) -> dict:
    try:
        data = json.loads(state_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise StateParseError(f"Uploaded file is not valid JSON: {e}") from e

    if not isinstance(data, dict) or "resources" not in data:
        raise StateParseError(
            "Uploaded file doesn't look like a Terraform state file "
            "(no top-level 'resources' key)"
        )
    return data


def _build_state_index(state_json: dict) -> dict[tuple[str, str], dict]:
    """{(resource_type, resource_name): real_attributes_dict} — skips data
    sources (mode != "managed", never something a diagram node represents)
    and any resource with zero instances (shouldn't happen in a real state
    file, but a hand-edited/corrupted one could have it)."""
    index: dict[tuple[str, str], dict] = {}
    for res in state_json.get("resources", []):
        if not isinstance(res, dict) or res.get("mode") != "managed":
            continue
        rtype = res.get("type")
        rname = res.get("name")
        instances = res.get("instances") or []
        if not rtype or not rname or not instances:
            continue
        attrs = instances[0].get("attributes") if isinstance(instances[0], dict) else None
        if isinstance(attrs, dict):
            index[(rtype, rname)] = attrs
    return index


def reconcile_from_state(resources: list[ParsedResource], state_bytes: bytes) -> list[str]:
    """
    Mutates each matched ParsedResource's `properties` in place, replacing
    catalog-default/placeholder values with the real value found in the
    uploaded state file for every property key the resource already has.
    Returns a human-readable summary line per resource actually changed
    (for job.log() — see pipeline_worker.py), empty list if nothing
    matched or the upload couldn't be parsed as state at all.
    """
    try:
        state_json = _parse_state(state_bytes)
    except StateParseError as e:
        log.warning("State reconciliation skipped: %s", e)
        return []

    state_index = _build_state_index(state_json)
    if not state_index:
        log.info("State reconciliation: uploaded state file has no managed resources — nothing to reconcile")
        return []

    summary: list[str] = []
    for resource in resources:
        key = (resource.aws_resource_type, resource.logical_name)
        real_attrs = state_index.get(key)
        if not real_attrs:
            continue

        changed_fields = []
        for field_key, current_value in list(resource.properties.items()):
            if field_key not in real_attrs:
                continue
            real_value = real_attrs[field_key]
            if real_value is None or real_value == current_value:
                continue
            resource.properties[field_key] = real_value
            changed_fields.append(field_key)

        if changed_fields:
            summary.append(
                f"{resource.aws_resource_type}.{resource.logical_name}: "
                f"{', '.join(sorted(changed_fields))}"
            )

    return summary
