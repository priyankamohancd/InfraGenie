"""
State Reconciler — unit tests
--------------------------------
Covers app/services/parser/state_reconciler.py, added 2026-07-29 per her
explicit follow-up request: the pre-existing state-upload path
(POST /jobs/{job_id}/upload-state) only ever fed a `terraform plan` DIFF,
never actually changed what the generated modules contain. This module is
what makes attaching a state file at /upload time (before planning starts)
actually reconcile the generated resource properties with real values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from shared.schemas.models import ParsedResource
from app.services.parser.state_reconciler import reconcile_from_state


def _state_bytes(resources: list[dict]) -> bytes:
    return json.dumps({"version": 4, "resources": resources}).encode()


def _managed_resource(rtype: str, rname: str, attributes: dict, module: str | None = None) -> dict:
    res = {"mode": "managed", "type": rtype, "name": rname, "instances": [{"attributes": attributes}]}
    if module:
        res["module"] = module
    return res


class TestReconcileFromState:
    def test_matched_resource_gets_real_values(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web Server", properties={"instance_type": "t3.micro", "ami": "ami-00000000000000000"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {
                "instance_type": "t3.large", "ami": "ami-0real0000000000", "id": "i-0abc123",
            }),
        ])

        summary = reconcile_from_state([r], state)

        assert r.properties["instance_type"] == "t3.large"
        assert r.properties["ami"] == "ami-0real0000000000"
        assert len(summary) == 1
        assert "aws_instance.web_server" in summary[0]

    def test_never_introduces_new_property_keys_not_already_tracked(self):
        """The real safety property: state carries dozens of computed
        attributes (arn, id, primary_network_interface_id, ...) that would
        be nonsensical as literal HCL input args — only keys the resource
        ALREADY has (from the catalog) are ever touched."""
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web Server", properties={"instance_type": "t3.micro"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {
                "instance_type": "t3.large", "id": "i-0abc123", "arn": "arn:aws:ec2:...",
            }),
        ])

        reconcile_from_state([r], state)

        assert r.properties == {"instance_type": "t3.large"}
        assert "id" not in r.properties
        assert "arn" not in r.properties

    def test_unmatched_resource_left_untouched(self):
        r = ParsedResource(
            id="db-1", aws_resource_type="aws_db_instance", logical_name="postgres_db",
            label="Postgres DB", properties={"engine": "postgres"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {"instance_type": "t3.large"}),
        ])

        summary = reconcile_from_state([r], state)

        assert r.properties == {"engine": "postgres"}
        assert summary == []

    def test_data_source_resources_are_ignored(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_ami", logical_name="ubuntu",
            label="Ubuntu AMI", properties={"name": "old-name"},
        )
        state_json = {
            "version": 4,
            "resources": [
                {"mode": "data", "type": "aws_ami", "name": "ubuntu",
                 "instances": [{"attributes": {"name": "real-name"}}]},
            ],
        }
        summary = reconcile_from_state([r], json.dumps(state_json).encode())

        assert r.properties == {"name": "old-name"}
        assert summary == []

    def test_identical_value_is_not_reported_as_changed(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web Server", properties={"instance_type": "t3.micro"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {"instance_type": "t3.micro"}),
        ])

        summary = reconcile_from_state([r], state)

        assert summary == []  # no actual change to report, even though matched

    def test_multiple_resources_each_reconciled_independently(self):
        r1 = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        r2 = ParsedResource(
            id="db-1", aws_resource_type="aws_db_instance", logical_name="postgres_db",
            label="DB", properties={"engine": "postgres", "instance_class": "db.t3.micro"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {"instance_type": "t3.xlarge"}),
            _managed_resource("aws_db_instance", "postgres_db", {"instance_class": "db.r5.large"}),
        ])

        summary = reconcile_from_state([r1, r2], state)

        assert r1.properties["instance_type"] == "t3.xlarge"
        assert r2.properties["instance_class"] == "db.r5.large"
        assert r2.properties["engine"] == "postgres"  # not in state attrs — untouched
        assert len(summary) == 2

    def test_malformed_json_degrades_gracefully_no_raise(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        summary = reconcile_from_state([r], b"not json at all")

        assert summary == []
        assert r.properties == {"instance_type": "t3.micro"}  # untouched

    def test_valid_json_but_not_a_state_file_degrades_gracefully(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        summary = reconcile_from_state([r], json.dumps({"hello": "world"}).encode())

        assert summary == []
        assert r.properties == {"instance_type": "t3.micro"}

    def test_empty_resources_list_in_state_degrades_gracefully(self):
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        summary = reconcile_from_state([r], _state_bytes([]))

        assert summary == []

    def test_multi_instance_resource_uses_first_instance_only(self):
        """count/for_each resources have multiple state instances with no
        single obvious diagram-node to reconcile each against — documented
        limitation, first instance is used as a best-effort default."""
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        state_json = {
            "version": 4,
            "resources": [
                {"mode": "managed", "type": "aws_instance", "name": "web_server",
                 "instances": [
                     {"attributes": {"instance_type": "t3.large"}},
                     {"attributes": {"instance_type": "t3.2xlarge"}},
                 ]},
            ],
        }
        reconcile_from_state([r], json.dumps(state_json).encode())

        assert r.properties["instance_type"] == "t3.large"

    def test_resource_with_module_path_still_matches_by_type_and_name(self):
        """Matching is by (type, name) alone, ignoring which module the
        resource sat in in the ORIGINAL state — same convention
        missing_info_detector.py's _resource_key() already uses for
        vars.yaml, so this needs no separate module-path bookkeeping."""
        r = ParsedResource(
            id="ec2-1", aws_resource_type="aws_instance", logical_name="web_server",
            label="Web", properties={"instance_type": "t3.micro"},
        )
        state = _state_bytes([
            _managed_resource("aws_instance", "web_server", {"instance_type": "t3.large"}, module="module.compute"),
        ])

        reconcile_from_state([r], state)

        assert r.properties["instance_type"] == "t3.large"
