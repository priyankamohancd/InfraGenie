"""
arch2terraform Bridge — unit tests
-------------------------------------
Covers the diagram-custom-data ("tags") pass-through added 2026-07-08: a
shape tagged in draw.io ("Edit Data") or Excalidraw (customData) should have
those key/value pairs land on Phase 2's ParsedResource.tags after going
through the full arch2terraform_bridge.py pipeline (parse -> classify ->
resolve -> map to Phase 2 contracts). ParsedResource.tags existed on the
model before this change but nothing populated it — this is the first real
test asserting it actually gets set.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "arch2terraform" / "src"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product"))
sys.path.insert(0, str(REPO_ROOT / "arch2tf-product" / "backend"))

from app.services.parser.arch2terraform_bridge import run_arch2terraform_pipeline

_TAGGED_DRAWIO_XML = """
<mxGraphModel>
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <object label="Web Server" tier="prod" pii="false" id="ec2_1">
      <mxCell style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.ec2;" vertex="1" parent="1">
        <mxGeometry x="40" y="40" width="60" height="60" as="geometry" />
      </mxCell>
    </object>
    <mxCell id="s3_1" value="Static Assets Bucket" style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.s3;" vertex="1" parent="1">
      <mxGeometry x="200" y="40" width="60" height="60" as="geometry" />
    </mxCell>
  </root>
</mxGraphModel>
"""


def test_drawio_custom_data_reaches_parsed_resource_tags(tmp_path):
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_DRAWIO_XML)

    parsed = run_arch2terraform_pipeline(str(f), "tagged.drawio")

    web_server = next(r for r in parsed.resources if r.id == "ec2_1")
    assert web_server.aws_resource_type == "aws_instance"
    assert web_server.tags == {"tier": "prod", "pii": "false"}


def test_untagged_resource_has_empty_tags(tmp_path):
    f = tmp_path / "tagged.drawio"
    f.write_text(_TAGGED_DRAWIO_XML)

    parsed = run_arch2terraform_pipeline(str(f), "tagged.drawio")

    bucket = next(r for r in parsed.resources if r.id == "s3_1")
    assert bucket.tags == {}


def test_real_fixture_untagged_resources_all_have_empty_tags():
    """The common case, run against the real fixture used throughout this
    test suite — no regression for diagrams that never use custom data."""
    fixture = REPO_ROOT / "arch2terraform/tests/fixtures/drawio/sample_architecture.drawio"
    parsed = run_arch2terraform_pipeline(str(fixture), fixture.name)
    assert len(parsed.resources) > 0
    for r in parsed.resources:
        assert r.tags == {}
