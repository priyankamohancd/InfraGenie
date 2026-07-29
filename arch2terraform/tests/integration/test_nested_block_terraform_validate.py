"""
Real `terraform validate` confirmation for the 11 resource types whose
required arguments include a nested HCL block (aws_autoscaling_group,
aws_eks_cluster, aws_batch_job_queue, aws_waf_web_acl,
aws_cloudfront_distribution, aws_dynamodb_table, aws_mq_broker,
aws_codepipeline, aws_codebuild_project, aws_glue_job,
aws_mwaa_environment).

Why this exists as its own test rather than relying on
test_pipeline_end_to_end.py / test_image_pipeline_end_to_end.py's existing
`test_generated_terraform_passes_real_validate`: neither of the two real
reference diagrams those tests exercise contains any of these 11 resource
types, so a passing run there does NOT actually validate this nested-block
HCL against a real provider schema — it only confirms no regression for the
resource types that already existed in those fixtures. python-hcl2 (used
elsewhere for sandbox-only syntax checks) confirms the output *parses*, but
parsing is not the same as passing the provider's real schema/cross-field
validation (see the ARN-format and aws_lb subnets/subnet_mapping bugs a real
`terraform validate` run caught previously that syntax-only checks missed).

None of these 11 types appear in hcl_generator.py's _CONTAINMENT_WIRING_RULES,
so they can be instantiated standalone (no VPC/subnet parent needed) without
tripping an unrelated "missing vpc_id/subnet_id" failure that would actually
be about containment wiring, not the nested-block feature this test targets.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from arch2terraform.classifier.catalog import CATALOG
from arch2terraform.generator.hcl_generator import (
    generate_main_tf,
    generate_outputs_tf,
    generate_provider_tf,
    generate_variables_tf,
)
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph

_NESTED_BLOCK_TYPES = {
    "aws_autoscaling_group",
    "aws_eks_cluster",
    "aws_batch_job_queue",
    "aws_waf_web_acl",
    "aws_cloudfront_distribution",
    "aws_dynamodb_table",
    "aws_mq_broker",
    "aws_codepipeline",
    "aws_codebuild_project",
    "aws_glue_job",
    "aws_mwaa_environment",
}


@pytest.fixture
def output_dir():
    d = tempfile.mkdtemp(prefix="arch2tf_nested_block_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _build_graph() -> ResourceGraph:
    by_type = {d.terraform_type: d for d in CATALOG}
    resources = []
    for i, terraform_type in enumerate(sorted(_NESTED_BLOCK_TYPES)):
        defn = by_type[terraform_type]
        resources.append(ClassifiedResource(
            node_id=f"n{i}",
            resource_type=defn.terraform_type,
            terraform_name=f"example_{i}",
            display_label=defn.terraform_type,
            confidence=1.0,
            attributes=dict(defn.default_attributes),
            nested_blocks=defn.nested_blocks,
            is_container=defn.is_container,
        ))
    return ResourceGraph(resources=resources, relationships=[])


def test_all_eleven_nested_block_types_are_exercised():
    """Guard against this test silently covering fewer types than intended if
    the catalog set drifts (e.g. a type gets removed from CATALOG)."""
    by_type = {d.terraform_type for d in CATALOG}
    missing = _NESTED_BLOCK_TYPES - by_type
    assert not missing, f"types no longer in catalog: {missing}"


def test_nested_block_output_has_no_unresolved_placeholders_syntax(output_dir):
    graph = _build_graph()
    main_tf = generate_main_tf(graph)
    assert "{{" not in main_tf  # no leftover template syntax


@pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform binary not installed in this environment")
def test_nested_block_terraform_passes_real_validate(output_dir):
    """The gold-standard check: real `terraform init` + `terraform validate`
    against generated HCL for all 11 nested-block resource types together."""
    graph = _build_graph()
    files = {
        "provider.tf": generate_provider_tf(),
        "variables.tf": generate_variables_tf(),
        "main.tf": generate_main_tf(graph),
        "outputs.tf": generate_outputs_tf(graph),
    }
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    init = subprocess.run(
        ["terraform", "init", "-backend=false", "-input=false"],
        cwd=output_dir, capture_output=True, text=True, timeout=120,
    )
    assert init.returncode == 0, f"terraform init failed:\n{init.stdout}\n{init.stderr}"

    validate = subprocess.run(
        ["terraform", "validate"],
        cwd=output_dir, capture_output=True, text=True, timeout=60,
    )
    assert validate.returncode == 0, f"terraform validate failed:\n{validate.stdout}\n{validate.stderr}"
