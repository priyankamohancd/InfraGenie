from arch2terraform.classifier.catalog import CATALOG
from arch2terraform.classifier.classifier import classify_diagram
from arch2terraform.schemas.diagram import BoundingBox, DiagramNode, NodeShape, ParsedDiagram


def _node(node_id, label="", image_ref=None, shape=NodeShape.ICON, parent_id=None):
    return DiagramNode(
        id=node_id,
        raw_label=label,
        shape=shape,
        bbox=BoundingBox(0, 0, 10, 10),
        image_ref=image_ref,
        parent_id=parent_id,
        source_format="test",
    )


def test_catalog_has_at_least_45_resource_types():
    assert len(CATALOG) >= 45


def test_catalog_terraform_types_are_unique():
    types = [d.terraform_type for d in CATALOG]
    assert len(types) == len(set(types)), "Duplicate terraform_type entries in catalog"


def test_classify_by_icon_ref_high_confidence():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Web Server", image_ref="mxgraph.aws4.ec2")],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_instance"
    assert classified[0].confidence >= 0.9
    assert classified[0].needs_clarification == []


def test_classify_by_label_lower_confidence():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Lambda function", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_lambda_function"
    assert classified[0].confidence < 0.9
    assert "resource_type" in classified[0].needs_clarification


def test_unrecognized_node_goes_to_unclassified():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Mystery Box", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 0
    assert unclassified == ["n1"]


def test_container_shape_without_label_defaults_to_vpc():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="", image_ref=None, shape=NodeShape.CONTAINER)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_vpc"
    assert classified[0].is_container


def test_terraform_names_are_unique_and_sanitized():
    diagram = ParsedDiagram(
        nodes=[
            _node("n1", label="Web Server!", image_ref="mxgraph.aws4.ec2"),
            _node("n2", label="Web Server!", image_ref="mxgraph.aws4.ec2"),
        ],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    names = {c.terraform_name for c in classified}
    assert len(names) == 2
    for name in names:
        assert name.replace("_", "a").isalnum()  # only [a-z0-9_]


def test_rds_instance_not_misclassified_as_ec2():
    """Regression test: 'RDS Instance' contains the substring 'instance', which is
    EC2's icon key. The classifier must prefer the more specific 'rds' match over
    the generic 'instance' match, regardless of catalog list order."""
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Orders DB", image_ref="AWS19 / Database RDS Instance")],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, unclassified = classify_diagram(diagram)
    assert len(classified) == 1
    assert classified[0].resource_type == "aws_db_instance"


def test_longest_label_keyword_wins_over_shorter_generic_one():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="aurora cluster", image_ref=None, shape=NodeShape.RECTANGLE)],
        edges=[],
        source_format="test",
        source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert classified[0].resource_type == "aws_rds_cluster"


def test_diagram_node_tags_pass_through_to_classified_resource():
    """2026-07-08: DiagramNode.tags (custom data from draw.io Edit Data /
    Excalidraw customData) must survive classification unchanged — this is
    the pass-through link between adapter-level parsing and Phase 2's
    ParsedResource.tags, so classify_diagram() must never drop or transform
    it, just carry it forward."""
    node = DiagramNode(
        id="n1", raw_label="Web Server", shape=NodeShape.ICON,
        bbox=BoundingBox(0, 0, 10, 10), image_ref="mxgraph.aws4.ec2",
        source_format="test", tags={"tier": "prod", "pii": "false"},
    )
    diagram = ParsedDiagram(nodes=[node], edges=[], source_format="test", source_file="test")
    classified, _ = classify_diagram(diagram)
    assert classified[0].tags == {"tier": "prod", "pii": "false"}


def test_untagged_node_yields_empty_tags_on_classified_resource():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Web Server", image_ref="mxgraph.aws4.ec2")],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert classified[0].tags == {}


def test_mq_broker_gets_generated_password_not_plaintext_placeholder():
    """2026-07-21: aws_mq_broker has no manage_master_user_password-style
    AWS-managed-secret flag (unlike RDS/Aurora), so the catalog's own
    placeholder password would otherwise be a real plaintext value baked
    into generated code. classify_diagram() must rewrite it to reference a
    generated random_password resource and emit the companion
    random_password/Secrets Manager resources needed to back it."""
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Message Broker", image_ref="mxgraph.aws4.amazonmq")],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert len(classified) == 1
    resource = classified[0]
    assert resource.resource_type == "aws_mq_broker"

    password_value = resource.nested_blocks["user"][0]["password"]
    assert password_value.startswith("random_password.")
    assert password_value.endswith(".result")
    assert "REPLACE_WITH" not in password_value

    assert len(resource.companion_blocks) == 3
    joined = "\n".join(resource.companion_blocks)
    assert 'resource "random_password"' in joined
    assert 'resource "aws_secretsmanager_secret"' in joined
    assert 'resource "aws_secretsmanager_secret_version"' in joined
    # The password reference used in the broker's nested block must match
    # the actual random_password resource address that gets emitted.
    pw_resource_name = password_value.split(".")[1]
    assert f'resource "random_password" "{pw_resource_name}"' in joined


def test_s3_bucket_gets_secure_by_default_companion_blocks():
    """2026-07-24: checkov flagged every generated aws_s3_bucket for missing
    versioning, KMS encryption, a public access block, and a lifecycle
    configuration — all fixed with unambiguous safe defaults, no user input
    needed, emitted as companion blocks (modern AWS provider versions split
    these off aws_s3_bucket into separate resource types)."""
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Assets Bucket", image_ref="mxgraph.aws4.s3")],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert len(classified) == 1
    resource = classified[0]
    assert resource.resource_type == "aws_s3_bucket"

    joined = "\n".join(resource.companion_blocks)
    assert 'resource "aws_s3_bucket_versioning"' in joined
    assert 'resource "aws_s3_bucket_server_side_encryption_configuration"' in joined
    assert 'sse_algorithm     = "aws:kms"' in joined
    assert 'resource "aws_kms_key"' in joined
    assert 'policy = jsonencode(' in joined  # CKV2_AWS_64 — key must have an explicit policy
    assert 'resource "aws_s3_bucket_public_access_block"' in joined
    assert 'block_public_acls       = true' in joined
    assert 'resource "aws_s3_bucket_lifecycle_configuration"' in joined

    # Each companion resource must reference this bucket's own terraform_name
    # (not a hardcoded/shared name) so multiple buckets in one diagram never
    # collide on duplicate resource addresses.
    assert f"aws_s3_bucket.{resource.terraform_name}.id" in joined


def test_two_s3_buckets_get_non_colliding_companion_resource_names():
    """A second bucket's companion blocks (including its caller-identity
    data source) must use unique resource names, or the generated HCL would
    contain duplicate resource/data addresses and fail terraform validate."""
    diagram = ParsedDiagram(
        nodes=[
            _node("n1", label="Bucket One", image_ref="mxgraph.aws4.s3"),
            _node("n2", label="Bucket Two", image_ref="mxgraph.aws4.s3"),
        ],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert len(classified) == 2
    all_blocks = "\n".join(b for r in classified for b in r.companion_blocks)

    kms_names = [line for line in all_blocks.splitlines() if line.startswith('resource "aws_kms_key"')]
    assert len(kms_names) == 2
    assert len(set(kms_names)) == 2  # no duplicate resource names

    caller_ids = [line for line in all_blocks.splitlines() if line.startswith('data "aws_caller_identity"')]
    assert len(caller_ids) == 2
    assert len(set(caller_ids)) == 2


def test_ec2_instance_has_hardened_defaults():
    """2026-07-24: checkov flagged every generated aws_instance for missing
    detailed monitoring, EBS optimization, IMDSv2 enforcement, and root
    volume encryption — all fixed as catalog-level defaults since none of
    them require per-deployment user input."""
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Web Server", image_ref="mxgraph.aws4.ec2")],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert len(classified) == 1
    resource = classified[0]
    assert resource.resource_type == "aws_instance"
    assert resource.attributes["monitoring"] is True
    assert resource.attributes["ebs_optimized"] is True
    assert resource.nested_blocks["metadata_options"][0]["http_tokens"] == "required"
    assert resource.nested_blocks["root_block_device"][0]["encrypted"] is True


def test_non_mq_broker_resources_have_no_companion_blocks():
    diagram = ParsedDiagram(
        nodes=[_node("n1", label="Web Server", image_ref="mxgraph.aws4.ec2")],
        edges=[], source_format="test", source_file="test",
    )
    classified, _ = classify_diagram(diagram)
    assert classified[0].companion_blocks == []
