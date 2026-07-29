"""
Unit tests for hcl_format.py's nested-block rendering (resource_block's
`nested_blocks` parameter and the `Block` marker for blocks nested inside
blocks). Added alongside the generator feature that closes the "11 catalog
entries need a nested HCL block the generator can't emit" gap tracked in
tests/unit/test_catalog.py's _REQUIRED_NESTED_BLOCKS.

These are string-shape tests (spacing/braces), not schema-correctness tests
— schema correctness (which types actually need which blocks) is covered by
test_catalog.py's test_required_nested_blocks_present and by parsing real
catalog output with python-hcl2 / real `terraform validate`.
"""

from __future__ import annotations

from arch2terraform.generator.hcl_format import Block, hcl_value, resource_block


def test_resource_block_with_no_nested_blocks_is_unchanged():
    """Backward-compat: omitting nested_blocks (or passing None) must produce
    exactly the old flat-attributes-only output — existing callers/tests must
    not see any difference."""
    text = resource_block("aws_vpc", "main", {"cidr_block": "10.0.0.0/16"})
    assert text == (
        'resource "aws_vpc" "main" {\n'
        '  cidr_block = "10.0.0.0/16"\n'
        "}"
    )


def test_single_nested_block_renders_with_brace_not_equals():
    text = resource_block(
        "aws_eks_cluster", "main", {"role_arn": "arn:aws:iam::000000000000:role/x"},
        nested_blocks={"vpc_config": [{"subnet_ids": ["subnet-a", "subnet-b"]}]},
    )
    assert 'vpc_config {' in text
    assert 'vpc_config =' not in text  # must be block syntax, not an attribute
    assert 'subnet_ids = ["subnet-a", "subnet-b"]' in text
    # braces balance
    assert text.count("{") == text.count("}")


def test_repeated_block_emits_one_block_per_list_item():
    """Some block types can repeat (e.g. aws_dynamodb_table's `attribute`,
    aws_codepipeline's `stage`) — nested_blocks[name] is a list precisely to
    support this; each item must become its own `name { ... }` block."""
    text = resource_block(
        "aws_dynamodb_table", "t", {"name": "t", "hash_key": "id"},
        nested_blocks={"attribute": [
            {"name": "id", "type": "S"},
            {"name": "created_at", "type": "N"},
        ]},
    )
    assert text.count("attribute {") == 2
    assert 'name = "id"' in text
    assert 'name = "created_at"' in text


def test_block_marker_nests_a_block_inside_a_block():
    """cloudfront-style deep nesting: default_cache_behavior contains a real
    nested block forwarded_values, which itself contains a real nested block
    cookies. Without the Block() marker these would be indistinguishable from
    map-typed attributes and render with `=`, which is the wrong HCL grammar
    for a schema-defined block."""
    text = resource_block(
        "aws_cloudfront_distribution", "cdn", {"enabled": True},
        nested_blocks={"default_cache_behavior": [{
            "target_origin_id": "primary",
            "forwarded_values": Block({
                "query_string": False,
                "cookies": Block({"forward": "none"}),
            }),
        }]},
    )
    assert "forwarded_values {" in text
    assert "forwarded_values =" not in text
    assert "cookies {" in text
    assert "cookies =" not in text
    assert 'forward = "none"' in text
    assert text.count("{") == text.count("}")


def test_plain_dict_value_still_renders_as_map_attribute_not_a_block():
    """The whole point of the Block marker is to disambiguate: an ordinary
    (unwrapped) dict value inside a block body — e.g. aws_codepipeline
    action's `configuration` map — must keep rendering as `key = { ... }`,
    not `key { ... }`, since the provider schema defines it as a map-typed
    attribute, not a block."""
    text = resource_block(
        "aws_codepipeline", "p", {"name": "p"},
        nested_blocks={"stage": [{
            "name": "Source",
            "action": [Block({
                "name": "Source",
                "configuration": {"S3Bucket": "my-bucket"},
            })],
        }]},
    )
    assert "configuration = {" in text
    assert "configuration {" not in text
    assert 'S3Bucket = "my-bucket"' in text


def test_hcl_value_unaffected_by_block_changes():
    """hcl_value() itself (used for ordinary flat/map/list attributes) must
    still work exactly as before — nested-block support is additive."""
    assert hcl_value(True) == "true"
    assert hcl_value(80) == "80"
    assert hcl_value("hello") == '"hello"'
    assert hcl_value(["a", "b"]) == '["a", "b"]'
    assert hcl_value({"k": "v"}) == '{\n    k = "v"\n  }'


def test_attributes_and_nested_blocks_together_are_separated_by_blank_line():
    text = resource_block(
        "aws_waf_web_acl", "acl", {"metric_name": "m"},
        nested_blocks={"default_action": [{"type": "ALLOW"}]},
    )
    lines = text.split("\n")
    # the blank line should sit between the last attribute and the block
    assert "" in lines
    blank_idx = lines.index("")
    assert "metric_name" in lines[blank_idx - 1]
    assert "default_action {" in lines[blank_idx + 1]
