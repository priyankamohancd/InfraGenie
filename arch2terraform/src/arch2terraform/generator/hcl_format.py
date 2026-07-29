"""
Minimal HCL formatting helpers.

We hand-roll HCL formatting rather than pulling in a templating engine —
the output shape is simple and fixed enough (resource blocks, provider
block, variable/output blocks) that explicit string building stays more
debuggable than a template layer for this stage of the project.

Nested HCL blocks (e.g. `vpc_config { subnet_ids = [...] }`) are a
different grammar construct from a map-typed attribute (e.g.
`tags = { Name = "x" }`) — blocks have no `=` between the name and `{`,
and the AWS provider schema decides per-argument which one is expected.
Using the wrong syntax for a given key fails `terraform validate` (or
worse, `terraform init`/parse). `Block` exists to make that distinction
explicit wherever it would otherwise be ambiguous (a plain dict value
inside a block body): wrap a dict in `Block(...)` when it must render as
a nested block rather than a map attribute.
"""

from __future__ import annotations


class Block(dict):
    """Marks a dict as an HCL nested-block body (`name { ... }`, no `=`)
    rather than a map-typed attribute value (`name = { ... }`).

    Only meaningful as a *value* inside a block body or attributes dict —
    the top-level `nested_blocks` argument to `resource_block()` is always
    a real block by definition, so its entries don't need this marker.
    """


def hcl_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if (
            value.startswith("var.")
            or value.startswith("data.")
            or (value.startswith(("aws_", "random_")) and "." in value)
        ):
            return value  # reference, not a literal string
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        items = ", ".join(hcl_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        lines = [f'    {k} = {hcl_value(v)}' for k, v in value.items()]
        body = "\n".join(lines)
        return f"{{\n{body}\n  }}"
    return f'"{value}"'


def _render_block_body(body: dict, indent: str) -> list[str]:
    """Renders the inside of a nested block, recursing into further nested
    blocks (marked with `Block`) while ordinary dict/list/scalar values are
    still rendered as flat `key = value` attributes via hcl_value()."""
    lines: list[str] = []
    for key, value in body.items():
        if isinstance(value, Block):
            lines.append(f"{indent}{key} {{")
            lines.extend(_render_block_body(value, indent + "  "))
            lines.append(f"{indent}}}")
        elif isinstance(value, list) and value and all(isinstance(v, Block) for v in value):
            for item in value:
                lines.append(f"{indent}{key} {{")
                lines.extend(_render_block_body(item, indent + "  "))
                lines.append(f"{indent}}}")
        else:
            lines.append(f"{indent}{key} = {hcl_value(value)}")
    return lines


def resource_block(
    resource_type: str,
    local_name: str,
    attributes: dict,
    comment: str = "",
    nested_blocks: dict[str, list[dict]] | None = None,
) -> str:
    """Builds a `resource "type" "name" { ... }` block.

    `attributes` are flat `key = value` lines (or map-typed values, via
    hcl_value's dict handling). `nested_blocks` is a mapping of block name
    to a list of block bodies (a list because some block types — e.g.
    aws_codepipeline's `stage`, aws_dynamodb_table's `attribute` — can
    repeat); each body can itself contain `Block`-wrapped values for
    further nesting (e.g. cloudfront's default_cache_behavior.forwarded_values.cookies).
    """
    lines = []
    if comment:
        lines.append(f"# {comment}")
    lines.append(f'resource "{resource_type}" "{local_name}" {{')

    body_started = False  # whether any attribute/block line has been emitted yet

    if attributes:
        max_key_len = max(len(k) for k in attributes)
        for key, value in attributes.items():
            padding = " " * (max_key_len - len(key))
            lines.append(f"  {key}{padding} = {hcl_value(value)}")
        body_started = True

    if nested_blocks:
        for block_name, bodies in nested_blocks.items():
            for body in bodies:
                if body_started:
                    lines.append("")  # blank line between attrs/blocks for readability
                lines.append(f"  {block_name} {{")
                lines.extend(_render_block_body(body, "    "))
                lines.append("  }")
                body_started = True

    lines.append("}")
    return "\n".join(lines)
