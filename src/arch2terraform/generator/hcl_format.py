"""
Minimal HCL formatting helpers.

We hand-roll HCL formatting rather than pulling in a templating engine —
the output shape is simple and fixed enough (resource blocks, provider
block, variable/output blocks) that explicit string building stays more
debuggable than a template layer for this stage of the project.
"""

from __future__ import annotations


def hcl_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value.startswith("var.") or value.startswith("aws_") and "." in value:
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


def resource_block(resource_type: str, local_name: str, attributes: dict, comment: str = "") -> str:
    lines = []
    if comment:
        lines.append(f"# {comment}")
    lines.append(f'resource "{resource_type}" "{local_name}" {{')

    if attributes:
        max_key_len = max(len(k) for k in attributes)
        for key, value in attributes.items():
            padding = " " * (max_key_len - len(key))
            lines.append(f"  {key}{padding} = {hcl_value(value)}")

    lines.append("}")
    return "\n".join(lines)
