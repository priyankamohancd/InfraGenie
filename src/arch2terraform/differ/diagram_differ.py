"""
Diagram Differ
----------------
Compares two versions of the SAME architecture diagram (already parsed +
classified + resolved into ResourceGraphs) and reports what actually
changed: resources added, removed, or modified, connections added or
removed, and — the specific case that matters most for someone about to
`terraform apply` a regenerated package — resources whose diagram LABEL
changed while their diagram node id stayed the same.

Why the rename case gets its own category instead of being folded into
"modified": classifier.py's `_unique_terraform_name()` derives a resource's
Terraform local name from `node.raw_label` (falling back to `node.id` only
when there's no label at all). So renaming a node in the diagram — even
though it's clearly "the same resource" to a human looking at the diagram
— changes its Terraform address (e.g. `aws_instance.web_server` becomes
`aws_instance.app_server`). Terraform has no concept of "this diagram node
stayed the same, only its label changed"; it only knows resource addresses,
so it will plan to DESTROY the old address and CREATE a new one — for what
might be a purely cosmetic rename. This differ catches that BEFORE anyone
applies it, since node_id (draw.io/Excalidraw's own stable element id,
preserved across normal edits) is a signal Terraform itself never sees.

This module intentionally does NOT need real Terraform state to do any of
this — it only compares two ResourceGraphs (parsed diagram + classify +
resolve), which is exactly what arch2tf-product already produces per job.
Real drift detection against ACTUALLY DEPLOYED infrastructure still needs
`terraform plan` against real persisted state (see terraform_planner.py's
_backend_tf / arch2tf-product's remote-state wiring) — that's a different,
complementary kind of drift (deployed vs. desired) from this module's
(diagram version A vs. diagram version B).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from arch2terraform.adapters.registry import parse_diagram
from arch2terraform.classifier.classifier import classify_diagram
from arch2terraform.resolver.resolver import resolve_relationships
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph, ResourceRelationship


@dataclass(frozen=True)
class ResourceChange:
    """One resource present in both diagram versions, but with a real
    attribute/type/nested-block difference (NOT a pure label rename — those
    are reported separately as RenameWarning, since they need different
    handling)."""

    node_id: str
    resource_type: str
    terraform_name: str
    display_label: str
    # attribute_name -> (old_value, new_value). Includes attributes only
    # present on one side (old_value or new_value will be None for those).
    changed_attributes: dict[str, tuple[object, object]] = field(default_factory=dict)
    nested_blocks_changed: bool = False
    # Set only when resource_type itself changed (e.g. a reclassification) —
    # (old_type, new_type). None otherwise.
    resource_type_change: tuple[str, str] | None = None


@dataclass(frozen=True)
class RenameWarning:
    """Same diagram node (node_id unchanged), different label — will cause
    Terraform to destroy+recreate this resource even though it's
    conceptually the same one. See this module's docstring for why."""

    node_id: str
    resource_type: str
    old_label: str
    new_label: str
    old_terraform_name: str
    new_terraform_name: str

    @property
    def suggested_state_mv_command(self) -> str:
        """`terraform state mv` command that preserves this resource's real
        deployed identity across the rename, instead of letting a plan
        silently destroy+recreate it. 2026-07-21: this only ever detected
        the rename before — it never told the user what to actually DO
        about it, which was the whole point of catching it early.

        Uses the bare single-file address (resource_type.terraform_name) —
        this is exactly right for arch2terraform's own single-file output.
        arch2tf-product's multi-module output prefixes real addresses with
        `module.<name>.`, which this differ has no visibility into (module
        assignment is a Phase 2-only concept, decided after this diff runs).
        Callers rendering this for Phase 2 (e.g. github_pusher.py's PR body)
        should prepend the correct module prefix to both sides before
        showing it to the user — see this dataclass's `old_terraform_name`/
        `new_terraform_name` fields, which are the raw material for that.
        """
        old_addr = f"{self.resource_type}.{self.old_terraform_name}"
        new_addr = f"{self.resource_type}.{self.new_terraform_name}"
        return f"terraform state mv '{old_addr}' '{new_addr}'"


@dataclass(frozen=True)
class ConnectionChange:
    source_node_id: str
    target_node_id: str
    relationship_type: str
    label: str = ""


@dataclass
class DiagramDiff:
    added: list[ClassifiedResource] = field(default_factory=list)
    removed: list[ClassifiedResource] = field(default_factory=list)
    modified: list[ResourceChange] = field(default_factory=list)
    rename_warnings: list[RenameWarning] = field(default_factory=list)
    connections_added: list[ConnectionChange] = field(default_factory=list)
    connections_removed: list[ConnectionChange] = field(default_factory=list)
    unchanged_count: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added or self.removed or self.modified or self.rename_warnings
            or self.connections_added or self.connections_removed
        )

    def summary(self) -> str:
        """Human-readable Markdown summary — meant to be dropped straight
        into a GitHub PR body so a reviewer sees what changed architecturally,
        not just a wall of generated HCL."""
        if not self.has_changes:
            return "No architectural changes detected — diagram is identical to the previous version."

        lines = ["### Architecture diff", ""]
        lines.append(f"- ➕ Added: {len(self.added)}")
        for r in self.added:
            lines.append(f"  - `{r.resource_type}.{r.terraform_name}` ({r.display_label})")
        lines.append(f"- ➖ Removed: {len(self.removed)}")
        for r in self.removed:
            lines.append(f"  - `{r.resource_type}.{r.terraform_name}` ({r.display_label})")
        lines.append(f"- ✏️ Modified: {len(self.modified)}")
        for m in self.modified:
            changed = ", ".join(sorted(m.changed_attributes)) or "nested block(s)"
            lines.append(f"  - `{m.resource_type}.{m.terraform_name}` ({m.display_label}) — {changed}")
        if self.rename_warnings:
            lines.append(f"- ⚠️ **Renamed — will cause Terraform to destroy+recreate**: {len(self.rename_warnings)}")
            for w in self.rename_warnings:
                lines.append(
                    f"  - \"{w.old_label}\" → \"{w.new_label}\" "
                    f"(`{w.resource_type}.{w.old_terraform_name}` → `{w.resource_type}.{w.new_terraform_name}`)"
                )
                lines.append(
                    f"    - To keep the existing infrastructure instead of destroying/recreating it, run: "
                    f"`{w.suggested_state_mv_command}`"
                )
                lines.append(
                    "    - (if this is Phase 2's multi-module output, prefix both addresses "
                    "with `module.<module_name>.` first — see the module the resource landed in)"
                )
        if self.connections_added or self.connections_removed:
            lines.append(f"- Connections: +{len(self.connections_added)} / -{len(self.connections_removed)}")
        lines.append(f"- Unchanged: {self.unchanged_count}")
        return "\n".join(lines)


def _connection_key(rel: ResourceRelationship) -> tuple[str, str, str]:
    return (rel.source_node_id, rel.target_node_id, rel.relationship_type)


def diff_resource_graphs(old: ResourceGraph, new: ResourceGraph) -> DiagramDiff:
    """
    Compares two already-resolved ResourceGraphs (same diagram, two points in
    time). Matches resources by `node_id` — the diagram tool's own stable
    element id — NOT by terraform_name/label, precisely so a pure rename can
    be told apart from a real add+remove (see RenameWarning).
    """
    old_by_id = {r.node_id: r for r in old.resources}
    new_by_id = {r.node_id: r for r in new.resources}

    added_ids = new_by_id.keys() - old_by_id.keys()
    removed_ids = old_by_id.keys() - new_by_id.keys()
    common_ids = old_by_id.keys() & new_by_id.keys()

    added = [new_by_id[i] for i in sorted(added_ids)]
    removed = [old_by_id[i] for i in sorted(removed_ids)]

    modified: list[ResourceChange] = []
    rename_warnings: list[RenameWarning] = []

    for node_id in sorted(common_ids):
        old_r = old_by_id[node_id]
        new_r = new_by_id[node_id]

        if old_r.display_label != new_r.display_label:
            rename_warnings.append(RenameWarning(
                node_id=node_id,
                resource_type=new_r.resource_type,
                old_label=old_r.display_label,
                new_label=new_r.display_label,
                old_terraform_name=old_r.terraform_name,
                new_terraform_name=new_r.terraform_name,
            ))

        changed_attrs: dict[str, tuple[object, object]] = {}
        for key in set(old_r.attributes) | set(new_r.attributes):
            old_val = old_r.attributes.get(key)
            new_val = new_r.attributes.get(key)
            if old_val != new_val:
                changed_attrs[key] = (old_val, new_val)

        nested_changed = old_r.nested_blocks != new_r.nested_blocks
        type_change = (old_r.resource_type, new_r.resource_type) if old_r.resource_type != new_r.resource_type else None

        if changed_attrs or nested_changed or type_change:
            modified.append(ResourceChange(
                node_id=node_id,
                resource_type=new_r.resource_type,
                terraform_name=new_r.terraform_name,
                display_label=new_r.display_label,
                changed_attributes=changed_attrs,
                nested_blocks_changed=nested_changed,
                resource_type_change=type_change,
            ))

    old_conns = {_connection_key(r): r for r in old.relationships}
    new_conns = {_connection_key(r): r for r in new.relationships}
    conn_added_keys = new_conns.keys() - old_conns.keys()
    conn_removed_keys = old_conns.keys() - new_conns.keys()

    connections_added = [
        ConnectionChange(*key, label=new_conns[key].label) for key in sorted(conn_added_keys)
    ]
    connections_removed = [
        ConnectionChange(*key, label=old_conns[key].label) for key in sorted(conn_removed_keys)
    ]

    # A resource can appear in `modified`, `rename_warnings`, both, or
    # neither — "unchanged" means genuinely neither, so dedupe by node_id
    # rather than subtracting the two list lengths (which would double-count
    # a resource that's both renamed AND attribute-modified).
    touched_ids = {m.node_id for m in modified} | {w.node_id for w in rename_warnings}
    unchanged_count = len(common_ids) - len(touched_ids)

    return DiagramDiff(
        added=added,
        removed=removed,
        modified=modified,
        rename_warnings=rename_warnings,
        connections_added=connections_added,
        connections_removed=connections_removed,
        unchanged_count=unchanged_count,
    )


def diff_diagram_files(old_path: str, new_path: str) -> DiagramDiff:
    """
    Convenience wrapper: runs arch2terraform's own parse -> classify ->
    resolve pipeline on two raw diagram files (need not be the same format —
    comparing a .drawio export against a later .excalidraw redraw of the
    same architecture is a valid, if unusual, use case) and diffs the
    results. This is what arch2tf-product's GitHub push flow uses to compare
    a newly-uploaded diagram against whatever's already committed at
    `diagrams/<environment>/<filename>` in the target repo.
    """
    old_diagram = parse_diagram(old_path)
    old_classified, _ = classify_diagram(old_diagram)
    old_graph = resolve_relationships(old_diagram, old_classified)

    new_diagram = parse_diagram(new_path)
    new_classified, _ = classify_diagram(new_diagram)
    new_graph = resolve_relationships(new_diagram, new_classified)

    return diff_resource_graphs(old_graph, new_graph)
