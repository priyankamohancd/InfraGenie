"""
Diagram Differ — unit tests
------------------------------
Synthetic ResourceGraph objects (same style as test_resolver.py) rather than
real diagram files, so these pin the exact diff semantics fast and
independent of any adapter/classifier behavior. See diagram_differ.py's
module docstring for the reasoning behind matching-by-node_id and the
rename-warning special case.
"""
from arch2terraform.differ.diagram_differ import diff_resource_graphs
from arch2terraform.schemas.resources import ClassifiedResource, ResourceGraph, ResourceRelationship


def _resource(node_id, resource_type, label=None, attributes=None, nested_blocks=None) -> ClassifiedResource:
    return ClassifiedResource(
        node_id=node_id,
        resource_type=resource_type,
        terraform_name=(label or node_id).lower().replace(" ", "_"),
        display_label=label or node_id,
        confidence=0.9,
        attributes=attributes or {},
        nested_blocks=nested_blocks or {},
    )


def _graph(resources, relationships=None) -> ResourceGraph:
    return ResourceGraph(resources=resources, relationships=relationships or [])


def test_identical_graphs_have_no_changes():
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server")])
    new = _graph([_resource("ec2-1", "aws_instance", "Web Server")])

    diff = diff_resource_graphs(old, new)
    assert not diff.has_changes
    assert diff.unchanged_count == 1
    assert "No architectural changes" in diff.summary()


def test_added_resource_detected():
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server")])
    new = _graph([
        _resource("ec2-1", "aws_instance", "Web Server"),
        _resource("rds-1", "aws_db_instance", "Postgres DB"),
    ])

    diff = diff_resource_graphs(old, new)
    assert diff.has_changes
    assert len(diff.added) == 1
    assert diff.added[0].node_id == "rds-1"
    assert len(diff.removed) == 0
    assert diff.unchanged_count == 1
    assert "aws_db_instance.postgres_db" in diff.summary()


def test_removed_resource_detected():
    old = _graph([
        _resource("ec2-1", "aws_instance", "Web Server"),
        _resource("s3-1", "aws_s3_bucket", "Old Bucket"),
    ])
    new = _graph([_resource("ec2-1", "aws_instance", "Web Server")])

    diff = diff_resource_graphs(old, new)
    assert len(diff.removed) == 1
    assert diff.removed[0].node_id == "s3-1"
    assert len(diff.added) == 0


def test_modified_attribute_detected():
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server", attributes={"instance_type": "t3.micro"})])
    new = _graph([_resource("ec2-1", "aws_instance", "Web Server", attributes={"instance_type": "t3.large"})])

    diff = diff_resource_graphs(old, new)
    assert len(diff.modified) == 1
    change = diff.modified[0]
    assert change.node_id == "ec2-1"
    assert change.changed_attributes["instance_type"] == ("t3.micro", "t3.large")
    assert not diff.rename_warnings  # label unchanged, this is a pure attribute change


def test_nested_block_change_detected():
    old = _graph([_resource("eks-1", "aws_eks_cluster", "Main Cluster",
                             nested_blocks={"vpc_config": [{"subnet_ids": ["subnet-1"]}]})])
    new = _graph([_resource("eks-1", "aws_eks_cluster", "Main Cluster",
                             nested_blocks={"vpc_config": [{"subnet_ids": ["subnet-1", "subnet-2"]}]})])

    diff = diff_resource_graphs(old, new)
    assert len(diff.modified) == 1
    assert diff.modified[0].nested_blocks_changed is True
    assert diff.modified[0].changed_attributes == {}


def test_resource_type_change_detected():
    """A reclassification (user corrected the resource type via the
    clarification UI between versions) is still a 'modified' resource, not
    silently ignored just because node_id matched."""
    old = _graph([_resource("node-1", "aws_instance", "Mystery Box")])
    new = _graph([_resource("node-1", "aws_lambda_function", "Mystery Box")])

    diff = diff_resource_graphs(old, new)
    assert len(diff.modified) == 1
    assert diff.modified[0].resource_type_change == ("aws_instance", "aws_lambda_function")


def test_rename_produces_warning_not_add_plus_remove():
    """The core case this differ exists for: same node_id, different label.
    Must be reported as ONE rename warning, never as a spurious add+remove
    pair (which would incorrectly suggest two unrelated resources)."""
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server")])
    new = _graph([_resource("ec2-1", "aws_instance", "App Server")])

    diff = diff_resource_graphs(old, new)
    assert len(diff.added) == 0
    assert len(diff.removed) == 0
    assert len(diff.rename_warnings) == 1

    warning = diff.rename_warnings[0]
    assert warning.old_label == "Web Server"
    assert warning.new_label == "App Server"
    assert warning.old_terraform_name == "web_server"
    assert warning.new_terraform_name == "app_server"
    assert "destroy+recreate" in diff.summary()


def test_rename_warning_includes_state_mv_remediation_command():
    """2026-07-21: detecting the rename alone doesn't help anyone avoid the
    destroy+recreate — the warning must also say what to actually run."""
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server")])
    new = _graph([_resource("ec2-1", "aws_instance", "App Server")])

    diff = diff_resource_graphs(old, new)
    warning = diff.rename_warnings[0]

    assert warning.suggested_state_mv_command == (
        "terraform state mv 'aws_instance.web_server' 'aws_instance.app_server'"
    )
    # Must appear in the rendered PR-body summary too, not just be available
    # as a property nobody surfaces.
    summary = diff.summary()
    assert "terraform state mv 'aws_instance.web_server' 'aws_instance.app_server'" in summary
    # Callers building Phase 2's multi-module output need to know this
    # command needs a module prefix - the summary must say so explicitly
    # rather than silently giving a command that's wrong in that context.
    assert "module." in summary


def test_rename_with_no_other_attribute_changes_is_not_also_listed_as_modified():
    """A pure rename (nothing else changed) should show up ONLY in
    rename_warnings, not ALSO clutter up `modified` with a no-op entry."""
    old = _graph([_resource("ec2-1", "aws_instance", "Web Server", attributes={"instance_type": "t3.micro"})])
    new = _graph([_resource("ec2-1", "aws_instance", "App Server", attributes={"instance_type": "t3.micro"})])

    diff = diff_resource_graphs(old, new)
    assert len(diff.rename_warnings) == 1
    assert len(diff.modified) == 0


def test_connection_added_and_removed():
    old = _graph(
        [_resource("ec2-1", "aws_instance", "Web Server"), _resource("s3-1", "aws_s3_bucket", "Bucket")],
        [ResourceRelationship(source_node_id="ec2-1", target_node_id="s3-1", relationship_type="network_ingress")],
    )
    new = _graph(
        [_resource("ec2-1", "aws_instance", "Web Server"), _resource("rds-1", "aws_db_instance", "DB")],
        [ResourceRelationship(source_node_id="ec2-1", target_node_id="rds-1", relationship_type="network_ingress")],
    )

    diff = diff_resource_graphs(old, new)
    assert len(diff.connections_added) == 1
    assert diff.connections_added[0].target_node_id == "rds-1"
    assert len(diff.connections_removed) == 1
    assert diff.connections_removed[0].target_node_id == "s3-1"


def test_summary_is_stable_and_readable_for_a_mixed_diff():
    old = _graph([
        _resource("ec2-1", "aws_instance", "Web Server"),
        _resource("s3-1", "aws_s3_bucket", "Old Bucket"),
    ])
    new = _graph([
        _resource("ec2-1", "aws_instance", "App Server"),  # renamed
        _resource("rds-1", "aws_db_instance", "New DB"),   # added
        # s3-1 removed
    ])

    diff = diff_resource_graphs(old, new)
    summary = diff.summary()
    assert "Added: 1" in summary
    assert "Removed: 1" in summary
    assert "Renamed" in summary
    assert diff.unchanged_count == 0
