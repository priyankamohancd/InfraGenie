"""
Systematic audit of the resource catalog against the AWS provider's actual
required arguments (schema Required: true, no Default). This exists because
the catalog previously covered `default_attributes` only for the handful of
resource types that happened to appear in the two real reference diagrams —
the other ~45 entries were unaudited and several turned out to be missing
required arguments (which produces syntactically valid but semantically
invalid HCL: it passes a naive syntax check but fails real
`terraform validate`), and two had outright wrong Terraform resource type
names (which fails even earlier, at `terraform init`/parse).

Where a required argument is itself a nested HCL block rather than a flat
`key = value` attribute (e.g. aws_autoscaling_group's `launch_template`,
aws_eks_cluster's `vpc_config`), it is NOT included in _REQUIRED_FLAT_ARGS —
the generator (hcl_format.py) only emits flat attribute lines, so those
resources are documented as incomplete in README's "Known limitations"
rather than faked with an attribute-equals-object that wouldn't satisfy the
provider's schema anyway. _KNOWN_INCOMPLETE_BLOCK_TYPES tracks those so this
test can assert every OTHER type has no missing required arguments, while
explicitly acknowledging the ones that still need manual completion.
"""

from __future__ import annotations

from arch2terraform.classifier.catalog import CATALOG

# resource_type -> set of required (no-default) flat arguments the AWS provider
# expects. Populated from the terraform-provider-aws schema for resources this
# catalog covers. `is_container`/wireable attributes (vpc_id, subnet_id, etc.
# handled by hcl_generator.py's _CONTAINMENT_WIRING_RULES) are intentionally
# excluded here since they're satisfied by wiring, not a catalog default.
_REQUIRED_FLAT_ARGS: dict[str, set[str]] = {
    "aws_vpc": {"cidr_block"},
    "aws_subnet": {"cidr_block"},  # vpc_id wired via containment
    "aws_internet_gateway": set(),  # vpc_id optional (can attach separately)
    "aws_nat_gateway": {"allocation_id"},  # subnet_id wired via containment
    "aws_route_table": set(),  # vpc_id wired via containment, no other required args
    "aws_security_group": set(),  # name/description/vpc_id all optional
    "aws_network_acl": set(),  # vpc_id wired via containment, no other required args
    "aws_vpc_peering_connection": {"vpc_id", "peer_vpc_id"},
    "aws_vpn_gateway": set(),
    "aws_transit_gateway": set(),
    "aws_instance": {"ami", "instance_type"},
    "aws_launch_template": set(),  # notably permissive schema — nothing strictly required
    "aws_lambda_function": {"function_name", "role", "filename", "runtime", "handler"},
    "aws_ecs_cluster": set(),  # name optional/auto
    "aws_ecs_service": {"name", "task_definition"},
    "aws_ecs_task_definition": {"family", "container_definitions"},
    "aws_eks_cluster": {"role_arn"},  # vpc_config block not covered — see module docstring
    "aws_batch_job_queue": {"name", "state", "priority"},  # compute_environment_order block not covered
    "aws_s3_bucket": {"bucket"},
    "aws_ebs_volume": {"availability_zone"},
    "aws_efs_file_system": set(),
    "aws_backup_vault": {"name"},
    "aws_glacier_vault": {"name"},
    "aws_db_instance": {"engine", "instance_class", "allocated_storage", "username"},
    "aws_dynamodb_table": {"name", "hash_key"},  # attribute block not covered
    "aws_elasticache_cluster": {"cluster_id", "engine", "node_type", "num_cache_nodes"},
    "aws_redshift_cluster": {"cluster_identifier", "node_type", "master_username"},
    "aws_rds_cluster": {"engine", "master_username"},
    # subnets/subnet_mapping is a "one of" cross-field constraint (not a simple
    # per-argument Required flag) — missed in the original audit and only caught
    # by a real `terraform validate` run. name itself is optional/auto.
    "aws_lb": {"subnets"},
    "aws_lb_target_group": {"port", "protocol"},  # vpc_id wired via containment
    "aws_route53_zone": {"name"},
    "aws_api_gateway_rest_api": {"name"},
    "aws_sqs_queue": set(),
    "aws_sns_topic": set(),
    "aws_kinesis_stream": {"name", "shard_count"},
    "aws_mq_broker": {"broker_name", "engine_type", "engine_version", "host_instance_type"},  # user block not covered
    "aws_cloudwatch_event_rule": set(),  # schedule_expression set but not schema-Required per se
    "aws_appsync_graphql_api": {"name", "authentication_type"},
    "aws_mwaa_environment": {
        "name", "execution_role_arn", "source_bucket_arn", "dag_s3_path",
    },  # network_configuration block not covered
    "aws_iam_role": {"assume_role_policy"},
    "aws_iam_policy": {"policy"},
    "aws_iam_user": {"name"},
    "aws_kms_key": set(),
    "aws_secretsmanager_secret": set(),
    "aws_acm_certificate": {"domain_name", "validation_method"},
    "aws_waf_web_acl": {"metric_name"},  # default_action block not covered
    "aws_cognito_user_pool": {"name"},
    "aws_cloudwatch_log_group": set(),
    "aws_cloudwatch_metric_alarm": {
        "alarm_name", "comparison_operator", "evaluation_periods",
        "metric_name", "namespace", "period", "statistic", "threshold",
    },
    "aws_cloudtrail": {"name", "s3_bucket_name"},
    "aws_xray_group": {"group_name", "filter_expression"},
    "aws_codepipeline": {"name", "role_arn"},  # artifact_store/stage blocks not covered
    "aws_codebuild_project": {"name", "service_role"},  # artifacts/environment/source blocks not covered
    "aws_ecr_repository": {"name"},
    "aws_sfn_state_machine": {"role_arn", "definition"},
    "aws_glue_job": {"name", "role_arn"},  # command block not covered
    "aws_athena_workgroup": {"name"},
}

# Resource types whose provider schema has at least one Required *nested block*
# argument (not a flat attribute) that the generator cannot currently emit.
# These are expected to remain incomplete — real terraform validate WILL still
# flag them until the generator supports nested blocks or a user hand-completes
# the resource. Tracked explicitly so this stays a documented, intentional gap
# rather than a silent one.
_KNOWN_INCOMPLETE_BLOCK_TYPES = {
    "aws_autoscaling_group",   # launch_template / launch_configuration / mixed_instances_policy
    "aws_eks_cluster",         # vpc_config
    "aws_batch_job_queue",     # compute_environment_order
    "aws_waf_web_acl",         # default_action
    "aws_cloudfront_distribution",  # origin, default_cache_behavior, restrictions, viewer_certificate
    "aws_dynamodb_table",      # attribute
    "aws_mq_broker",           # user
    "aws_codepipeline",        # artifact_store, stage
    "aws_codebuild_project",   # artifacts, environment, source
    "aws_glue_job",            # command
    "aws_mwaa_environment",    # network_configuration
}


def test_no_duplicate_or_obviously_wrong_resource_type_names():
    """Regression test: two catalog entries previously used resource type names
    that don't exist in the AWS provider at all (aws_eventbridge_rule,
    aws_step_function_state_machine) — terraform validate/init would reject
    these immediately with 'Invalid resource type', which is worse than any
    missing-argument issue. Every terraform_type must start with 'aws_' and
    the specific historical typos must not reappear."""
    types = [d.terraform_type for d in CATALOG]
    assert "aws_eventbridge_rule" not in types
    assert "aws_step_function_state_machine" not in types
    assert "aws_cloudwatch_event_rule" in types
    assert "aws_sfn_state_machine" in types
    for t in types:
        assert t.startswith("aws_"), f"suspicious resource type name: {t}"


def test_required_flat_arguments_present_for_audited_types():
    """Every catalog entry we've audited against the real AWS provider schema
    must supply all of its required-with-no-default flat arguments — otherwise
    the generated HCL is syntactically valid but fails real terraform validate."""
    by_type = {d.terraform_type: d for d in CATALOG}
    for terraform_type, required in _REQUIRED_FLAT_ARGS.items():
        assert terraform_type in by_type, f"{terraform_type} no longer in catalog — update this test"
        defn = by_type[terraform_type]
        missing = required - set(defn.default_attributes)
        assert not missing, f"{terraform_type} missing required args: {missing}"


def test_every_catalog_entry_is_categorized():
    """Every catalog entry should either be in the audited required-args map or
    explicitly acknowledged as needing nested-block support — this makes the
    fully-unaudited state (the situation before this test existed) impossible
    to silently regress back into."""
    all_types = {d.terraform_type for d in CATALOG}
    audited = set(_REQUIRED_FLAT_ARGS) | _KNOWN_INCOMPLETE_BLOCK_TYPES
    unaudited = all_types - audited
    assert not unaudited, (
        f"Catalog entries added without a required-argument audit: {unaudited}. "
        "Add them to _REQUIRED_FLAT_ARGS (or _KNOWN_INCOMPLETE_BLOCK_TYPES if they "
        "need a nested block the generator can't emit yet)."
    )


# Attribute keys the AWS provider format-validates client-side as an ARN
# (verify.ValidARN in the provider source) — a plain descriptive placeholder
# like the old "REPLACE_WITH_IAM_ROLE_ARN" fails terraform validate outright
# with "is an invalid ARN: arn: invalid prefix". Real terraform validate (not
# this sandbox's syntax-only checks) is what caught this the first time.
_ARN_VALIDATED_KEYS = {"role", "role_arn", "service_role", "execution_role_arn", "source_bucket_arn"}


def test_arn_typed_placeholders_are_actually_shaped_like_arns():
    """Regression test: terraform validate rejected role = "REPLACE_WITH_IAM_ROLE_ARN"
    because the AWS provider validates ARN-typed arguments client-side, not just
    their presence. Every placeholder for an ARN-validated key must at least look
    like arn:partition:service:region:account-id:resource, even with fake values."""
    for defn in CATALOG:
        for key, value in defn.default_attributes.items():
            if key in _ARN_VALIDATED_KEYS:
                assert isinstance(value, str) and value.startswith("arn:"), (
                    f"{defn.terraform_type}.{key} = {value!r} is not shaped like an ARN "
                    "and will fail real terraform validate"
                )
