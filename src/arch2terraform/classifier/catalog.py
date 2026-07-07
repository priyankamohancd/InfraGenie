"""
Catalog of AWS resource types the classifier can recognize.

Each entry defines:
  - terraform_type: the actual Terraform resource block type
  - icon_keys: substrings matched against DiagramNode.image_ref (draw.io/Lucidchart
    AWS stencil names, e.g. "mxgraph.aws4.ec2" or "ec2" from Lucidchart shape lib)
  - label_keywords: substrings matched against the lowercased node label, used for
    Excalidraw (no icon system) and as a fallback everywhere else
  - is_container: true for boxes that other resources nest inside (VPC, subnet, SG)
  - default_attributes: minimal viable attribute set so generated HCL is valid
    out of the box; users override via the Phase 2 clarifier or by hand after.

This is intentionally a plain data structure (no classification logic) so it's
easy to extend past 45 entries without touching classifier.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResourceDefinition:
    terraform_type: str
    icon_keys: tuple[str, ...]
    label_keywords: tuple[str, ...]
    is_container: bool = False
    default_attributes: dict = field(default_factory=dict)


# Placeholder ARNs — NOT just descriptive strings. A real `terraform validate`
# run (2026-07) caught that AWS provider fields named `role`/`*_role_arn` etc.
# run client-side format validation (verify.ValidARN) at validate time, not
# just plan/apply — a free-text placeholder like the old "REPLACE_WITH_IAM_ROLE_ARN"
# fails with "is an invalid ARN: arn: invalid prefix" before the user even
# gets to the "supply a real one" step. These must be syntactically valid
# ARNs (arn:partition:service:region:account-id:resource) even though the
# account ID is fake, so validate passes and only `plan`/`apply` against a
# real AWS account will (correctly) fail until the user swaps in a real ARN.
_FAKE_IAM_ROLE_ARN = "arn:aws:iam::000000000000:role/REPLACE_WITH_ROLE_NAME"
_FAKE_S3_BUCKET_ARN = "arn:aws:s3:::replace-with-globally-unique-name"


CATALOG: list[ResourceDefinition] = [
    # --- Networking / containers ---------------------------------------
    ResourceDefinition("aws_vpc", ("vpc",), ("vpc", "virtual private cloud"),
                        is_container=True, default_attributes={"cidr_block": "10.0.0.0/16"}),
    ResourceDefinition("aws_subnet", ("subnet", "privatesubnet", "publicsubnet"), ("subnet",),
                        is_container=True, default_attributes={"cidr_block": "10.0.1.0/24"}),
    ResourceDefinition("aws_internet_gateway", ("internetgateway", "internet_gateway"), ("internet gateway", "igw")),
    # subnet_id is wired via _CONTAINMENT_WIRING_RULES (see hcl_generator.py) when the
    # diagram nests this inside a subnet. allocation_id (the EIP) has no containment
    # analogue — it's a sibling resource, not something the NAT gateway sits "inside" —
    # so it gets a placeholder like the other externally-provisioned-ID cases.
    ResourceDefinition("aws_nat_gateway", ("natgateway",), ("nat gateway", "nat"),
                        default_attributes={"allocation_id": "REPLACE_WITH_EIP_ALLOCATION_ID"}),
    # vpc_id is wired via containment (see hcl_generator.py's wiring rules) — no flat
    # default needed here, same pattern as aws_subnet's vpc_id.
    ResourceDefinition("aws_route_table", ("routetable",), ("route table",)),
    ResourceDefinition("aws_security_group", ("securitygroup",), ("security group", "sg"),
                        is_container=True),
    # vpc_id wired via containment, same as aws_route_table above.
    ResourceDefinition("aws_network_acl", ("nacl", "networkacl"), ("network acl", "nacl")),
    # peer_vpc_id and vpc_id are both required with no containment analogue (a peering
    # connection spans two VPCs, not "inside" either one in the diagram-nesting sense).
    ResourceDefinition("aws_vpc_peering_connection", ("vpcpeering",), ("vpc peering", "peering connection"),
                        default_attributes={
                            "vpc_id": "REPLACE_WITH_REQUESTER_VPC_ID",
                            "peer_vpc_id": "REPLACE_WITH_ACCEPTER_VPC_ID",
                        }),
    ResourceDefinition("aws_vpn_gateway", ("vpngateway",), ("vpn gateway",)),
    ResourceDefinition("aws_transit_gateway", ("transitgateway",), ("transit gateway",)),

    # --- Compute ---------------------------------------------------------
    # `ami` is a required argument with no valid universal default (AMI IDs are
    # region- and architecture-specific) — a clearly-fake placeholder in the same
    # style as the S3 bucket name below lets `terraform validate` pass while still
    # forcing the user to supply a real one before `apply`.
    ResourceDefinition("aws_instance", ("ec2",), ("ec2 instance", "ec2", "virtual machine"),
                        default_attributes={"instance_type": "t3.micro", "ami": "ami-00000000000000000"}),
    ResourceDefinition("aws_launch_template", ("launchtemplate",), ("launch template",)),
    ResourceDefinition("aws_autoscaling_group", ("autoscaling",), ("auto scaling", "asg")),
    # function_name, role, and a deployment package source (filename here) are all
    # required by the provider with no valid default — same placeholder philosophy.
    ResourceDefinition("aws_lambda_function", ("lambda",), ("lambda", "function"),
                        default_attributes={
                            "runtime": "python3.12",
                            "handler": "index.handler",
                            "function_name": "REPLACE_WITH_FUNCTION_NAME",
                            "role": _FAKE_IAM_ROLE_ARN,
                            "filename": "REPLACE_WITH_DEPLOYMENT_PACKAGE_ZIP",
                        }),
    ResourceDefinition("aws_ecs_cluster", ("ecscluster", "elasticcontainerservice"), ("ecs cluster", "ecs")),
    # name and task_definition are both required with no default.
    ResourceDefinition("aws_ecs_service", ("ecsservice",), ("ecs service",),
                        default_attributes={
                            "name": "replace-with-service-name",
                            "task_definition": "REPLACE_WITH_TASK_DEFINITION_ARN",
                        }),
    # family and container_definitions (a JSON string, itself required to describe at
    # least one container by the provider's own validation) are both required.
    ResourceDefinition("aws_ecs_task_definition", ("taskdefinition",), ("task definition",),
                        default_attributes={
                            "family": "replace-with-task-family",
                            "container_definitions": (
                                '[{"name":"placeholder","image":"REPLACE_WITH_IMAGE_URI",'
                                '"cpu":128,"memory":256,"essential":true}]'
                            ),
                        }),
    # role_arn is a required flat argument. vpc_config is ALSO required but is a nested
    # HCL block (`vpc_config { subnet_ids = [...] }`), not a flat attribute — the
    # generator currently only emits `key = value` attribute lines (see hcl_format.py),
    # so this resource is emitted incomplete. Flagged in README's "Known limitations".
    ResourceDefinition("aws_eks_cluster", ("eks", "elastickubernetes"), ("eks", "kubernetes"),
                        default_attributes={"role_arn": _FAKE_IAM_ROLE_ARN}),
    # name, state, priority are required flat arguments. compute_environment_order is
    # ALSO required but is a nested block — same generator limitation as aws_eks_cluster
    # above; this resource is emitted incomplete without it.
    ResourceDefinition("aws_batch_job_queue", ("batch",), ("batch job", "batch queue"),
                        default_attributes={
                            "name": "replace-with-queue-name",
                            "state": "ENABLED",
                            "priority": 1,
                        }),

    # --- Storage -----------------------------------------------------------
    # S3 bucket names are validated client-side by the provider (lowercase letters,
    # digits, dots, hyphens only — no underscores/uppercase); a placeholder that
    # violates that format fails `terraform validate` before the user even gets to
    # the "pick a real globally-unique name" step, so the placeholder itself must
    # already be a legal bucket name.
    ResourceDefinition("aws_s3_bucket", ("s3", "simplestorageservice"), ("s3", "bucket", "object storage"),
                        default_attributes={"bucket": "replace-with-globally-unique-name"}),
    # availability_zone is required with no default — us-east-1a is a real AZ name in
    # the default region this project generates (see variables.tf's aws_region default).
    ResourceDefinition("aws_ebs_volume", ("ebs", "elasticblockstore"), ("ebs", "block storage"),
                        default_attributes={"availability_zone": "us-east-1a"}),
    ResourceDefinition("aws_efs_file_system", ("efs", "elasticfilesystem"), ("efs", "file system")),
    ResourceDefinition("aws_backup_vault", ("backup",), ("backup vault",),
                        default_attributes={"name": "replace-with-backup-vault-name"}),
    ResourceDefinition("aws_glacier_vault", ("glacier",), ("glacier",),
                        default_attributes={"name": "replace-with-glacier-vault-name"}),

    # --- Database ----------------------------------------------------------
    # allocated_storage is required with no default. username is required unless
    # manage_master_user_password handles auth — set both so the block is complete
    # without ever writing a literal password into generated code.
    ResourceDefinition("aws_db_instance", ("rds", "relationaldatabaseservice"), ("rds", "database", "postgres", "mysql"),
                        default_attributes={
                            "engine": "postgres",
                            "instance_class": "db.t3.micro",
                            "allocated_storage": 20,
                            "username": "admin",
                            "manage_master_user_password": True,
                        }),
    # name and hash_key are required flat arguments. PAY_PER_REQUEST billing mode is
    # used specifically to avoid also needing read_capacity/write_capacity (which are
    # conditionally required under the default PROVISIONED mode). The `attribute` block
    # (declaring hash_key's type) is ALSO required but is a nested HCL block the
    # generator can't emit yet — flagged in README's "Known limitations".
    ResourceDefinition("aws_dynamodb_table", ("dynamodb",), ("dynamodb", "nosql"),
                        default_attributes={
                            "name": "replace-with-table-name",
                            "billing_mode": "PAY_PER_REQUEST",
                            "hash_key": "id",
                        }),
    ResourceDefinition("aws_elasticache_cluster", ("elasticache",), ("elasticache", "redis", "memcached"),
                        default_attributes={
                            "cluster_id": "replace-with-cluster-id",
                            "engine": "redis",
                            "node_type": "cache.t3.micro",
                            "num_cache_nodes": 1,
                        }),
    ResourceDefinition("aws_redshift_cluster", ("redshift",), ("redshift", "data warehouse"),
                        default_attributes={
                            "cluster_identifier": "replace-with-cluster-id",
                            "node_type": "dc2.large",
                            "master_username": "admin",
                            "manage_master_password": True,
                        }),
    ResourceDefinition("aws_rds_cluster", ("aurora",), ("aurora",),
                        default_attributes={
                            "engine": "aurora-postgresql",
                            "master_username": "admin",
                            "manage_master_user_password": True,
                        }),

    # --- Load balancing / networking edge ----------------------------------
    # A real `terraform validate` run caught this: one of `subnets` or
    # `subnet_mapping` is required (a cross-field check, not a simple per-argument
    # Required flag, which is why it wasn't caught by the earlier per-argument
    # audit). A single-element placeholder list satisfies it; a real deployment
    # normally wants 2+ subnets across AZs, which the user must supply.
    ResourceDefinition("aws_lb", ("elasticloadbalancing", "alb", "nlb"), ("load balancer", "alb", "nlb"),
                        default_attributes={"subnets": ["subnet-00000000000000000"]}),
    # port, protocol, and vpc_id are all required when target_type is "instance" (the
    # default) — port/protocol have no containment analogue so get flat defaults;
    # vpc_id is wired via containment like aws_subnet's vpc_id.
    ResourceDefinition("aws_lb_target_group", ("targetgroup",), ("target group",),
                        default_attributes={"port": 80, "protocol": "HTTP"}),
    # Required blocks (origin, default_cache_behavior, restrictions, viewer_certificate)
    # make this resource fundamentally incomplete without nested-block generator
    # support — flagged in README's "Known limitations" rather than faked with flat
    # attributes that wouldn't satisfy the schema anyway.
    ResourceDefinition("aws_cloudfront_distribution", ("cloudfront",), ("cloudfront", "cdn")),
    # `name` (the domain) is required with no default; "example.com" is a
    # syntactically valid placeholder domain reserved for documentation use
    # (RFC 2606), so it passes validation without looking like a real zone.
    ResourceDefinition("aws_route53_zone", ("route53",), ("route 53", "dns"),
                        default_attributes={"name": "replace-with-your-domain.example.com"}),
    ResourceDefinition("aws_api_gateway_rest_api", ("apigateway",), ("api gateway", "rest api"),
                        default_attributes={"name": "replace-with-api-name"}),

    # --- Messaging / integration --------------------------------------------
    ResourceDefinition("aws_sqs_queue", ("sqs", "simplequeueservice"), ("sqs", "queue")),
    ResourceDefinition("aws_sns_topic", ("sns", "simplenotificationservice"), ("sns", "topic", "notification")),
    # name required; shard_count required under the default PROVISIONED stream mode
    # (avoids also needing a stream_mode_details block for ON_DEMAND mode).
    ResourceDefinition("aws_kinesis_stream", ("kinesis",), ("kinesis", "stream"),
                        default_attributes={"name": "replace-with-stream-name", "shard_count": 1}),
    # broker_name/engine_type/engine_version/host_instance_type are required flat
    # arguments. `user` is ALSO required but is a nested block (at least one, with
    # username/password) — same generator limitation noted elsewhere, flagged in README.
    ResourceDefinition("aws_mq_broker", ("mq", "amazonmq"), ("amazon mq", "message broker"),
                        default_attributes={
                            "broker_name": "replace-with-broker-name",
                            "engine_type": "ActiveMQ",
                            "engine_version": "5.17.6",
                            "host_instance_type": "mq.t3.micro",
                        }),
    # Real Terraform AWS provider resource type is aws_cloudwatch_event_rule — there is
    # no "aws_eventbridge_rule" resource type in the provider (EventBridge is the AWS
    # *service* rebrand; the Terraform resource kept its original CloudWatch Events
    # name). The old value here would have made terraform validate/init fail outright
    # with "Invalid resource type", which is worse than any missing-argument issue.
    # Neither schedule_expression nor event_pattern is individually marked Required,
    # but the provider rejects a rule with both unset — schedule_expression is set here
    # since it needs no companion resource, unlike event_pattern's JSON matching a
    # specific event source.
    ResourceDefinition("aws_cloudwatch_event_rule", ("eventbridge", "cloudwatchevents"), ("eventbridge", "event bus"),
                        default_attributes={"schedule_expression": "rate(1 day)"}),
    # name and authentication_type are both required flat arguments.
    ResourceDefinition("aws_appsync_graphql_api", ("appsync",), ("appsync", "graphql"),
                        default_attributes={
                            "name": "replace-with-api-name",
                            "authentication_type": "API_KEY",
                        }),
    # name, execution_role_arn, source_bucket_arn, dag_s3_path are all required flat
    # arguments. network_configuration is ALSO required but is a nested block — same
    # generator limitation noted elsewhere, flagged in README's "Known limitations".
    ResourceDefinition("aws_mwaa_environment", ("mwaa", "managedairflow"), ("airflow", "mwaa"),
                        default_attributes={
                            "name": "replace-with-environment-name",
                            "execution_role_arn": _FAKE_IAM_ROLE_ARN,
                            "source_bucket_arn": _FAKE_S3_BUCKET_ARN,
                            "dag_s3_path": "dags/",
                        }),

    # --- IAM / Security ------------------------------------------------------
    # assume_role_policy is required (a JSON trust policy) with no default; a generic
    # Lambda-service trust policy is a plausible, syntactically-valid placeholder.
    ResourceDefinition("aws_iam_role", ("iamrole",), ("iam role", "role"),
                        default_attributes={
                            "assume_role_policy": (
                                '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
                                '"Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
                            ),
                        }),
    # policy (a JSON IAM policy document) is required with no default. A deny-all
    # placeholder is syntactically valid and, unlike an arbitrary allow, is safe to
    # accidentally apply — it just does nothing until the user defines real permissions.
    ResourceDefinition("aws_iam_policy", ("iampolicy",), ("iam policy", "policy"),
                        default_attributes={
                            "policy": '{"Version":"2012-10-17","Statement":[{"Effect":"Deny","Action":"*","Resource":"*"}]}',
                        }),
    ResourceDefinition("aws_iam_user", ("iamuser",), ("iam user",),
                        default_attributes={"name": "replace-with-iam-user-name"}),
    ResourceDefinition("aws_kms_key", ("kms",), ("kms", "key management")),
    ResourceDefinition("aws_secretsmanager_secret", ("secretsmanager",), ("secrets manager", "secret")),
    # domain_name is required unless importing an existing certificate (private_key +
    # certificate_body) — validation_method is required alongside it for ACM-issued certs.
    ResourceDefinition("aws_acm_certificate", ("acm", "certificatemanager"), ("acm", "certificate"),
                        default_attributes={
                            "domain_name": "replace-with-your-domain.example.com",
                            "validation_method": "DNS",
                        }),
    # metric_name is a required flat argument. default_action is ALSO required but is a
    # nested block (`default_action { allow {} }` / `block {}`) — same generator
    # limitation as elsewhere, flagged in README's "Known limitations".
    ResourceDefinition("aws_waf_web_acl", ("waf",), ("waf", "web acl"),
                        default_attributes={"metric_name": "replaceWithMetricName"}),
    ResourceDefinition("aws_cognito_user_pool", ("cognito",), ("cognito", "user pool"),
                        default_attributes={"name": "replace-with-user-pool-name"}),

    # --- Observability ---------------------------------------------------
    ResourceDefinition("aws_cloudwatch_log_group", ("cloudwatch", "cloudwatchlogs"), ("cloudwatch", "log group")),
    # All eight of these are required with no default (comparison_operator,
    # evaluation_periods, metric_name, namespace, period, statistic, threshold, plus
    # alarm_name) — all flat scalars, fully fixable without any nested block.
    ResourceDefinition("aws_cloudwatch_metric_alarm", ("cloudwatchalarm",), ("cloudwatch alarm", "alarm"),
                        default_attributes={
                            "alarm_name": "replace-with-alarm-name",
                            "comparison_operator": "GreaterThanThreshold",
                            "evaluation_periods": 1,
                            "metric_name": "CPUUtilization",
                            "namespace": "AWS/EC2",
                            "period": 300,
                            "statistic": "Average",
                            "threshold": 80,
                        }),
    # name and s3_bucket_name are both required with no default; s3_bucket_name isn't
    # wired via containment since a CloudTrail trail isn't drawn "inside" its target
    # bucket the way a subnet sits inside a VPC.
    ResourceDefinition("aws_cloudtrail", ("cloudtrail",), ("cloudtrail",),
                        default_attributes={
                            "name": "replace-with-trail-name",
                            "s3_bucket_name": "replace-with-globally-unique-name",
                        }),
    ResourceDefinition("aws_xray_group", ("xray",), ("x-ray", "xray"),
                        default_attributes={
                            "group_name": "replace-with-group-name",
                            "filter_expression": 'service("*") { fault OR error }',
                        }),

    # --- Misc -------------------------------------------------------------
    # name and role_arn are required flat arguments. artifact_store and stage are ALSO
    # required but are nested HCL blocks — same generator limitation noted elsewhere,
    # flagged in README's "Known limitations".
    ResourceDefinition("aws_codepipeline", ("codepipeline",), ("codepipeline", "ci/cd pipeline"),
                        default_attributes={
                            "name": "replace-with-pipeline-name",
                            "role_arn": _FAKE_IAM_ROLE_ARN,
                        }),
    # name and service_role are required flat arguments. artifacts/environment/source
    # are ALSO required but are nested blocks — same generator limitation, flagged.
    ResourceDefinition("aws_codebuild_project", ("codebuild",), ("codebuild",),
                        default_attributes={
                            "name": "replace-with-project-name",
                            "service_role": _FAKE_IAM_ROLE_ARN,
                        }),
    ResourceDefinition("aws_ecr_repository", ("ecr", "elasticcontainerregistry"), ("ecr", "container registry"),
                        default_attributes={"name": "replace-with-repository-name"}),
    # Real Terraform AWS provider resource type is aws_sfn_state_machine — there is no
    # "aws_step_function_state_machine" resource type in the provider (same class of
    # naming bug as aws_eventbridge_rule above: AWS's service-facing name "Step
    # Functions" differs from the Terraform resource name "sfn", inherited from the
    # underlying API's own naming). role_arn and definition (Amazon States Language,
    # itself required to have a StartAt/States structure by the provider's own JSON
    # validation) are both required flat string arguments.
    ResourceDefinition("aws_sfn_state_machine", ("stepfunctions",), ("step functions", "state machine"),
                        default_attributes={
                            "role_arn": _FAKE_IAM_ROLE_ARN,
                            "definition": (
                                '{"Comment":"Placeholder state machine","StartAt":"PlaceholderState",'
                                '"States":{"PlaceholderState":{"Type":"Pass","End":true}}}'
                            ),
                        }),
    # name and role_arn are required flat arguments. command is ALSO required but is a
    # nested block — same generator limitation noted elsewhere, flagged in README.
    ResourceDefinition("aws_glue_job", ("glue",), ("glue job", "glue"),
                        default_attributes={
                            "name": "replace-with-job-name",
                            "role_arn": _FAKE_IAM_ROLE_ARN,
                        }),
    ResourceDefinition("aws_athena_workgroup", ("athena",), ("athena",),
                        default_attributes={"name": "replace-with-workgroup-name"}),
]

assert len(CATALOG) >= 45, f"Catalog must cover 45+ resource types, has {len(CATALOG)}"
