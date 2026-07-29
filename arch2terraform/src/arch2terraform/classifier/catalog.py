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

from arch2terraform.generator.hcl_format import Block


@dataclass
class ResourceDefinition:
    terraform_type: str
    icon_keys: tuple[str, ...]
    label_keywords: tuple[str, ...]
    is_container: bool = False
    default_attributes: dict = field(default_factory=dict)
    # Required nested HCL blocks (e.g. aws_eks_cluster's `vpc_config { ... }`)
    # that hcl_format.py's resource_block() renders as real blocks (no `=`),
    # distinct from default_attributes' flat/map-attribute values. Keyed by
    # block name -> list of block bodies (a list because some block types can
    # repeat, e.g. aws_codepipeline's `stage`). See hcl_format.Block for how
    # to nest a block inside a block body.
    nested_blocks: dict = field(default_factory=dict)


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
# Same ARN-format-validation story as the two above, caught by a real
# `terraform validate` run (2026-07) against aws_batch_job_queue's
# compute_environment_order.compute_environment nested-block argument.
_FAKE_BATCH_COMPUTE_ENV_ARN = "arn:aws:batch:us-east-1:000000000000:compute-environment/REPLACE_WITH_COMPUTE_ENVIRONMENT_NAME"


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
    # monitoring/ebs_optimized/metadata_options/root_block_device added
    # 2026-07-24: checkov flagged every generated instance for CKV_AWS_126
    # (detailed monitoring), CKV_AWS_135 (EBS-optimized), CKV_AWS_79
    # (IMDSv2 — http_tokens defaults to "optional", i.e. IMDSv1 still
    # reachable, a real SSRF-adjacent risk), and CKV_AWS_8 (encrypted root
    # volume). All four have one unambiguous secure default with no
    # required user input, unlike e.g. instance_type/ami which are genuine
    # per-deployment choices — so these are safe to bake in rather than ask.
    ResourceDefinition("aws_instance", ("ec2",), ("ec2 instance", "ec2", "virtual machine"),
                        default_attributes={
                            "instance_type": "t3.micro",
                            "ami": "ami-00000000000000000",
                            "monitoring": True,
                            "ebs_optimized": True,
                        },
                        nested_blocks={
                            "metadata_options": [{
                                "http_tokens": "required",
                            }],
                            "root_block_device": [{
                                "encrypted": True,
                            }],
                        }),
    ResourceDefinition("aws_launch_template", ("launchtemplate",), ("launch template",)),
    # max_size/min_size are required flat arguments with no default. One of
    # availability_zones/vpc_zone_identifier is also required (a cross-field "one
    # of" constraint, same class of bug as aws_lb's subnets — not a simple
    # per-argument Required flag, so it was previously missed). launch_template is
    # a required *nested block* (or launch_configuration/mixed_instances_policy as
    # alternatives) — now emitted via nested_blocks now that the generator supports
    # real HCL blocks, not just flat attributes.
    ResourceDefinition("aws_autoscaling_group", ("autoscaling",), ("auto scaling", "asg"),
                        default_attributes={
                            "max_size": 1,
                            "min_size": 1,
                            "vpc_zone_identifier": ["subnet-00000000000000000"],
                        },
                        nested_blocks={
                            # A real `terraform validate` run (2026-07) caught that
                            # launch_template.id is format-validated client-side —
                            # it must look like a real launch template ID (`lt-`
                            # prefix + alphanumeric), not a free-text placeholder;
                            # a descriptive placeholder like the old
                            # "REPLACE_WITH_LAUNCH_TEMPLATE_ID" fails validate outright.
                            "launch_template": [{
                                "id": "lt-00000000000000000",
                                "version": "$Latest",
                            }],
                        }),
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
    # role_arn is a required flat argument. vpc_config is ALSO required and is a nested
    # HCL block (`vpc_config { subnet_ids = [...] }`), now emitted via nested_blocks.
    # A real `terraform validate` run (2026-07) caught that `name` is also required
    # with no default — the original per-argument audit only recorded role_arn for
    # this type since it was already flagged block-incomplete and got less scrutiny.
    ResourceDefinition("aws_eks_cluster", ("eks", "elastickubernetes"), ("eks", "kubernetes"),
                        default_attributes={
                            "name": "replace-with-cluster-name",
                            "role_arn": _FAKE_IAM_ROLE_ARN,
                        },
                        nested_blocks={
                            "vpc_config": [{
                                "subnet_ids": ["subnet-00000000000000000", "subnet-00000000000000001"],
                            }],
                        }),
    # name, state, priority are required flat arguments. compute_environment_order is
    # ALSO required and is a nested block (at least one entry) — now emitted via
    # nested_blocks.
    ResourceDefinition("aws_batch_job_queue", ("batch",), ("batch job", "batch queue"),
                        default_attributes={
                            "name": "replace-with-queue-name",
                            "state": "ENABLED",
                            "priority": 1,
                        },
                        nested_blocks={
                            "compute_environment_order": [{
                                "order": 1,
                                "compute_environment": _FAKE_BATCH_COMPUTE_ENV_ARN,
                            }],
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
    # (declaring hash_key's type — "S" for string, matching the "id" hash_key below) is
    # ALSO required and is a nested HCL block, now emitted via nested_blocks.
    ResourceDefinition("aws_dynamodb_table", ("dynamodb",), ("dynamodb", "nosql"),
                        default_attributes={
                            "name": "replace-with-table-name",
                            "billing_mode": "PAY_PER_REQUEST",
                            "hash_key": "id",
                        },
                        nested_blocks={
                            "attribute": [{"name": "id", "type": "S"}],
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
    # Required blocks: origin (>=1), default_cache_behavior, restrictions, and
    # viewer_certificate — now emitted via nested_blocks. default_cache_behavior
    # itself needs a nested forwarded_values block (with a further-nested cookies
    # block), and origin needs a nested custom_origin_config block (Block() marks
    # these as blocks rather than map-typed attributes at each level). enabled is
    # technically optional (defaults true) but set explicitly for clarity.
    ResourceDefinition(
        "aws_cloudfront_distribution", ("cloudfront",), ("cloudfront", "cdn"),
        default_attributes={"enabled": True},
        nested_blocks={
            "origin": [{
                "domain_name": "replace-with-origin-domain.example.com",
                "origin_id": "primary-origin",
                "custom_origin_config": Block({
                    "http_port": 80,
                    "https_port": 443,
                    "origin_protocol_policy": "https-only",
                    "origin_ssl_protocols": ["TLSv1.2"],
                }),
            }],
            "default_cache_behavior": [{
                "allowed_methods": ["GET", "HEAD"],
                "cached_methods": ["GET", "HEAD"],
                "target_origin_id": "primary-origin",
                "viewer_protocol_policy": "redirect-to-https",
                "forwarded_values": Block({
                    "query_string": False,
                    "cookies": Block({"forward": "none"}),
                }),
            }],
            "restrictions": [{
                "geo_restriction": Block({"restriction_type": "none"}),
            }],
            "viewer_certificate": [{
                "cloudfront_default_certificate": True,
            }],
        },
    ),
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
    # arguments. `user` is ALSO required and is a nested block (at least one, with
    # username/password) — now emitted via nested_blocks. AmazonMQ enforces a
    # >=12-character password, so the placeholder is deliberately long enough to
    # pass that client-side check rather than fail validate on password length.
    ResourceDefinition("aws_mq_broker", ("mq", "amazonmq"), ("amazon mq", "message broker"),
                        default_attributes={
                            "broker_name": "replace-with-broker-name",
                            "engine_type": "ActiveMQ",
                            "engine_version": "5.17.6",
                            "host_instance_type": "mq.t3.micro",
                        },
                        nested_blocks={
                            "user": [{
                                "username": "replace-with-username",
                                "password": "REPLACE_WITH_STRONG_PASSWORD_12CHARS",
                            }],
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
    # arguments. network_configuration is ALSO required and is a nested block (needs
    # exactly 2 subnet_ids across different AZs plus at least one security_group_id)
    # — now emitted via nested_blocks.
    ResourceDefinition("aws_mwaa_environment", ("mwaa", "managedairflow"), ("airflow", "mwaa"),
                        default_attributes={
                            "name": "replace-with-environment-name",
                            "execution_role_arn": _FAKE_IAM_ROLE_ARN,
                            "source_bucket_arn": _FAKE_S3_BUCKET_ARN,
                            "dag_s3_path": "dags/",
                        },
                        nested_blocks={
                            "network_configuration": [{
                                "subnet_ids": ["subnet-00000000000000000", "subnet-00000000000000001"],
                                "security_group_ids": ["sg-00000000000000000"],
                            }],
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
    # metric_name is a required flat argument. default_action is ALSO required and is a
    # nested block (`default_action { type = "ALLOW" }`) — now emitted via nested_blocks.
    # A real `terraform validate` run (2026-07) caught that `name` is also required
    # with no default — same class of miss as aws_eks_cluster above (this type was
    # already flagged block-incomplete, so its flat args got less audit scrutiny).
    ResourceDefinition("aws_waf_web_acl", ("waf",), ("waf", "web acl"),
                        default_attributes={
                            "name": "replace-with-web-acl-name",
                            "metric_name": "replaceWithMetricName",
                        },
                        nested_blocks={
                            "default_action": [{"type": "ALLOW"}],
                        }),
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
    # name and role_arn are required flat arguments. artifact_store and stage (at
    # least two: a Source stage and a downstream stage) are ALSO required and are
    # nested HCL blocks — now emitted via nested_blocks. Each stage's `action` is
    # itself a nested block (can repeat) inside the stage block, while `configuration`
    # inside an action is a genuine map-typed *attribute* (not a block) — Block() is
    # used only where the provider schema actually expects block syntax, so
    # `configuration` stays a plain dict and renders with `=` via hcl_value.
    ResourceDefinition(
        "aws_codepipeline", ("codepipeline",), ("codepipeline", "ci/cd pipeline"),
        default_attributes={
            "name": "replace-with-pipeline-name",
            "role_arn": _FAKE_IAM_ROLE_ARN,
        },
        nested_blocks={
            "artifact_store": [{
                "location": "replace-with-artifact-bucket-name",
                "type": "S3",
            }],
            "stage": [
                {
                    "name": "Source",
                    "action": [Block({
                        "name": "Source",
                        "category": "Source",
                        "owner": "AWS",
                        "provider": "S3",
                        "version": "1",
                        "output_artifacts": ["source_output"],
                        "configuration": {
                            "S3Bucket": "replace-with-artifact-bucket-name",
                            "S3ObjectKey": "replace-with-source.zip",
                        },
                    })],
                },
                {
                    "name": "Build",
                    "action": [Block({
                        "name": "Build",
                        "category": "Build",
                        "owner": "AWS",
                        "provider": "CodeBuild",
                        "version": "1",
                        "input_artifacts": ["source_output"],
                        "output_artifacts": ["build_output"],
                        "configuration": {
                            "ProjectName": "replace-with-codebuild-project-name",
                        },
                    })],
                },
            ],
        },
    ),
    # name and service_role are required flat arguments. artifacts/environment/source
    # are ALSO required and are nested blocks — now emitted via nested_blocks.
    # artifacts.type="NO_ARTIFACTS" and source.type="NO_SOURCE" are both valid enum
    # values that need no companion location, keeping the placeholder minimal while
    # still satisfying the schema.
    ResourceDefinition(
        "aws_codebuild_project", ("codebuild",), ("codebuild",),
        default_attributes={
            "name": "replace-with-project-name",
            "service_role": _FAKE_IAM_ROLE_ARN,
        },
        nested_blocks={
            "artifacts": [{"type": "NO_ARTIFACTS"}],
            "environment": [{
                "compute_type": "BUILD_GENERAL1_SMALL",
                "image": "aws/codebuild/standard:7.0",
                "type": "LINUX_CONTAINER",
            }],
            "source": [{"type": "NO_SOURCE"}],
        },
    ),
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
    # name and role_arn are required flat arguments. command is ALSO required and is a
    # nested block (script_location is its only required sub-argument; the job "type"
    # via `name` defaults to "glueetl") — now emitted via nested_blocks.
    ResourceDefinition("aws_glue_job", ("glue",), ("glue job", "glue"),
                        default_attributes={
                            "name": "replace-with-job-name",
                            "role_arn": _FAKE_IAM_ROLE_ARN,
                        },
                        nested_blocks={
                            "command": [{
                                "script_location": "s3://replace-with-scripts-bucket/replace-with-script.py",
                            }],
                        }),
    ResourceDefinition("aws_athena_workgroup", ("athena",), ("athena",),
                        default_attributes={"name": "replace-with-workgroup-name"}),
]

assert len(CATALOG) >= 45, f"Catalog must cover 45+ resource types, has {len(CATALOG)}"
