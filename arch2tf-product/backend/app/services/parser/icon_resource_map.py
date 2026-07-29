"""
SUPERSEDED (2026-07-08): diagram_parser.py no longer calls into this module.
Classification now goes through arch2terraform.classifier.classifier +
arch2terraform.classifier.catalog (via arch2terraform_bridge.py), which has
been individually audited against the real AWS provider schema (required
arguments, ARN formats, nested HCL blocks) — the mappings below never went
through that audit and may be missing required arguments or use placeholder
styles (e.g. "# TODO: ...") that don't satisfy real `terraform validate`.
Kept in the repo for reference rather than deleted; not imported anywhere
in the active pipeline.

AWS icon-name → Terraform resource-type mapper.

The hash table keys are canonical icon pack names (e.g. "Amazon-EC2",
"Elastic-Load-Balancing"). This module converts them to the primary
Terraform resource type and default properties so the planner can generate
correct HCL without guessing.

Lookup order
------------
1. Exact match in ICON_TO_RESOURCE.
2. Keyword scan of the icon name (covers variant names like
   "Amazon-Elastic-Kubernetes-Service" mapping the same as "Amazon-EKS-Anywhere").
3. Fallback → "aws_unknown_resource" (emits a TODO comment in HCL).

Container nodes (VPC, Subnet, AZ) are handled by _container_label_to_resource,
which inspects the OCR label rather than the icon name.
"""
from __future__ import annotations
import re


# ── Exact icon-name → (aws_resource_type, default_properties) ────────────────

ICON_TO_RESOURCE: dict[str, tuple[str, dict]] = {
    # ── Compute ───────────────────────────────────────────────────────────────
    "Amazon-EC2":                           ("aws_instance",               {"instance_type": "t3.micro",  "ami": "# TODO: specify AMI ID"}),
    "Amazon-EC2-Auto-Scaling":              ("aws_autoscaling_group",      {"min_size": 1, "max_size": 3, "desired_capacity": 1}),
    "Amazon-EC2-Image-Builder":             ("aws_imagebuilder_image_pipeline", {}),
    "Auto-Scaling-group":                   ("aws_autoscaling_group",      {"min_size": 1, "max_size": 3, "desired_capacity": 1}),
    "AWS-Elastic-Beanstalk":               ("aws_elastic_beanstalk_environment", {}),
    "AWS-Fargate":                          ("aws_ecs_cluster",            {"setting": [{"name": "containerInsights", "value": "enabled"}]}),
    "AWS-Batch":                            ("aws_batch_job_definition",   {"type": "container"}),
    "AWS-Compute-Optimizer":               ("aws_resourcegroups_group",    {}),
    "EC2-instance-contents":               ("aws_instance",                {"instance_type": "t3.micro", "ami": "# TODO: specify AMI ID"}),
    "Spot-Fleet":                           ("aws_spot_fleet_request",     {}),

    # ── Load Balancing ────────────────────────────────────────────────────────
    "Elastic-Load-Balancing":              ("aws_lb",                      {"load_balancer_type": "application", "internal": False}),

    # ── Containers ────────────────────────────────────────────────────────────
    "Amazon-Elastic-Container-Service":    ("aws_ecs_cluster",             {}),
    "Amazon-ECS-Anywhere":                 ("aws_ecs_cluster",             {}),
    "Amazon-Elastic-Kubernetes-Service":   ("aws_eks_cluster",             {"version": "1.29"}),
    "Amazon-EKS-Anywhere":                 ("aws_eks_cluster",             {"version": "1.29"}),
    "Amazon-EKS-Distro":                   ("aws_eks_cluster",             {"version": "1.29"}),
    "Amazon-Elastic-Container-Registry":   ("aws_ecr_repository",         {"image_tag_mutability": "MUTABLE"}),

    # ── Serverless ────────────────────────────────────────────────────────────
    "AWS-Lambda":                           ("aws_lambda_function",        {"runtime": "python3.12", "handler": "index.handler", "memory_size": 128, "timeout": 30}),
    "Amazon-API-Gateway":                  ("aws_api_gateway_rest_api",   {}),
    "AWS-AppSync":                         ("aws_appsync_graphql_api",    {"authentication_type": "API_KEY"}),
    "AWS-Step-Functions":                  ("aws_sfn_state_machine",      {"type": "STANDARD"}),
    "AWS-Express-Workflows":               ("aws_sfn_state_machine",      {"type": "EXPRESS"}),
    "AWS-App-Runner":                      ("aws_apprunner_service",       {}),

    # ── Database ──────────────────────────────────────────────────────────────
    "Amazon-RDS":                           ("aws_db_instance",            {"engine": "postgres", "instance_class": "db.t3.micro", "allocated_storage": 20, "skip_final_snapshot": True}),
    "Amazon-Aurora":                        ("aws_rds_cluster",            {"engine": "aurora-postgresql", "engine_mode": "provisioned"}),
    "Amazon-DynamoDB":                     ("aws_dynamodb_table",         {"billing_mode": "PAY_PER_REQUEST"}),
    "Amazon-ElastiCache":                  ("aws_elasticache_cluster",    {"engine": "redis", "node_type": "cache.t3.micro", "num_cache_nodes": 1}),
    "Amazon-MemoryDB":                     ("aws_memorydb_cluster",       {"node_type": "db.t4g.small"}),
    "Amazon-Keyspaces":                    ("aws_keyspaces_table",        {}),
    "Amazon-DocumentDB":                   ("aws_docdb_cluster",          {}),
    "Amazon-Managed-Service-for-Apache-Flink": ("aws_kinesisanalyticsv2_application", {"runtime_environment": "FLINK-1_18"}),
    "Amazon-OpenSearch-Service":           ("aws_opensearch_domain",      {"engine_version": "OpenSearch_2.11"}),
    "Amazon-Timestream":                   ("aws_timestreamwrite_database", {}),
    "Amazon-QLDB":                         ("aws_qldb_ledger",            {"permissions_mode": "ALLOW_ALL"}),
    "Amazon-Neptune":                      ("aws_neptune_cluster",        {}),

    # ── Storage ───────────────────────────────────────────────────────────────
    "Amazon-Simple-Storage-Service":       ("aws_s3_bucket",              {"force_destroy": False}),
    "Amazon-S3-on-Outposts":              ("aws_s3_bucket",               {"force_destroy": False}),
    "Amazon-Simple-Storage-Service-Glacier": ("aws_s3_bucket",           {"force_destroy": False}),
    "Amazon-EFS":                          ("aws_efs_file_system",        {"performance_mode": "generalPurpose"}),
    "Amazon-Elastic-Block-Store":         ("aws_ebs_volume",              {"type": "gp3", "size": 20}),
    "Amazon-FSx":                          ("aws_fsx_lustre_file_system", {}),
    "Amazon-FSx-for-Lustre":              ("aws_fsx_lustre_file_system",  {}),
    "Amazon-FSx-for-NetApp-ONTAP":        ("aws_fsx_ontap_file_system",   {}),
    "Amazon-FSx-for-OpenZFS":             ("aws_fsx_openzfs_file_system", {}),
    "Amazon-FSx-for-WFS":                 ("aws_fsx_windows_file_system", {}),
    "Amazon-File-Cache":                   ("aws_fsx_file_cache",         {}),
    "AWS-Backup":                          ("aws_backup_plan",            {}),
    "AWS-DataSync":                        ("aws_datasync_task",          {}),
    "AWS-Storage-Gateway":                ("aws_storagegateway_gateway",  {}),
    "AWS-Elastic-Disaster-Recovery":      ("aws_drs_replication_configuration_template", {}),

    # ── Networking ────────────────────────────────────────────────────────────
    "Virtual-private-cloud-VPC":          ("aws_vpc",                     {"cidr_block": "10.0.0.0/16", "enable_dns_hostnames": True, "enable_dns_support": True}),
    "Public-subnet":                       ("aws_subnet",                  {"map_public_ip_on_launch": True}),
    "Private-subnet":                      ("aws_subnet",                  {"map_public_ip_on_launch": False}),
    "Region":                              ("aws_vpc",                     {"cidr_block": "10.0.0.0/16"}),
    "Amazon-VPC-Lattice":                 ("aws_vpclattice_service_network", {}),
    "AWS-Client-VPN":                     ("aws_ec2_client_vpn_endpoint", {}),
    "AWS-Cloud-WAN":                      ("aws_networkmanager_core_network", {}),
    "AWS-Cloud-Map":                      ("aws_service_discovery_private_dns_namespace", {}),
    "AWS-Direct-Connect":                 ("aws_dx_connection",           {"bandwidth": "1Gbps"}),

    # ── Messaging ─────────────────────────────────────────────────────────────
    "Amazon-Simple-Queue-Service":        ("aws_sqs_queue",               {}),
    "Amazon-Simple-Notification-Service": ("aws_sns_topic",               {}),
    "Amazon-Kinesis":                     ("aws_kinesis_stream",          {"shard_count": 1}),
    "Amazon-Kinesis-Data-Streams":        ("aws_kinesis_stream",          {"shard_count": 1}),
    "Amazon-Kinesis-Video-Streams":       ("aws_kinesis_video_stream",    {}),
    "Amazon-Data-Firehose":               ("aws_kinesis_firehose_delivery_stream", {}),
    "Amazon-EventBridge":                 ("aws_cloudwatch_event_bus",    {}),
    "Amazon-MQ":                          ("aws_mq_broker",               {"broker_name": "# TODO", "engine_type": "ActiveMQ", "engine_version": "5.15.14", "host_instance_type": "mq.t3.micro"}),
    "Amazon-Managed-Streaming-for-Apache-Kafka": ("aws_msk_cluster",     {}),

    # ── Security & IAM ────────────────────────────────────────────────────────
    "AWS-IAM-Identity-Center":            ("aws_iam_role",                {"assume_role_policy": "# TODO: specify trust policy"}),
    "AWS-Certificate-Manager":            ("aws_acm_certificate",        {"validation_method": "DNS"}),
    "AWS-Key-Management-Service":         ("aws_kms_key",                 {"enable_key_rotation": True}),
    "AWS-KMS":                            ("aws_kms_key",                 {"enable_key_rotation": True}),
    "AWS-CloudHSM":                       ("aws_cloudhsm_v2_cluster",    {}),
    "AWS-Secrets-Manager":                ("aws_secretsmanager_secret",  {}),
    "AWS-WAF":                            ("aws_wafv2_web_acl",          {"scope": "REGIONAL"}),
    "AWS-Firewall-Manager":              ("aws_fms_policy",               {}),
    "AWS-Shield":                         ("aws_shield_protection",       {}),
    "Amazon-Cognito":                     ("aws_cognito_user_pool",       {}),
    "Amazon-Detective":                   ("aws_detective_graph",         {}),
    "Amazon-GuardDuty":                   ("aws_guardduty_detector",     {"enable": True}),
    "Amazon-Inspector":                   ("aws_inspector2_enabler",     {}),
    "Amazon-Macie":                       ("aws_macie2_account",          {}),
    "AWS-Audit-Manager":                 ("aws_auditmanager_framework",   {}),
    "AWS-Control-Tower":                 ("aws_controltower_landing_zone", {}),
    "AWS-Security-Hub":                  ("aws_securityhub_account",     {}),
    "AWS-Artifact":                       ("aws_artifact_report_definition", {}),

    # ── CDN & DNS ─────────────────────────────────────────────────────────────
    "Amazon-CloudFront":                  ("aws_cloudfront_distribution", {"enabled": True, "price_class": "PriceClass_100"}),
    "Amazon-Route-53":                    ("aws_route53_zone",            {"comment": "Managed by Terraform"}),

    # ── Observability ─────────────────────────────────────────────────────────
    "Amazon-CloudWatch":                  ("aws_cloudwatch_log_group",   {"retention_in_days": 30}),
    "AWS-CloudTrail":                     ("aws_cloudtrail",             {"is_multi_region_trail": True, "include_global_service_events": True}),
    "AWS-Config":                         ("aws_config_configuration_recorder", {}),
    "AWS-Distro-for-OpenTelemetry":      ("aws_oam_sink",                {}),
    "Amazon-Managed-Grafana":            ("aws_grafana_workspace",       {"account_access_type": "CURRENT_ACCOUNT"}),
    "Amazon-Managed-Service-for-Prometheus": ("aws_prometheus_workspace", {}),
    "Amazon-DevOps-Guru":                ("aws_devopsguru_service_integration", {}),
    "AWS-Compute-Optimizer":             ("aws_resourcegroups_group",     {}),
    "AWS-Chatbot":                        ("aws_chatbot_slack_channel_configuration", {}),
    "AWS-User-Notifications":            ("aws_notifications_notification_hub", {}),

    # ── CI/CD & DevOps ────────────────────────────────────────────────────────
    "AWS-CodePipeline":                   ("aws_codepipeline",           {}),
    "AWS-CodeBuild":                      ("aws_codebuild_project",      {"build_timeout": 30}),
    "AWS-CodeCommit":                     ("aws_codecommit_repository",  {}),
    "AWS-CodeDeploy":                     ("aws_codedeploy_app",         {}),
    "AWS-CodeArtifact":                   ("aws_codeartifact_repository", {}),
    "AWS-CloudFormation":                 ("aws_cloudformation_stack",   {}),
    "AWS-Cloud9":                         ("aws_cloud9_environment_ec2", {}),
    "AWS-Cloud-Development-Kit":         ("aws_cloudformation_stack",    {}),
    "Amazon-CodeCatalyst":               ("aws_codecatalyst_project",     {}),
    "AWS-CloudShell":                     ("aws_cloudshell_environment",  {}),
    "AWS-Device-Farm":                    ("aws_devicefarm_project",     {}),

    # ── Analytics ─────────────────────────────────────────────────────────────
    "Amazon-Athena":                      ("aws_athena_workgroup",        {}),
    "Amazon-EMR":                         ("aws_emr_cluster",             {}),
    "Amazon-Redshift":                    ("aws_redshift_cluster",        {"node_type": "dc2.large", "number_of_nodes": 1}),
    "AWS-Glue":                           ("aws_glue_job",                {}),
    "Amazon-OpenSearch-Service":         ("aws_opensearch_domain",       {"engine_version": "OpenSearch_2.11"}),
    "AWS-Data-Exchange":                  ("aws_dataexchange_data_set",  {}),
    "Amazon-FinSpace":                    ("aws_finspace_environment",   {}),
    "Amazon-DataZone":                    ("aws_datazone_domain",         {}),
    "Amazon-AppFlow":                     ("aws_appflow_flow",            {}),
    "Amazon-CloudSearch":                 ("aws_cloudsearch_domain",     {}),

    # ── AI/ML ─────────────────────────────────────────────────────────────────
    "Amazon-SageMaker":                   ("aws_sagemaker_endpoint",     {}),
    "Amazon-Bedrock":                     ("aws_bedrock_model_invocation_logging_configuration", {}),
    "Amazon-Bedrock-AgentCore":          ("aws_bedrockagent_agent",       {}),
    "Amazon-Kendra":                      ("aws_kendra_index",            {}),
    "Amazon-Lex":                         ("aws_lex_bot",                 {}),
    "Amazon-Rekognition":                ("aws_rekognition_project",      {}),
    "Amazon-Comprehend":                  ("aws_comprehend_document_classifier", {}),
    "Amazon-Polly":                       ("aws_polly_voice",             {}),
    "Amazon-Transcribe":                 ("aws_transcribe_vocabulary",    {}),
    "Amazon-Translate":                   ("aws_translate_parallel_data", {}),
    "Amazon-Textract":                    ("aws_textract_document_analysis", {}),
    "Amazon-Fraud-Detector":             ("aws_frauddetector_detector",   {}),
    "Amazon-Forecast":                    ("aws_forecast_dataset",        {}),
    "Amazon-Personalize":                ("aws_personalizeruntime_get_recommendations", {}),
    "Amazon-Augmented-AI-A2I":           ("aws_sagemaker_human_task_ui", {}),
    "Amazon-Braket":                      ("aws_braket_quantum_task",     {}),
    "Amazon-CodeGuru":                    ("aws_codegurureviewer_repository_association", {}),

    # ── Communication & Business ──────────────────────────────────────────────
    "Amazon-Simple-Email-Service":       ("aws_ses_email_identity",      {}),
    "Amazon-Chime":                       ("aws_chime_voice_connector",  {}),
    "Amazon-Chime-SDK":                   ("aws_chimesdkmediapipelines_media_capture_pipeline", {}),
    "Amazon-Connect":                     ("aws_connect_instance",        {}),
    "Amazon-Interactive-Video-Service":  ("aws_ivs_channel",              {}),
    "Amazon-GameLift-Servers":           ("aws_gamelift_fleet",           {}),
    "Amazon-GameLift-Streams":           ("aws_gamelift_game_server_group", {}),
    "AWS-End-User-Messaging":            ("aws_pinpoint_app",             {}),

    # ── IoT ───────────────────────────────────────────────────────────────────
    "AWS-IoT-Core":                       ("aws_iot_thing",               {}),
    "AWS-IoT-Greengrass":                ("aws_greengrassv2_component_version", {}),

    # ── Migration ─────────────────────────────────────────────────────────────
    "AWS-Database-Migration-Service":    ("aws_dms_replication_instance", {}),
    "AWS-Application-Migration-Service": ("aws_mgn_replication_configuration_template", {}),
    "AWS-Application-Discovery-Service": ("aws_applicationdiscovery_application", {}),

    # ── Management ────────────────────────────────────────────────────────────
    "AWS-Organizations":                  ("aws_organizations_organization", {}),
    "AWS-Budgets":                        ("aws_budgets_budget",          {}),
    "AWS-Cost-Explorer":                  ("aws_ce_cost_category",        {}),
    "AWS-Billing-Conductor":             ("aws_billingconductor_billing_group", {}),
    "AWS-Auto-Scaling":                   ("aws_autoscaling_group",       {"min_size": 1, "max_size": 3, "desired_capacity": 1}),
    "AWS-Application-Auto-Scaling":     ("aws_appautoscaling_target",    {}),
    "AWS-Systems-Manager":               ("aws_ssm_parameter",            {}),
    "AWS-License-Manager":               ("aws_licensemanager_association", {}),
    "AWS-Service-Catalog":               ("aws_servicecatalog_portfolio", {}),
    "AWS-Trusted-Advisor":               ("aws_trustedadvisor_enrollment_status", {}),
    "AWS-Health-Dashboard":              ("aws_health_event",             {}),
    "AWS-Resilience-Hub":                ("aws_resiliencehub_app",        {}),
    "AWS-Fault-Injection-Service":       ("aws_fis_experiment_template",  {}),

    # ── Lightweight resources ────────────────────────────────────────────────-
    "Amazon-Lightsail":                   ("aws_lightsail_instance",      {"bundle_id": "nano_3_0", "blueprint_id": "# TODO"}),
    "AWS-Amplify":                        ("aws_amplify_app",             {}),
    "Amazon-Location-Service":           ("aws_location_place_index",     {}),
    "Amazon-Managed-Blockchain":         ("aws_managedblockchain_network", {}),
}


# ── Keyword fallback ─────────────────────────────────────────────────────────
# Ordered list of (substring_lower, aws_resource_type, default_properties)
_KEYWORD_FALLBACKS: list[tuple[str, str, dict]] = [
    ("elastic-load-balanc", "aws_lb",                        {"load_balancer_type": "application"}),
    ("ec2",                  "aws_instance",                   {"instance_type": "t3.micro", "ami": "# TODO: specify AMI ID"}),
    ("lambda",               "aws_lambda_function",            {"runtime": "python3.12", "handler": "index.handler"}),
    ("eks",                  "aws_eks_cluster",                {"version": "1.29"}),
    ("ecs",                  "aws_ecs_cluster",                {}),
    ("ecr",                  "aws_ecr_repository",             {"image_tag_mutability": "MUTABLE"}),
    ("fargate",              "aws_ecs_cluster",                {}),
    ("rds",                  "aws_db_instance",                {"engine": "postgres", "instance_class": "db.t3.micro", "allocated_storage": 20, "skip_final_snapshot": True}),
    ("aurora",               "aws_rds_cluster",               {"engine": "aurora-postgresql"}),
    ("dynamodb",             "aws_dynamodb_table",             {"billing_mode": "PAY_PER_REQUEST"}),
    ("elasticache",          "aws_elasticache_cluster",        {"engine": "redis", "node_type": "cache.t3.micro", "num_cache_nodes": 1}),
    ("simple-storage",       "aws_s3_bucket",                  {}),
    ("s3",                   "aws_s3_bucket",                  {}),
    ("cloudfront",           "aws_cloudfront_distribution",    {"enabled": True}),
    ("route-53",             "aws_route53_zone",               {}),
    ("route53",              "aws_route53_zone",               {}),
    ("simple-notification",  "aws_sns_topic",                  {}),
    ("sns",                  "aws_sns_topic",                  {}),
    ("simple-queue",         "aws_sqs_queue",                  {}),
    ("sqs",                  "aws_sqs_queue",                  {}),
    ("kinesis",              "aws_kinesis_stream",             {"shard_count": 1}),
    ("firehose",             "aws_kinesis_firehose_delivery_stream", {}),
    ("eventbridge",          "aws_cloudwatch_event_bus",       {}),
    ("api-gateway",          "aws_api_gateway_rest_api",       {}),
    ("step-function",        "aws_sfn_state_machine",          {"type": "STANDARD"}),
    ("cloudwatch",           "aws_cloudwatch_log_group",       {"retention_in_days": 30}),
    ("cloudtrail",           "aws_cloudtrail",                 {}),
    ("kms",                  "aws_kms_key",                    {"enable_key_rotation": True}),
    ("key-management",       "aws_kms_key",                    {"enable_key_rotation": True}),
    ("secrets-manager",      "aws_secretsmanager_secret",      {}),
    ("certificate-manager",  "aws_acm_certificate",            {"validation_method": "DNS"}),
    ("cognito",              "aws_cognito_user_pool",           {}),
    ("guardduty",            "aws_guardduty_detector",         {"enable": True}),
    ("waf",                  "aws_wafv2_web_acl",              {"scope": "REGIONAL"}),
    ("iam",                  "aws_iam_role",                   {}),
    ("codepipeline",         "aws_codepipeline",               {}),
    ("codebuild",            "aws_codebuild_project",          {}),
    ("codecommit",           "aws_codecommit_repository",      {}),
    ("vpc",                  "aws_vpc",                        {"cidr_block": "10.0.0.0/16"}),
    ("subnet",               "aws_subnet",                     {}),
    ("efs",                  "aws_efs_file_system",            {}),
    ("fsx",                  "aws_fsx_lustre_file_system",     {}),
    ("glue",                 "aws_glue_job",                   {}),
    ("athena",               "aws_athena_workgroup",           {}),
    ("sagemaker",            "aws_sagemaker_endpoint",         {}),
    ("bedrock",              "aws_bedrock_model_invocation_logging_configuration", {}),
    ("auto-scaling",         "aws_autoscaling_group",          {"min_size": 1, "max_size": 3}),
    ("batch",                "aws_batch_job_definition",       {"type": "container"}),
    ("mq",                   "aws_mq_broker",                  {"engine_type": "ActiveMQ"}),
    ("kafka",                "aws_msk_cluster",                {}),
    ("ses",                  "aws_ses_email_identity",         {}),
    ("email",                "aws_ses_email_identity",         {}),
    ("direct-connect",       "aws_dx_connection",              {"bandwidth": "1Gbps"}),
    ("cloudformation",       "aws_cloudformation_stack",       {}),
    ("backup",               "aws_backup_plan",                {}),
]


# ── Container-label → resource type ──────────────────────────────────────────

def container_label_to_resource(label: str) -> tuple[str, dict] | None:
    """
    Map an OCR-derived container label to a Terraform resource type.
    Returns None for AZ/Region labels which are not TF-managed resources.
    """
    lower = label.lower()

    if "availability zone" in lower or "az" == lower.strip():
        return None  # AZ is a placement constraint, not a resource

    if "subnet" in lower:
        is_public = "public" in lower
        return ("aws_subnet", {
            "cidr_block":                "# TODO: specify CIDR",
            "availability_zone":         "# TODO: specify AZ",
            "map_public_ip_on_launch":   is_public,
        })

    if "vpc" in lower or "virtual private cloud" in lower:
        return ("aws_vpc", {
            "cidr_block":            "10.0.0.0/16",
            "enable_dns_hostnames":  True,
            "enable_dns_support":    True,
        })

    if "region" in lower:
        return None  # Region is an AWS global concept, not a TF resource block

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def icon_name_to_resource(icon_name: str) -> tuple[str, dict]:
    """
    Convert an icon pack name to (aws_resource_type, default_properties).

    Parameters
    ----------
    icon_name : str
        Canonical icon name from the hash table (e.g. "Amazon-EC2").

    Returns
    -------
    (aws_resource_type, default_properties)
        If no mapping is found, returns ("aws_unknown_resource", {}).
    """
    # 1. Exact match
    if icon_name in ICON_TO_RESOURCE:
        rt, props = ICON_TO_RESOURCE[icon_name]
        return rt, dict(props)

    # 2. Keyword scan
    lower = icon_name.lower()
    for kw, rt, props in _KEYWORD_FALLBACKS:
        if kw in lower:
            return rt, dict(props)

    # 3. Fallback
    return "aws_unknown_resource", {}


def slugify(text: str) -> str:
    """
    Convert a human label to a valid Terraform identifier.

    e.g. "Web EC2 Instance" → "web_ec2_instance"
         "Amazon-RDS (Primary)" → "amazon_rds_primary"
    """
    slug = re.sub(r"[^\w]+", "_", text.lower()).strip("_")
    # Terraform names must start with a letter or underscore
    if slug and slug[0].isdigit():
        slug = "r_" + slug
    return slug or "resource"
