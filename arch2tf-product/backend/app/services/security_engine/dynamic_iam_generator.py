"""
Dynamic IAM Policy Generator - Works with ANY AWS service combination
Comprehensive, extensible, service-agnostic approach
"""
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field


@dataclass
class ServiceDefinition:
    """Defines a single AWS service"""
    name: str  # "S3", "RDS", "DynamoDB"
    resource_type: str  # "aws_s3_bucket", "aws_db_instance"
    arn_format: str  # "arn:aws:s3:::{bucket_name}"
    default_actions: Dict[str, List[str]] = field(default_factory=dict)

    # Possible actions by operation type
    read_actions: List[str] = field(default_factory=list)
    write_actions: List[str] = field(default_factory=list)
    manage_actions: List[str] = field(default_factory=list)


class DynamicActionRegistry:
    """
    Registry of AWS services and their possible IAM actions.
    Designed to be extended with new services easily.
    """

    def __init__(self):
        self.services: Dict[str, ServiceDefinition] = {}
        self.service_by_type: Dict[str, str] = {}
        self._initialize_aws_services()

    def _initialize_aws_services(self):
        """Initialize comprehensive AWS service definitions"""

        # S3
        self.register_service(ServiceDefinition(
            name="S3",
            resource_type="aws_s3_bucket",
            arn_format="arn:aws:s3:::{bucket_name}/*",
            read_actions=["s3:GetObject", "s3:GetObjectVersion"],
            write_actions=["s3:PutObject", "s3:PutObjectAcl"],
            manage_actions=["s3:DeleteObject", "s3:ListBucket"]
        ))

        # RDS
        self.register_service(ServiceDefinition(
            name="RDS",
            resource_type="aws_db_instance",
            arn_format="arn:aws:rds:{region}:{account_id}:db/{db_name}",
            read_actions=["rds-db:connect"],
            write_actions=["rds:CreateDBSnapshot"],
            manage_actions=["rds:ModifyDBInstance", "rds:DeleteDBInstance"]
        ))

        # DynamoDB
        self.register_service(ServiceDefinition(
            name="DynamoDB",
            resource_type="aws_dynamodb_table",
            arn_format="arn:aws:dynamodb:{region}:{account_id}:table/{table_name}",
            read_actions=["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"],
            write_actions=["dynamodb:PutItem", "dynamodb:UpdateItem"],
            manage_actions=["dynamodb:DeleteItem", "dynamodb:DeleteTable"]
        ))

        # SQS
        self.register_service(ServiceDefinition(
            name="SQS",
            resource_type="aws_sqs_queue",
            arn_format="arn:aws:sqs:{region}:{account_id}:{queue_name}",
            read_actions=["sqs:ReceiveMessage", "sqs:GetQueueAttributes"],
            write_actions=["sqs:SendMessage"],
            manage_actions=["sqs:DeleteMessage", "sqs:PurgeQueue"]
        ))

        # SNS
        self.register_service(ServiceDefinition(
            name="SNS",
            resource_type="aws_sns_topic",
            arn_format="arn:aws:sns:{region}:{account_id}:{topic_name}",
            read_actions=["sns:GetTopicAttributes"],
            write_actions=["sns:Publish"],
            manage_actions=["sns:DeleteTopic"]
        ))

        # Lambda
        self.register_service(ServiceDefinition(
            name="Lambda",
            resource_type="aws_lambda_function",
            arn_format="arn:aws:lambda:{region}:{account_id}:function:{function_name}",
            read_actions=["lambda:GetFunction"],
            write_actions=["lambda:InvokeFunction"],
            manage_actions=["lambda:DeleteFunction"]
        ))

        # ElastiCache
        self.register_service(ServiceDefinition(
            name="ElastiCache",
            resource_type="aws_elasticache_cluster",
            arn_format="arn:aws:elasticache:{region}:{account_id}:cluster:{cluster_name}",
            read_actions=[],  # No IAM-based access control
            write_actions=[],
            manage_actions=[]
        ))

        # Secrets Manager
        self.register_service(ServiceDefinition(
            name="Secrets Manager",
            resource_type="aws_secretsmanager_secret",
            arn_format="arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}",
            read_actions=["secretsmanager:GetSecretValue"],
            write_actions=["secretsmanager:PutSecretValue"],
            manage_actions=["secretsmanager:DeleteSecret"]
        ))

        # Parameter Store (Systems Manager)
        self.register_service(ServiceDefinition(
            name="Parameter Store",
            resource_type="aws_ssm_parameter",
            arn_format="arn:aws:ssm:{region}:{account_id}:parameter{parameter_name}",
            read_actions=["ssm:GetParameter", "ssm:GetParameters"],
            write_actions=["ssm:PutParameter"],
            manage_actions=["ssm:DeleteParameter"]
        ))

        # KMS
        self.register_service(ServiceDefinition(
            name="KMS",
            resource_type="aws_kms_key",
            arn_format="arn:aws:kms:{region}:{account_id}:key/{key_id}",
            read_actions=["kms:Decrypt"],
            write_actions=["kms:Encrypt", "kms:GenerateDataKey"],
            manage_actions=["kms:ScheduleKeyDeletion"]
        ))

        # CloudWatch Logs
        self.register_service(ServiceDefinition(
            name="CloudWatch Logs",
            resource_type="aws_cloudwatch_log_group",
            arn_format="arn:aws:logs:{region}:{account_id}:log-group:{log_group_name}",
            read_actions=["logs:GetLogEvents"],
            write_actions=["logs:PutLogEvents", "logs:CreateLogStream"],
            manage_actions=["logs:DeleteLogGroup"]
        ))

        # SageMaker
        self.register_service(ServiceDefinition(
            name="SageMaker",
            resource_type="aws_sagemaker_endpoint",
            arn_format="arn:aws:sagemaker:{region}:{account_id}:endpoint/{endpoint_name}",
            read_actions=["sagemaker:InvokeEndpoint"],
            write_actions=[],
            manage_actions=[]
        ))

        # Kinesis
        self.register_service(ServiceDefinition(
            name="Kinesis",
            resource_type="aws_kinesis_stream",
            arn_format="arn:aws:kinesis:{region}:{account_id}:stream/{stream_name}",
            read_actions=["kinesis:GetRecords", "kinesis:GetShardIterator"],
            write_actions=["kinesis:PutRecord", "kinesis:PutRecords"],
            manage_actions=["kinesis:DeleteStream"]
        ))

        # API Gateway
        self.register_service(ServiceDefinition(
            name="API Gateway",
            resource_type="aws_api_gateway_rest_api",
            arn_format="arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*",
            read_actions=["execute-api:Invoke"],
            write_actions=[],
            manage_actions=[]
        ))

        # EventBridge
        self.register_service(ServiceDefinition(
            name="EventBridge",
            resource_type="aws_cloudwatch_event_rule",
            arn_format="arn:aws:events:{region}:{account_id}:rule/{rule_name}",
            read_actions=["events:DescribeRule"],
            write_actions=["events:PutEvents"],
            manage_actions=["events:DeleteRule"]
        ))

        # Step Functions
        self.register_service(ServiceDefinition(
            name="Step Functions",
            resource_type="aws_sfn_state_machine",
            arn_format="arn:aws:states:{region}:{account_id}:stateMachine:{state_machine_name}",
            read_actions=["states:DescribeExecution", "states:GetExecutionHistory"],
            write_actions=["states:StartExecution"],
            manage_actions=[]
        ))

        # ElasticSearch
        self.register_service(ServiceDefinition(
            name="Elasticsearch",
            resource_type="aws_elasticsearch_domain",
            arn_format="arn:aws:es:{region}:{account_id}:domain/{domain_name}/*",
            read_actions=["es:ESHttpGet"],
            write_actions=["es:ESHttpPut", "es:ESHttpPost"],
            manage_actions=["es:ESHttpDelete"]
        ))

        # Redshift
        self.register_service(ServiceDefinition(
            name="Redshift",
            resource_type="aws_redshift_cluster",
            arn_format="arn:aws:redshift:{region}:{account_id}:cluster:{cluster_name}",
            read_actions=["redshift:DescribeClusters"],
            write_actions=[],
            manage_actions=[]
        ))

        # DocumentDB
        self.register_service(ServiceDefinition(
            name="DocumentDB",
            resource_type="aws_docdb_cluster",
            arn_format="arn:aws:rds:{region}:{account_id}:cluster:{cluster_name}",
            read_actions=["rds-db:connect"],
            write_actions=[],
            manage_actions=[]
        ))

        # Neptune
        self.register_service(ServiceDefinition(
            name="Neptune",
            resource_type="aws_neptune_cluster",
            arn_format="arn:aws:rds:{region}:{account_id}:cluster:{cluster_name}",
            read_actions=["rds-db:connect"],
            write_actions=[],
            manage_actions=[]
        ))

    def register_service(self, service: ServiceDefinition):
        """Register a new AWS service"""
        self.services[service.name] = service
        self.service_by_type[service.resource_type] = service.name

    def get_service(self, name_or_type: str) -> Optional[ServiceDefinition]:
        """Get service by name or resource type"""
        if name_or_type in self.services:
            return self.services[name_or_type]
        if name_or_type in self.service_by_type:
            return self.services[self.service_by_type[name_or_type]]
        return None

    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self.services.keys())


class EdgeOperationInferencer:
    """
    Infers operation type (read, write, manage) from edge label and context
    """

    def __init__(self):
        # Edge label to operation type mapping
        self.label_operations = {
            # Read operations
            "read": "read",
            "get": "read",
            "query": "read",
            "scan": "read",
            "describe": "read",
            "list": "read",
            "fetch": "read",
            "retrieve": "read",
            "select": "read",

            # Write operations
            "write": "write",
            "put": "write",
            "post": "write",
            "send": "write",
            "publish": "write",
            "update": "write",
            "insert": "write",
            "upsert": "write",

            # Manage operations
            "delete": "manage",
            "destroy": "manage",
            "drop": "manage",
            "remove": "manage",
            "purge": "manage",
            "manage": "manage",

            # Bidirectional (assume read+write)
            "sync": "read_write",
            "replicate": "read_write",
            "backup": "read",  # Usually read source, write target
            "restore": "write",
        }

    def infer_operation(self, edge_label: str) -> str:
        """
        Infer operation type from edge label

        Returns: "read", "write", "manage", "read_write", or "custom"
        """
        if not edge_label:
            return "read_write"  # Default: both read and write

        label_lower = edge_label.lower()

        # Check exact matches first
        if label_lower in self.label_operations:
            return self.label_operations[label_lower]

        # Check if label contains operation keywords
        for keyword, operation in self.label_operations.items():
            if keyword in label_lower:
                return operation

        # Default: assume read (safer)
        return "read_write"


class DynamicIAMPolicyGenerator:
    """
    Generates IAM policies dynamically for ANY AWS service combination.
    """

    # Compute/network resource types that are never reached via an AWS-API
    # IAM policy (they're plain network traffic, handled by security groups
    # instead). An edge into one of these must never produce a policy - not
    # even a NEEDS_REVIEW placeholder - since there's nothing to review.
    NON_IAM_TARGET_TYPES = {
        "aws_instance",
        "aws_lb",
        "aws_lb_target_group",
        "aws_elasticache_cluster",
        "aws_nat_gateway",
        "aws_internet_gateway",
        "aws_vpc",
        "aws_subnet",
        "aws_route_table",
        "aws_security_group",
    }

    def __init__(self):
        self.service_registry = DynamicActionRegistry()
        self.operation_inferencer = EdgeOperationInferencer()
        self.account_id = "${data.aws_caller_identity.current.account_id}"
        self.region = "${data.aws_region.current.name}"

    def generate_policy_for_edge(self, from_node: Dict, to_node: Dict,
                                 edge: Dict) -> Optional[Dict]:
        """
        Generate IAM policy for a single edge connection

        Works with ANY source and target service combination
        """
        to_type = to_node.get('type')
        to_label = to_node.get('label', '')
        to_metadata = to_node.get('metadata', {})
        edge_label = edge.get('label', '')
        # Vision-LLM's own semantic read of this connection (see
        # security_bridge.py's _build_resource_graph) — "" for every edge
        # from the classical (non-Vision-LLM) pipeline, in which case this
        # falls through to the existing label-keyword inference exactly as
        # before. Added 2026-07-31 per explicit follow-up: the model has
        # actual diagram context (both endpoints, surrounding architecture)
        # that a label-keyword match on the edge's own caption text alone
        # can't see.
        operation_hint = str(edge.get('operation_hint') or '').strip().lower()

        # A model-confirmed "network" connection (plain traffic, not an
        # AWS-API call) is a strictly better signal than NON_IAM_TARGET_TYPES
        # below — it's a judgment about THIS specific connection using real
        # context, not a blanket rule keyed only on the target's type. Still
        # keep NON_IAM_TARGET_TYPES as the fallback for the classical
        # pipeline (operation_hint == "" there) and as a defensive backstop
        # even when Vision-LLM is enabled but didn't classify this edge.
        if operation_hint == "network":
            return None

        # Plain compute/network traffic (e.g. ALB -> EC2, EC2 -> EC2) is
        # handled by security groups, not IAM - skip silently rather than
        # emitting a NEEDS_REVIEW placeholder policy for it.
        if to_type in self.NON_IAM_TARGET_TYPES:
            return None

        # Step 1: Lookup target service
        target_service = self.service_registry.get_service(to_type)
        if not target_service:
            return self._handle_unknown_service(to_type, to_label, edge_label)

        # Step 2: Infer operation type — prefer the model's own hint
        # (read/write/manage/read_write) when it gave one; only fall back to
        # keyword-guessing the edge's label text when it didn't.
        if operation_hint in ("read", "write", "manage", "read_write"):
            operation = operation_hint
        else:
            operation = self.operation_inferencer.infer_operation(edge_label)

        # Step 3: Get required actions based on operation
        actions = self._get_actions_for_operation(target_service, operation)

        if not actions:
            return None  # Service doesn't support IAM (e.g., ElastiCache)

        # Step 4: Generate ARN
        arn = self._generate_arn(target_service, to_label, to_metadata)

        # Step 5: Create policy
        policy_name = self._generate_policy_name(to_label, operation)

        return {
            "name": policy_name,
            "service": target_service.name,
            "actions": actions,
            "resource_arn": arn,
            "description": f"{operation.title()} access to {to_label}"
        }

    def _get_actions_for_operation(self, service: ServiceDefinition,
                                   operation: str) -> List[str]:
        """Get IAM actions based on operation type"""
        if operation == "read":
            return service.read_actions
        elif operation == "write":
            return service.write_actions
        elif operation == "manage":
            return service.manage_actions
        elif operation == "read_write":
            return service.read_actions + service.write_actions
        return []

    def _generate_arn(self, service: ServiceDefinition, resource_label: str,
                      metadata: Dict) -> str:
        """Generate ARN for a resource"""
        arn = service.arn_format

        # Replace placeholders with actual values
        resource_name = metadata.get('name', resource_label.lower().replace(' ', '-'))
        arn = arn.replace("{bucket_name}", resource_name)
        arn = arn.replace("{db_name}", resource_name)
        arn = arn.replace("{table_name}", resource_name)
        arn = arn.replace("{queue_name}", resource_name)
        arn = arn.replace("{topic_name}", resource_name)
        arn = arn.replace("{stream_name}", resource_name)
        arn = arn.replace("{function_name}", resource_name)
        arn = arn.replace("{secret_name}", resource_name)
        arn = arn.replace("{parameter_name}", resource_name)
        arn = arn.replace("{log_group_name}", resource_name)
        arn = arn.replace("{cluster_name}", resource_name)
        arn = arn.replace("{domain_name}", resource_name)
        arn = arn.replace("{rule_name}", resource_name)
        arn = arn.replace("{state_machine_name}", resource_name)

        # Replace region and account ID
        arn = arn.replace("{region}", self.region)
        arn = arn.replace("{account_id}", self.account_id)

        return arn

    def _generate_policy_name(self, resource_label: str, operation: str) -> str:
        """Generate policy name"""
        clean_label = resource_label.lower().replace(' ', '-')
        return f"{clean_label}-{operation}-access"

    def _handle_unknown_service(self, resource_type: str, resource_label: str,
                               edge_label: str) -> Optional[Dict]:
        """
        Handle unknown AWS services gracefully

        Return a template that user can customize
        """
        print(f"⚠️  Unknown service type: {resource_type}")
        print(f"   Resource: {resource_label}")
        print(f"   Edge: {edge_label}")
        print()

        # Return generic template
        operation = self.operation_inferencer.infer_operation(edge_label)
        return {
            "name": f"{resource_label.lower()}-{operation}-access",
            "service": resource_type,
            "actions": ["ACTION_PLACEHOLDER"],  # User must fill in
            "resource_arn": "arn:aws:PLACEHOLDER",
            "description": f"{operation.title()} access to {resource_label}",
            "status": "NEEDS_REVIEW"
        }


# AWS-managed policy ARNs that a resource type needs attached to its
# generated role regardless of what the diagram's edges imply — the
# baseline permissions AWS itself requires the service to function at all,
# as opposed to the edge-derived custom policies above (which express what
# the diagram says THIS specific role should additionally be allowed to
# call). Added 2026-07-31 generalizing a fix originally scoped to just
# aws_eks_cluster (her explicit follow-up: "this shouldn't happen just in
# case of eks ... whenever any resource requires a role then they should
# just be created") — every entry here is a real, AWS-documented minimum
# requirement for that resource type to be created/operate at all, not a
# guessed convenience default. Deliberately conservative: only add an entry
# here when there's one unambiguous AWS-managed policy that's genuinely
# mandatory (matches the same bar catalog.py's default_attributes uses for
# "safe to bake in without asking") — MWAA/CodePipeline/CodeBuild/Glue are
# still made IAM_ROLE_ELIGIBLE (see complete_security_orchestrator.py) so
# they get a real role + trust policy + any edge-derived policies, but no
# entry is added here for them since their real IAM requirements are
# genuinely diagram/config-dependent (e.g. CodeBuild's service role depends
# entirely on what it builds) rather than one fixed managed policy.
MANDATORY_MANAGED_POLICY_ARNS: Dict[str, List[str]] = {
    "aws_eks_cluster": ["arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"],
    # An EKS worker node needs these three unconditionally to function at
    # all (join the cluster, run the CNI plugin, pull container images) —
    # not diagram/edge-dependent, same "real AWS-documented minimum"
    # standard as the cluster's own policy above.
    "aws_eks_node_group": [
        "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
        "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    ],
}


class UniversalPolicyBuilder:
    """
    Build complete IAM role with policies for ANY architecture
    """

    def __init__(self):
        self.generator = DynamicIAMPolicyGenerator()

    def generate_policies_for_compute_resource(self, compute_node: Dict,
                                               edges: List[Dict],
                                               all_nodes: Dict) -> Dict:
        """
        Generate all policies for a compute/service resource (EC2, Lambda,
        EKS cluster, Glue job, etc.)

        Works with ANY number of outbound connections — including zero, for
        resource types AWS requires a role for unconditionally (see
        MANDATORY_MANAGED_POLICY_ARNS and
        complete_security_orchestrator.MANDATORY_ROLE_TYPES), where the role
        still needs to exist even with no diagram-derived permissions to add
        to it.
        """
        compute_label = compute_node.get('label', '')
        compute_type = compute_node.get('type', '')

        role_name = f"role-{compute_label.lower().replace(' ', '-')}"
        policies = []

        # Find all outbound edges from this resource
        outbound_edges = [e for e in edges if e.get('from') == compute_node.get('id')]

        for edge in outbound_edges:
            to_id = edge.get('to')
            to_node = all_nodes.get(to_id)

            if not to_node:
                continue

            # Generate policy for this connection
            policy = self.generator.generate_policy_for_edge(
                compute_node, to_node, edge
            )

            if policy:
                policies.append(policy)

        managed_policy_arns = list(MANDATORY_MANAGED_POLICY_ARNS.get(compute_type, []))

        return {
            "role_name": role_name,
            "service_principal": self._get_service_principal(compute_type),
            "policies": policies,
            "managed_policy_arns": managed_policy_arns,
            "resource_count": len(policies),
        }

    def _get_service_principal(self, resource_type: str) -> str:
        """Get AWS service principal for assume role policy"""
        service_principals = {
            "aws_instance": "ec2.amazonaws.com",
            "aws_lambda_function": "lambda.amazonaws.com",
            "aws_ecs_task_definition": "ecs-tasks.amazonaws.com",
            "aws_batch_job_definition": "batch.amazonaws.com",
            # Was "aws_states_state_machine" — a real, previously-silent bug
            # (found 2026-07-31): the actual Terraform/AWS provider resource
            # type is aws_sfn_state_machine (see arch2terraform's catalog.py
            # and complete_security_orchestrator.IAM_ROLE_ELIGIBLE_TYPES,
            # both of which already use the correct name) — this dict's key
            # never matched, so every Step Functions role silently fell back
            # to "ec2.amazonaws.com" below instead of "states.amazonaws.com".
            "aws_sfn_state_machine": "states.amazonaws.com",
            "aws_eks_cluster": "eks.amazonaws.com",
            "aws_eks_node_group": "ec2.amazonaws.com",
            "aws_mwaa_environment": "airflow.amazonaws.com",
            "aws_codepipeline": "codepipeline.amazonaws.com",
            "aws_codebuild_project": "codebuild.amazonaws.com",
            "aws_glue_job": "glue.amazonaws.com",
        }
        return service_principals.get(resource_type, "ec2.amazonaws.com")


# Demo service registry output
def print_available_services():
    """Print all available AWS services"""
    registry = DynamicActionRegistry()
    print("=" * 80)
    print("DYNAMICALLY SUPPORTED AWS SERVICES")
    print("=" * 80)
    print()

    for service_name in sorted(registry.list_services()):
        service = registry.get_service(service_name)
        print(f"{service_name}:")
        print(f"  Resource Type: {service.resource_type}")
        print(f"  ARN Format: {service.arn_format}")
        print(f"  Read Actions: {len(service.read_actions)}")
        print(f"  Write Actions: {len(service.write_actions)}")
        print(f"  Manage Actions: {len(service.manage_actions)}")
        print()
