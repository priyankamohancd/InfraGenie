"""
Data models for security configuration generation
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class RuleType(Enum):
    INGRESS = "ingress"
    EGRESS = "egress"


class ResourceType(Enum):
    EC2 = "aws_instance"
    ALB = "aws_lb"
    RDS = "aws_db_instance"
    DYNAMODB = "aws_dynamodb_table"
    S3 = "aws_s3_bucket"
    LAMBDA = "aws_lambda_function"
    ELASTICACHE = "aws_elasticache_cluster"
    SQS = "aws_sqs_queue"
    SNS = "aws_sns_topic"
    SECRETS_MANAGER = "aws_secretsmanager_secret"


@dataclass
class SecurityGroupRule:
    """Represents a single security group rule (ingress or egress)"""
    rule_id: str
    type: RuleType
    protocol: str  # tcp, udp, icmp, -1 (all)
    from_port: int
    to_port: int
    source_sg_id: Optional[str] = None  # For ingress from security group
    source_cidr: Optional[str] = None  # For ingress from CIDR
    destination_sg_id: Optional[str] = None  # For egress to security group
    destination_cidr: Optional[str] = None  # For egress to CIDR
    description: str = ""
    source_resource_name: Optional[str] = None
    destination_resource_name: Optional[str] = None


@dataclass
class SecurityGroup:
    """Represents a security group"""
    name: str
    resource_name: str  # Terraform resource identifier
    vpc_id: str  # Reference to VPC
    inbound_rules: List[SecurityGroupRule] = field(default_factory=list)
    outbound_rules: List[SecurityGroupRule] = field(default_factory=list)
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def add_inbound_rule(self, rule: SecurityGroupRule):
        self.inbound_rules.append(rule)

    def add_outbound_rule(self, rule: SecurityGroupRule):
        self.outbound_rules.append(rule)


@dataclass
class IAMPolicyStatement:
    """Single statement in an IAM policy"""
    effect: str  # Allow or Deny
    actions: List[str]
    resources: List[str]
    conditions: Dict[str, any] = field(default_factory=dict)


@dataclass
class IAMPolicy:
    """Inline IAM policy"""
    name: str
    statements: List[IAMPolicyStatement] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": stmt.effect,
                    "Action": stmt.actions,
                    "Resource": stmt.resources,
                    **({"Condition": stmt.conditions} if stmt.conditions else {})
                }
                for stmt in self.statements
            ]
        }


@dataclass
class IAMRole:
    """IAM role with trust policy and attached policies"""
    name: str
    resource_name: str
    service: str  # ec2.amazonaws.com, lambda.amazonaws.com, etc.
    inline_policies: List[IAMPolicy] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    def add_policy(self, policy: IAMPolicy):
        self.inline_policies.append(policy)

    def get_trust_policy(self) -> Dict:
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": self.service},
                    "Action": "sts:AssumeRole"
                }
            ]
        }


@dataclass
class SecurityConfiguration:
    """Complete security configuration for an architecture"""
    security_groups: Dict[str, SecurityGroup] = field(default_factory=dict)
    iam_roles: Dict[str, IAMRole] = field(default_factory=dict)
    namespace: str = "default"

    def add_security_group(self, sg: SecurityGroup):
        self.security_groups[sg.resource_name] = sg

    def add_iam_role(self, role: IAMRole):
        self.iam_roles[role.resource_name] = role


# Port mapping for AWS services
SERVICE_PORTS = {
    ResourceType.RDS: {
        "mysql": 3306,
        "postgres": 5432,
        "mariadb": 3306,
        "oracle": 1521,
        "sqlserver": 1433
    },
    ResourceType.ELASTICACHE: {
        "redis": 6379,
        "memcached": 11211
    },
    ResourceType.ALB: {"http": 80, "https": 443},
    ResourceType.EC2: {"ssh": 22, "http": 80, "https": 443},
    ResourceType.SQS: {"sqs": None},  # No security group port
}


# IAM action mapping for resource connections
IAM_ACTION_MAPPING = {
    (ResourceType.EC2, ResourceType.S3): {
        "actions": ["s3:GetObject", "s3:PutObject"],
        "resource_format": "arn:aws:s3:::{bucket_name}/*"
    },
    (ResourceType.EC2, ResourceType.RDS): {
        "actions": ["rds-db:connect"],
        "resource_format": "arn:aws:rds:{region}:{account_id}:db/{db_name}"
    },
    (ResourceType.EC2, ResourceType.DYNAMODB): {
        "actions": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:Query", "dynamodb:Scan"],
        "resource_format": "arn:aws:dynamodb:{region}:{account_id}:table/{table_name}"
    },
    (ResourceType.LAMBDA, ResourceType.S3): {
        "actions": ["s3:GetObject"],
        "resource_format": "arn:aws:s3:::{bucket_name}/*"
    },
    (ResourceType.LAMBDA, ResourceType.SQS): {
        "actions": ["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage"],
        "resource_format": "arn:aws:sqs:{region}:{account_id}:{queue_name}"
    },
    (ResourceType.LAMBDA, ResourceType.SNS): {
        "actions": ["sns:Publish"],
        "resource_format": "arn:aws:sns:{region}:{account_id}:{topic_name}"
    },
    (ResourceType.EC2, ResourceType.SECRETS_MANAGER): {
        "actions": ["secretsmanager:GetSecretValue"],
        "resource_format": "arn:aws:secretsmanager:{region}:{account_id}:secret:{secret_name}-*"
    },
}
