"""
IAM policy generation for least-privilege access
"""
from typing import Dict, List, Optional, Set
from models import (
    IAMRole, IAMPolicy, IAMPolicyStatement,
    ResourceType, IAM_ACTION_MAPPING
)


class IAMPolicyGenerator:
    """Generates least-privilege IAM policies from diagram edges"""

    def __init__(self, namespace: str = "default", account_id: str = "${data.aws_caller_identity.current.account_id}",
                 region: str = "${data.aws_region.current.name}"):
        self.namespace = namespace
        self.account_id = account_id
        self.region = region
        self.roles: Dict[str, IAMRole] = {}

    def generate_from_diagram(self, resource_graph: Dict) -> Dict[str, IAMRole]:
        """
        Generate IAM roles and policies from resource graph

        Args:
            resource_graph: {
                nodes: [{id, label, type, metadata}],
                edges: [{from, to, type, label}]
            }
        """
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}
        edges = resource_graph.get('edges', [])

        # Step 1: Create IAM role for each compute resource
        self._create_roles_for_compute(nodes)

        # Step 2: Generate policies based on edges
        self._generate_policies_from_edges(edges, nodes)

        return self.roles

    def _create_roles_for_compute(self, nodes: Dict):
        """Create IAM role for EC2 instances and Lambda functions"""
        for node_id, node in nodes.items():
            resource_type = node.get('type')
            resource_label = node.get('label', node_id)

            # Create role for compute resources
            if resource_type == ResourceType.EC2.value:
                role = self._create_ec2_role(resource_label)
                self.roles[role.resource_name] = role
            elif resource_type == ResourceType.LAMBDA.value:
                role = self._create_lambda_role(resource_label)
                self.roles[role.resource_name] = role

    def _create_ec2_role(self, resource_label: str) -> IAMRole:
        """Create EC2 instance role"""
        role_name = f"{self.namespace}-{resource_label.lower().replace(' ', '-')}-role"
        resource_name = f"{self.namespace}_{resource_label.lower().replace(' ', '_')}_role"

        return IAMRole(
            name=role_name,
            resource_name=resource_name,
            service="ec2.amazonaws.com",
            tags={
                "Name": role_name,
                "ManagedBy": "terraform-accelerators",
                "Resource": resource_label
            }
        )

    def _create_lambda_role(self, resource_label: str) -> IAMRole:
        """Create Lambda execution role"""
        role_name = f"{self.namespace}-{resource_label.lower().replace(' ', '-')}-role"
        resource_name = f"{self.namespace}_{resource_label.lower().replace(' ', '_')}_role"

        return IAMRole(
            name=role_name,
            resource_name=resource_name,
            service="lambda.amazonaws.com",
            tags={
                "Name": role_name,
                "ManagedBy": "terraform-accelerators",
                "Resource": resource_label
            }
        )

    def _generate_policies_from_edges(self, edges: List[Dict], nodes: Dict):
        """Generate IAM policies based on edges (connections)"""
        for edge in edges:
            from_node_id = edge.get('from')
            to_node_id = edge.get('to')

            from_node = nodes.get(from_node_id, {})
            to_node = nodes.get(to_node_id, {})

            from_type = from_node.get('type')
            to_type = to_node.get('type')
            to_label = to_node.get('label', to_node_id)

            # Get source role (only for compute resources)
            from_role_key = self._get_role_resource_name(from_node.get('label', from_node_id), from_type)
            from_role = self.roles.get(from_role_key)

            if not from_role:
                continue

            # Generate policy for this edge
            policy = self._create_policy_for_edge(from_type, to_type, to_label, to_node)

            if policy:
                from_role.add_policy(policy)

    def _create_policy_for_edge(self, from_type: str, to_type: str,
                                to_label: str, to_node: Dict) -> Optional[IAMPolicy]:
        """Create IAM policy for a source→target connection"""

        # Look up in mapping
        resource_type_from = self._normalize_resource_type(from_type)
        resource_type_to = self._normalize_resource_type(to_type)

        mapping_key = (resource_type_from, resource_type_to)

        if mapping_key not in IAM_ACTION_MAPPING:
            return None

        mapping = IAM_ACTION_MAPPING[mapping_key]
        actions = mapping.get('actions', [])
        resource_format = mapping.get('resource_format', '')

        # Generate ARN based on target resource
        arn = self._generate_arn(resource_type_to, to_label, resource_format)

        policy_name = f"{to_label.lower().replace(' ', '-')}-access"

        statement = IAMPolicyStatement(
            effect="Allow",
            actions=actions,
            resources=[arn]
        )

        policy = IAMPolicy(name=policy_name)
        policy.statements.append(statement)

        return policy

    def _generate_arn(self, resource_type: ResourceType, resource_label: str, format_str: str) -> str:
        """Generate ARN for a resource"""
        # Replace placeholders in ARN format
        arn = format_str
        arn = arn.replace("{bucket_name}", resource_label.lower().replace(' ', '-'))
        arn = arn.replace("{db_name}", resource_label.lower().replace(' ', '_'))
        arn = arn.replace("{table_name}", resource_label.lower().replace(' ', '_'))
        arn = arn.replace("{queue_name}", resource_label.lower().replace(' ', '-'))
        arn = arn.replace("{topic_name}", resource_label.lower().replace(' ', '-'))
        arn = arn.replace("{secret_name}", resource_label.lower().replace(' ', '-'))
        arn = arn.replace("{region}", self.region)
        arn = arn.replace("{account_id}", self.account_id)

        return arn

    def _normalize_resource_type(self, resource_type: str) -> ResourceType:
        """Convert resource type string to ResourceType enum"""
        for rt in ResourceType:
            if rt.value == resource_type:
                return rt
        return None

    def _get_role_resource_name(self, resource_label: str, resource_type: str) -> str:
        """Get IAM role resource name for a resource"""
        normalized_type = self._normalize_resource_type(resource_type)
        if normalized_type not in [ResourceType.EC2, ResourceType.LAMBDA]:
            return None
        return f"{self.namespace}_{resource_label.lower().replace(' ', '_')}_role"

    def add_custom_policy(self, role_name: str, policy: IAMPolicy):
        """Add custom policy to a role"""
        if role_name in self.roles:
            self.roles[role_name].add_policy(policy)


class IAMPolicyValidator:
    """Validates IAM policies for least-privilege principles"""

    @staticmethod
    def validate_policy(policy: IAMPolicy) -> List[str]:
        """
        Validate policy for security issues

        Returns:
            List of warnings/issues found
        """
        issues = []

        for stmt in policy.statements:
            # Check for wildcard actions
            if "*" in stmt.actions:
                issues.append(f"Policy '{policy.name}' has wildcard actions: {stmt.actions}")

            # Check for wildcard resources
            for resource in stmt.resources:
                if resource.endswith("*"):
                    issues.append(f"Policy '{policy.name}' has overly broad resource: {resource}")

        return issues

    @staticmethod
    def validate_role(role: IAMRole) -> List[str]:
        """Validate IAM role for issues"""
        issues = []

        if not role.inline_policies:
            issues.append(f"Role '{role.name}' has no policies attached")

        for policy in role.inline_policies:
            policy_issues = IAMPolicyValidator.validate_policy(policy)
            issues.extend(policy_issues)

        return issues
