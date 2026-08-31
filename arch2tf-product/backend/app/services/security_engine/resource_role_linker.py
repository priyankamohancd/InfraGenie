"""
Resource-Role Linker
Maps resources to roles and generates attachment code
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Same fix as terraform_generator.py's _NON_ALNUM_RUN_RE (found 2026-08-20,
# see _sanitize_tf_id's docstring below) — collapses any run of
# non-alphanumeric characters into a single underscore, instead of only
# handling spaces/hyphens/periods.
_NON_ALNUM_RUN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ResourceRoleMapping:
    """Maps a resource to its IAM role"""
    resource_id: str
    resource_label: str
    resource_type: str
    role_name: str
    role_arn: str
    policies: List[str] = field(default_factory=list)
    attachment_code: str = ""


class ResourceRoleLinker:
    """
    Links resources to their IAM roles
    Generates attachment code for Terraform
    """

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.mappings: Dict[str, ResourceRoleMapping] = {}
        self.compute_resources = [
            "aws_instance",
            "aws_lambda_function",
            "aws_ecs_task_definition",
            "aws_batch_job_definition",
            "aws_sfn_state_machine"
        ]

    def link_resources_to_roles(self, resource_graph: Dict,
                                policies_by_resource: Dict,
                                iam_roles: Optional[Dict] = None) -> Dict[str, ResourceRoleMapping]:
        """
        Link each resource to its role and generate attachment code

        Args:
            resource_graph: nodes and edges from diagram
            policies_by_resource: {resource_id: [policy_names]}
            iam_roles: the roles dict actually produced for this graph (keyed
                by role_name, e.g. from UniversalPolicyBuilder /
                CompleteSecurityOrchestrator._generate_iam_policies). When
                given, a resource only gets attachment code if it actually
                has a role in here - a compute resource with zero AWS-service
                outbound edges gets no role generated at all, so attaching it
                to one anyway would reference a resource that's never
                declared in iam_roles.tf.
        """
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}

        for node_id, node in nodes.items():
            resource_type = node.get('type')

            # Only compute resources need roles
            if resource_type not in self.compute_resources:
                continue

            if iam_roles is not None and self._policy_builder_role_name(node) not in iam_roles:
                # No policies were generated for this resource - it has no
                # role to attach to, so skip it rather than emit a dangling
                # reference to an aws_iam_role that's never declared.
                continue

            # Create unique role name for this resource
            role_name = self._generate_role_name(node)
            role_arn = self._generate_role_arn(role_name)

            # Get policies for this resource
            policies = policies_by_resource.get(node_id, [])

            # Generate attachment code
            attachment_code = self._generate_attachment_code(node, role_name)

            # Create mapping
            mapping = ResourceRoleMapping(
                resource_id=node_id,
                resource_label=node.get('label', node_id),
                resource_type=resource_type,
                role_name=role_name,
                role_arn=role_arn,
                policies=policies,
                attachment_code=attachment_code
            )

            self.mappings[node_id] = mapping

        return self.mappings

    def _generate_role_name(self, resource_node: Dict) -> str:
        """Generate unique role name for a resource"""
        resource_label = resource_node.get('label', resource_node.get('id', 'resource'))
        clean_label = resource_label.lower().replace(' ', '-').replace('_', '-')
        return f"{self.namespace}-{clean_label}-role"

    @staticmethod
    def _policy_builder_role_name(resource_node: Dict) -> str:
        """
        Reproduce UniversalPolicyBuilder.generate_policies_for_compute_resource's
        role_name scheme exactly, so we can check whether a given resource
        actually has an entry in the `iam_roles` dict that
        CompleteSecurityOrchestrator._generate_iam_policies produced (that
        dict is keyed by this name, not by ResourceRoleLinker's own
        namespace-prefixed _generate_role_name).
        """
        compute_label = resource_node.get('label', '')
        return f"role-{compute_label.lower().replace(' ', '-')}"

    @staticmethod
    def _role_tf_id(resource_label: str) -> str:
        """
        Reproduce CompleteSecurityOrchestrator._generate_iam_hcl's
        `role_tf_id` exactly, so attachment code's `aws_iam_role.<id>`
        references actually match the resource address iam_roles.tf
        declares the role under.

        Updated 2026-08-24 alongside complete_security_orchestrator.py's
        `_tf_id` fix: that file switched its role_tf_id derivation from a
        naive `.replace('-', '_')` to a regex collapse of ANY
        non-alphanumeric run (found via a real retest — a Vision-LLM label
        with parentheses broke `terraform init` on modules/security). Both
        still build `role_name` from `resource_label`/`compute_label` via
        the identical `f"role-{label.lower().replace(' ', '-')}"` step
        first, so as long as the FINAL collapse step here matches that
        file's `_tf_id` exactly, the two independently-computed ids stay
        identical for the same input label. If that file's collapsing
        logic changes again, this must change identically or attachment
        code will reference a role address that's never declared.
        """
        role_name = f"role-{resource_label.lower().replace(' ', '-')}"
        return _NON_ALNUM_RUN_RE.sub("_", role_name).strip("_")

    def _generate_role_arn(self, role_name: str) -> str:
        """Generate ARN for role"""
        return f"arn:aws:iam::${{data.aws_caller_identity.current.account_id}}:role/{role_name}"

    def _generate_attachment_code(self, resource_node: Dict, role_name: str) -> str:
        """
        Generate Terraform code to attach role to resource

        Returns appropriate attachment code based on resource type
        """
        resource_type = resource_node.get('type')
        resource_id = resource_node.get('id', 'resource')
        resource_label = resource_node.get('label', resource_id)

        # Terraform resource identifier for the attachment snippet itself,
        # and the role's real resource address as declared in iam_roles.tf
        # (these are two different naming schemes - see _role_tf_id).
        tf_resource_id = self._sanitize_tf_id(resource_label)
        role_tf_id = self._role_tf_id(resource_label)

        if resource_type == "aws_instance":
            return self._generate_ec2_attachment(tf_resource_id, role_tf_id)

        elif resource_type == "aws_lambda_function":
            return self._generate_lambda_attachment(tf_resource_id, role_tf_id)

        elif resource_type == "aws_ecs_task_definition":
            return self._generate_ecs_attachment(tf_resource_id, role_tf_id)

        elif resource_type == "aws_batch_job_definition":
            return self._generate_batch_attachment(tf_resource_id, role_tf_id)

        elif resource_type == "aws_sfn_state_machine":
            return self._generate_stepfunctions_attachment(tf_resource_id, role_tf_id)

        return ""

    def _generate_ec2_attachment(self, resource_id: str, role_id: str) -> str:
        """Generate attachment code for EC2 instance"""
        return f"""
# Instance Profile for EC2
resource "aws_iam_instance_profile" "{role_id}_profile" {{
  name = "{self.namespace}-{resource_id}-profile"
  role = aws_iam_role.{role_id}.name
}}

# EC2 Instance with IAM Profile attached
resource "aws_instance" "{resource_id}" {{
  # ... other configuration ...
  iam_instance_profile = aws_iam_instance_profile.{role_id}_profile.name  # ← Role attached here
  # ... other configuration ...
}}
"""

    def _generate_lambda_attachment(self, resource_id: str, role_id: str) -> str:
        """Generate attachment code for Lambda function"""
        return f"""
# Lambda Function with Execution Role
resource "aws_lambda_function" "{resource_id}" {{
  filename      = "lambda_function.zip"
  function_name = "{self.namespace}-{resource_id}"
  role          = aws_iam_role.{role_id}.arn  # ← Role attached here
  handler       = "index.handler"
  # ... other configuration ...
}}
"""

    def _generate_ecs_attachment(self, resource_id: str, role_id: str) -> str:
        """Generate attachment code for ECS Task"""
        return f"""
# ECS Task Definition with Task Role
resource "aws_ecs_task_definition" "{resource_id}" {{
  family                   = "{self.namespace}-{resource_id}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.{role_id}.arn  # ← Role attached here

  container_definitions = jsonencode([{{
    name      = "{resource_id}"
    image     = "my-image:latest"
    # ... other configuration ...
  }}])
}}
"""

    def _generate_batch_attachment(self, resource_id: str, role_id: str) -> str:
        """Generate attachment code for Batch Job"""
        return f"""
# Batch Job Definition with Role
resource "aws_batch_job_definition" "{resource_id}" {{
  name  = "{self.namespace}-{resource_id}"
  type  = "container"
  container_properties = jsonencode({{
    image = "my-image:latest"
    # ... other configuration ...
  }})

  # Container execution role
  # (Attach via container properties)
}}
"""

    def _generate_stepfunctions_attachment(self, resource_id: str, role_id: str) -> str:
        """Generate attachment code for Step Functions"""
        return f"""
# Step Functions State Machine with Execution Role
resource "aws_sfn_state_machine" "{resource_id}" {{
  name       = "{self.namespace}-{resource_id}"
  role_arn   = aws_iam_role.{role_id}.arn  # ← Role attached here
  definition = jsonencode({{
    Comment = "{resource_id}"
    StartAt = "FirstState"
    States = {{
      FirstState = {{
        Type = "Pass"
        End  = true
      }}
    }}
  }})
}}
"""

    def _sanitize_tf_id(self, text: str) -> str:
        """
        Sanitize an arbitrary string into a valid Terraform resource
        identifier (letters, digits, underscores, hyphens only; must not
        start with a digit).

        Same bug as terraform_generator.py's _sanitize_id, found 2026-08-20
        via the same "Invalid resource name" terraform init failure: the old
        .replace() chain here only handled spaces/hyphens/periods, so any
        other punctuation carried through a diagram label (e.g. Vision-LLM
        edge labels like "EKS Node Group (2 worker nodes) to ElastiCache
        Cluster" — see vision_llm_detector.py) reached generated resource
        names as literal invalid characters. Fixed identically: collapse
        every run of non-alphanumeric characters to a single underscore,
        strip stray leading/trailing underscores, and guard against a
        leading digit.
        """
        slug = _NON_ALNUM_RUN_RE.sub("_", text.lower()).strip("_")
        if slug and slug[0].isdigit():
            slug = f"r_{slug}"
        return slug or "unnamed"

    def generate_linking_report(self) -> str:
        """Generate a report showing resource-role mappings"""
        report = "=" * 80 + "\n"
        report += "RESOURCE-ROLE LINKING REPORT\n"
        report += "=" * 80 + "\n\n"

        if not self.mappings:
            report += "No compute resources found\n"
            return report

        report += f"Total Compute Resources: {len(self.mappings)}\n\n"

        for resource_id, mapping in self.mappings.items():
            report += f"Resource: {mapping.resource_label}\n"
            report += f"  Type: {mapping.resource_type}\n"
            report += f"  Role Name: {mapping.role_name}\n"
            report += f"  Role ARN: {mapping.role_arn}\n"
            report += f"  Policies: {', '.join(mapping.policies) if mapping.policies else 'None'}\n"
            report += f"  Attachment: ✓ (see code below)\n"
            report += "\n"

        return report

    def generate_complete_terraform(self, roles_and_policies: Dict,
                                   resource_graph: Dict) -> Dict[str, str]:
        """
        Generate complete Terraform files with attachments

        Returns:
            {
                "iam_roles.tf": role definitions,
                "resource_attachments.tf": attachment code,
                "linking_report.txt": mapping report
            }
        """
        # First, link resources to roles
        policies_by_resource = self._extract_policies_by_resource(resource_graph)
        self.link_resources_to_roles(resource_graph, policies_by_resource)

        # Generate IAM roles file
        iam_roles_code = self._generate_iam_roles_code(roles_and_policies)

        # Generate resource attachments file
        attachments_code = self._generate_attachments_code()

        # Generate report
        report = self.generate_linking_report()

        return {
            "iam_roles.tf": iam_roles_code,
            "resource_attachments.tf": attachments_code,
            "linking_report.txt": report
        }

    def _extract_policies_by_resource(self, resource_graph: Dict) -> Dict:
        """Extract which policies belong to which resource"""
        # This is a simplified version - in practice, connect to IAM generator
        policies_by_resource = {}
        edges = resource_graph.get('edges', [])

        for edge in edges:
            from_id = edge.get('from')
            if from_id not in policies_by_resource:
                policies_by_resource[from_id] = []
            # Add policy names based on target
            to_node_id = edge.get('to')
            to_node = next((n for n in resource_graph['nodes'] if n['id'] == to_node_id), {})
            if to_node:
                policy_name = f"{to_node.get('label', to_node_id).lower()}-access"
                policies_by_resource[from_id].append(policy_name)

        return policies_by_resource

    def _generate_iam_roles_code(self, roles_data: Dict) -> str:
        """Generate HCL code for IAM roles and policies"""
        code = "# Auto-generated IAM Roles and Policies\n\n"

        for role_id, role_data in self.mappings.items():
            role_tf_id = self._sanitize_tf_id(role_data.resource_label)

            # Role definition
            code += f'resource "aws_iam_role" "{role_tf_id}" {{\n'
            code += f'  name = "{role_data.role_name}"\n'
            code += f'  \n'
            code += f'  assume_role_policy = jsonencode({{\n'
            code += f'    Version = "2012-10-17"\n'
            code += f'    Statement = [{{\n'
            code += f'      Effect = "Allow"\n'
            code += f'      Principal = {{ Service = "{self._get_service_principal(role_data.resource_type)}" }}\n'
            code += f'      Action = "sts:AssumeRole"\n'
            code += f'    }}]\n'
            code += f'  }})\n'
            code += f'\n'
            code += f'  tags = {{\n'
            code += f'    Name = "{role_data.role_name}"\n'
            code += f'    ManagedBy = "terraform-accelerators"\n'
            code += f'  }}\n'
            code += f'}}\n\n'

        return code

    def _generate_attachments_code(self) -> str:
        """Generate resource attachment code"""
        code = "# Auto-generated Resource-Role Attachments\n\n"

        for resource_id, mapping in self.mappings.items():
            code += mapping.attachment_code
            code += "\n\n"

        return code

    def _get_service_principal(self, resource_type: str) -> str:
        """Get AWS service principal for resource type"""
        principals = {
            "aws_instance": "ec2.amazonaws.com",
            "aws_lambda_function": "lambda.amazonaws.com",
            "aws_ecs_task_definition": "ecs-tasks.amazonaws.com",
            "aws_batch_job_definition": "batch.amazonaws.com",
            "aws_sfn_state_machine": "states.amazonaws.com"
        }
        return principals.get(resource_type, "ec2.amazonaws.com")

    def get_role_for_resource(self, resource_id: str) -> Optional[ResourceRoleMapping]:
        """Get role mapping for a specific resource"""
        return self.mappings.get(resource_id)

    def get_all_mappings(self) -> Dict[str, ResourceRoleMapping]:
        """Get all resource-role mappings"""
        return self.mappings

    def validate_mappings(self) -> List[str]:
        """Validate resource-role mappings"""
        issues = []

        if not self.mappings:
            issues.append("⚠️  No compute resources found in diagram")
            return issues

        # Check for duplicate role names
        role_names = [m.role_name for m in self.mappings.values()]
        duplicates = set([r for r in role_names if role_names.count(r) > 1])
        if duplicates:
            issues.append(f"❌ Duplicate role names: {duplicates}")

        # Check for resources without policies
        for resource_id, mapping in self.mappings.items():
            if not mapping.policies:
                issues.append(f"⚠️  Resource '{mapping.resource_label}' has no policies")

        # Check for missing attachment code
        for resource_id, mapping in self.mappings.items():
            if not mapping.attachment_code:
                issues.append(f"❌ No attachment code for '{mapping.resource_label}'")

        return issues if issues else ["✓ All mappings validated successfully"]
