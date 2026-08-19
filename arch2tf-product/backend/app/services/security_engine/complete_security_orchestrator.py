"""
Complete Security Implementation Orchestrator
Integrates all components:
  - Traffic Flow Analysis
  - Security Group Generation
  - Dynamic IAM Policy Generation
  - Resource-Role Linking
  - Terraform Code Generation
"""
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from traffic_flow_analyzer import TrafficFlowAnalyzer, TrafficFlowVisualizer
from security_group_generator import SecurityGroupGenerator
from dynamic_iam_generator import UniversalPolicyBuilder, DynamicActionRegistry
from resource_role_linker import ResourceRoleLinker


@dataclass
class SecurityImplementationResult:
    """Complete security implementation result"""
    status: str  # "success" or "error"
    namespace: str

    # Components
    traffic_analysis: Dict = None
    security_groups: Dict = None
    iam_roles: Dict = None
    resource_mappings: Dict = None

    # Generated files
    terraform_files: Dict[str, str] = None

    # Validation results
    validation_issues: List[str] = None
    warnings: List[str] = None

    # Summary statistics
    stats: Dict = None

    # Error info
    error_message: str = None


class CompleteSecurityOrchestrator:
    """
    Main orchestrator that coordinates complete security implementation

    Workflow:
    1. Analyze traffic flow from diagram
    2. Generate security groups
    3. Generate IAM policies dynamically
    4. Link resources to roles
    5. Generate complete Terraform code
    6. Validate everything
    """

    # Resource types whose catalog definition (arch2terraform's catalog.py)
    # requires a role_arn/execution_role_arn/service_role flat argument with
    # NO valid default — AWS itself won't create the resource without one,
    # regardless of what (if anything) the diagram wires it to. These always
    # get a role generated, even with zero qualifying outbound edges — see
    # _generate_iam_policies below and dynamic_iam_generator's
    # MANDATORY_MANAGED_POLICY_ARNS for the baseline permissions each of
    # these needs beyond just an assume-role trust policy.
    #
    # Added 2026-07-31 (generalized from an EKS-only fix per her explicit
    # follow-up: "this shouldn't happen just in case of eks, but whenever
    # any resource requires a role then they should just be created by
    # analysing the resource's connections with other resources"). Not
    # included: aws_batch_job_definition's role_arn-equivalent isn't in
    # arch2terraform's catalog yet (that resource type isn't classifiable
    # from a diagram at all currently), so there's no real placeholder this
    # fixes — left out rather than guessing at a shape that isn't real yet.
    MANDATORY_ROLE_TYPES = {
        "aws_eks_cluster",
        "aws_eks_node_group",
        "aws_mwaa_environment",
        "aws_codepipeline",
        "aws_codebuild_project",
        "aws_glue_job",
    }

    # Resource types that can actually assume an IAM role. Anything else
    # (ALB, RDS, S3, ...) never gets a role/policies generated for it, even
    # if it happens to have outbound edges in the diagram. Includes both the
    # original edge-driven-only set (a role only appears if the diagram
    # gives the resource qualifying outbound edges) and MANDATORY_ROLE_TYPES
    # above (a role always appears, edges or not).
    IAM_ROLE_ELIGIBLE_TYPES = {
        "aws_instance",
        "aws_lambda_function",
        "aws_ecs_task_definition",
        "aws_batch_job_definition",
        "aws_sfn_state_machine",
    } | MANDATORY_ROLE_TYPES

    def __init__(self, namespace: str = "default",
                 vpc_id: str = "var.vpc_id",
                 region: str = "${data.aws_region.current.name}",
                 internal_cidr: str = SecurityGroupGenerator.DEFAULT_INTERNAL_CIDR):
        self.namespace = namespace
        self.vpc_id = vpc_id
        self.region = region

        # Initialize all components
        self.traffic_analyzer = TrafficFlowAnalyzer()
        self.sg_generator = SecurityGroupGenerator(
            namespace, vpc_id, traffic_analyzer=self.traffic_analyzer, internal_cidr=internal_cidr
        )
        self.iam_builder = UniversalPolicyBuilder()
        self.resource_linker = ResourceRoleLinker(namespace)
        self.service_registry = DynamicActionRegistry()

    def execute_complete_implementation(self, resource_graph: Dict) -> SecurityImplementationResult:
        """
        Execute complete security implementation pipeline

        Args:
            resource_graph: {nodes: [...], edges: [...]}

        Returns:
            SecurityImplementationResult with all components
        """
        try:
            print("[*] Starting Complete Security Implementation Pipeline")
            print("=" * 80)
            print()

            # Step 1: Analyze Traffic Flow
            print("[Step 1/6] Analyzing Traffic Flow...")
            traffic_analysis = self._analyze_traffic_flow(resource_graph)
            print(f"  ✓ Analyzed {len(traffic_analysis['edges'])} connections")
            print()

            # Step 2: Generate Security Groups
            print("[Step 2/6] Generating Security Groups...")
            security_config = self._generate_security_groups(resource_graph)
            print(f"  ✓ Generated {len(security_config.security_groups)} security groups")
            print()

            # Step 3: Generate IAM Policies
            print("[Step 3/6] Generating IAM Policies...")
            iam_roles = self._generate_iam_policies(resource_graph)
            print(f"  ✓ Generated {len(iam_roles)} IAM roles")
            print()

            # Step 4: Link Resources to Roles
            print("[Step 4/6] Linking Resources to Roles...")
            resource_mappings = self._link_resources_to_roles(resource_graph, iam_roles)
            print(f"  ✓ Linked {len(resource_mappings)} resources to roles")
            print()

            # Step 5: Generate Terraform Code
            print("[Step 5/6] Generating Terraform Code...")
            terraform_files = self._generate_terraform_code(
                security_config, iam_roles, resource_mappings
            )
            print(f"  ✓ Generated {len(terraform_files)} Terraform files")
            print()

            # Step 6: Validate & Compile Results
            print("[Step 6/6] Validating Implementation...")
            validation_issues, warnings = self._validate_implementation(
                security_config, iam_roles, resource_mappings
            )
            print(f"  ✓ Validation complete ({len(warnings)} warnings)")
            print()

            # Compile statistics
            stats = self._compile_statistics(
                traffic_analysis, security_config, iam_roles, resource_mappings
            )

            return SecurityImplementationResult(
                status="success",
                namespace=self.namespace,
                traffic_analysis=traffic_analysis,
                security_groups=asdict(security_config) if security_config else None,
                iam_roles=iam_roles,
                resource_mappings=resource_mappings,
                terraform_files=terraform_files,
                validation_issues=validation_issues,
                warnings=warnings,
                stats=stats
            )

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return SecurityImplementationResult(
                status="error",
                namespace=self.namespace,
                error_message=str(e)
            )

    def _analyze_traffic_flow(self, resource_graph: Dict) -> Dict:
        """
        Step 1: Analyze traffic flow from diagram

        Determines protocol, port, and direction for each connection
        """
        edges_analysis = []
        nodes = {n['id']: n for n in resource_graph.get('nodes', [])}

        for edge in resource_graph.get('edges', []):
            from_node = nodes.get(edge['from'])
            to_node = nodes.get(edge['to'])

            if not from_node or not to_node:
                continue

            # Analyze this edge
            protocol, from_port, to_port = self.traffic_analyzer.analyze_edge(
                from_node, to_node, edge
            )

            edges_analysis.append({
                'from': from_node.get('label', edge['from']),
                'from_type': from_node.get('type'),
                'to': to_node.get('label', edge['to']),
                'to_type': to_node.get('type'),
                'edge_label': edge.get('label', ''),
                'protocol': protocol,
                'port': f"{from_port}:{to_port}" if from_port else "N/A (IAM)",
                'direction': f"{from_node.get('label')} → {to_node.get('label')}"
            })

        # Detect implicit flows
        implicit_flows = self.traffic_analyzer.detect_implicit_flows(resource_graph)

        # Validate traffic flow
        flow_issues = self.traffic_analyzer.validate_traffic_flow(resource_graph)

        return {
            'edges': edges_analysis,
            'implicit_flows': implicit_flows,
            'issues': flow_issues,
            'total_connections': len(edges_analysis)
        }

    def _generate_security_groups(self, resource_graph: Dict):
        """
        Step 2: Generate security groups from traffic analysis

        Creates security group rules based on diagram edges
        """
        return self.sg_generator.generate_from_diagram(resource_graph)

    def _generate_iam_policies(self, resource_graph: Dict) -> Dict:
        """
        Step 3: Generate IAM policies dynamically

        Works with ANY AWS service combination
        """
        nodes = {n['id']: n for n in resource_graph.get('nodes', [])}
        edges = resource_graph.get('edges', [])

        iam_roles = {}

        # Only resources that can actually assume an IAM role get one -
        # e.g. an ALB has outbound "edges" to web servers in the diagram but
        # never assumes a role or calls AWS APIs, so it must never get a
        # role/policy generated for it.
        for node_id, node in nodes.items():
            node_type = node.get('type')
            if node_type not in self.IAM_ROLE_ELIGIBLE_TYPES:
                continue

            result = self.iam_builder.generate_policies_for_compute_resource(
                node, edges, nodes
            )

            # MANDATORY_ROLE_TYPES always get a role even with zero
            # edge-derived policies (AWS requires the role attribute to be
            # set regardless of what the diagram wires the resource to) —
            # everything else keeps the original "only if it actually has
            # something to grant" behavior, so e.g. an EC2 instance with no
            # outbound edges still gets no role/instance-profile noise.
            if result['resource_count'] > 0 or node_type in self.MANDATORY_ROLE_TYPES:
                iam_roles[result['role_name']] = {
                    'role_name': result['role_name'],
                    'service_principal': result['service_principal'],
                    'policies': result['policies'],
                    'managed_policy_arns': result.get('managed_policy_arns', []),
                    'node_id': node_id,
                    'resource_label': node.get('label', node_id),
                    'resource_type': node_type,
                }

        return iam_roles

    def _link_resources_to_roles(self, resource_graph: Dict,
                                iam_roles: Dict) -> Dict:
        """
        Step 4: Link resources to their roles

        Maps each compute resource to its unique IAM role
        """
        # Build mapping from resource to policies
        policies_by_resource = {}
        edges = resource_graph.get('edges', [])
        nodes = {n['id']: n for n in resource_graph.get('nodes', [])}

        for edge in edges:
            from_id = edge['from']
            if from_id not in policies_by_resource:
                policies_by_resource[from_id] = []

            to_node = nodes.get(edge['to'], {})
            policy_name = f"{to_node.get('label', 'resource').lower().replace(' ', '-')}-access"
            policies_by_resource[from_id].append(policy_name)

        # Link resources to roles - only resources that actually got a role
        # in `iam_roles` (Step 3) get attachment code; a compute resource
        # with no AWS-service outbound edges has no role to attach.
        mappings = self.resource_linker.link_resources_to_roles(
            resource_graph, policies_by_resource, iam_roles=iam_roles
        )

        return mappings

    def _generate_terraform_code(self, security_config, iam_roles: Dict,
                                resource_mappings: Dict) -> Dict[str, str]:
        """
        Step 5: Generate complete Terraform code

        Creates HCL files for:
        - Security groups and rules
        - IAM roles and policies
        - Resource attachments
        """
        from terraform_generator import TerraformSecurityGenerator

        tf_gen = TerraformSecurityGenerator()

        terraform_files = {}

        # 1. Security Groups
        if security_config:
            terraform_files['security_groups.tf'] = tf_gen.generate_security_groups_hcl(security_config)

        # 2. IAM Roles and Policies
        terraform_files['iam_roles.tf'] = self._generate_iam_hcl(iam_roles)

        # 3. Resource Attachments
        terraform_files['attachments.tf'] = self._generate_attachments_hcl(resource_mappings)

        # 4. Validation Script
        terraform_files['validate_security.sh'] = tf_gen.generate_security_validation_script()

        # 5. Variables
        terraform_files['variables.tf'] = self._generate_variables_hcl()

        # 6. Outputs
        terraform_files['outputs.tf'] = self._generate_outputs_hcl(
            security_config, iam_roles, resource_mappings
        )

        return terraform_files

    def _generate_iam_hcl(self, iam_roles: Dict) -> str:
        """Generate IAM HCL code"""
        code = "# Auto-generated IAM Roles and Policies\n\n"

        for role_name, role_data in iam_roles.items():
            role_tf_id = role_name.replace('-', '_')

            # Role
            code += f'resource "aws_iam_role" "{role_tf_id}" {{\n'
            code += f'  name = "{role_name}"\n'
            code += f'  assume_role_policy = jsonencode({{\n'
            code += f'    Version = "2012-10-17"\n'
            code += f'    Statement = [{{\n'
            code += f'      Effect = "Allow"\n'
            code += f'      Principal = {{ Service = "{role_data["service_principal"]}" }}\n'
            code += f'      Action = "sts:AssumeRole"\n'
            code += f'    }}]\n'
            code += f'  }})\n'
            code += f'}}\n\n'

            # EC2 cannot attach an IAM role directly — it needs an instance
            # profile wrapping it. security_bridge.py's
            # ATTACHMENT_ATTR_BY_RESOURCE_TYPE["aws_instance"] wires an
            # EC2 instance's iam_instance_profile attribute onto
            # aws_iam_instance_profile.{role_tf_id}_profile.name unconditionally
            # for every EC2-attached role, so that resource must always be
            # declared here or the reference dangles. Found 2026-08-18: this
            # method is a separate, actually-wired reimplementation of the
            # same responsibility as terraform_generator.py's
            # generate_iam_hcl() (unused in this pipeline), which already had
            # this instance-profile step — it was simply never carried over
            # when this method was written, so every diagram generating an
            # EC2 role failed real `terraform validate` with "Reference to
            # undeclared resource" on modules/security/outputs.tf.
            if "ec2" in role_data.get("service_principal", ""):
                code += f'resource "aws_iam_instance_profile" "{role_tf_id}_profile" {{\n'
                code += f'  name = "{role_name}-profile"\n'
                code += f'  role = aws_iam_role.{role_tf_id}.name\n'
                code += f'}}\n\n'

            # Mandatory AWS-managed policy attachments (e.g. EKS's required
            # AmazonEKSClusterPolicy) — separate from the edge-derived
            # inline policies below since these are fixed per-service
            # baseline requirements, not something the diagram's
            # connections determine. See dynamic_iam_generator's
            # MANDATORY_MANAGED_POLICY_ARNS.
            for i, policy_arn in enumerate(role_data.get('managed_policy_arns', []), start=1):
                suffix = "" if i == 1 else f"_{i}"
                code += f'resource "aws_iam_role_policy_attachment" "{role_tf_id}_managed{suffix}" {{\n'
                code += f'  role       = aws_iam_role.{role_tf_id}.name\n'
                code += f'  policy_arn = "{policy_arn}"\n'
                code += f'}}\n\n'

            # Policies - each gets its own resource address (role_tf_id alone
            # would collide when a role has more than one policy, which
            # Terraform rejects as a duplicate resource definition)
            for policy in role_data.get('policies', []):
                policy_tf_id = policy['name'].replace('-', '_').replace(' ', '_')
                code += f'resource "aws_iam_role_policy" "{role_tf_id}_{policy_tf_id}" {{\n'
                code += f'  name = "{policy["name"]}"\n'
                code += f'  role = aws_iam_role.{role_tf_id}.id\n'
                code += f'  policy = jsonencode({{\n'
                code += f'    Version = "2012-10-17"\n'
                code += f'    Statement = [{{\n'
                code += f'      Effect = "Allow"\n'
                code += f'      Action = {json.dumps(policy["actions"])}\n'
                code += f'      Resource = "{policy["resource_arn"]}"\n'
                code += f'    }}]\n'
                code += f'  }})\n'
                code += f'}}\n\n'

        return code

    def _generate_attachments_hcl(self, resource_mappings: Dict) -> str:
        """Generate resource attachment HCL code"""
        code = "# Auto-generated Resource Attachments\n\n"

        for resource_id, mapping in resource_mappings.items():
            code += mapping.attachment_code
            code += "\n"

        return code

    def _generate_variables_hcl(self) -> str:
        """Generate variables.tf"""
        return '''variable "namespace" {
  description = "Namespace for resources"
  type        = string
  default     = "default"
}

variable "vpc_id" {
  description = "VPC ID for security groups"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {
  provider = aws
}
'''

    def _generate_outputs_hcl(self, security_config, iam_roles: Dict,
                             resource_mappings: Dict) -> str:
        """Generate outputs.tf"""
        code = "# Outputs for Security Configuration\n\n"

        # Security Group outputs
        if security_config:
            code += "# Security Group Outputs\n"
            for sg_name, sg in security_config.security_groups.items():
                sg_tf_id = sg_name.replace('-', '_')
                code += f'output "{sg_tf_id}_id" {{\n'
                code += f'  description = "Security Group ID for {sg.name}"\n'
                code += f'  value       = aws_security_group.{sg_tf_id}.id\n'
                code += f'}}\n\n'

        # IAM Role outputs
        code += "# IAM Role Outputs\n"
        for role_name, role_data in iam_roles.items():
            role_tf_id = role_name.replace('-', '_')
            code += f'output "{role_tf_id}_arn" {{\n'
            code += f'  description = "ARN of IAM role {role_name}"\n'
            code += f'  value       = aws_iam_role.{role_tf_id}.arn\n'
            code += f'}}\n\n'

        return code

    def _validate_implementation(self, security_config, iam_roles: Dict,
                                resource_mappings: Dict) -> Tuple[List[str], List[str]]:
        """
        Step 6: Validate security implementation

        Returns: (critical_issues, warnings)
        """
        issues = []
        warnings = []

        # Validate security groups
        if not security_config or not security_config.security_groups:
            warnings.append("⚠️  No security groups generated")

        # Validate IAM roles
        if not iam_roles:
            warnings.append("⚠️  No IAM roles generated")

        for role_name, role_data in iam_roles.items():
            # A mandatory-role type (e.g. EKS) with no edge-derived policies
            # but a real managed-policy attachment is expected, not a gap —
            # only warn when the role has neither.
            if not role_data.get('policies') and not role_data.get('managed_policy_arns'):
                warnings.append(f"⚠️  Role '{role_name}' has no policies")

        # Validate resource linkings
        if not resource_mappings:
            warnings.append("⚠️  No resource-role linkings generated")

        # Validate for issues
        linker_issues = self.resource_linker.validate_mappings()
        if any("❌" in issue for issue in linker_issues):
            issues.extend([i for i in linker_issues if "❌" in i])
        else:
            warnings.extend(linker_issues)

        return issues, warnings

    def _compile_statistics(self, traffic_analysis: Dict, security_config,
                           iam_roles: Dict, resource_mappings: Dict) -> Dict:
        """Compile statistics for the implementation"""
        return {
            'connections_analyzed': traffic_analysis.get('total_connections', 0),
            'security_groups_generated': len(security_config.security_groups) if security_config else 0,
            'iam_roles_generated': len(iam_roles),
            'resources_linked': len(resource_mappings),
            'total_sg_rules': sum(
                len(sg.inbound_rules) + len(sg.outbound_rules)
                for sg in (security_config.security_groups.values() if security_config else [])
            ),
            'total_policies': sum(
                len(role.get('policies', []))
                for role in iam_roles.values()
            )
        }

    def print_summary(self, result: SecurityImplementationResult):
        """Print implementation summary"""
        if result.status == "error":
            print(f"❌ ERROR: {result.error_message}")
            return

        print("=" * 80)
        print("SECURITY IMPLEMENTATION SUMMARY")
        print("=" * 80)
        print()

        print("Traffic Flow Analysis:")
        print(f"  Connections Analyzed: {result.traffic_analysis.get('total_connections', 0)}")
        if result.traffic_analysis.get('issues'):
            for issue in result.traffic_analysis['issues']:
                print(f"  {issue}")
        print()

        print("Security Groups Generated:")
        print(f"  Total: {result.stats['security_groups_generated']}")
        print(f"  Total Rules: {result.stats['total_sg_rules']}")
        print()

        print("IAM Roles Generated:")
        print(f"  Total Roles: {result.stats['iam_roles_generated']}")
        print(f"  Total Policies: {result.stats['total_policies']}")
        print()

        print("Resource Linkings:")
        print(f"  Resources Linked: {result.stats['resources_linked']}")
        print()

        print("Terraform Files Generated:")
        for filename in result.terraform_files.keys():
            print(f"  ✓ {filename}")
        print()

        if result.validation_issues:
            print("Critical Issues:")
            for issue in result.validation_issues:
                print(f"  ❌ {issue}")
            print()

        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  {warning}")
            print()

        print("=" * 80)
        print("Next Steps:")
        print("  1. Review generated Terraform files")
        print("  2. Run: terraform plan -out=tfplan")
        print("  3. Run: bash validate_security.sh")
        print("  4. Run: terraform apply tfplan")
        print("=" * 80)
