"""
Security configuration orchestrator - ties all components together
"""
from typing import Dict, Tuple
from security_group_generator import SecurityGroupGenerator
from iam_policy_generator import IAMPolicyGenerator, IAMPolicyValidator
from terraform_generator import TerraformSecurityGenerator
from models import SecurityConfiguration


class SecurityConfigOrchestrator:
    """Orchestrates security configuration generation from diagrams"""

    def __init__(self, namespace: str = "default", vpc_id: str = "var.vpc_id"):
        self.namespace = namespace
        self.vpc_id = vpc_id

    def generate_complete_security_config(self, resource_graph: Dict) -> Tuple[
        SecurityConfiguration, Dict, Dict[str, list]
    ]:
        """
        Generate complete security configuration (SG + IAM) from diagram

        Returns:
            (security_config, iam_roles, validation_issues)
        """
        print(f"[*] Generating security configuration for namespace: {self.namespace}")

        # Step 1: Generate security groups
        print("[*] Step 1: Generating security groups...")
        sg_generator = SecurityGroupGenerator(self.namespace, self.vpc_id)
        security_config = sg_generator.generate_from_diagram(resource_graph)
        print(f"    ✓ Generated {len(security_config.security_groups)} security groups")

        # Step 2: Generate IAM roles and policies
        print("[*] Step 2: Generating IAM policies...")
        iam_generator = IAMPolicyGenerator(self.namespace)
        iam_roles = iam_generator.generate_from_diagram(resource_graph)
        print(f"    ✓ Generated {len(iam_roles)} IAM roles")

        # Step 3: Validate policies
        print("[*] Step 3: Validating policies...")
        validation_issues = {}
        for role_name, role in iam_roles.items():
            issues = IAMPolicyValidator.validate_role(role)
            if issues:
                validation_issues[role_name] = issues
                for issue in issues:
                    print(f"    ⚠️  {issue}")

        if not validation_issues:
            print("    ✓ All policies validated successfully")

        return security_config, iam_roles, validation_issues

    def generate_terraform_code(self, security_config: SecurityConfiguration, iam_roles: Dict) -> Dict[str, str]:
        """Generate Terraform HCL code for all security resources"""
        print("[*] Step 4: Generating Terraform code...")

        tf_generator = TerraformSecurityGenerator()

        # Generate security groups
        sg_hcl = tf_generator.generate_security_groups_hcl(security_config)
        print(f"    ✓ Generated security group HCL ({len(sg_hcl)} characters)")

        # Generate IAM
        iam_hcl = tf_generator.generate_iam_hcl(iam_roles)
        print(f"    ✓ Generated IAM HCL ({len(iam_hcl)} characters)")

        # Generate validation script
        validation_script = tf_generator.generate_security_validation_script()
        print(f"    ✓ Generated validation script")

        return {
            "security_groups.tf": sg_hcl,
            "iam.tf": iam_hcl,
            "validate_security.sh": validation_script
        }

    def generate_full_pipeline(self, resource_graph: Dict) -> Dict[str, any]:
        """
        Full pipeline: diagram → security config → Terraform code

        Returns:
            {
                "status": "success|error",
                "security_config": SecurityConfiguration,
                "iam_roles": Dict,
                "terraform_files": Dict[str, str],
                "validation_issues": Dict,
                "summary": str
            }
        """
        try:
            # Generate security configuration
            security_config, iam_roles, validation_issues = self.generate_complete_security_config(resource_graph)

            # Generate Terraform code
            terraform_files = self.generate_terraform_code(security_config, iam_roles)

            # Build summary
            summary = self._build_summary(security_config, iam_roles, validation_issues)

            return {
                "status": "success",
                "security_config": security_config,
                "iam_roles": iam_roles,
                "terraform_files": terraform_files,
                "validation_issues": validation_issues,
                "summary": summary
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "security_config": None,
                "iam_roles": None,
                "terraform_files": None,
                "validation_issues": None,
                "summary": None
            }

    def _build_summary(self, security_config: SecurityConfiguration, iam_roles: Dict,
                      validation_issues: Dict) -> str:
        """Build human-readable summary of generated configuration"""
        summary = f"""
=== Security Configuration Summary ===

Security Groups:
  Total SGs: {len(security_config.security_groups)}
  {self._format_sg_summary(security_config)}

IAM Roles:
  Total Roles: {len(iam_roles)}
  {self._format_iam_summary(iam_roles)}

Validation:
  Issues Found: {len(validation_issues)}
  {self._format_validation_summary(validation_issues)}

Terraform Files Generated:
  - security_groups.tf (security group resources)
  - iam.tf (IAM roles and policies)
  - validate_security.sh (security validation script)

Next Steps:
  1. Review generated Terraform code
  2. Run: terraform plan -out=tfplan
  3. Review plan output
  4. Run: bash validate_security.sh
  5. If validation passes: terraform apply tfplan
"""
        return summary

    def _format_sg_summary(self, config: SecurityConfiguration) -> str:
        """Format security group summary"""
        lines = []
        for sg_name, sg in config.security_groups.items():
            lines.append(f"  - {sg.name}: {len(sg.inbound_rules)} inbound, {len(sg.outbound_rules)} outbound")
        return "\n  ".join(lines) if lines else "  (none)"

    def _format_iam_summary(self, iam_roles: Dict) -> str:
        """Format IAM summary"""
        lines = []
        for role_name, role in iam_roles.items():
            lines.append(f"  - {role.name}: {len(role.inline_policies)} policies")
        return "\n  ".join(lines) if lines else "  (none)"

    def _format_validation_summary(self, issues: Dict) -> str:
        """Format validation issues summary"""
        if not issues:
            return "  ✓ No issues found"
        lines = []
        for role_name, role_issues in issues.items():
            for issue in role_issues:
                lines.append(f"  - {issue}")
        return "\n  ".join(lines)


class SecurityConfigExporter:
    """Export security configuration to various formats"""

    @staticmethod
    def export_to_json(security_config: SecurityConfiguration, iam_roles: Dict) -> Dict:
        """Export to JSON-serializable format"""
        return {
            "security_groups": {
                sg_name: {
                    "name": sg.name,
                    "vpc_id": sg.vpc_id,
                    "description": sg.description,
                    "inbound_rules": [
                        {
                            "rule_id": rule.rule_id,
                            "protocol": rule.protocol,
                            "from_port": rule.from_port,
                            "to_port": rule.to_port,
                            "source_sg_id": rule.source_sg_id,
                            "source_cidr": rule.source_cidr,
                            "description": rule.description
                        }
                        for rule in sg.inbound_rules
                    ],
                    "outbound_rules": [
                        {
                            "rule_id": rule.rule_id,
                            "protocol": rule.protocol,
                            "from_port": rule.from_port,
                            "to_port": rule.to_port,
                            "destination_sg_id": rule.destination_sg_id,
                            "destination_cidr": rule.destination_cidr,
                            "description": rule.description
                        }
                        for rule in sg.outbound_rules
                    ]
                }
                for sg_name, sg in security_config.security_groups.items()
            },
            "iam_roles": {
                role_name: {
                    "name": role.name,
                    "service": role.service,
                    "policies": [
                        {
                            "name": policy.name,
                            "policy_document": policy.to_dict()
                        }
                        for policy in role.inline_policies
                    ]
                }
                for role_name, role in iam_roles.items()
            }
        }
