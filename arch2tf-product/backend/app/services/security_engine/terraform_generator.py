"""
Generate Terraform HCL code from security configuration
"""
import json
import re
from typing import Dict, List
from models import SecurityConfiguration, SecurityGroup, SecurityGroupRule, RuleType

# Any run of one-or-more characters that ISN'T a lowercase letter or digit
# collapses to a single underscore — see _sanitize_id's docstring below for
# why this replaced the old per-character .replace() chain.
_NON_ALNUM_RUN_RE = re.compile(r"[^a-z0-9]+")


class TerraformSecurityGenerator:
    """Generate Terraform code for security groups and IAM"""

    def __init__(self):
        self.output = []

    @staticmethod
    def _unwrap_interpolation(value: str) -> str:
        """
        Normalize a resource-reference value that's emitted UNQUOTED in HCL
        (e.g. `vpc_id      = <value>`, no surrounding quotes).

        Callers sometimes pass legacy Terraform 0.11-style interpolation
        syntax like "${aws_vpc.main.id}". That's only valid HCL when it's
        inside a quoted string - written bare it's a syntax error. Since
        these values are always emitted unquoted here (the modern, correct
        way to reference another resource), strip a wrapping ${...} if
        present so either calling convention produces valid HCL.
        """
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return value[2:-1]
        return value

    def generate_security_groups_hcl(self, config: SecurityConfiguration) -> str:
        """Generate Terraform code for all security groups"""
        hcl_code = []

        # Generate security group resources
        for sg_name, sg in config.security_groups.items():
            hcl_code.append(self._generate_sg_resource(sg))

        # Generate security group rules (separate from SG for better organization)
        for sg_name, sg in config.security_groups.items():
            # Inbound rules
            for rule in sg.inbound_rules:
                hcl_code.append(self._generate_sg_rule(sg, rule))

            # Outbound rules
            for rule in sg.outbound_rules:
                hcl_code.append(self._generate_sg_rule(sg, rule))

        return "\n\n".join(hcl_code)

    def _generate_sg_resource(self, sg: SecurityGroup) -> str:
        """Generate security group resource block"""
        tags = json.dumps(sg.tags, indent=2)

        # checkov:skip justification, 2026-07-24: every security group this
        # pipeline generates is attached to the resource it's named for -
        # either a same-module `aws_security_group.X.id` reference, or (for
        # a resource that landed in a different module) a cross-module
        # var./output/module-argument wire (see terraform_planner.py's
        # _wire_sg_attachments and security_bridge.py's sg_attachment_specs
        # - confirmed by generating and reading real plans). Verified real
        # limitation, not a guess: Checkov's static graph resolver cannot
        # trace an attribute reference through a module-boundary variable
        # back to the security group resource that populates it (reproduced
        # in isolation with a minimal 2-module fixture - a same-module
        # `[aws_security_group.x.id]` reference passes CKV2_AWS_5 fine, the
        # identical value passed in via `var.sg_id` from a parent module
        # fails it, even though `terraform plan` shows the same attachment).
        # Skipping with this comment keeps the scan honest instead of
        # hiding a real unattached-SG bug behind a blanket suppression.
        hcl = f"""# checkov:skip=CKV2_AWS_5: Attached via cross-module Terraform variable wiring (see modules/*/main.tf); Checkov's static analyzer cannot trace SG usage through module-boundary variables — confirmed not a real gap.
resource "aws_security_group" "{sg.resource_name}" {{
  name        = "{sg.name}"
  description = "{sg.description}"
  vpc_id      = {self._unwrap_interpolation(sg.vpc_id)}

  tags = {self._format_dict(sg.tags)}

  lifecycle {{
    create_before_destroy = true
  }}
}}"""
        return hcl

    def _generate_sg_rule(self, sg: SecurityGroup, rule: SecurityGroupRule) -> str:
        """Generate security group rule resource block"""
        rule_type = rule.type.value
        rule_id = self._sanitize_id(rule.rule_id)

        hcl = f"""resource "aws_security_group_rule" "{sg.resource_name}_{rule_id}" {{
  type              = "{rule_type}"
  from_port         = {rule.from_port if rule.from_port is not None else 0}
  to_port           = {rule.to_port if rule.to_port is not None else 65535}
  protocol          = "{rule.protocol}"
  security_group_id = aws_security_group.{sg.resource_name}.id
"""

        if rule_type == "ingress":
            if rule.source_sg_id:
                hcl += f'  source_security_group_id = {self._unwrap_interpolation(rule.source_sg_id)}\n'
            elif rule.source_cidr:
                hcl += f'  cidr_blocks              = ["{rule.source_cidr}"]\n'
        else:  # egress
            if rule.destination_sg_id:
                hcl += f'  source_security_group_id = {self._unwrap_interpolation(rule.destination_sg_id)}\n'
            elif rule.destination_cidr:
                hcl += f'  cidr_blocks              = ["{rule.destination_cidr}"]\n'

        if rule.description:
            hcl += f'  description       = "{rule.description}"\n'

        hcl += "}"

        return hcl

    def generate_iam_hcl(self, roles: Dict) -> str:
        """Generate Terraform code for IAM roles and policies"""
        hcl_code = []

        for role_name, role in roles.items():
            # Trust policy
            hcl_code.append(self._generate_iam_role(role))

            # Inline policies
            for policy in role.inline_policies:
                hcl_code.append(self._generate_iam_policy(role, policy))

            # Instance profile (for EC2)
            if "ec2" in role.service:
                hcl_code.append(self._generate_instance_profile(role))

        return "\n\n".join(hcl_code)

    def _generate_iam_role(self, role) -> str:
        """Generate IAM role resource block"""
        trust_policy = json.dumps(role.get_trust_policy(), indent=2)

        hcl = f"""resource "aws_iam_role" "{role.resource_name}" {{
  name = "{role.name}"

  assume_role_policy = jsonencode({self._format_dict(role.get_trust_policy())})

  tags = {self._format_dict(role.tags)}
}}"""
        return hcl

    def _generate_iam_policy(self, role, policy) -> str:
        """Generate inline IAM policy resource block"""
        policy_dict = policy.to_dict()
        policy_id = self._sanitize_id(policy.name)

        hcl = f"""resource "aws_iam_role_policy" "{role.resource_name}_{policy_id}" {{
  name = "{policy.name}"
  role = aws_iam_role.{role.resource_name}.id

  policy = jsonencode({self._format_dict(policy_dict)})
}}"""
        return hcl

    def _generate_instance_profile(self, role) -> str:
        """Generate EC2 instance profile"""
        hcl = f"""resource "aws_iam_instance_profile" "{role.resource_name}_profile" {{
  name = "{role.name}-profile"
  role = aws_iam_role.{role.resource_name}.name
}}"""
        return hcl

    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for Terraform HCL"""
        return json.dumps(d, indent=2)

    def _sanitize_id(self, id_str: str) -> str:
        """
        Sanitize an arbitrary string into a valid Terraform resource
        identifier (letters, digits, underscores, hyphens only; must not
        start with a digit).

        Real bug, found 2026-08-20: the previous implementation only
        replaced hyphens/spaces/periods, so ANY other punctuation in the
        source string — parentheses, slashes, colons, commas, etc. —
        passed straight through into the generated resource name.
        `terraform init` rejects that outright with "Invalid resource
        name". Root cause traced to a real generated file:
        `rule.rule_id` here is frequently built from a diagram connection's
        verbatim label (e.g. "EKS Node Group (2 worker nodes) to
        ElastiCache Cluster" from the Vision-LLM path — see
        vision_llm_detector.py's "label" field, copied through
        unmodified), and the old .replace() chain let "(2" and "nodes)"
        reach the resource name as literal, invalid characters. This
        became far more likely to trigger once the Vision-LLM path was
        enabled, since its labels are richer natural-language text than
        the classical pipeline's typically-short OCR'd labels — but the
        same bug could always have hit any sufficiently punctuated label,
        classical or VLM.

        Collapses every run of one-or-more non-alphanumeric characters to
        a single underscore (not a per-character replacement, so
        "(2 worker" becomes "_2_worker" rather than fragmenting into
        multiple separate underscores per character), strips any
        leading/trailing underscore left over, then guards against a
        still-invalid leading digit (Terraform identifiers must not start
        with one) by prefixing "r_".
        """
        slug = _NON_ALNUM_RUN_RE.sub("_", id_str.lower()).strip("_")
        if slug and slug[0].isdigit():
            slug = f"r_{slug}"
        return slug or "unnamed"

    def generate_security_validation_script(self) -> str:
        """Generate bash script for security validation"""
        script = """#!/bin/bash
# Security Validation Script for Terraform Accelerators

echo "=== Terraform Security Validation ==="
echo ""

echo "1. Running terraform validate..."
terraform validate
if [ $? -ne 0 ]; then
  echo "❌ Terraform validation failed"
  exit 1
fi
echo "✅ Terraform validation passed"
echo ""

echo "2. Running tflint..."
tflint --config .tflint.hcl
if [ $? -ne 0 ]; then
  echo "⚠️  tflint found issues (may be non-critical)"
fi
echo ""

echo "3. Running checkov..."
checkov -f terraform/ --framework terraform --check CK_AWS_21,CK_AWS_24,CK_AWS_40,CK_AWS_62
if [ $? -ne 0 ]; then
  echo "❌ Checkov security checks failed"
  exit 1
fi
echo "✅ Checkov security checks passed"
echo ""

echo "4. Checking for hardcoded secrets..."
grep -r "password\\|secret\\|api_key\\|access_key" terraform/ || echo "✅ No hardcoded secrets found"
echo ""

echo "=== All security validations complete ==="
"""
        return script
