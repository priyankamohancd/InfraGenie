"""
Security configuration generation module for Terraform Accelerators

This module provides:
- SecurityGroupGenerator: Generate security groups from diagram edges
- IAMPolicyGenerator: Generate least-privilege IAM policies
- TerraformSecurityGenerator: Generate Terraform HCL code
- SecurityConfigOrchestrator: Orchestrate the complete pipeline
"""

from models import (
    SecurityGroup,
    SecurityGroupRule,
    IAMRole,
    IAMPolicy,
    SecurityConfiguration,
    ResourceType,
    RuleType
)
from security_group_generator import SecurityGroupGenerator
from iam_policy_generator import IAMPolicyGenerator, IAMPolicyValidator
from terraform_generator import TerraformSecurityGenerator
from orchestrator import SecurityConfigOrchestrator, SecurityConfigExporter

__all__ = [
    "SecurityGroup",
    "SecurityGroupRule",
    "IAMRole",
    "IAMPolicy",
    "SecurityConfiguration",
    "ResourceType",
    "RuleType",
    "SecurityGroupGenerator",
    "IAMPolicyGenerator",
    "IAMPolicyValidator",
    "TerraformSecurityGenerator",
    "SecurityConfigOrchestrator",
    "SecurityConfigExporter"
]
