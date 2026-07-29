"""
Example: Dynamic IAM Policy Generation for ANY AWS Architecture
Demonstrates the system working with diverse AWS services
"""
import json
from dynamic_iam_generator import (
    DynamicIAMPolicyGenerator,
    UniversalPolicyBuilder,
    DynamicActionRegistry
)


# Example 1: Traditional Web Architecture
WEB_ARCHITECTURE = {
    "nodes": [
        {"id": "ec2-1", "label": "Web Server", "type": "aws_instance", "metadata": {}},
        {"id": "rds-1", "label": "PostgreSQL DB", "type": "aws_db_instance",
         "metadata": {"engine": "postgres"}},
        {"id": "s3-1", "label": "Static Assets", "type": "aws_s3_bucket", "metadata": {}},
    ],
    "edges": [
        {"from": "ec2-1", "to": "rds-1", "label": "PostgreSQL"},
        {"from": "ec2-1", "to": "s3-1", "label": "Read"},
    ]
}

# Example 2: Serverless Data Pipeline
SERVERLESS_ARCHITECTURE = {
    "nodes": [
        {"id": "lambda-1", "label": "Data Processor", "type": "aws_lambda_function", "metadata": {}},
        {"id": "sqs-1", "label": "Input Queue", "type": "aws_sqs_queue", "metadata": {}},
        {"id": "dynamodb-1", "label": "Results Table", "type": "aws_dynamodb_table", "metadata": {}},
        {"id": "s3-2", "label": "Data Lake", "type": "aws_s3_bucket", "metadata": {}},
        {"id": "sns-1", "label": "Notifications", "type": "aws_sns_topic", "metadata": {}},
        {"id": "secrets-1", "label": "API Keys", "type": "aws_secretsmanager_secret", "metadata": {}},
    ],
    "edges": [
        {"from": "lambda-1", "to": "sqs-1", "label": "Receive"},
        {"from": "lambda-1", "to": "dynamodb-1", "label": "Write"},
        {"from": "lambda-1", "to": "s3-2", "label": "Put"},
        {"from": "lambda-1", "to": "sns-1", "label": "Publish"},
        {"from": "lambda-1", "to": "secrets-1", "label": "Get Secret"},
    ]
}

# Example 3: Analytics Platform
ANALYTICS_ARCHITECTURE = {
    "nodes": [
        {"id": "lambda-2", "label": "Analytics Engine", "type": "aws_lambda_function", "metadata": {}},
        {"id": "kinesis-1", "label": "Data Stream", "type": "aws_kinesis_stream", "metadata": {}},
        {"id": "redshift-1", "label": "Data Warehouse", "type": "aws_redshift_cluster", "metadata": {}},
        {"id": "dynamodb-2", "label": "Cache", "type": "aws_dynamodb_table", "metadata": {}},
        {"id": "s3-3", "label": "Raw Data", "type": "aws_s3_bucket", "metadata": {}},
        {"id": "logs-1", "label": "Audit Logs", "type": "aws_cloudwatch_log_group", "metadata": {}},
        {"id": "kms-1", "label": "Encryption Key", "type": "aws_kms_key", "metadata": {}},
    ],
    "edges": [
        {"from": "lambda-2", "to": "kinesis-1", "label": "Get Records"},
        {"from": "lambda-2", "to": "redshift-1", "label": "Query"},
        {"from": "lambda-2", "to": "dynamodb-2", "label": "Write"},
        {"from": "lambda-2", "to": "s3-3", "label": "Read"},
        {"from": "lambda-2", "to": "logs-1", "label": "Write Logs"},
        {"from": "lambda-2", "to": "kms-1", "label": "Decrypt"},
    ]
}

# Example 4: ML Pipeline
ML_ARCHITECTURE = {
    "nodes": [
        {"id": "lambda-3", "label": "ML Handler", "type": "aws_lambda_function", "metadata": {}},
        {"id": "sagemaker-1", "label": "Model Endpoint", "type": "aws_sagemaker_endpoint", "metadata": {}},
        {"id": "s3-4", "label": "Training Data", "type": "aws_s3_bucket", "metadata": {}},
        {"id": "dynamodb-3", "label": "Predictions", "type": "aws_dynamodb_table", "metadata": {}},
        {"id": "sns-2", "label": "Alerts", "type": "aws_sns_topic", "metadata": {}},
    ],
    "edges": [
        {"from": "lambda-3", "to": "sagemaker-1", "label": "Invoke"},
        {"from": "lambda-3", "to": "s3-4", "label": "Read"},
        {"from": "lambda-3", "to": "dynamodb-3", "label": "Put"},
        {"from": "lambda-3", "to": "sns-2", "label": "Publish"},
    ]
}


def analyze_architecture(name, architecture):
    """Analyze an architecture and generate IAM policies"""
    print("=" * 80)
    print(f"ARCHITECTURE: {name}")
    print("=" * 80)
    print()

    # Step 1: List resources
    print(f"Resources ({len(architecture['nodes'])}):")
    for node in architecture['nodes']:
        print(f"  • {node['label']:<25} ({node['type']})")
    print()

    # Step 2: List connections
    print(f"Connections ({len(architecture['edges'])}):")
    all_nodes = {n['id']: n for n in architecture['nodes']}
    for edge in architecture['edges']:
        from_node = all_nodes.get(edge['from'], {})
        to_node = all_nodes.get(edge['to'], {})
        print(f"  • {from_node.get('label', 'Unknown'):<20} → {to_node.get('label', 'Unknown'):<20} ({edge.get('label', 'N/A')})")
    print()

    # Step 3: Generate IAM policies
    print("Generated IAM Policies:")
    print("-" * 80)

    builder = UniversalPolicyBuilder()
    all_nodes = {n['id']: n for n in architecture['nodes']}
    edges = architecture['edges']

    # Find all compute resources
    compute_types = ["aws_lambda_function", "aws_instance", "aws_ecs_task_definition"]
    compute_resources = [n for n in architecture['nodes'] if n['type'] in compute_types]

    for compute_node in compute_resources:
        result = builder.generate_policies_for_compute_resource(
            compute_node, edges, all_nodes
        )

        print()
        print(f"Role: {result['role_name']}")
        print(f"Service Principal: {result['service_principal']}")
        print(f"Total Policies: {result['resource_count']}")
        print()

        for policy in result['policies']:
            print(f"  Policy: {policy['name']}")
            print(f"    Service: {policy['service']}")
            print(f"    Actions: {', '.join(policy['actions'][:2])}{'...' if len(policy['actions']) > 2 else ''}")
            print(f"    Resource ARN: {policy['resource_arn']}")
            print(f"    Status: {'✓ OK' if policy.get('status') != 'NEEDS_REVIEW' else '⚠️  NEEDS_REVIEW'}")
            print()

    print()


def show_service_registry():
    """Show all supported services"""
    print("=" * 80)
    print("ALL SUPPORTED AWS SERVICES (Dynamic Registry)")
    print("=" * 80)
    print()

    registry = DynamicActionRegistry()
    services = sorted(registry.list_services())

    print(f"Total Services: {len(services)}\n")

    for service_name in services:
        service = registry.get_service(service_name)
        total_actions = len(service.read_actions) + len(service.write_actions) + len(service.manage_actions)
        print(f"{service_name:<20} | Type: {service.resource_type:<35} | Actions: {total_actions}")

    print()


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "DYNAMIC IAM POLICY GENERATOR - ANY AWS ARCHITECTURE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Show service registry
    show_service_registry()
    input("Press Enter to continue...")
    print()

    # Analyze each architecture
    architectures = [
        ("Web Application", WEB_ARCHITECTURE),
        ("Serverless Data Pipeline", SERVERLESS_ARCHITECTURE),
        ("Analytics Platform", ANALYTICS_ARCHITECTURE),
        ("ML Pipeline", ML_ARCHITECTURE),
    ]

    for name, arch in architectures:
        analyze_architecture(name, arch)
        print()
        print("=" * 80)
        print()

    # Show generated Terraform code for one example
    print("EXAMPLE: Generated Terraform for Serverless Data Pipeline")
    print("=" * 80)
    print()

    builder = UniversalPolicyBuilder()
    all_nodes = {n['id']: n for n in SERVERLESS_ARCHITECTURE['nodes']}
    edges = SERVERLESS_ARCHITECTURE['edges']

    lambda_node = next(n for n in SERVERLESS_ARCHITECTURE['nodes']
                       if n['type'] == 'aws_lambda_function')

    result = builder.generate_policies_for_compute_resource(
        lambda_node, edges, all_nodes
    )

    # Generate Terraform code
    role_name_tf = result['role_name'].replace('-', '_')
    print(f'resource "aws_iam_role" "{role_name_tf}" {{')
    print(f'  name = "{result["role_name"]}"')
    print()
    print(f'  assume_role_policy = jsonencode({{')
    print(f'    Version = "2012-10-17"')
    print(f'    Statement = [{{')
    print(f'      Effect = "Allow"')
    print(f'      Principal = {{ Service = "{result["service_principal"]}" }}')
    print(f'      Action = "sts:AssumeRole"')
    print(f'    }}]')
    print(f'  }})')
    print(f'}}')
    print()

    for i, policy in enumerate(result['policies'], 1):
        policy_name_tf = policy['name'].replace('-', '_')
        print(f'resource "aws_iam_role_policy" "{role_name_tf}_{policy_name_tf}" {{')
        print(f'  name = "{policy["name"]}"')
        print(f'  role = aws_iam_role.{role_name_tf}.id')
        print()
        print(f'  policy = jsonencode({{')
        print(f'    Version = "2012-10-17"')
        print(f'    Statement = [{{')
        print(f'      Effect = "Allow"')
        print(f'      Action = {json.dumps(policy["actions"])}')
        print(f'      Resource = "{policy["resource_arn"]}"')
        print(f'    }}]')
        print(f'  }})')
        print(f'}}')
        if i < len(result['policies']):
            print()

    print()
    print("=" * 80)
    print()
    print("✅ Key Features Demonstrated:")
    print("  • Works with ANY AWS service (20+ supported)")
    print("  • Infers operation type from edge labels")
    print("  • Generates least-privilege policies")
    print("  • Handles unknown services gracefully")
    print("  • Extensible architecture for adding new services")
    print()


if __name__ == "__main__":
    main()
