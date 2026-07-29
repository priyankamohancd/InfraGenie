"""
Complete Implementation Example
Demonstrates the full security pipeline with a realistic architecture
"""
from complete_security_orchestrator import CompleteSecurityOrchestrator


# Example: Complete 5-tier E-Commerce Architecture
ECOMMERCE_ARCHITECTURE = {
    "nodes": [
        # Web Tier
        {
            "id": "alb-1",
            "label": "Public ALB",
            "type": "aws_lb",
            "metadata": {"scheme": "internet-facing"}
        },

        # Application Tier
        {
            "id": "ec2-web-1",
            "label": "Web Server 1",
            "type": "aws_instance",
            "metadata": {"role": "web"}
        },
        {
            "id": "ec2-web-2",
            "label": "Web Server 2",
            "type": "aws_instance",
            "metadata": {"role": "web"}
        },

        # API Tier
        {
            "id": "ec2-api",
            "label": "API Server",
            "type": "aws_instance",
            "metadata": {"role": "api"}
        },

        # Data Tier
        {
            "id": "rds-primary",
            "label": "Primary Database",
            "type": "aws_db_instance",
            "metadata": {"engine": "postgres", "allocated_storage": 100}
        },
        {
            "id": "rds-replica",
            "label": "Read Replica",
            "type": "aws_db_instance",
            "metadata": {"engine": "postgres", "allocated_storage": 100}
        },

        # Cache Tier
        {
            "id": "cache-1",
            "label": "Redis Cache",
            "type": "aws_elasticache_cluster",
            "metadata": {"engine": "redis"}
        },

        # Storage Tier
        {
            "id": "s3-products",
            "label": "Product Images",
            "type": "aws_s3_bucket",
            "metadata": {}
        },
        {
            "id": "s3-backups",
            "label": "Backup Storage",
            "type": "aws_s3_bucket",
            "metadata": {}
        },

        # Logging & Monitoring
        {
            "id": "logs-1",
            "label": "Application Logs",
            "type": "aws_cloudwatch_log_group",
            "metadata": {}
        },

        # Message Queue
        {
            "id": "sqs-1",
            "label": "Background Jobs",
            "type": "aws_sqs_queue",
            "metadata": {}
        },

        # Notifications
        {
            "id": "sns-1",
            "label": "Order Notifications",
            "type": "aws_sns_topic",
            "metadata": {}
        },

        # Encryption
        {
            "id": "kms-1",
            "label": "Data Encryption Key",
            "type": "aws_kms_key",
            "metadata": {}
        },

        # Worker Lambda
        {
            "id": "lambda-worker",
            "label": "Order Processor",
            "type": "aws_lambda_function",
            "metadata": {}
        }
    ],

    "edges": [
        # Internet to ALB
        {"from": "internet", "to": "alb-1", "label": "HTTPS", "type": "connection"},

        # ALB to Web Servers
        {"from": "alb-1", "to": "ec2-web-1", "label": "HTTP", "type": "connection"},
        {"from": "alb-1", "to": "ec2-web-2", "label": "HTTP", "type": "connection"},

        # Web Servers to API Server
        {"from": "ec2-web-1", "to": "ec2-api", "label": "REST API", "type": "connection"},
        {"from": "ec2-web-2", "to": "ec2-api", "label": "REST API", "type": "connection"},

        # API Server to Databases
        {"from": "ec2-api", "to": "rds-primary", "label": "PostgreSQL", "type": "connection"},
        {"from": "ec2-api", "to": "rds-replica", "label": "SELECT", "type": "connection"},

        # API Server to Cache
        {"from": "ec2-api", "to": "cache-1", "label": "Redis", "type": "connection"},

        # API Server to Storage
        {"from": "ec2-api", "to": "s3-products", "label": "Read", "type": "connection"},

        # API Server to Logs
        {"from": "ec2-api", "to": "logs-1", "label": "Write Logs", "type": "connection"},

        # API Server to Queue
        {"from": "ec2-api", "to": "sqs-1", "label": "Send", "type": "connection"},

        # API Server to KMS
        {"from": "ec2-api", "to": "kms-1", "label": "Decrypt", "type": "connection"},

        # Lambda Worker to SQS
        {"from": "lambda-worker", "to": "sqs-1", "label": "Receive", "type": "connection"},

        # Lambda Worker to Database
        {"from": "lambda-worker", "to": "rds-primary", "label": "PostgreSQL", "type": "connection"},

        # Lambda Worker to Storage
        {"from": "lambda-worker", "to": "s3-backups", "label": "Write", "type": "connection"},

        # Lambda Worker to SNS
        {"from": "lambda-worker", "to": "sns-1", "label": "Publish", "type": "connection"},

        # Lambda Worker to Logs
        {"from": "lambda-worker", "to": "logs-1", "label": "Write Logs", "type": "connection"},

        # Lambda Worker to KMS
        {"from": "lambda-worker", "to": "kms-1", "label": "Decrypt", "type": "connection"},
    ]
}


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "COMPLETE SECURITY IMPLEMENTATION - E-COMMERCE ARCHITECTURE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Initialize orchestrator
    orchestrator = CompleteSecurityOrchestrator(
        namespace="ecommerce-prod",
        vpc_id="var.vpc_id",
        region="${data.aws_region.current.name}"
    )

    # Execute complete implementation
    result = orchestrator.execute_complete_implementation(ECOMMERCE_ARCHITECTURE)

    # Print summary
    orchestrator.print_summary(result)

    # Save generated files
    if result.terraform_files:
        print()
        print("=" * 80)
        print("TERRAFORM FILES GENERATED")
        print("=" * 80)
        print()

        # Show a sample of each file
        if "security_groups.tf" in result.terraform_files:
            print("--- security_groups.tf (first 50 lines) ---")
            lines = result.terraform_files["security_groups.tf"].split("\n")[:50]
            print("\n".join(lines))
            print(f"\n... ({len(result.terraform_files['security_groups.tf'].split(chr(10)))} total lines)")
            print()

        if "iam_roles.tf" in result.terraform_files:
            print("--- iam_roles.tf (first 50 lines) ---")
            lines = result.terraform_files["iam_roles.tf"].split("\n")[:50]
            print("\n".join(lines))
            print(f"\n... ({len(result.terraform_files['iam_roles.tf'].split(chr(10)))} total lines)")
            print()

        if "attachments.tf" in result.terraform_files:
            print("--- attachments.tf (first 30 lines) ---")
            lines = result.terraform_files["attachments.tf"].split("\n")[:30]
            print("\n".join(lines))
            print(f"\n... ({len(result.terraform_files['attachments.tf'].split(chr(10)))} total lines)")
            print()

        # Save to files
        print("=" * 80)
        print("SAVING FILES")
        print("=" * 80)
        for filename, content in result.terraform_files.items():
            output_path = f"/tmp/terraform_{filename}"
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"✓ {filename} ({len(content)} bytes)")

    # Print statistics
    print()
    print("=" * 80)
    print("IMPLEMENTATION STATISTICS")
    print("=" * 80)
    print()
    if result.stats:
        for key, value in result.stats.items():
            print(f"  {key}: {value}")
    print()

    # Print detailed traffic analysis
    if result.traffic_analysis:
        print("=" * 80)
        print("TRAFFIC FLOW ANALYSIS")
        print("=" * 80)
        print()
        print(f"Total Connections: {result.traffic_analysis.get('total_connections', 0)}")
        print()
        print("Connection Details:")
        for edge in result.traffic_analysis.get('edges', [])[:15]:
            protocol_display = edge['protocol'] or "N/A"
            print(f"  {edge['direction']:<45} Protocol: {protocol_display:<6} Port: {edge['port']}")
        if len(result.traffic_analysis.get('edges', [])) > 15:
            print(f"  ... and {len(result.traffic_analysis['edges']) - 15} more")
        print()

    print("=" * 80)
    print("✅ COMPLETE SECURITY IMPLEMENTATION FINISHED")
    print("=" * 80)
    print()
    print("The generated Terraform files are production-ready and include:")
    print("  • Security groups with intelligent rule generation")
    print("  • IAM roles with least-privilege policies")
    print("  • Resource-to-role linkings")
    print("  • Complete validation pipeline")
    print()


if __name__ == "__main__":
    main()
