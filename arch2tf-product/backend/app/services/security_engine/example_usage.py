"""
Example: Using the security configuration generator
Demonstrates complete workflow from diagram to Terraform code
"""
import json
from orchestrator import SecurityConfigOrchestrator, SecurityConfigExporter


# Example: 3-tier architecture diagram
EXAMPLE_RESOURCE_GRAPH = {
    "nodes": [
        {
            "id": "vpc-1",
            "label": "VPC",
            "type": "aws_vpc",
            "metadata": {"cidr": "10.0.0.0/16"}
        },
        {
            "id": "alb-1",
            "label": "Public ALB",
            "type": "aws_lb",
            "metadata": {"scheme": "internet-facing"}
        },
        {
            "id": "ec2-1",
            "label": "App Server",
            "type": "aws_instance",
            "metadata": {"ami": "ami-0c55b159cbfafe1f0"}
        },
        {
            "id": "rds-1",
            "label": "Database",
            "type": "aws_db_instance",
            "metadata": {"engine": "mysql", "allocated_storage": 20}
        },
        {
            "id": "s3-1",
            "label": "Data Bucket",
            "type": "aws_s3_bucket",
            "metadata": {}
        }
    ],
    "edges": [
        {
            "from": "alb-1",
            "to": "ec2-1",
            "type": "connection",
            "label": "HTTP/HTTPS"
        },
        {
            "from": "ec2-1",
            "to": "rds-1",
            "type": "connection",
            "label": "MySQL"
        },
        {
            "from": "ec2-1",
            "to": "s3-1",
            "type": "connection",
            "label": "Read/Write"
        }
    ]
}


def main():
    print("=" * 80)
    print("TERRAFORM ACCELERATORS - Security Configuration Generation")
    print("=" * 80)
    print()

    # Initialize orchestrator
    orchestrator = SecurityConfigOrchestrator(
        namespace="prod",
        vpc_id="${aws_vpc.main.id}"
    )

    # Generate complete security configuration
    print("Generating security configuration from diagram...")
    print()
    result = orchestrator.generate_full_pipeline(EXAMPLE_RESOURCE_GRAPH)

    # Print summary
    if result["status"] == "success":
        print(result["summary"])

        # Export to JSON for inspection
        json_export = SecurityConfigExporter.export_to_json(
            result["security_config"],
            result["iam_roles"]
        )

        print("\n" + "=" * 80)
        print("GENERATED TERRAFORM CODE")
        print("=" * 80)
        print()

        # Print security_groups.tf
        print("--- security_groups.tf ---")
        print(result["terraform_files"]["security_groups.tf"])
        print()

        # Print iam.tf
        print("--- iam.tf ---")
        print(result["terraform_files"]["iam.tf"])
        print()

        # Print validation script
        print("--- validate_security.sh ---")
        print(result["terraform_files"]["validate_security.sh"])
        print()

        # Print JSON export for programmatic use
        print("=" * 80)
        print("SECURITY CONFIGURATION (JSON)")
        print("=" * 80)
        print(json.dumps(json_export, indent=2))

    else:
        print(f"❌ Error: {result['error']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
