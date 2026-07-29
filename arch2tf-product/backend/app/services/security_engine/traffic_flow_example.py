"""
Example: How traffic flow is detected from diagram
"""
import json
from traffic_flow_analyzer import TrafficFlowAnalyzer, TrafficFlowVisualizer


# Example architecture diagram with traffic flow hints
EXAMPLE_DIAGRAM = {
    "nodes": [
        {
            "id": "alb-1",
            "label": "Public ALB",
            "type": "aws_lb",
            "metadata": {
                "scheme": "internet-facing",
                "port": 80
            }
        },
        {
            "id": "ec2-1",
            "label": "App Server",
            "type": "aws_instance",
            "metadata": {
                "role": "web",
                "port": 8080
            }
        },
        {
            "id": "ec2-2",
            "label": "App Server 2",
            "type": "aws_instance",
            "metadata": {
                "role": "api",
                "port": 5000
            }
        },
        {
            "id": "rds-1",
            "label": "Primary Database",
            "type": "aws_db_instance",
            "metadata": {
                "engine": "postgres",
                "allocated_storage": 100
            }
        },
        {
            "id": "rds-2",
            "label": "Read Replica",
            "type": "aws_db_instance",
            "metadata": {
                "engine": "postgres",
                "allocated_storage": 100
            }
        },
        {
            "id": "cache-1",
            "label": "Redis Cache",
            "type": "aws_elasticache_cluster",
            "metadata": {
                "engine": "redis",
                "node_type": "cache.t3.small"
            }
        },
        {
            "id": "s3-1",
            "label": "Static Assets",
            "type": "aws_s3_bucket",
            "metadata": {}
        }
    ],
    "edges": [
        # Internet → ALB
        {
            "from": "internet",
            "to": "alb-1",
            "label": "HTTP/HTTPS",
            "type": "connection"
        },
        # ALB → App Servers
        {
            "from": "alb-1",
            "to": "ec2-1",
            "label": "HTTP",
            "type": "connection"
        },
        {
            "from": "alb-1",
            "to": "ec2-2",
            "label": "REST API",
            "type": "connection"
        },
        # App → Database
        {
            "from": "ec2-1",
            "to": "rds-1",
            "label": "PostgreSQL",
            "type": "connection"
        },
        {
            "from": "ec2-2",
            "to": "rds-1",
            "label": "PostgreSQL",
            "type": "connection"
        },
        # Read Replica (for reporting)
        {
            "from": "ec2-2",
            "to": "rds-2",
            "label": "SELECT",
            "type": "connection"
        },
        # App → Cache
        {
            "from": "ec2-1",
            "to": "cache-1",
            "label": "Redis",
            "type": "connection"
        },
        # App → S3
        {
            "from": "ec2-1",
            "to": "s3-1",
            "label": "GetObject",
            "type": "connection"
        }
    ]
}


def main():
    print("=" * 80)
    print("TRAFFIC FLOW ANALYSIS")
    print("=" * 80)
    print()

    # Initialize analyzer
    analyzer = TrafficFlowAnalyzer()

    # Step 1: Visualize traffic flow
    print("Step 1: Diagram Traffic Flow")
    print("-" * 80)
    viz = TrafficFlowVisualizer.visualize_flow(EXAMPLE_DIAGRAM)
    print(viz)
    print()

    # Step 2: Analyze each edge
    print("Step 2: Protocol Inference for Each Connection")
    print("-" * 80)

    nodes = {node['id']: node for node in EXAMPLE_DIAGRAM.get('nodes', [])}

    for i, edge in enumerate(EXAMPLE_DIAGRAM.get('edges', []), 1):
        from_id = edge['from']
        to_id = edge['to']
        from_node = nodes.get(from_id, {'label': from_id})
        to_node = nodes.get(to_id, {'label': to_id})

        protocol, from_port, to_port = analyzer.analyze_edge(from_node, to_node, edge)

        print(f"{i}. {from_node['label']} → {to_node['label']}")
        print(f"   Edge Label: '{edge.get('label', 'none')}'")
        print(f"   Protocol: {protocol}, Port: {from_port}:{to_port}")
        print()

    # Step 3: Identify implicit flows
    print("Step 3: Implicit Traffic Flows (Auto-Generated)")
    print("-" * 80)
    implicit = analyzer.detect_implicit_flows(EXAMPLE_DIAGRAM)
    for flow in implicit:
        if flow.get('implicit'):
            print(f"• {flow['description']}")
            print(f"  From: {nodes.get(flow['from'], {}).get('label', flow['from'])}")
            print(f"  To: {flow['to']}")
            print()

    # Step 4: Resource security levels
    print("Step 4: Security Level Classification")
    print("-" * 80)
    for node_id, node in nodes.items():
        outbound = [e for e in EXAMPLE_DIAGRAM.get('edges', []) if e['from'] == node_id]
        level = analyzer.infer_resource_security_level(node, outbound)
        is_public = analyzer.is_resource_public(node)
        print(f"• {node['label']:<20} Level: {level:<8} Public: {is_public}")
    print()

    # Step 5: Validate traffic flow
    print("Step 5: Traffic Flow Validation")
    print("-" * 80)
    issues = analyzer.validate_traffic_flow(EXAMPLE_DIAGRAM)
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ No traffic flow issues detected")
    print()

    # Step 6: Show how this maps to security groups
    print("Step 6: Resulting Security Group Rules")
    print("-" * 80)
    print()
    print("PUBLIC ALB Security Group (alb-1-sg)")
    print("  Inbound:")
    print("    ✓ 0.0.0.0/0:80 (HTTP from Internet)")
    print("    ✓ 0.0.0.0/0:443 (HTTPS from Internet)")
    print("  Outbound:")
    print("    ✓ app-server-sg:8080 (to App Server 1)")
    print("    ✓ app-server-sg:5000 (to App Server 2)")
    print()

    print("APP SERVER Security Group (app-server-sg)")
    print("  Inbound:")
    print("    ✓ alb-sg:80 (HTTP from ALB)")
    print("    ✓ alb-sg:443 (HTTPS from ALB)")
    print("  Outbound:")
    print("    ✓ rds-sg:5432 (PostgreSQL to Database)")
    print("    ✓ cache-sg:6379 (Redis to Cache)")
    print("    ✓ 0.0.0.0/0:443 (HTTPS to Internet for S3)")
    print()

    print("DATABASE Security Group (rds-sg)")
    print("  Inbound:")
    print("    ✓ app-server-sg:5432 (PostgreSQL from App Server 1)")
    print("    ✓ app-server-sg:5432 (PostgreSQL from App Server 2)")
    print("  Outbound:")
    print("    ✓ 0.0.0.0/0:443 (HTTPS for backups/updates)")
    print()

    print("CACHE Security Group (cache-sg)")
    print("  Inbound:")
    print("    ✓ app-server-sg:6379 (Redis from App Server)")
    print("  Outbound:")
    print("    ✓ 0.0.0.0/0:443 (HTTPS for updates)")
    print()

    # Step 7: Export as JSON for programmatic use
    print("Step 7: Protocol Mapping (JSON)")
    print("-" * 80)
    protocol_map = {}
    for i, edge in enumerate(EXAMPLE_DIAGRAM.get('edges', []), 1):
        from_id = edge['from']
        to_id = edge['to']
        from_node = nodes.get(from_id, {'label': from_id})
        to_node = nodes.get(to_id, {'label': to_id})

        protocol, from_port, to_port = analyzer.analyze_edge(from_node, to_node, edge)
        protocol_map[f"{from_node['label']} → {to_node['label']}"] = {
            "protocol": protocol,
            "port": f"{from_port}:{to_port}" if from_port else "N/A (IAM only)",
            "edge_label": edge.get('label', 'implicit'),
            "from_type": from_node.get('type'),
            "to_type": to_node.get('type'),
            "to_metadata": to_node.get('metadata', {})
        }

    print(json.dumps(protocol_map, indent=2))
    print()

    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. EDGE LABEL determines protocol first")
    print("   - 'PostgreSQL' → tcp:5432")
    print("   - 'Redis' → tcp:6379")
    print("   - 'HTTP' → tcp:80")
    print()
    print("2. RESOURCE TYPE determines default if label unclear")
    print("   - ALB → always 80/443")
    print("   - RDS → check metadata for engine type")
    print("   - ElastiCache → check metadata for redis/memcached")
    print()
    print("3. TO_NODE METADATA refines the port")
    print("   - RDS 'engine: postgres' → 5432 (not default 3306)")
    print("   - ElastiCache 'engine: redis' → 6379")
    print()
    print("4. IMPLICIT FLOWS are auto-generated")
    print("   - All resources: outbound HTTPS for updates")
    print("   - EC2: DNS (53/udp)")
    print("   - Databases: NTP (123/udp)")
    print()
    print("5. DIRECTION is edge direction")
    print("   - 'from' → 'to' creates inbound rule on 'to'")
    print("   - Symmetric: app→db creates inbound on db")
    print()


if __name__ == "__main__":
    main()
