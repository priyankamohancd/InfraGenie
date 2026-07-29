"""
Traffic flow analyzer - determines protocol, port, and direction from diagram
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum
from models import ResourceType


class ProtocolType(Enum):
    HTTP = "tcp:80"
    HTTPS = "tcp:443"
    SSH = "tcp:22"
    MYSQL = "tcp:3306"
    POSTGRESQL = "tcp:5432"
    REDIS = "tcp:6379"
    MEMCACHED = "tcp:11211"
    MONGODB = "tcp:27017"
    REST_API = "tcp:8080"
    WEBSOCKET = "tcp:8080"
    ICMP = "icmp"
    UDP = "udp:53"


class TrafficFlowAnalyzer:
    """
    Analyzes diagram to determine:
    - Traffic direction (source → destination)
    - Protocol and port
    - Bandwidth/latency requirements
    - Security sensitivity level
    """

    # AWS-managed services reached over the AWS API (IAM), not a routable
    # network port a security group could reference. Edges into these
    # resource types have no protocol/port - they need an IAM policy, not
    # a security group rule.
    IAM_ONLY_TARGET_TYPES = {
        "aws_s3_bucket",
        "aws_lambda_function",
        "aws_sqs_queue",
        "aws_sns_topic",
        "aws_dynamodb_table",
        "aws_secretsmanager_secret",
        "aws_kms_key",
        "aws_cloudwatch_log_group",
        "aws_ssm_parameter",
        "aws_sfn_state_machine",
        "aws_events_rule",
    }

    def __init__(self):
        # Resource type to default ports
        self.resource_protocols = {
            ResourceType.ALB: {
                "default": (80, 443),
                "protocol": "tcp"
            },
            ResourceType.RDS: {
                "mysql": (3306, 3306),
                "postgres": (5432, 5432),
                "mariadb": (3306, 3306),
                "oracle": (1521, 1521),
                "sqlserver": (1433, 1433),
                "default": (3306, 3306)
            },
            ResourceType.EC2: {
                "ssh": (22, 22),
                "http": (80, 80),
                "https": (443, 443),
                "default": (8080, 8080)
            },
            ResourceType.ELASTICACHE: {
                "redis": (6379, 6379),
                "memcached": (11211, 11211),
                "default": (6379, 6379)
            }
        }

        # Edge labels to protocol inference
        self.label_to_protocol = {
            "http": ("tcp", 80, 80),
            "https": ("tcp", 443, 443),
            "ssh": ("tcp", 22, 22),
            "mysql": ("tcp", 3306, 3306),
            "postgres": ("tcp", 5432, 5432),
            "postgresql": ("tcp", 5432, 5432),
            "redis": ("tcp", 6379, 6379),
            "memcached": ("tcp", 11211, 11211),
            "mongodb": ("tcp", 27017, 27017),
            "rest": ("tcp", 8080, 8080),
            "api": ("tcp", 8080, 8080),
            "websocket": ("tcp", 8080, 8080),
            "dns": ("udp", 53, 53),
            "icmp": ("icmp", None, None),
        }

    def analyze_edge(self, from_node: Dict, to_node: Dict, edge: Dict) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Determine protocol and port from edge connection

        Args:
            from_node: Source resource node
            to_node: Destination resource node
            edge: Edge connecting them

        Returns:
            (protocol, from_port, to_port) - protocol is None when the target
            is an AWS-managed service reached via IAM/API, not a network port.
        """
        to_type = to_node.get('type')

        # A connection into an IAM-only service (S3, SQS, SNS, DynamoDB, KMS,
        # etc.) never has a network protocol/port, regardless of edge label -
        # checked first so a misleading label (e.g. "Write") can't override it.
        if to_type in self.IAM_ONLY_TARGET_TYPES:
            return None, None, None

        # Step 1: Check edge label for explicit protocol
        edge_label = (edge.get('label') or '').lower()
        if edge_label in self.label_to_protocol:
            return self.label_to_protocol[edge_label]

        # Step 2: Infer from resource type combination
        from_type = from_node.get('type')
        protocol, from_port, to_port = self._infer_from_types(from_type, to_type, to_node)

        if protocol:
            return protocol, from_port, to_port

        # Step 3: Default to TCP 8080 (compute-to-compute traffic with no
        # more specific signal available)
        return "tcp", 8080, 8080

    def _infer_from_types(self, from_type: str, to_type: str, to_node: Dict) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Infer protocol based on resource types"""

        # ALB → * : HTTP/HTTPS
        if from_type == ResourceType.ALB.value:
            return "tcp", 80, 80

        # EC2 → RDS : MySQL (check metadata)
        if from_type == ResourceType.EC2.value and to_type == ResourceType.RDS.value:
            db_engine = to_node.get('metadata', {}).get('engine', 'mysql')
            if db_engine == 'postgres':
                return "tcp", 5432, 5432
            return "tcp", 3306, 3306

        # EC2 → ElastiCache : Redis (check metadata)
        if from_type == ResourceType.EC2.value and to_type == ResourceType.ELASTICACHE.value:
            cache_type = to_node.get('metadata', {}).get('engine', 'redis')
            if cache_type == 'memcached':
                return "tcp", 11211, 11211
            return "tcp", 6379, 6379

        # EC2 → S3 : No network port (IAM only)
        if to_type == ResourceType.S3.value:
            return None, None, None

        # EC2 → Lambda : No network port (API Gateway)
        if to_type == ResourceType.LAMBDA.value:
            return None, None, None

        return None, None, None

    def is_resource_public(self, node: Dict) -> bool:
        """Determine if resource is public-facing"""
        label = (node.get('label') or '').lower()
        resource_type = node.get('type')

        # Explicit labels
        if 'public' in label:
            return True
        if 'private' in label:
            return False

        # Type-based defaults
        if resource_type == ResourceType.ALB.value:
            # Default ALB is internet-facing unless labeled private
            return True

        return False

    def infer_resource_security_level(self, node: Dict, outbound_connections: List[Dict]) -> str:
        """
        Infer security sensitivity level

        Returns: "public", "internal", "data"
        """
        label = (node.get('label') or '').lower()
        resource_type = node.get('type')

        # Public resources
        if resource_type == ResourceType.ALB.value:
            return "public"
        if 'public' in label:
            return "public"

        # Data resources (high sensitivity)
        if resource_type in [ResourceType.RDS.value, ResourceType.DYNAMODB.value]:
            return "data"
        if 'database' in label or 'db' in label:
            return "data"

        # Internal resources (default)
        return "internal"

    def get_minimum_outbound_rules(self, from_type: str) -> List[Tuple[str, int, int, str]]:
        """
        Get minimum necessary outbound rules for resource type

        Returns:
            [(protocol, from_port, to_port, description), ...]
        """
        rules = []

        # All resources need HTTPS for package updates
        rules.append(("tcp", 443, 443, "HTTPS for updates"))

        # EC2 instances might need DNS
        if from_type == ResourceType.EC2.value:
            rules.append(("udp", 53, 53, "DNS queries"))

        # RDS needs NTP
        if from_type == ResourceType.RDS.value:
            rules.append(("udp", 123, 123, "NTP time sync"))

        return rules

    def detect_implicit_flows(self, resource_graph: Dict) -> List[Dict]:
        """
        Detect implicit traffic flows not explicitly in diagram edges

        Examples:
        - All resources need outbound HTTPS for updates
        - EC2 needs DNS (53/udp)
        - RDS needs NTP (123/udp)
        """
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}
        implicit_edges = []

        for node_id, node in nodes.items():
            resource_type = node.get('type')

            # All resources: outbound HTTPS (for package managers, API calls)
            implicit_edges.append({
                'from': node_id,
                'to': 'internet',
                'label': 'HTTPS',
                'implicit': True,
                'description': 'Outbound HTTPS for updates/APIs'
            })

            # EC2: needs DNS
            if resource_type == ResourceType.EC2.value:
                implicit_edges.append({
                    'from': node_id,
                    'to': 'dns-resolver',
                    'label': 'DNS',
                    'implicit': True,
                    'description': 'DNS name resolution'
                })

        return implicit_edges

    def validate_traffic_flow(self, resource_graph: Dict) -> List[str]:
        """
        Validate traffic flow for issues

        Returns:
            List of warnings/issues
        """
        issues = []
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}
        edges = resource_graph.get('edges', [])

        # Check for isolated resources
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge['from'])
            connected_nodes.add(edge['to'])

        for node_id, node in nodes.items():
            if node_id not in connected_nodes:
                issues.append(f"⚠️  Resource '{node.get('label', node_id)}' is not connected to any other resource")

        # Check for ALB without downstream resources
        for node_id, node in nodes.items():
            if node.get('type') == ResourceType.ALB.value:
                has_outbound = any(e['from'] == node_id for e in edges)
                if not has_outbound:
                    issues.append(f"⚠️  ALB '{node.get('label', node_id)}' has no connected backend resources")

        # Check for data resources without access controls
        for node_id, node in nodes.items():
            if node.get('type') in [ResourceType.RDS.value, ResourceType.DYNAMODB.value]:
                has_inbound = any(e['to'] == node_id for e in edges)
                if not has_inbound:
                    issues.append(f"⚠️  Data resource '{node.get('label', node_id)}' accepts connections from nowhere (will deny all)")

        return issues


class TrafficFlowVisualizer:
    """Generate ASCII diagram of traffic flow"""

    @staticmethod
    def visualize_flow(resource_graph: Dict) -> str:
        """Create ASCII visualization of traffic flow"""
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}
        edges = resource_graph.get('edges', [])

        # Build adjacency
        graph_viz = {}
        for edge in edges:
            from_id = edge['from']
            to_id = edge['to']
            label = edge.get('label', '')

            from_label = nodes.get(from_id, {}).get('label', from_id)
            to_label = nodes.get(to_id, {}).get('label', to_id)

            if from_label not in graph_viz:
                graph_viz[from_label] = []
            graph_viz[from_label].append((to_label, label))

        # Generate ASCII
        lines = ["Traffic Flow Diagram:", ""]

        for source, targets in graph_viz.items():
            for i, (target, label) in enumerate(targets):
                connector = "└→" if i == len(targets) - 1 else "├→"
                lines.append(f"{source} {connector} {target} ({label})")

        return "\n".join(lines)


# Traffic flow patterns (for documentation)
TRAFFIC_PATTERNS = {
    "web_tier": {
        "description": "Web tier (ALB)",
        "inbound": ["0.0.0.0/0:80", "0.0.0.0/0:443"],
        "outbound": ["app_tier:8080", "internet:443"],
    },
    "app_tier": {
        "description": "Application tier (EC2)",
        "inbound": ["web_tier:80", "web_tier:443"],
        "outbound": ["data_tier:3306", "cache_tier:6379", "internet:443"],
    },
    "data_tier": {
        "description": "Data tier (RDS)",
        "inbound": ["app_tier:3306"],
        "outbound": ["internet:443"],  # For backups, updates
    },
    "cache_tier": {
        "description": "Cache tier (ElastiCache)",
        "inbound": ["app_tier:6379"],
        "outbound": ["internet:443"],
    }
}
