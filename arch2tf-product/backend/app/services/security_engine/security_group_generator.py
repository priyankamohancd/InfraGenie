"""
Security group generation from architecture diagrams
"""
from typing import Dict, List, Set, Tuple, Optional
from models import (
    SecurityGroup, SecurityGroupRule, SecurityConfiguration,
    RuleType, ResourceType, SERVICE_PORTS
)


class SecurityGroupGenerator:
    """Generates security groups from diagram resource graph"""

    # Resource types that get a real aws_security_group resource
    SG_ELIGIBLE_TYPES = {
        ResourceType.EC2.value,
        ResourceType.ALB.value,
        ResourceType.RDS.value,
        ResourceType.ELASTICACHE.value,
    }

    # Diagram custom-data tag values (case-insensitive, read from
    # node['tags']['tier']) that explicitly mark a resource's network
    # exposure. These take priority over the older "public" in node-label
    # heuristic below, which remains as the fallback for diagrams that don't
    # use tags at all so existing behavior is unchanged for them. See
    # arch2terraform's DiagramNode.tags / drawio+excalidraw "Edit
    # Data"/customData ingestion for how a diagram author sets this tag.
    PUBLIC_TIER_VALUES = {"public"}
    PRIVATE_TIER_VALUES = {"private", "internal"}

    # Fallback CIDR used for ingress that's explicitly tier=private/internal
    # but would otherwise (by resource type / edge shape) have been opened to
    # 0.0.0.0/0 - broad enough to cover any RFC1918 10.x VPC/subnet scheme
    # without needing to know the diagram's actual VPC CIDR at generation
    # time (the same "safe generic default" convention used elsewhere in this
    # project, e.g. the fake ARN/ID placeholders in arch2terraform's catalog).
    DEFAULT_INTERNAL_CIDR = "10.0.0.0/8"

    def __init__(self, namespace: str = "default", vpc_id: str = "var.vpc_id",
                 traffic_analyzer=None, internal_cidr: str = DEFAULT_INTERNAL_CIDR):
        """
        Args:
            namespace: resource name prefix
            vpc_id: bare HCL reference for the VPC id (no ${...} wrapper -
                this is emitted unquoted, e.g. "var.vpc_id" or "aws_vpc.main.id")
            traffic_analyzer: optional TrafficFlowAnalyzer instance. When given,
                protocol/port for each rule are taken from its 3-level inference
                instead of this generator's own crude target-type-only defaults.
            internal_cidr: CIDR to scope ingress to when a resource is tagged
                tier=private/internal but would otherwise have gotten an
                internet-open rule.
        """
        self.namespace = namespace
        self.vpc_id = vpc_id
        self.traffic_analyzer = traffic_analyzer
        self.internal_cidr = internal_cidr
        self.config = SecurityConfiguration(namespace=namespace)

    def _resolve_tier_public(self, node: Optional[Dict]) -> Optional[bool]:
        """
        Returns True if node['tags']['tier'] explicitly marks it public,
        False if explicitly private/internal, or None if untagged (caller
        should fall back to its own legacy heuristic, e.g. "public" in label).
        """
        if not node:
            return None
        tier = (node.get('tags') or {}).get('tier', '')
        tier = tier.strip().lower() if isinstance(tier, str) else ''
        if tier in self.PUBLIC_TIER_VALUES:
            return True
        if tier in self.PRIVATE_TIER_VALUES:
            return False
        return None

    def generate_from_diagram(self, resource_graph: Dict) -> SecurityConfiguration:
        """
        Generate security configuration from resource graph

        Args:
            resource_graph: {
                nodes: [{id, label, type, metadata}],
                edges: [{from, to, type, label}]
            }
        """
        nodes = {node['id']: node for node in resource_graph.get('nodes', [])}
        edges = resource_graph.get('edges', [])

        # Step 1: Create security groups for all resources that need them
        self._create_security_groups(nodes)

        # Step 2: Extract rules from edges
        self._extract_rules_from_edges(edges, nodes)

        # Step 3: Add default rules
        self._add_default_rules(nodes)

        return self.config

    def _create_security_groups(self, nodes: Dict):
        """Create security group for each resource that needs one"""
        for node_id, node in nodes.items():
            resource_type = node.get('type')

            # Determine if resource needs a security group
            needs_sg = resource_type in self.SG_ELIGIBLE_TYPES

            if needs_sg:
                sg = self._create_security_group_for_resource(node_id, node)
                self.config.add_security_group(sg)

    def _create_security_group_for_resource(self, resource_id: str, node: Dict) -> SecurityGroup:
        """Create a security group for a single resource"""
        resource_label = node.get('label', resource_id)
        sg_name = f"{self.namespace}-{resource_label.lower().replace(' ', '-')}-sg"
        # Real bug found 2026-07-21 while testing tier-tag ingress rules: this
        # used to build its own resource-name string inline, replacing only
        # spaces (not hyphens) — while _get_sg_resource_name() (used by every
        # OTHER lookup: _extract_rules_from_edges, _add_default_rules) also
        # replaces hyphens. Any hyphenated label (e.g. "public-alb",
        # "web-server" — an extremely common naming style) meant the two
        # naming schemes diverged, so config.security_groups.get(sg_resource)
        # silently returned None and _add_default_rules's `if not sg:
        # continue` skipped default-rule generation entirely for that
        # resource. Fixed by reusing the one real naming function instead of
        # a second, subtly different implementation of the same thing.
        sg_resource_name = self._get_sg_resource_name(resource_label)

        sg = SecurityGroup(
            name=sg_name,
            resource_name=sg_resource_name,
            vpc_id=self.vpc_id,
            description=f"Security group for {resource_label}",
            tags={
                "Name": sg_name,
                "ManagedBy": "terraform-accelerators",
                "Resource": resource_label
            }
        )
        return sg

    def _extract_rules_from_edges(self, edges: List[Dict], nodes: Dict):
        """Extract security group rules from diagram edges"""
        for edge in edges:
            from_node_id = edge.get('from')
            to_node_id = edge.get('to')

            from_node = nodes.get(from_node_id, {})
            to_node = nodes.get(to_node_id, {})

            from_type = from_node.get('type')
            to_type = to_node.get('type')

            from_label = from_node.get('label', from_node_id)
            to_label = to_node.get('label', to_node_id)

            # Get target security group (inbound rule)
            to_sg_resource = self._get_sg_resource_name(to_label)

            if to_type in [ResourceType.EC2.value, ResourceType.RDS.value, ResourceType.ELASTICACHE.value]:
                # Create inbound rule on target
                rule = self._create_inbound_rule(
                    from_node_id, from_label, from_type,
                    to_node_id, to_label, to_type,
                    from_node=from_node, to_node=to_node, edge=edge
                )
                if rule:
                    to_sg = self.config.security_groups.get(to_sg_resource)
                    if to_sg:
                        to_sg.add_inbound_rule(rule)

    def _create_inbound_rule(self, from_id: str, from_label: str, from_type: str,
                            to_id: str, to_label: str, to_type: str,
                            from_node: Optional[Dict] = None,
                            to_node: Optional[Dict] = None,
                            edge: Optional[Dict] = None) -> Optional[SecurityGroupRule]:
        """Create inbound rule based on source and target resource types"""

        # Prefer the traffic analyzer's 3-level inference (edge label -> resource
        # type -> node metadata) when available; fall back to the crude
        # target-type-only default otherwise.
        protocol, from_port, to_port = None, None, None
        if self.traffic_analyzer is not None and from_node is not None and to_node is not None:
            protocol, from_port, to_port = self.traffic_analyzer.analyze_edge(
                from_node, to_node, edge or {}
            )
        if protocol is None:
            protocol, from_port, to_port = self._infer_protocol_and_port(to_type)

        if protocol is None:
            return None

        # Only reference a source security group if the source resource
        # actually gets one created (see SG_ELIGIBLE_TYPES) - otherwise this
        # would be a dangling reference to an aws_security_group resource
        # that's never declared (e.g. Lambda has no SG in this model).
        source_sg_resource = None
        if from_type in self.SG_ELIGIBLE_TYPES:
            source_sg_resource = self._get_sg_resource_name(from_label)

        # Tag takes priority over the legacy "public" in label heuristic;
        # None (untagged) falls back to that heuristic unchanged.
        tier_public = self._resolve_tier_public(from_node)
        is_from_public = tier_public if tier_public is not None else (
            from_type == ResourceType.ALB.value and "public" in from_label.lower()
        )
        source_cidr = None
        if not source_sg_resource and from_type == ResourceType.ALB.value:
            if is_from_public:
                source_cidr = "0.0.0.0/0"
            elif tier_public is False:
                # Explicitly tagged private/internal - scope to the internal
                # CIDR instead of silently emitting no rule at all, so an
                # "internal ALB" still reaches its target from within the VPC.
                source_cidr = self.internal_cidr

        rule = SecurityGroupRule(
            rule_id=f"{from_label}-to-{to_label}",
            type=RuleType.INGRESS,
            protocol=protocol,
            from_port=from_port,
            to_port=to_port,
            source_sg_id=f"aws_security_group.{source_sg_resource}.id"
            if source_sg_resource else None,
            source_cidr=source_cidr,
            description=f"{from_label} to {to_label}",
            source_resource_name=from_label,
            destination_resource_name=to_label
        )

        return rule

    def _infer_protocol_and_port(self, resource_type: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Infer protocol and port based on resource type"""
        if resource_type == ResourceType.RDS.value:
            # Default to MySQL
            return "tcp", 3306, 3306
        elif resource_type == ResourceType.ELASTICACHE.value:
            # Default to Redis
            return "tcp", 6379, 6379
        elif resource_type == ResourceType.EC2.value:
            # Allow application traffic (assume 8080)
            return "tcp", 8080, 8080
        return None, None, None

    def _add_default_rules(self, nodes: Dict):
        """Add default rules for public resources"""
        for node_id, node in nodes.items():
            node_label = node.get('label', node_id).lower()
            resource_type = node.get('type')

            sg_resource = self._get_sg_resource_name(node.get('label', node_id))
            sg = self.config.security_groups.get(sg_resource)

            if not sg:
                continue

            # ALB ingress: tier=public tag (or, if untagged, the legacy
            # "public" in label heuristic) opens HTTP/HTTPS to the internet;
            # an explicit tier=private/internal tag scopes the same rules to
            # the internal CIDR instead of skipping them, so an internal ALB
            # still gets reachable from within the VPC.
            if resource_type == ResourceType.ALB.value:
                tier_public = self._resolve_tier_public(node)
                is_public = tier_public if tier_public is not None else ("public" in node_label)

                if is_public:
                    alb_cidr = "0.0.0.0/0"
                elif tier_public is False:
                    alb_cidr = self.internal_cidr
                else:
                    alb_cidr = None

                if alb_cidr:
                    self._add_rule_to_sg(sg, SecurityGroupRule(
                        rule_id=f"{sg_resource}-http",
                        type=RuleType.INGRESS,
                        protocol="tcp",
                        from_port=80,
                        to_port=80,
                        source_cidr=alb_cidr,
                        description="HTTP from Internet" if alb_cidr == "0.0.0.0/0" else "HTTP from internal network"
                    ))
                    self._add_rule_to_sg(sg, SecurityGroupRule(
                        rule_id=f"{sg_resource}-https",
                        type=RuleType.INGRESS,
                        protocol="tcp",
                        from_port=443,
                        to_port=443,
                        source_cidr=alb_cidr,
                        description="HTTPS from Internet" if alb_cidr == "0.0.0.0/0" else "HTTPS from internal network"
                    ))

            # All resources: scoped outbound instead of a blanket allow-all.
            # Real fix 2026-07-24: protocol="-1"/0-65535/0.0.0.0/0 (allow
            # every port to anywhere) is exactly what Checkov's CKV_AWS_382
            # flags ("Ensure no security groups allow egress from 0.0.0.0/0
            # to port -1"). Every generated resource still needs to reach
            # the internet/AWS APIs for package installs, TLS calls, etc.,
            # so instead of removing egress outright (which would silently
            # break real deployments) this scopes it to the handful of
            # ports that cover that need: HTTPS (443, the overwhelming
            # majority of AWS API/package-registry/outbound-webhook
            # traffic), HTTP (80, redirects/legacy endpoints) and DNS
            # (UDP/53, required for the above to resolve at all).
            for rule_suffix, protocol, port, description in (
                ("outbound-https", "tcp", 443, "HTTPS outbound (AWS APIs, package registries, TLS calls)"),
                ("outbound-http", "tcp", 80, "HTTP outbound (redirects, legacy endpoints)"),
                ("outbound-dns", "udp", 53, "DNS resolution"),
            ):
                self._add_rule_to_sg(sg, SecurityGroupRule(
                    rule_id=f"{sg_resource}-{rule_suffix}",
                    type=RuleType.EGRESS,
                    protocol=protocol,
                    from_port=port,
                    to_port=port,
                    destination_cidr="0.0.0.0/0",
                    description=description
                ))

    def _add_rule_to_sg(self, sg: SecurityGroup, rule: SecurityGroupRule):
        """Add rule to security group, avoiding duplicates"""
        if rule.type == RuleType.INGRESS:
            # Check for duplicates
            for existing_rule in sg.inbound_rules:
                if self._rules_are_equal(existing_rule, rule):
                    return
            sg.add_inbound_rule(rule)
        else:
            for existing_rule in sg.outbound_rules:
                if self._rules_are_equal(existing_rule, rule):
                    return
            sg.add_outbound_rule(rule)

    def _rules_are_equal(self, rule1: SecurityGroupRule, rule2: SecurityGroupRule) -> bool:
        """Check if two rules are equivalent"""
        return (
            rule1.protocol == rule2.protocol and
            rule1.from_port == rule2.from_port and
            rule1.to_port == rule2.to_port and
            rule1.source_cidr == rule2.source_cidr and
            rule1.source_sg_id == rule2.source_sg_id
        )

    def _get_sg_resource_name(self, resource_label: str) -> str:
        """Convert resource label to security group resource name"""
        return f"{self.namespace}_{resource_label.lower().replace(' ', '_').replace('-', '_')}_sg"
