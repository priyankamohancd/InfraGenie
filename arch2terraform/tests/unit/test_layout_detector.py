"""
Unit tests for Stage 1 — layout_detector.py

All tests create synthetic BGR images with cv2.rectangle / cv2.line so no
binary fixture files are needed. Each test draws one or more colored
rectangles on a white background and verifies the detector finds them.

Color choices match the actual AWS SVG color values used in DEFAULT_COLOR_PROFILES:
  Green  #7AA116 → BGR(22, 161, 122)   → VPC
  Orange #ED7100 → BGR(0,  113, 237)   → Subnet
  Teal   #00A4A6 → BGR(166, 164, 0)    → Availability Zone
  Purple #8C4FFF → BGR(255, 79,  140)  → VPC (newer style)
  Navy   #242F3E → BGR(62,  47,  36)   → AWS Cloud
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from arch2terraform.adapters.image.layout_detector import (
    ContainerType,
    DetectorConfig,
    detect_containers_from_array,
)
from arch2terraform.schemas.diagram import NodeShape


# ---------------------------------------------------------------------------
# Image-drawing helpers
# ---------------------------------------------------------------------------

# BGR values derived from the hex colors in DEFAULT_COLOR_PROFILES
GREEN_BGR  = (22,  161, 122)   # #7AA116 — VPC classic green
ORANGE_BGR = (0,   113, 237)   # #ED7100 — Subnet / Security Group
TEAL_BGR   = (166, 164,   0)   # #00A4A6 — Availability Zone
PURPLE_BGR = (255,  79, 140)   # #8C4FFF — VPC newer purple
NAVY_BGR   = (62,   47,  36)   # #242F3E — AWS Cloud


def _white_canvas(width: int = 800, height: int = 600) -> np.ndarray:
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def _draw_solid_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                     color: tuple, thickness: int = 4) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def _draw_dashed_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                      color: tuple, thickness: int = 3,
                      dash_len: int = 12, gap_len: int = 8) -> None:
    """Draw a rectangle with dashed borders."""

    def _dashed_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        total = int(np.hypot(x2 - x1, y2 - y1))
        dx = (x2 - x1) / total if total else 0
        dy = (y2 - y1) / total if total else 0
        pos = 0
        while pos < total:
            end = min(pos + dash_len, total)
            sx, sy = int(x1 + pos * dx), int(y1 + pos * dy)
            ex, ey = int(x1 + end * dx), int(y1 + end * dy)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
            pos += dash_len + gap_len

    _dashed_line((x,     y),     (x + w, y    ))  # top
    _dashed_line((x + w, y),     (x + w, y + h))  # right
    _dashed_line((x + w, y + h), (x,     y + h))  # bottom
    _dashed_line((x,     y + h), (x,     y    ))  # left


# ---------------------------------------------------------------------------
# Config tuned for synthetic images (smaller min areas, tighter params)
# ---------------------------------------------------------------------------

def _test_cfg(**overrides) -> DetectorConfig:
    cfg = DetectorConfig(
        min_bbox_area     = 2_000,
        min_side_length   = 30,
        close_kernel_size = 25,
        close_iterations  = 2,
        dash_sample_width = 5,
        dash_threshold    = 0.60,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Helpers for common assertions
# ---------------------------------------------------------------------------

def _nodes_of_type(nodes, container_type_value: str):
    return [n for n in nodes if n.extra.get("container_type") == container_type_value]


def _assert_contains_bbox(nodes, x, y, w, h, tolerance=30):
    """Assert at least one node's bbox is within `tolerance` pixels of the target."""
    for n in nodes:
        b = n.bbox
        if (abs(b.x - x) <= tolerance and abs(b.y - y) <= tolerance and
                abs(b.width - w) <= tolerance and abs(b.height - h) <= tolerance):
            return
    bboxes = [(n.bbox.x, n.bbox.y, n.bbox.width, n.bbox.height) for n in nodes]
    pytest.fail(
        f"No node near bbox ({x},{y},{w},{h}) within tolerance={tolerance}. Found: {bboxes}"
    )


# ---------------------------------------------------------------------------
# Single-container tests
# ---------------------------------------------------------------------------

class TestSingleContainerDetection:

    def test_solid_green_rect_detected_as_vpc(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, GREEN_BGR, thickness=4)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes = _nodes_of_type(nodes, ContainerType.VPC.value)
        assert len(vpc_nodes) >= 1, f"Expected at least 1 VPC, got: {[n.extra for n in nodes]}"

    def test_solid_purple_rect_detected_as_vpc(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, PURPLE_BGR, thickness=4)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes = _nodes_of_type(nodes, ContainerType.VPC.value)
        assert len(vpc_nodes) >= 1

    def test_dashed_orange_rect_detected_as_subnet(self):
        img = _white_canvas()
        _draw_dashed_rect(img, 100, 100, 500, 300, ORANGE_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)
        assert len(subnet_nodes) >= 1, f"Expected SUBNET, got: {[n.extra for n in nodes]}"

    def test_dashed_teal_rect_detected_as_availability_zone(self):
        img = _white_canvas()
        _draw_dashed_rect(img, 80, 80, 600, 400, TEAL_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        az_nodes = _nodes_of_type(nodes, ContainerType.AVAILABILITY_ZONE.value)
        assert len(az_nodes) >= 1, f"Expected AZ, got: {[n.extra for n in nodes]}"

    def test_solid_navy_rect_detected_as_aws_cloud(self):
        img = _white_canvas(1000, 800)
        _draw_solid_rect(img, 20, 20, 960, 760, NAVY_BGR, thickness=3)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        cloud_nodes = _nodes_of_type(nodes, ContainerType.AWS_CLOUD.value)
        assert len(cloud_nodes) >= 1

    def test_all_detected_nodes_have_container_shape(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, GREEN_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        for n in nodes:
            assert n.shape == NodeShape.CONTAINER

    def test_detected_node_has_image_ref(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, GREEN_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes = _nodes_of_type(nodes, ContainerType.VPC.value)
        assert len(vpc_nodes) >= 1
        assert vpc_nodes[0].image_ref == "Virtual-private-cloud-VPC"

    def test_detected_node_has_source_format_image(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, GREEN_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        for n in nodes:
            assert n.source_format == "image"

    def test_empty_image_returns_no_nodes(self):
        img = _white_canvas()  # pure white, no containers
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        assert nodes == []


# ---------------------------------------------------------------------------
# Bounding box accuracy
# ---------------------------------------------------------------------------

class TestBoundingBoxAccuracy:

    def test_bbox_close_to_drawn_rect(self):
        img = _white_canvas(800, 600)
        x, y, w, h = 80, 60, 620, 480
        _draw_solid_rect(img, x, y, w, h, GREEN_BGR, thickness=4)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes = _nodes_of_type(nodes, ContainerType.VPC.value)
        assert len(vpc_nodes) >= 1
        _assert_contains_bbox(vpc_nodes, x, y, w, h, tolerance=25)

    def test_dashed_subnet_bbox_accurate(self):
        img = _white_canvas(800, 600)
        x, y, w, h = 120, 100, 500, 350
        _draw_dashed_rect(img, x, y, w, h, ORANGE_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)
        assert len(subnet_nodes) >= 1
        _assert_contains_bbox(subnet_nodes, x, y, w, h, tolerance=30)


# ---------------------------------------------------------------------------
# Containment hierarchy
# ---------------------------------------------------------------------------

class TestContainmentHierarchy:

    def test_inner_subnet_parent_is_vpc(self):
        """A subnet drawn inside a VPC should get the VPC as its parent."""
        img = _white_canvas(900, 700)
        # VPC — large green solid
        _draw_solid_rect(img, 50, 50, 800, 600, GREEN_BGR, thickness=4)
        # Subnet — smaller orange dashed inside VPC
        _draw_dashed_rect(img, 150, 150, 500, 300, ORANGE_BGR)

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes    = _nodes_of_type(nodes, ContainerType.VPC.value)
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)

        assert len(vpc_nodes)    >= 1, "VPC not detected"
        assert len(subnet_nodes) >= 1, "Subnet not detected"

        vpc_id = vpc_nodes[0].id
        assert subnet_nodes[0].parent_id == vpc_id, (
            f"Subnet parent_id={subnet_nodes[0].parent_id!r}, expected {vpc_id!r}"
        )

    def test_vpc_inside_aws_cloud_has_cloud_as_parent(self):
        """VPC inside AWS Cloud boundary should have AWS Cloud as parent."""
        img = _white_canvas(1000, 800)
        # AWS Cloud — large navy solid
        _draw_solid_rect(img, 20, 20, 960, 760, NAVY_BGR, thickness=3)
        # VPC — green solid inside cloud
        _draw_solid_rect(img, 80, 80, 800, 600, GREEN_BGR, thickness=4)

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        cloud_nodes = _nodes_of_type(nodes, ContainerType.AWS_CLOUD.value)
        vpc_nodes   = _nodes_of_type(nodes, ContainerType.VPC.value)

        assert len(cloud_nodes) >= 1, "AWS Cloud not detected"
        assert len(vpc_nodes)   >= 1, "VPC not detected"

        cloud_id = cloud_nodes[0].id
        assert vpc_nodes[0].parent_id == cloud_id, (
            f"VPC parent_id={vpc_nodes[0].parent_id!r}, expected {cloud_id!r}"
        )

    def test_three_level_nesting(self):
        """AWS Cloud → VPC → Subnet: three-level hierarchy."""
        img = _white_canvas(1000, 800)
        _draw_solid_rect(img, 10, 10, 980, 780, NAVY_BGR,   thickness=3)
        _draw_solid_rect(img, 60, 60, 860, 660, GREEN_BGR,  thickness=4)
        _draw_dashed_rect(img, 150, 150, 600, 400, ORANGE_BGR)

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        cloud_nodes  = _nodes_of_type(nodes, ContainerType.AWS_CLOUD.value)
        vpc_nodes    = _nodes_of_type(nodes, ContainerType.VPC.value)
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)

        assert len(cloud_nodes)  >= 1, "AWS Cloud not detected"
        assert len(vpc_nodes)    >= 1, "VPC not detected"
        assert len(subnet_nodes) >= 1, "Subnet not detected"

        cloud_id = cloud_nodes[0].id
        vpc_id   = vpc_nodes[0].id
        assert vpc_nodes[0].parent_id    == cloud_id, "VPC parent should be AWS Cloud"
        assert subnet_nodes[0].parent_id == vpc_id,   "Subnet parent should be VPC"

    def test_sibling_subnets_share_same_parent(self):
        """Two subnets inside one VPC should both point to the same VPC parent.
        Subnets are placed 100px apart so the close kernel (25px) does not bridge
        the gap between them and merge them into a single contour."""
        img = _white_canvas(1200, 700)
        _draw_solid_rect(img, 40, 40, 1120, 620, GREEN_BGR, thickness=4)
        _draw_dashed_rect(img, 80,  100, 440, 400, ORANGE_BGR)
        _draw_dashed_rect(img, 640, 100, 440, 400, ORANGE_BGR)  # 640-(80+440)=120px gap

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes    = _nodes_of_type(nodes, ContainerType.VPC.value)
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)

        assert len(vpc_nodes)    >= 1, "VPC not detected"
        assert len(subnet_nodes) >= 2, f"Expected 2 subnets, got {len(subnet_nodes)}"

        vpc_id = vpc_nodes[0].id
        for sn in subnet_nodes:
            assert sn.parent_id == vpc_id, f"Subnet {sn.id[:8]} has wrong parent"

    def test_top_level_container_has_no_parent(self):
        """The outermost region should have parent_id=None."""
        img = _white_canvas(1000, 800)
        _draw_solid_rect(img, 10, 10, 980, 780, NAVY_BGR, thickness=3)
        _draw_solid_rect(img, 80, 80, 800, 600, GREEN_BGR, thickness=4)

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        cloud_nodes = _nodes_of_type(nodes, ContainerType.AWS_CLOUD.value)
        assert len(cloud_nodes) >= 1
        assert cloud_nodes[0].parent_id is None, "Top-level container should have no parent"


# ---------------------------------------------------------------------------
# Border style detection
# ---------------------------------------------------------------------------

class TestBorderStyleDetection:

    def test_solid_border_classified_correctly(self):
        img = _white_canvas()
        _draw_solid_rect(img, 50, 50, 700, 500, GREEN_BGR, thickness=5)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        vpc_nodes = _nodes_of_type(nodes, ContainerType.VPC.value)
        assert len(vpc_nodes) >= 1
        assert vpc_nodes[0].extra["border_style"] == "solid"

    def test_dashed_border_classified_correctly(self):
        img = _white_canvas()
        _draw_dashed_rect(img, 100, 100, 550, 350, ORANGE_BGR, dash_len=14, gap_len=9)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        subnet_nodes = _nodes_of_type(nodes, ContainerType.SUBNET.value)
        assert len(subnet_nodes) >= 1
        assert subnet_nodes[0].extra["border_style"] == "dashed"


# ---------------------------------------------------------------------------
# Security Group disambiguation
# ---------------------------------------------------------------------------

class TestSecurityGroupDisambiguation:

    def test_small_orange_rect_reclassified_as_security_group(self):
        """
        A small orange dashed rect adjacent to (not inside) a large subnet,
        but both inside the same VPC, should be reclassified as SECURITY_GROUP.

        Placement:  VPC 40,40,1100,700
                    Subnet 100,100,700,500  (large, right side)
                    Small rect 860,200,120,120  (far right, clear of Subnet border)

        Note: in real AWS diagrams security groups appear as padlock icons, not
        dashed rectangles. This test covers the heuristic that disambiguates
        a small same-color same-parent region from a large one.
        """
        img = _white_canvas(1200, 800)
        _draw_solid_rect(img, 40,  40, 1120, 720, GREEN_BGR, thickness=4)    # VPC
        _draw_dashed_rect(img, 100, 100,  700, 500, ORANGE_BGR)               # Subnet (large)
        _draw_dashed_rect(img, 870, 200,  120, 120, ORANGE_BGR)               # Small rect — 70px gap from Subnet

        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        sg_nodes = _nodes_of_type(nodes, ContainerType.SECURITY_GROUP.value)
        assert len(sg_nodes) >= 1, (
            f"Expected at least 1 SECURITY_GROUP but got: "
            f"{[n.extra.get('container_type') for n in nodes]}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_very_small_rect_filtered_out(self):
        """Rectangles below min_bbox_area should not appear in output."""
        img = _white_canvas()
        _draw_solid_rect(img, 100, 100, 30, 20, GREEN_BGR)  # tiny — below threshold
        nodes = detect_containers_from_array(img, cfg=_test_cfg(min_bbox_area=5_000))
        assert nodes == []

    def test_node_ids_are_unique(self):
        img = _white_canvas(1000, 700)
        _draw_solid_rect(img, 40, 40, 920, 620, GREEN_BGR)
        _draw_dashed_rect(img, 100, 100, 350, 400, ORANGE_BGR)
        _draw_dashed_rect(img, 520, 100, 350, 400, ORANGE_BGR)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids)), "Node IDs must be unique"

    def test_output_sorted_largest_first(self):
        """Largest containers should appear first in the output list."""
        img = _white_canvas(1000, 800)
        _draw_solid_rect(img, 20, 20, 960, 760, NAVY_BGR,  thickness=3)
        _draw_solid_rect(img, 60, 60, 860, 660, GREEN_BGR, thickness=4)
        nodes = detect_containers_from_array(img, cfg=_test_cfg())
        if len(nodes) >= 2:
            areas = [n.bbox.width * n.bbox.height for n in nodes]
            assert areas == sorted(areas, reverse=True), (
                f"Nodes not sorted largest-first: {areas}"
            )

    def test_file_not_found_raises(self, tmp_path):
        from arch2terraform.adapters.image.layout_detector import detect_containers
        with pytest.raises(FileNotFoundError):
            detect_containers(tmp_path / "nonexistent.png")
