"""
Unit tests for the vision-LLM detection path (vision_llm_detector.py).

No test here ever performs real network I/O or requires an API key — the
low-level API call is injected via `call_api=`, so these tests exercise the
prompt construction, JSON-response parsing, and DiagramNode mapping in full
isolation, exactly the parts of this module that don't need a live model.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from arch2terraform.adapters.image.vision_llm_detector import (
    VisionLLMError,
    _build_prompt,
    _extract_json,
    _parse_response,
    detect_via_vision_llm,
)
from arch2terraform.schemas.diagram import NodeShape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(path, width=400, height=300):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(path), img)


def _canned_response(elements: list[dict]) -> str:
    return json.dumps({"elements": elements})


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_includes_exact_pixel_dimensions(self):
        prompt = _build_prompt(1920, 1080)
        assert "1920x1080" in prompt

    def test_prompt_requests_json_only(self):
        prompt = _build_prompt(800, 600)
        assert "JSON" in prompt
        assert "no markdown fences" in prompt or "ONLY" in prompt

    def test_prompt_distinguishes_container_from_group(self):
        prompt = _build_prompt(800, 600)
        assert '"container"' in prompt
        assert '"group"' in prompt


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json_parses(self):
        raw = '{"elements": []}'
        assert _extract_json(raw) == {"elements": []}

    def test_strips_markdown_fence(self):
        raw = "```json\n{\"elements\": []}\n```"
        assert _extract_json(raw) == {"elements": []}

    def test_strips_bare_fence_no_language_tag(self):
        raw = "```\n{\"elements\": []}\n```"
        assert _extract_json(raw) == {"elements": []}

    def test_invalid_json_raises_json_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_basic_icon_and_container(self):
        data = {
            "elements": [
                {
                    "id": "e1", "type": "container", "service_name": "VPC",
                    "label": "Main VPC", "bbox": {"x": 10, "y": 10, "width": 500, "height": 400},
                    "parent_id": None, "confidence": 0.9,
                },
                {
                    "id": "e2", "type": "icon", "service_name": "Amazon EC2",
                    "label": "Web Server", "bbox": {"x": 50, "y": 50, "width": 60, "height": 60},
                    "parent_id": "e1", "confidence": 0.95,
                },
            ]
        }
        nodes, warnings = _parse_response(data)
        assert len(nodes) == 2
        assert warnings == []

        container = next(n for n in nodes if n.shape == NodeShape.CONTAINER)
        icon = next(n for n in nodes if n.shape == NodeShape.ICON)

        assert container.image_ref == "VPC"
        assert container.raw_label == "Main VPC"
        assert container.parent_id is None

        assert icon.image_ref == "Amazon EC2"
        assert icon.parent_id == container.id  # local "e1" -> resolved to container's real uuid

    def test_group_type_becomes_container_shape_with_no_image_ref(self):
        """
        See module docstring: a "group" (organizational box, not a real AWS
        resource) is kept as CONTAINER shape (so containment/geometry still
        works) but with image_ref=None, so the classifier's existing
        no-signal-fallback correctly routes it to unclassified instead of
        guessing a resource type — the exact bug (phantom subnets from
        non-resource grouping boxes) found and fixed 2026-07-28.
        """
        data = {
            "elements": [
                {
                    "id": "g1", "type": "group", "service_name": None,
                    "label": "", "bbox": {"x": 0, "y": 0, "width": 300, "height": 200},
                    "parent_id": None, "confidence": 0.4,
                },
            ]
        }
        nodes, warnings = _parse_response(data)
        assert len(nodes) == 1
        assert nodes[0].shape == NodeShape.CONTAINER
        assert nodes[0].image_ref is None
        assert nodes[0].extra["vision_type"] == "group"

    def test_missing_bbox_is_skipped_with_warning(self):
        data = {"elements": [{"id": "e1", "type": "icon", "service_name": "S3", "label": ""}]}
        nodes, warnings = _parse_response(data)
        assert nodes == []
        assert len(warnings) == 1
        assert "bbox" in warnings[0]

    def test_non_positive_bbox_size_is_skipped(self):
        data = {
            "elements": [
                {
                    "id": "e1", "type": "icon", "service_name": "S3", "label": "",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 40},
                }
            ]
        }
        nodes, warnings = _parse_response(data)
        assert nodes == []
        assert any("non-positive" in w for w in warnings)

    def test_unrecognized_type_becomes_unknown_shape_with_warning(self):
        data = {
            "elements": [
                {
                    "id": "e1", "type": "mystery-blob", "service_name": None, "label": "",
                    "bbox": {"x": 0, "y": 0, "width": 40, "height": 40},
                }
            ]
        }
        nodes, warnings = _parse_response(data)
        assert len(nodes) == 1
        assert nodes[0].shape == NodeShape.UNKNOWN
        assert any("unrecognized type" in w for w in warnings)

    def test_malformed_element_not_a_dict_is_skipped(self):
        data = {"elements": ["not-a-dict", 123]}
        nodes, warnings = _parse_response(data)
        assert nodes == []
        assert len(warnings) == 2

    def test_missing_elements_key_raises(self):
        with pytest.raises(VisionLLMError):
            _parse_response({})

    def test_dangling_parent_id_resolves_to_none(self):
        """A parent_id referencing an id that isn't in the element list
        (hallucinated by the model) must not crash — falls back to no parent
        rather than raising, since a child missing its precise parent is far
        less harmful than dropping the child entirely."""
        data = {
            "elements": [
                {
                    "id": "e1", "type": "icon", "service_name": "S3", "label": "",
                    "bbox": {"x": 0, "y": 0, "width": 40, "height": 40},
                    "parent_id": "nonexistent-id",
                }
            ]
        }
        nodes, warnings = _parse_response(data)
        assert len(nodes) == 1
        assert nodes[0].parent_id is None

    def test_label_falls_back_to_service_name_when_no_visible_text(self):
        data = {
            "elements": [
                {
                    "id": "e1", "type": "icon", "service_name": "Amazon RDS", "label": "",
                    "bbox": {"x": 0, "y": 0, "width": 40, "height": 40},
                }
            ]
        }
        nodes, _ = _parse_response(data)
        assert nodes[0].raw_label == "Amazon RDS"


# ---------------------------------------------------------------------------
# detect_via_vision_llm (full flow, API call injected)
# ---------------------------------------------------------------------------

class TestDetectViaVisionLLM:
    def test_full_flow_with_injected_api_call(self, tmp_path):
        img_path = tmp_path / "diagram.png"
        _write_png(img_path, width=800, height=600)

        canned = _canned_response([
            {
                "id": "e1", "type": "container", "service_name": "VPC", "label": "Main VPC",
                "bbox": {"x": 10, "y": 10, "width": 700, "height": 500},
                "parent_id": None, "confidence": 0.9,
            },
        ])

        def fake_call(image_b64, media_type, prompt, model, api_key, max_tokens):
            # Confirm the real image dimensions were threaded into the prompt.
            assert "800x600" in prompt
            return canned

        result = detect_via_vision_llm(img_path, call_api=fake_call)
        assert len(result.nodes) == 1
        assert result.nodes[0].image_ref == "VPC"
        assert result.source_file == str(img_path)
        assert result.warnings == []

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detect_via_vision_llm(tmp_path / "does_not_exist.png", call_api=lambda *a: "{}")

    def test_api_call_exception_wrapped_as_vision_llm_error(self, tmp_path):
        img_path = tmp_path / "diagram.png"
        _write_png(img_path)

        def failing_call(*args):
            raise RuntimeError("network unreachable")

        with pytest.raises(VisionLLMError, match="Vision LLM API call failed"):
            detect_via_vision_llm(img_path, call_api=failing_call)

    def test_invalid_json_response_raises_vision_llm_error(self, tmp_path):
        img_path = tmp_path / "diagram.png"
        _write_png(img_path)

        with pytest.raises(VisionLLMError, match="did not return valid JSON"):
            detect_via_vision_llm(img_path, call_api=lambda *a: "this is not json")

    def test_empty_response_raises_vision_llm_error(self, tmp_path):
        img_path = tmp_path / "diagram.png"
        _write_png(img_path)

        with pytest.raises(VisionLLMError, match="empty response"):
            detect_via_vision_llm(img_path, call_api=lambda *a: "   ")

    def test_zero_elements_surfaces_as_warning_not_error(self, tmp_path):
        """An empty-but-valid element list is a real (if surprising) answer
        from the model, not a malformed response — surfaced as a warning so
        the caller can decide, not raised as an error."""
        img_path = tmp_path / "diagram.png"
        _write_png(img_path)

        result = detect_via_vision_llm(img_path, call_api=lambda *a: _canned_response([]))
        assert result.nodes == []
        assert any("zero usable elements" in w for w in result.warnings)

    def test_markdown_fenced_response_is_handled(self, tmp_path):
        img_path = tmp_path / "diagram.png"
        _write_png(img_path)

        fenced = "```json\n" + _canned_response([
            {
                "id": "e1", "type": "icon", "service_name": "Amazon EC2", "label": "",
                "bbox": {"x": 0, "y": 0, "width": 40, "height": 40}, "parent_id": None,
            }
        ]) + "\n```"

        result = detect_via_vision_llm(img_path, call_api=lambda *a: fenced)
        assert len(result.nodes) == 1
        assert result.nodes[0].image_ref == "Amazon EC2"
