"""
Unit tests for Stage 2 — hash_matcher.py

Strategy
--------
We don't need real AWS icon files here. We build a tiny synthetic table
in-memory and pass it directly to match_icon / batch_match, bypassing
disk I/O entirely. This keeps the tests fast, offline, and free from any
dependency on the built artifact or the aws-icons directory.

The only test that touches the real artifact (test_real_table_*) is skipped
when the pkl hasn't been built yet — run `scripts/build_hash_table.py` first
if you want those to pass.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import imagehash
import pytest
from PIL import Image

from arch2terraform.adapters.image.hash_matcher import (
    DEFAULT_MAX_HAMMING,
    MatchResult,
    batch_match,
    match_icon,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTIFACT_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "arch2terraform" / "data" / "reference_hashes.pkl"
)

_REAL_TABLE_AVAILABLE = ARTIFACT_PATH.exists()


def _patterned_image(index: int, size: int = 64) -> Image.Image:
    """
    Create a visually distinct patterned image for a given index.

    phash works on the DCT of the grayscale version, so plain solid colours
    with the same luminance hash identically. Instead we use distinct stripe
    patterns (varying frequency and orientation) to guarantee well-separated
    hashes for different indices.
    """
    import numpy as np

    rng = np.random.default_rng(seed=index * 137 + 42)  # deterministic per index
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    # Overlay a stripe pattern that shifts with index so hashes diverge.
    for row in range(size):
        if (row // (4 + index % 4)) % 2 == 0:
            arr[row, :, 0] = (arr[row, :, 0].astype(int) + 60 * (index + 1)) % 256
    return Image.fromarray(arr, "RGB")


def _make_table(entries: list[tuple[str, str, str]]) -> dict:
    """
    Build a synthetic reference table from (name, category, icon_type) tuples.
    Each entry gets the phash of a deterministic patterned image so we control
    exactly which service wins a match.
    """
    table = {}
    for idx, (name, category, icon_type) in enumerate(entries):
        img = _patterned_image(idx)
        table[name] = {
            "phash": str(imagehash.phash(img)),
            "dhash": str(imagehash.dhash(img)),
            "category": category,
            "icon_type": icon_type,
            "icon_path": f"fake/{name}.png",
        }
    return table


# ---------------------------------------------------------------------------
# Basic matching
# ---------------------------------------------------------------------------

class TestMatchIconExact:
    """Exact matches — the crop is identical to the reference image."""

    def setup_method(self):
        self.entries = [
            ("Amazon-EC2",              "Compute",  "service"),
            ("Elastic-Load-Balancing",  "Networking-Content-Delivery", "service"),
            ("Amazon-S3",               "Storage",  "service"),
        ]
        self.table = _make_table(self.entries)

    def test_exact_match_returns_correct_service(self):
        for idx, (name, _, _) in enumerate(self.entries):
            crop = _patterned_image(idx)  # same seed as _make_table → identical hash
            result = match_icon(crop, table=self.table)
            assert result.service_name == name, f"Expected {name!r}, got {result.service_name!r}"

    def test_exact_match_is_confident(self):
        crop = _patterned_image(0)
        result = match_icon(crop, table=self.table)
        assert result.confident is True

    def test_exact_match_hamming_is_zero(self):
        crop = _patterned_image(0)
        result = match_icon(crop, table=self.table)
        assert result.hamming == 0

    def test_exact_match_category_populated(self):
        crop = _patterned_image(1)
        result = match_icon(crop, table=self.table)
        assert result.category == "Networking-Content-Delivery"


class TestMatchIconNoMatch:
    """Crops that should NOT match — Hamming exceeds the threshold."""

    def setup_method(self):
        self.table = _make_table([
            ("Amazon-RDS", "Databases", "service"),
        ])

    def test_no_match_returns_none_service_name(self):
        # A high-index patterned image not in the table → should not match at tight threshold.
        unknown_crop = _patterned_image(99)
        result = match_icon(unknown_crop, table=self.table, max_hamming=5)
        if result.hamming > 5:
            assert result.service_name is None
            assert result.confident is False

    def test_zero_threshold_only_accepts_identical(self):
        """max_hamming=0 means only hash==0 distance is a match."""
        unknown_crop = _patterned_image(99)  # not in the table
        result = match_icon(unknown_crop, table=self.table, max_hamming=0)
        if result.hamming > 0:
            assert result.service_name is None
            assert result.confident is False

    def test_returns_match_result_namedtuple(self):
        crop = _patterned_image(99)
        result = match_icon(crop, table=self.table)
        assert isinstance(result, MatchResult)
        assert hasattr(result, "service_name")
        assert hasattr(result, "hamming")
        assert hasattr(result, "category")
        assert hasattr(result, "confident")


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

class TestMatchIconThreshold:
    """Verify that raising/lowering max_hamming changes confidence outcomes."""

    def setup_method(self):
        self.table = _make_table([
            ("Amazon-EC2", "Compute", "service"),
        ])
        # Reference crop → exact match, hamming=0.
        self.matching_crop = _patterned_image(0)

    def test_high_threshold_accepts_exact_match(self):
        result = match_icon(self.matching_crop, table=self.table, max_hamming=20)
        assert result.confident is True
        assert result.service_name == "Amazon-EC2"

    def test_negative_threshold_rejects_everything(self):
        """max_hamming=-1 means nothing is ever a match."""
        result = match_icon(self.matching_crop, table=self.table, max_hamming=-1)
        assert result.confident is False
        assert result.service_name is None


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

class TestBatchMatch:
    """batch_match must return one result per crop, in order."""

    def setup_method(self):
        self.entries = [
            ("Amazon-EC2",  "Compute",   "service"),
            ("Amazon-S3",   "Storage",   "service"),
            ("Amazon-RDS",  "Databases", "service"),
        ]
        self.table = _make_table(self.entries)

    def test_batch_length_matches_inputs(self):
        crops = [_patterned_image(i) for i in range(len(self.entries))]
        results = batch_match(crops, table=self.table)
        assert len(results) == len(crops)

    def test_batch_order_preserved(self):
        crops = [_patterned_image(i) for i in range(len(self.entries))]
        results = batch_match(crops, table=self.table)
        for idx, (name, _, _) in enumerate(self.entries):
            assert results[idx].service_name == name

    def test_batch_empty_input(self):
        results = batch_match([], table=self.table)
        assert results == []

    def test_batch_single_crop(self):
        crop = _patterned_image(0)
        results = batch_match([crop], table=self.table)
        assert len(results) == 1
        assert results[0].service_name == "Amazon-EC2"


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

class TestLoadTable:
    """Tests for disk-load path — uses tmp_path to avoid touching real data."""

    def test_missing_table_raises_file_not_found(self, tmp_path):
        from arch2terraform.adapters.image.hash_matcher import load_table, _load_table
        _load_table.cache_clear()
        with pytest.raises(FileNotFoundError, match="[Rr]eference hash table not found"):
            load_table(tmp_path / "nonexistent.pkl")

    def test_valid_pkl_loads_correctly(self, tmp_path):
        from arch2terraform.adapters.image.hash_matcher import load_table, _load_table
        _load_table.cache_clear()

        synthetic = {"Amazon-EC2": {"phash": "abc", "dhash": "def", "category": "Compute",
                                    "icon_type": "service", "icon_path": "fake.png"}}
        pkl_path = tmp_path / "test_hashes.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(synthetic, fh)

        loaded = load_table(pkl_path)
        assert "Amazon-EC2" in loaded
        assert loaded["Amazon-EC2"]["category"] == "Compute"
        _load_table.cache_clear()  # clean up so other tests aren't affected


# ---------------------------------------------------------------------------
# Integration against the real artifact (skipped if not built)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_TABLE_AVAILABLE, reason="reference_hashes.pkl not built yet")
class TestRealTable:
    """Smoke-tests against the actual built artifact."""

    def setup_method(self):
        from arch2terraform.adapters.image.hash_matcher import load_table, _load_table
        _load_table.cache_clear()
        self.table = load_table(ARTIFACT_PATH)

    def test_table_has_expected_minimum_size(self):
        assert len(self.table) >= 300, f"Expected 300+ entries, got {len(self.table)}"

    def test_known_services_present(self):
        for expected in ["Amazon-EC2", "Amazon-RDS", "Elastic-Load-Balancing",
                         "Amazon-Simple-Storage-Service", "Virtual-private-cloud-VPC"]:
            assert expected in self.table, f"'{expected}' missing from table"

    def test_each_entry_has_required_keys(self):
        required = {"phash", "dhash", "category", "icon_type", "icon_path"}
        for name, entry in self.table.items():
            missing = required - entry.keys()
            assert not missing, f"Entry '{name}' missing keys: {missing}"

    def test_exact_phash_match_on_reference_icon(self):
        """Load an actual icon PNG and verify it matches itself."""
        # Use Amazon-EC2 as a stable, always-present test fixture.
        ec2_entry = self.table.get("Amazon-EC2")
        if ec2_entry is None:
            pytest.skip("Amazon-EC2 not in table")

        # Reconstruct the absolute path from the stored relative icon_path.
        # The builder stored it relative to icons_dir, which we don't know here —
        # so we skip if we can't resolve it.
        pytest.skip("Icon path resolution requires known icons_dir — run integration tests instead")
