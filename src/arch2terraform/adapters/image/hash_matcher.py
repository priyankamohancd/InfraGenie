"""
Stage 2 — Perceptual-hash icon matcher.

At parse time, this module:
  1. Loads the pre-built reference hash table from disk (once, cached in-process).
  2. Accepts a PIL Image crop (the region around a detected candidate icon).
  3. Returns the best-matching AWS service name and the Hamming distance, or
     (None, distance) when the distance exceeds max_hamming — signalling that
     the Stage 3 YOLO fallback should handle this crop.

The reference table is built separately by scripts/build_hash_table.py and
committed as arch2terraform/src/arch2terraform/data/reference_hashes.pkl.
Nothing in this module knows about the aws-icons directory — it only reads
the pre-built artifact.
"""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Default location of the pre-built hash table artifact.
_DEFAULT_TABLE_PATH = Path(__file__).parent.parent.parent / "data" / "reference_hashes.pkl"

# Default Hamming distance ceiling for a confident match (64-bit hash).
# Icons within the same AWS category can be visually similar, so this is
# intentionally conservative. Tune via the collision report produced by
# scripts/build_hash_table.py.
DEFAULT_MAX_HAMMING = 10


class MatchResult(NamedTuple):
    service_name: str | None   # e.g. "Amazon-EC2", "Elastic-Load-Balancing", or None
    hamming: int               # distance from query to best candidate (0 = identical)
    category: str | None       # e.g. "Compute", "Storage", "Group"
    confident: bool            # True when hamming <= max_hamming


@lru_cache(maxsize=1)
def _load_table(table_path: str) -> dict:
    """Load and cache the reference hash table. Called once per process."""
    path = Path(table_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference hash table not found at '{path}'. "
            "Run scripts/build_hash_table.py to generate it."
        )
    with path.open("rb") as fh:
        table = pickle.load(fh)
    logger.info("Loaded reference hash table: %d entries from '%s'", len(table), path)
    return table


def load_table(table_path: Path | str | None = None) -> dict:
    """Public wrapper — resolves default path and delegates to the cached loader."""
    resolved = str(table_path) if table_path else str(_DEFAULT_TABLE_PATH)
    return _load_table(resolved)


def match_icon(
    crop,  # PIL.Image.Image — typed loosely to avoid hard import at module level
    table: dict | None = None,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    table_path: Path | str | None = None,
) -> MatchResult:
    """
    Match a PIL Image crop against the reference hash table.

    Parameters
    ----------
    crop        : PIL.Image.Image — the region cropped from the input diagram.
    table       : pre-loaded reference dict (pass to avoid repeated disk reads).
    max_hamming : Hamming distance ceiling for a confident match.
    table_path  : override the default artifact location.

    Returns
    -------
    MatchResult with service_name=None and confident=False when nothing matches.
    """
    try:
        import imagehash
    except ImportError as exc:
        raise ImportError(
            "imagehash is required for Stage 2 matching. "
            "Install with: pip install imagehash"
        ) from exc

    if table is None:
        table = load_table(table_path)

    query_phash = imagehash.phash(crop)

    best_name: str | None = None
    best_category: str | None = None
    best_dist: int = max_hamming + 1

    for service_name, entry in table.items():
        ref_phash = imagehash.hex_to_hash(entry["phash"])
        dist = query_phash - ref_phash
        if dist < best_dist:
            best_dist = dist
            best_name = service_name
            best_category = entry.get("category")

    confident = best_dist <= max_hamming
    if not confident:
        logger.debug(
            "No confident phash match (best=%s dist=%d > threshold=%d) — Stage 3 fallback needed",
            best_name,
            best_dist,
            max_hamming,
        )
        return MatchResult(None, best_dist, None, False)

    logger.debug("phash match: %s (dist=%d, category=%s)", best_name, best_dist, best_category)
    return MatchResult(best_name, best_dist, best_category, True)


def batch_match(
    crops: list,  # list[PIL.Image.Image]
    table: dict | None = None,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    table_path: Path | str | None = None,
) -> list[MatchResult]:
    """
    Match a list of crops in one call, sharing one table load.
    Returns results in the same order as inputs.
    """
    if table is None:
        table = load_table(table_path)
    return [match_icon(crop, table=table, max_hamming=max_hamming) for crop in crops]
