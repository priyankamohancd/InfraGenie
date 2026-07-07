#!/usr/bin/env python3
"""
Stage 2 reference hash table builder.

One-time build step — run whenever the AWS icon pack is updated.

Usage
-----
    python scripts/build_hash_table.py --icons-dir ~/work/thesis/aws-icons

Output (both written to arch2terraform/src/arch2terraform/data/):
    reference_hashes.pkl   — fast binary artifact loaded at parse time
    reference_hashes.json  — human-readable audit copy

Collision report
----------------
After building, the script prints pairs of services whose phash Hamming
distance is below --collision-threshold (default 10). These are icons the
perceptual hash stage may confuse — Stage 3 (YOLO) is the safety net for them.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "src" / "arch2terraform" / "data"

OUTPUT_PKL = DATA_DIR / "reference_hashes.pkl"
OUTPUT_JSON = DATA_DIR / "reference_hashes.json"

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches:  Arch_Amazon-EC2_64.png  →  group(1) = "Amazon-EC2"
_ARCH_PATTERN = re.compile(r"^Arch_(.+?)_\d+(?:@\d+x)?\.png$", re.IGNORECASE)

# Matches:  Virtual-private-cloud-VPC_32.png  →  group(1) = "Virtual-private-cloud-VPC"
_GROUP_PATTERN = re.compile(r"^(.+?)_\d+(?:@\d+x)?\.png$", re.IGNORECASE)

# Category is encoded in the parent dir name: Arch_Compute → "Compute"
_CATEGORY_PATTERN = re.compile(r"^(?:Arch_|Res_)?(.+)$")


def _parse_arch_filename(filename: str) -> str | None:
    """Return service name from an Architecture-Service-Icons filename, or None."""
    m = _ARCH_PATTERN.match(filename)
    return m.group(1) if m else None


def _parse_group_filename(filename: str) -> str | None:
    """Return group name from an Architecture-Group-Icons filename, or None."""
    m = _GROUP_PATTERN.match(filename)
    return m.group(1) if m else None


def _extract_category(dir_name: str) -> str:
    """'Arch_Compute' → 'Compute',  'Arch_Networking-Content-Delivery' → 'Networking-Content-Delivery'."""
    m = _CATEGORY_PATTERN.match(dir_name)
    return m.group(1) if m else dir_name


# ---------------------------------------------------------------------------
# Icon discovery
# ---------------------------------------------------------------------------

def collect_service_icons(icons_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Walk Architecture-Service-Icons_*/*/64/*.png (excluding @5x variants).

    Returns list of (service_name, category, file_path).
    """
    results = []
    service_root = next(icons_dir.glob("Architecture-Service-Icons_*"), None)
    if service_root is None:
        log.error("No Architecture-Service-Icons_* directory found under %s", icons_dir)
        return results

    for cat_dir in sorted(service_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = _extract_category(cat_dir.name)
        size_dir = cat_dir / "64"
        if not size_dir.exists():
            continue
        for png in sorted(size_dir.glob("*.png")):
            if "@" in png.name:          # skip @5x hi-res duplicates
                continue
            service_name = _parse_arch_filename(png.name)
            if service_name is None:
                log.warning("Unexpected filename format, skipping: %s", png)
                continue
            results.append((service_name, category, png))

    log.info("Found %d service icons", len(results))
    return results


def collect_group_icons(icons_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Walk Architecture-Group-Icons_*/*.png (32px, no size subdir).

    Returns list of (group_name, "Group", file_path).
    """
    results = []
    group_root = next(icons_dir.glob("Architecture-Group-Icons_*"), None)
    if group_root is None:
        log.warning("No Architecture-Group-Icons_* directory found under %s", icons_dir)
        return results

    for png in sorted(group_root.glob("*.png")):
        if "@" in png.name or png.name.startswith("."):
            continue
        group_name = _parse_group_filename(png.name)
        if group_name is None:
            log.warning("Unexpected filename format, skipping: %s", png)
            continue
        results.append((group_name, "Group", png))

    log.info("Found %d group/boundary icons", len(results))
    return results


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_hashes(png_path: Path) -> tuple[str, str]:
    """Return (phash_hex, dhash_hex) for a PNG file."""
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        log.error("Missing dependency: %s  —  pip install Pillow imagehash", exc)
        sys.exit(1)

    with Image.open(png_path) as img:
        img = img.convert("RGBA")
        # Composite onto white background so transparent icons hash consistently.
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        rgb = background.convert("RGB")

    return str(imagehash.phash(rgb)), str(imagehash.dhash(rgb))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_table(icons_dir: Path) -> dict:
    """
    Build the full reference hash table from the icon pack.

    Returns dict keyed by service/group name:
    {
        "Amazon-EC2": {
            "phash": "a3f2...",
            "dhash": "b1c4...",
            "category": "Compute",
            "icon_type": "service",
            "icon_path": "Architecture-Service-Icons_.../Arch_Compute/64/Arch_Amazon-EC2_64.png",
        },
        "Virtual-private-cloud-VPC": { ... "icon_type": "group" },
        ...
    }
    """
    table: dict = {}

    service_icons = collect_service_icons(icons_dir)
    group_icons = collect_group_icons(icons_dir)
    all_icons = [(n, c, p, "service") for n, c, p in service_icons] + \
                [(n, c, p, "group") for n, c, p in group_icons]

    total = len(all_icons)
    for idx, (name, category, path, icon_type) in enumerate(all_icons, 1):
        if idx % 50 == 0 or idx == total:
            log.info("  Hashing %d/%d  (%s)", idx, total, name)

        if name in table:
            log.warning("Duplicate service name '%s' — keeping first, skipping: %s", name, path)
            continue

        try:
            phash_hex, dhash_hex = compute_hashes(path)
        except Exception as exc:
            log.warning("Failed to hash '%s': %s — skipping", path, exc)
            continue

        table[name] = {
            "phash": phash_hex,
            "dhash": dhash_hex,
            "category": category,
            "icon_type": icon_type,
            "icon_path": str(path.relative_to(icons_dir)),
        }

    log.info("Hash table built: %d entries", len(table))
    return table


# ---------------------------------------------------------------------------
# Collision report
# ---------------------------------------------------------------------------

def collision_report(table: dict, threshold: int) -> list[tuple[str, str, int]]:
    """
    Return pairs (name_a, name_b, hamming) where phash distance < threshold.
    Sorted ascending by distance (closest collisions first).
    """
    try:
        import imagehash
    except ImportError as exc:
        log.error("Missing dependency: %s", exc)
        return []

    entries = [(name, imagehash.hex_to_hash(entry["phash"])) for name, entry in table.items()]
    collisions = []
    for (name_a, hash_a), (name_b, hash_b) in combinations(entries, 2):
        dist = hash_a - hash_b
        if dist < threshold:
            collisions.append((name_a, name_b, dist))

    collisions.sort(key=lambda t: t[2])
    return collisions


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save(table: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PKL.open("wb") as fh:
        pickle.dump(table, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote %s", OUTPUT_PKL)

    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2, ensure_ascii=False, sort_keys=True)
    log.info("Wrote %s", OUTPUT_JSON)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the perceptual-hash reference table from the AWS icon pack."
    )
    parser.add_argument(
        "--icons-dir",
        required=True,
        type=Path,
        help="Path to the root aws-icons directory (contains Architecture-Service-Icons_* etc.)",
    )
    parser.add_argument(
        "--collision-threshold",
        type=int,
        default=10,
        metavar="BITS",
        help="Hamming distance below which two icons are flagged as a collision risk (default: 10)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry-run: build the table but don't write files (useful for collision checks)",
    )
    args = parser.parse_args()

    icons_dir = args.icons_dir.expanduser().resolve()
    if not icons_dir.exists():
        log.error("icons-dir not found: %s", icons_dir)
        sys.exit(1)

    log.info("Building hash table from: %s", icons_dir)
    table = build_table(icons_dir)

    if not args.no_save:
        save(table)

    # Collision report
    log.info("Running collision check (threshold=%d bits) ...", args.collision_threshold)
    collisions = collision_report(table, args.collision_threshold)

    if collisions:
        print(f"\n{'='*70}")
        print(f"COLLISION REPORT  —  {len(collisions)} pairs with Hamming < {args.collision_threshold}")
        print(f"{'='*70}")
        print(f"{'Distance':>8}  {'Service A':<45}  {'Service B'}")
        print(f"{'-'*8}  {'-'*45}  {'-'*45}")
        for name_a, name_b, dist in collisions:
            print(f"{dist:>8}  {name_a:<45}  {name_b}")
        print(
            f"\n  These pairs may confuse Stage 2. Stage 3 (YOLO) is the fallback.\n"
            f"  Consider raising --collision-threshold or tuning DEFAULT_MAX_HAMMING\n"
            f"  in hash_matcher.py if false positives appear at runtime.\n"
        )
    else:
        print(f"\nNo collisions found below {args.collision_threshold} bits. Hash table looks clean.")

    print(f"\nDone. Table size: {len(table)} entries.")


if __name__ == "__main__":
    main()
