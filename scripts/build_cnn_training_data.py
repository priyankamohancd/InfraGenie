#!/usr/bin/env python3
"""
Stage 3b training-data builder — synthetic augmented crops for the CNN
fallback classifier.

Why synthetic instead of waiting for real diagram misses: Stage 2 (phash)
and Stage 3 (NCC) already fail on the SAME kinds of inputs (rotated icons,
color-shifted icons, icons composited on a non-white background, slightly
blurred/rescaled icons) — that failure mode is exactly what this script
reproduces from the reference icon pack itself, using reference_hashes.pkl
(built by build_hash_table.py) as the source of truth for which 64x64 PNG
belongs to which `service_name` label. This guarantees the CNN's output
vocabulary is IDENTICAL to what hash_matcher.py/stage3_matcher.py already
use — no separate label mapping to keep in sync.

Usage
-----
    python scripts/build_cnn_training_data.py --icons-dir ~/work/thesis/aws-icons \\
        --samples-per-class 40

Output (written to arch2terraform/src/arch2terraform/data/):
    cnn_train.npz   — (images uint8 [N,64,64,3], labels int64 [N])
    cnn_val.npz     — held-out split (default 15%), same shape
    cnn_classes.json — index -> service_name, so stage3b_cnn_classifier.py's
                        model output indices map back to real service names
                        without needing reference_hashes.pkl at inference time.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "src" / "arch2terraform" / "data"
HASH_TABLE_PATH = DATA_DIR / "reference_hashes.pkl"

OUTPUT_TRAIN = DATA_DIR / "cnn_train.npz"
OUTPUT_VAL = DATA_DIR / "cnn_val.npz"
OUTPUT_CLASSES = DATA_DIR / "cnn_classes.json"

IMG_SIZE = 64  # matches Stage 3's NCC template size, and Stage 2's icon crop convention


def _augment(base_rgb, rng: random.Random):
    """
    One augmented 64x64 RGB variant of a reference icon, using only numpy +
    cv2 (already dependencies of this project - no new imaging library
    needed). Mirrors the actual failure modes Stage 2/3 exhibit on real
    diagram crops: rotation (diagrams aren't always axis-aligned after
    cropping), scale jitter (icon size varies with diagram zoom/DPI), color/
    contrast jitter (screenshots, exports, and re-renders shift colors
    slightly), mild blur (downscaled/re-encoded diagrams), and a non-white
    background composite (icons sitting on a container's fill color instead
    of a plain white canvas — the single biggest reason NCC's white-background
    assumption breaks in practice).
    """
    import cv2
    import numpy as np

    img = base_rgb.copy()
    h, w = img.shape[:2]

    # 1. Background composite: paste onto a randomly colored canvas first
    #    (icons are RGBA; the caller has already flattened to RGB against
    #    white — here we occasionally recolor that background instead).
    if rng.random() < 0.5:
        bg_color = np.array([rng.randint(200, 255) for _ in range(3)], dtype=np.uint8)
        mask = np.all(img > 250, axis=-1)  # approximate "was transparent/white" pixels
        img[mask] = bg_color

    # 2. Rotation (small angles - diagram icons are rarely rotated far)
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Brightness / contrast jitter
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-15, 15)      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 4. Mild Gaussian blur (simulates re-encoding / downscaling artifacts)
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 5. Small translation jitter (crop wasn't perfectly centered)
    tx, ty = rng.randint(-4, 4), rng.randint(-4, 4)
    M2 = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def _load_reference_rgb(icons_dir: Path, rel_path: str):
    """Load one reference icon, composited onto white, resized to IMG_SIZE."""
    import cv2
    import numpy as np
    from PIL import Image

    abs_path = icons_dir / rel_path
    with Image.open(abs_path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        rgb = np.array(bg.convert("RGB"))

    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def build_dataset(icons_dir: Path, samples_per_class: int, val_fraction: float, seed: int):
    import numpy as np

    with HASH_TABLE_PATH.open("rb") as fh:
        table: dict = pickle.load(fh)

    class_names = sorted(table.keys())
    class_index = {name: i for i, name in enumerate(class_names)}
    log.info("Loaded %d classes from %s", len(class_names), HASH_TABLE_PATH)

    rng = random.Random(seed)
    train_images, train_labels = [], []
    val_images, val_labels = [], []

    for idx, name in enumerate(class_names, 1):
        if idx % 50 == 0 or idx == len(class_names):
            log.info("  Augmenting %d/%d (%s)", idx, len(class_names), name)

        entry = table[name]
        try:
            base_rgb = _load_reference_rgb(icons_dir, entry["icon_path"])
        except Exception as exc:
            log.warning("Skipping '%s' - failed to load reference icon: %s", name, exc)
            continue

        label = class_index[name]
        n_val = max(1, round(samples_per_class * val_fraction))
        n_train = samples_per_class - n_val

        for _ in range(n_train):
            train_images.append(_augment(base_rgb, rng))
            train_labels.append(label)
        for _ in range(n_val):
            val_images.append(_augment(base_rgb, rng))
            val_labels.append(label)

    train_images = np.stack(train_images).astype(np.uint8)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_images = np.stack(val_images).astype(np.uint8)
    val_labels = np.array(val_labels, dtype=np.int64)

    log.info("Built %d train / %d val samples across %d classes",
              len(train_labels), len(val_labels), len(class_names))

    return train_images, train_labels, val_images, val_labels, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic training data for the Stage 3b CNN fallback.")
    parser.add_argument("--icons-dir", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=40,
                         help="Total augmented samples per class, split into train/val (default: 40)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    icons_dir = args.icons_dir.expanduser().resolve()
    if not icons_dir.exists():
        log.error("icons-dir not found: %s", icons_dir)
        sys.exit(1)
    if not HASH_TABLE_PATH.exists():
        log.error("%s not found - run build_hash_table.py first", HASH_TABLE_PATH)
        sys.exit(1)

    import numpy as np

    train_images, train_labels, val_images, val_labels, class_names = build_dataset(
        icons_dir, args.samples_per_class, args.val_fraction, args.seed,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_TRAIN, images=train_images, labels=train_labels)
    np.savez_compressed(OUTPUT_VAL, images=val_images, labels=val_labels)
    with OUTPUT_CLASSES.open("w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)

    log.info("Wrote %s (%.1f MB)", OUTPUT_TRAIN, OUTPUT_TRAIN.stat().st_size / 1e6)
    log.info("Wrote %s (%.1f MB)", OUTPUT_VAL, OUTPUT_VAL.stat().st_size / 1e6)
    log.info("Wrote %s (%d classes)", OUTPUT_CLASSES, len(class_names))


if __name__ == "__main__":
    main()
