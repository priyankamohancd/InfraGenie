"""
Stage 3b — CNN fallback classifier.

Runs only on icon crops that both Stage 2 (phash) and Stage 3 (NCC template
matching) failed to confidently identify. Unlike Stage 3, it does not need
icons_dir at inference time — the trained weights already encode the
appearance of all reference icons (see scripts/build_cnn_training_data.py
and scripts/train_cnn_classifier.py for how the checkpoint was produced).

Why this exists (see stage3_matcher.py's docstring for the original
YOLOv8-was-rejected reasoning, which still applies to *replacing* NCC):
NCC's exact-alignment template match still fails on icon crops with
real-world distortions it doesn't tolerate well — rotation, off-white
backgrounds, blur, and scale jitter beyond a few percent. Rather than
replace NCC (fast, exact, needs zero training data) with a learned model
everywhere, this CNN is scoped narrowly as a THIRD-LEVEL fallback: only the
small residual of icons that fail *both* Stage 2 and Stage 3 ever reach it.
It was trained purely on synthetic augmentations of the same reference icon
pack (rotation, scale, brightness/contrast, background recolor, blur — see
build_cnn_training_data.py) so its output vocabulary is guaranteed identical
to hash_matcher's / stage3_matcher's service_name strings.

Optional dependency: requires torch. If unavailable or the checkpoint hasn't
been trained yet, callers should catch ImportError/FileNotFoundError and
skip Stage 3b gracefully — icons then fall through to Stage 4 OCR exactly as
they did before this stage existed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Default location of the trained checkpoint artifact.
_DEFAULT_CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "data" / "cnn_classifier.pt"

# Softmax-probability floor for a confident match. Deliberately higher than
# Stage 2/3's thresholds (which operate on raw distance/correlation, not a
# calibrated probability) because Stage 3b is the last automated identification
# stage before OCR — a wrong high-confidence label is worse than falling
# through to OCR/UNKNOWN, where a human reviewing the diagram can still tell.
DEFAULT_MIN_CONFIDENCE: float = 0.85


class Stage3bResult(NamedTuple):
    service_name: str | None
    confidence: float          # softmax probability of the top predicted class
    category: str | None
    confident: bool            # True when confidence >= min_confidence


# ---------------------------------------------------------------------------
# Checkpoint cache (module-level so it survives across multiple parse() calls)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_checkpoint(checkpoint_path: str):
    """Load and cache the trained CNN checkpoint. Called once per process."""
    import torch  # local import — torch is an optional, heavy dependency

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found at '{path}'. Run "
            "scripts/build_cnn_training_data.py then scripts/train_cnn_classifier.py "
            "to generate it."
        )
    ckpt = torch.load(path, map_location="cpu")

    from arch2terraform.adapters.image.cnn_model import IconCNN

    model = IconCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        "Loaded Stage 3b CNN checkpoint: %d classes, val_acc=%.4f, from '%s'",
        ckpt["num_classes"], ckpt.get("best_val_acc", -1.0), path,
    )
    return model, ckpt["class_names"], ckpt.get("img_size", 64)


class Stage3bCNNClassifier:
    """
    CNN classifier for Stage 3b icon identification.

    Parameters
    ----------
    table            : phash reference table (dict from hash_matcher.load_table),
                        used only to resolve service_name -> category for the
                        response (the model itself has no notion of category).
    checkpoint_path  : override the default trained-weights location.
    min_confidence   : softmax-probability floor for a confident match.
    """

    def __init__(
        self,
        table: dict,
        checkpoint_path: str | Path | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._table = table
        self._checkpoint_path = str(checkpoint_path) if checkpoint_path else str(_DEFAULT_CHECKPOINT_PATH)
        self._min_confidence = min_confidence
        self._model = None
        self._class_names: list[str] | None = None
        self._img_size = 64

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._class_names, self._img_size = _load_checkpoint(self._checkpoint_path)
        return self._model, self._class_names, self._img_size

    def _crop_to_tensor(self, crop_bgr: np.ndarray):
        import cv2
        import torch

        resized = cv2.resize(crop_bgr, (self._img_size, self._img_size), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, crop_bgr: np.ndarray) -> Stage3bResult:
        """
        Run CNN classification for a single BGR crop.

        Raises
        ------
        ImportError       if torch is not installed.
        FileNotFoundError if the trained checkpoint hasn't been built yet.
        Callers should catch both and treat Stage 3b as unavailable.
        """
        import torch

        model, class_names, img_size = self._ensure_loaded()
        self._img_size = img_size

        x = self._crop_to_tensor(crop_bgr)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        best_name = class_names[int(top_idx)]
        best_conf = float(top_prob)
        confident = best_conf >= self._min_confidence

        best_cat = self._table.get(best_name, {}).get("category") if confident else None

        if confident:
            logger.debug("Stage3b CNN match: %s (confidence=%.4f)", best_name, best_conf)
        else:
            logger.debug(
                "Stage3b CNN: no confident match (best=%s confidence=%.4f < %.2f)",
                best_name, best_conf, self._min_confidence,
            )

        return Stage3bResult(
            service_name=best_name if confident else None,
            confidence=best_conf,
            category=best_cat,
            confident=confident,
        )

    def batch_match(self, crops: list[np.ndarray]) -> list[Stage3bResult]:
        """Match a list of BGR crops, sharing the loaded model."""
        return [self.match(crop) for crop in crops]
