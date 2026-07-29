"""
Stage 3b — CNN architecture definition.

Shared between the training script (scripts/train_cnn_classifier.py) and the
inference wrapper (stage3b_cnn_classifier.py) so both use the exact same
graph. Kept in its own module (rather than inlined in either caller) so a
training-time change can never accidentally drift from what inference loads.

Deliberately small: this is a CPU-only, no-GPU sandbox (see stage3_matcher.py's
docstring for why YOLOv8 was rejected for the same reason). 4 conv blocks with
modest channel counts keep both training time and the on-disk weight file
small, while still comfortably separating ~300 visually distinct flat-icon
classes — this is a much easier problem than natural-image classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

IMG_SIZE = 64  # must match build_cnn_training_data.py's IMG_SIZE


class IconCNN(nn.Module):
    """
    4 conv blocks (conv -> batchnorm -> relu -> maxpool), halving spatial
    dims each time: 64 -> 32 -> 16 -> 8 -> 4. Channels 32/64/128/128.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
