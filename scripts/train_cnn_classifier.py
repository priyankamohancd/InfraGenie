#!/usr/bin/env python3
"""
Stage 3b — trains the CNN fallback classifier on the synthetic dataset built
by build_cnn_training_data.py.

CPU-only, step-based & time-budgeted / resumable: this sandbox enforces a
hard wall-clock limit per invocation that's shorter than a full epoch takes
to train, so this script tracks a global step counter (not an epoch counter)
and trains for --seconds-budget wall-clock seconds per invocation, then
checkpoints (model + optimizer + global_step + best_val_acc) and exits.
Re-run the same command until it reports step == total_steps.

Batches are drawn from a deterministic per-epoch permutation (numpy
RandomState seeded with seed+epoch), so resuming mid-epoch reproduces the
exact same batch sequence without needing to persist a DataLoader/iterator.

Usage
-----
    python scripts/train_cnn_classifier.py --total-epochs 10 --seconds-budget 38
    # re-run the same command repeatedly until "reached total_steps"

Output:
    src/arch2terraform/data/cnn_classifier.pt   — latest checkpoint (+ best_val_acc)
    src/arch2terraform/data/cnn_train_log.json  — per-epoch loss/accuracy history
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "src" / "arch2terraform" / "data"
SRC_DIR = REPO_ROOT / "src"

TRAIN_NPZ = DATA_DIR / "cnn_train.npz"
VAL_NPZ = DATA_DIR / "cnn_val.npz"
CLASSES_JSON = DATA_DIR / "cnn_classes.json"

CHECKPOINT_PATH = DATA_DIR / "cnn_classifier.pt"
LOG_PATH = DATA_DIR / "cnn_train_log.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Stage 3b CNN fallback classifier.")
    parser.add_argument("--total-epochs", type=int, default=10)
    parser.add_argument("--seconds-budget", type=float, default=38.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not TRAIN_NPZ.exists() or not VAL_NPZ.exists() or not CLASSES_JSON.exists():
        log.error("Training data not found — run build_cnn_training_data.py first.")
        sys.exit(1)

    sys.path.insert(0, str(SRC_DIR))
    import numpy as np
    import torch
    import torch.nn as nn

    from arch2terraform.adapters.image.cnn_model import IconCNN

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    with CLASSES_JSON.open() as fh:
        class_names = json.load(fh)
    num_classes = len(class_names)

    t_load0 = time.time()
    train_data = np.load(TRAIN_NPZ)
    train_images = torch.from_numpy(
        np.transpose(train_data["images"].astype(np.float32) / 255.0, (0, 3, 1, 2)).copy()
    )
    train_labels = torch.from_numpy(train_data["labels"].astype(np.int64))

    val_data = np.load(VAL_NPZ)
    val_images = torch.from_numpy(
        np.transpose(val_data["images"].astype(np.float32) / 255.0, (0, 3, 1, 2)).copy()
    )
    val_labels = torch.from_numpy(val_data["labels"].astype(np.int64))
    log.info("Data loaded in %.1fs (train=%d, val=%d)", time.time() - t_load0, len(train_labels), len(val_labels))

    N = train_images.shape[0]
    batch_size = args.batch_size
    steps_per_epoch = (N + batch_size - 1) // batch_size
    total_steps = args.total_epochs * steps_per_epoch

    model = IconCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    best_val_acc = 0.0
    history: list[dict] = []

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if ckpt.get("num_classes") == num_classes:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            global_step = ckpt.get("global_step", 0)
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            log.info("Resumed from checkpoint at step %d/%d (best_val_acc=%.4f)",
                      global_step, total_steps, best_val_acc)
        else:
            log.warning("Checkpoint class count mismatch — starting fresh.")

    if LOG_PATH.exists():
        with LOG_PATH.open() as fh:
            history = json.load(fh)

    if global_step >= total_steps:
        log.info("Already reached total_steps=%d (current=%d). Nothing to do.", total_steps, global_step)
        return

    def evaluate() -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val_labels), 256):
                xb = val_images[i:i + 256]
                yb = val_labels[i:i + 256]
                out = model(xb)
                correct += (out.argmax(dim=1) == yb).sum().item()
        model.train()
        return correct / len(val_labels)

    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    step = global_step

    while step < total_steps and (time.time() - start_time) < args.seconds_budget:
        epoch = step // steps_per_epoch
        batch_idx = step % steps_per_epoch

        rng = np.random.RandomState(args.seed + epoch)
        perm = rng.permutation(N)
        idx = perm[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        images = train_images[idx]
        labels = train_labels[idx]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        running_total += images.size(0)

        step += 1

        if batch_idx == steps_per_epoch - 1:
            # Completed a full epoch — validate and log.
            train_loss = running_loss / running_total
            train_acc = running_correct / running_total
            val_acc = evaluate()
            log.info(
                "Epoch %d/%d complete (step %d/%d)  train_loss=%.4f train_acc=%.4f val_acc=%.4f",
                epoch + 1, args.total_epochs, step, total_steps, train_loss, train_acc, val_acc,
            )
            history.append({
                "epoch": epoch + 1, "step": step, "train_loss": train_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            })
            best_val_acc = max(best_val_acc, val_acc)
            running_loss = running_correct = running_total = 0

    elapsed = time.time() - start_time
    log.info("Trained %d steps (%d -> %d) in %.1fs this run", step - global_step, global_step, step, elapsed)

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": step,
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "class_names": class_names,
        "img_size": 64,
    }, CHECKPOINT_PATH)
    with LOG_PATH.open("w") as fh:
        json.dump(history, fh, indent=2)

    if step >= total_steps:
        log.info("REACHED total_steps=%d. Final best_val_acc=%.4f. Training complete.", total_steps, best_val_acc)
    else:
        log.info("Progress: step %d/%d (%.0f%%). Re-run the same command to continue.",
                  step, total_steps, 100 * step / total_steps)


if __name__ == "__main__":
    main()
