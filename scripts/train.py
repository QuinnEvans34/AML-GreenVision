"""Top-level training CLI: full GreenVision pipeline under nested MLflow runs.

Two modes:

1. **From-scratch** (default — what produced v3):
    PYTHONPATH=src .venv/bin/python scripts/train.py --attempt-id 002

2. **Fine-tune from registered version** (Decision 15 — produces v4):
    PYTHONPATH=src .venv/bin/python scripts/train.py \\
        --attempt-id 003 \\
        --fine-tune-from 3 \\
        --robust-augmentation \\
        --phase2-max-epochs 12

In fine-tune mode the script loads ``models:/GreenVision/<version>`` from the
MLflow Registry, skips Phase 1 (the head is already trained), and runs a
single ``finetune`` phase with optional robust augmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from greenvision.data.datasets import build_dataloaders
from greenvision.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    train_tfms_robust,
)
from greenvision.models.efficientnet_head import build_model
from greenvision.training.loop import (
    EARLY_STOP_PATIENCE,
    evaluate,
    run_phase1,
    run_phase2,
    save_checkpoint,
    train_one_epoch,
)
from greenvision.training.mlflow_utils import (
    init_mlflow,
    log_model_to_registry,
    log_test_artifacts,
    log_training_curves,
    parent_run,
    phase_run,
)
from greenvision.training.optim import build_optimizer
from greenvision.training.phases import freeze_all_backbone, unfreeze_from_block
from greenvision.training.schedulers import build_scheduler
from greenvision.training.seed import set_seed

BEST_CHECKPOINT = "artifacts/checkpoints/best.pt"
CLASS_NAMES_ARTIFACT = "artifacts/checkpoints/class_names.json"
HEAD_LR = 1e-3

# Fine-tune LRs (Decision 15) — lower than Phase 2 because v3 is already
# converged; we're just adapting to the new augmentation distribution.
FT_HEAD_LR = 5e-4
FT_BACKBONE_LR = 5e-5
FT_EARLY_STOP_PATIENCE = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training attempt."""
    parser = argparse.ArgumentParser(description="Train GreenVision end-to-end.")
    parser.add_argument("--data-root", default="data/raw/PlantVillage")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-max-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--fine-tune-from",
        type=int,
        default=None,
        help=(
            "If set, load models:/GreenVision/<version> from the registry "
            "instead of building from scratch. Skips Phase 1 entirely and "
            "runs a single 'finetune' phase. Use to produce v4 from v3."
        ),
    )
    parser.add_argument(
        "--robust-augmentation",
        action="store_true",
        help=(
            "Use train_tfms_robust (Decision 15) instead of train_tfms. "
            "Recommended whenever fine-tuning for out-of-distribution "
            "generalization."
        ),
    )
    return parser.parse_args()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference over a loader and collect true/predicted label indices."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────────
# Fine-tune phase — used by Track 2 of the W10 resilience plan
# ──────────────────────────────────────────────────────────────────────


def run_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int,
) -> tuple[dict[str, list[float]], float, int]:
    """Fine-tune a fully-unfrozen model with conservative LRs and early stop.

    No gradual unfreezing — v3 is already fully trained, so we unfreeze
    everything upfront and just adapt to whatever new augmentation
    distribution we're running on. The best checkpoint by val accuracy is
    saved to ``BEST_CHECKPOINT``.

    Args:
        model: The model to fine-tune (loaded from the registry).
        train_loader: Training data loader (likely using ``train_tfms_robust``).
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        max_epochs: Max epochs before stopping.

    Returns:
        ``(history, best_val_acc, best_epoch)`` — same shape as ``run_phase2``.
    """
    # Ensure every parameter is trainable from the start
    unfreeze_from_block(model, from_idx=0)

    optimizer = build_optimizer(model, head_lr=FT_HEAD_LR, backbone_lr=FT_BACKBONE_LR)
    scheduler = build_scheduler(optimizer)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_val_acc_gap": [],
        "lr_head": [],
        "lr_backbone": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(max_epochs):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m["val_loss"])

        # ReduceLROnPlateau exposes the *current* LR per group via param_groups
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[2]["lr"]
        gap = train_m["train_acc"] - val_m["val_acc"]

        metrics = {
            **train_m,
            **val_m,
            "train_val_acc_gap": gap,
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
        }
        mlflow.log_metrics(metrics, step=epoch)
        for key, value in metrics.items():
            history[key].append(value)

        print(
            f"[finetune] epoch {epoch:2d} · "
            f"train_acc {train_m['train_acc']:.4f} · "
            f"val_acc {val_m['val_acc']:.4f} · "
            f"lr_head {lr_head:.2e} · lr_backbone {lr_backbone:.2e}"
        )

        if val_m["val_acc"] > best_val_acc:
            best_val_acc = val_m["val_acc"]
            best_epoch = epoch
            epochs_since_improve = 0
            save_checkpoint(model, BEST_CHECKPOINT, epoch, best_val_acc)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= FT_EARLY_STOP_PATIENCE:
                print(
                    f"[finetune] early stop · no improvement for "
                    f"{FT_EARLY_STOP_PATIENCE} epochs (best {best_val_acc:.4f} "
                    f"at epoch {best_epoch})"
                )
                break

    return history, best_val_acc, best_epoch


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the training pipeline for one attempt."""
    args = parse_args()
    set_seed(args.seed)
    init_mlflow()

    # Pick the training transform: original (v3) or robust (Decision 15)
    if args.robust_augmentation:
        print("Using train_tfms_robust (Decision 15 — heavy augmentation)")
        train_transform = train_tfms_robust
    else:
        print("Using train_tfms (standard light augmentation)")
        train_transform = None  # build_dataloaders defaults to train_tfms

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transform=train_transform,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    parent_params = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_classes": len(class_names),
        "image_size": IMG_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "attempt_id": args.attempt_id,
        "augmentation": "robust" if args.robust_augmentation else "standard",
        "fine_tune_from": args.fine_tune_from,
    }

    # ── FINE-TUNE branch ───────────────────────────────────────────────
    if args.fine_tune_from is not None:
        print(
            f"\n→ Fine-tune mode · loading models:/GreenVision/"
            f"{args.fine_tune_from} from MLflow Registry…"
        )
        model = mlflow.pytorch.load_model(
            f"models:/GreenVision/{args.fine_tune_from}"
        )
        model.to(device)
        print(f"  loaded · type={type(model).__name__} · device={device}")

        with parent_run(args.attempt_id, parent_params):
            with phase_run(
                "finetune",
                {
                    "max_epochs": args.phase2_max_epochs,
                    "head_lr": FT_HEAD_LR,
                    "backbone_lr": FT_BACKBONE_LR,
                    "early_stop_patience": FT_EARLY_STOP_PATIENCE,
                    "base_version": args.fine_tune_from,
                },
            ):
                history, best_val_acc, best_epoch = run_finetune(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    device,
                    max_epochs=args.phase2_max_epochs,
                )
                # Load the best-epoch weights for registration
                checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"\nBest val_acc={best_val_acc:.4f} at epoch {best_epoch}")

                log_training_curves(history)
                log_model_to_registry(model)
                mlflow.log_artifact(CLASS_NAMES_ARTIFACT)
                mlflow.log_artifact(BEST_CHECKPOINT)

            # Final test-set evaluation on the parent run
            y_true, y_pred = collect_predictions(model, test_loader, device)
            log_test_artifacts(y_true, y_pred, class_names)
        return

    # ── FROM-SCRATCH branch (original v3 training path) ────────────────
    model = build_model().to(device)

    with parent_run(args.attempt_id, parent_params):
        with phase_run("phase1", {"epochs": args.phase1_epochs, "head_lr": HEAD_LR}):
            freeze_all_backbone(model)
            run_phase1(
                model,
                train_loader,
                val_loader,
                criterion,
                device,
                epochs=args.phase1_epochs,
                head_lr=HEAD_LR,
            )

        with phase_run(
            "phase2",
            {
                "max_epochs": args.phase2_max_epochs,
                "early_stop_patience": EARLY_STOP_PATIENCE,
            },
        ):
            history, best_val_acc, best_epoch = run_phase2(
                model,
                train_loader,
                val_loader,
                criterion,
                device,
                max_epochs=args.phase2_max_epochs,
            )
            # Register the BEST checkpoint, not the last-epoch weights.
            checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

            log_training_curves(history)
            log_model_to_registry(model)
            mlflow.log_artifact(CLASS_NAMES_ARTIFACT)
            mlflow.log_artifact(BEST_CHECKPOINT)

        # Final test-set evaluation logged under the parent run.
        y_true, y_pred = collect_predictions(model, test_loader, device)
        log_test_artifacts(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()
