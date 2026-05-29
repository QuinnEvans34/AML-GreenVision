"""Top-level training CLI: full GreenVision pipeline under nested MLflow runs.

Usage:
    PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train.py \\
        --data-root data/raw/PlantVillage --attempt-id 001
"""

import argparse

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from greenvision.data.datasets import build_dataloaders
from greenvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from greenvision.models.efficientnet_head import build_model
from greenvision.training.loop import (
    EARLY_STOP_PATIENCE,
    run_phase1,
    run_phase2,
)
from greenvision.training.mlflow_utils import (
    init_mlflow,
    log_model_to_registry,
    log_test_artifacts,
    log_training_curves,
    parent_run,
    phase_run,
)
from greenvision.training.phases import freeze_all_backbone
from greenvision.training.seed import set_seed

BEST_CHECKPOINT = "artifacts/checkpoints/best.pt"
CLASS_NAMES_ARTIFACT = "artifacts/checkpoints/class_names.json"
HEAD_LR = 1e-3


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
    return parser.parse_args()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run inference over a loader and collect true/predicted label indices.

    Args:
        model: The model to evaluate.
        loader: The data loader to iterate.
        device: Device to run on.

    Returns:
        A ``(y_true, y_pred)`` tuple of label-index lists.
    """
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


def main() -> None:
    """Run the full training pipeline for one attempt."""
    args = parse_args()
    set_seed(args.seed)
    init_mlflow()

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    parent_params = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_classes": len(class_names),
        "image_size": IMG_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "attempt_id": args.attempt_id,
    }

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
