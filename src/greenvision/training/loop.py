"""Training loop: per-epoch train/eval plus the two-phase orchestrators.

``run_phase1`` warms up the head on a frozen backbone; ``run_phase2`` gradually
unfreezes the backbone on the ``PHASE2_SCHEDULE``, stepping the LR scheduler on
val loss, checkpointing the best model by val accuracy, and early-stopping when
val accuracy stops improving. Both orchestrators assume an MLflow run is already
active and log per-epoch metrics with ``step=epoch``.
"""

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from greenvision.models.efficientnet_head import NUM_CLASSES
from greenvision.training.optim import build_optimizer
from greenvision.training.phases import freeze_all_backbone, unfreeze_from_block
from greenvision.training.schedulers import build_scheduler

MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
PHASE2_SCHEDULE = [(0, 6), (4, 3), (8, 0)]


def from_idx_for(epoch: int) -> int:
    """Return the unfreeze ``from_idx`` for a given Phase 2 epoch.

    Args:
        epoch: Zero-based Phase 2 epoch index.

    Returns:
        The ``from_idx`` defined by ``PHASE2_SCHEDULE`` for that epoch.
    """
    for start, idx in reversed(PHASE2_SCHEDULE):
        if epoch >= start:
            return idx
    return 6


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run one training epoch and return loss/accuracy.

    Args:
        model: The model to train (set to train mode here).
        loader: Training data loader.
        optimizer: Optimizer stepped once per batch.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        device: Device to run on.

    Returns:
        ``{"train_loss": float, "train_acc": float}`` averaged over all samples.
    """
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return {
        "train_loss": total_loss / total,
        "train_acc": total_correct / total,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a loader and return loss/accuracy.

    Args:
        model: The model to evaluate (set to eval mode here).
        loader: Validation or test data loader.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        device: Device to run on.

    Returns:
        ``{"val_loss": float, "val_acc": float}`` averaged over all samples.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return {
        "val_loss": total_loss / total,
        "val_acc": total_correct / total,
    }


def save_checkpoint(model: nn.Module, path: str, epoch: int, val_acc: float) -> None:
    """Save a model checkpoint with the metadata needed to reload it.

    Args:
        model: The model whose state dict is saved.
        path: Destination file path; parent directories are created.
        epoch: Epoch index this checkpoint was taken at.
        val_acc: Validation accuracy at this checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "num_classes": NUM_CLASSES,
            "arch": "efficientnet_b0",
        },
        path,
    )


def run_phase1(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 3,
    head_lr: float = 1e-3,
) -> dict[str, list[float]]:
    """Phase 1: warm up the classifier head on a frozen backbone.

    Freezes the backbone, builds a head-only optimizer, and trains for a fixed
    number of epochs. Per-epoch metrics are logged to the active MLflow run.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        epochs: Number of warm-up epochs.
        head_lr: Learning rate for the head.

    Returns:
        A history dict mapping each metric name to a per-epoch list.
    """
    freeze_all_backbone(model)
    optimizer = build_optimizer(model, head_lr=head_lr)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    for epoch in range(epochs):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)
        mlflow.log_metrics({**train_m, **val_m}, step=epoch)
        for key, value in {**train_m, **val_m}.items():
            history[key].append(value)

    return history


def run_phase2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[dict[str, list[float]], float, int]:
    """Phase 2: gradually unfreeze the backbone with early stopping.

    Rebuilds the optimizer and scheduler whenever new blocks come online (per
    ``PHASE2_SCHEDULE``), steps ``ReduceLROnPlateau`` on val loss, checkpoints
    the best model by val accuracy, and early-stops after
    ``EARLY_STOP_PATIENCE`` epochs without improvement. Per-epoch metrics
    (including the train/val accuracy gap and current LRs) are logged to the
    active MLflow run.

    Args:
        model: The model to fine-tune.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to run on.
        max_epochs: Maximum number of Phase 2 epochs.

    Returns:
        A ``(history, best_val_acc, best_epoch)`` tuple.
    """
    # Start below zero so the first epoch always produces a best checkpoint,
    # even if val accuracy is 0 (e.g. tiny smoke-test datasets).
    best_val_acc, best_epoch, bad_epochs = -1.0, -1, 0
    current_from_idx = -1  # force rebuild on first iteration
    optimizer: Optimizer | None = None
    scheduler = None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_val_acc_gap": [],
    }

    for epoch in range(max_epochs):
        # Gradual unfreezing: rebuild optimizer when new blocks come online.
        new_from_idx = from_idx_for(epoch)
        if new_from_idx != current_from_idx:
            current_from_idx = new_from_idx
            unfreeze_from_block(model, current_from_idx)
            optimizer = build_optimizer(model)
            scheduler = build_scheduler(optimizer)
            mlflow.log_metric("phase2_from_idx", current_from_idx, step=epoch)

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)

        gap = train_m["train_acc"] - val_m["val_acc"]
        mlflow.log_metrics(
            {
                **train_m,
                **val_m,
                "train_val_acc_gap": gap,
                "lr_head": optimizer.param_groups[2]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        for key, value in {**train_m, **val_m, "train_val_acc_gap": gap}.items():
            history[key].append(value)

        # LR reduction on val loss (more sensitive than val accuracy).
        scheduler.step(val_m["val_loss"])

        # Best checkpoint + early stopping on val accuracy (what we report).
        if val_m["val_acc"] > best_val_acc:
            best_val_acc = val_m["val_acc"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                model,
                "artifacts/checkpoints/best.pt",
                epoch=epoch,
                val_acc=val_m["val_acc"],
            )
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                mlflow.log_metric("early_stopped_epoch", epoch)
                break

    return history, best_val_acc, best_epoch
