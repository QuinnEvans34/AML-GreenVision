"""MLflow helpers: nested-run context managers, artifact logging, registration.

These wrap the MLflow calls used during training so the orchestrator stays
readable: one parent run per attempt, nested child runs per phase, end-of-
training plots/reports, and a one-line model-registration call.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch.nn as nn
from mlflow.entities import Run
from sklearn.metrics import classification_report, confusion_matrix

# NOTE: the tracking store must NOT live under ``artifacts/`` — MLflow's file
# store reserves the folder name "artifacts" (FileStore.ARTIFACTS_FOLDER_NAME),
# which breaks run discovery. Keep it at the repo root instead.
TRACKING_URI = "file:./mlruns"
EXPERIMENT = "greenvision"


def init_mlflow() -> None:
    """Point MLflow at the project tracking store and experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


@contextmanager
def parent_run(attempt_id: str, params: dict) -> Iterator[Run]:
    """Open the outer run for one full training attempt.

    Args:
        attempt_id: Identifier suffixed onto the run name (``attempt_<id>``).
        params: Run-level params logged at the start of the run.

    Yields:
        The active MLflow ``Run``.
    """
    with mlflow.start_run(run_name=f"attempt_{attempt_id}") as run:
        mlflow.log_params(params)
        yield run


@contextmanager
def phase_run(phase: str, params: dict) -> Iterator[Run]:
    """Open a nested child run for a single training phase.

    Args:
        phase: Child run name (e.g. ``"phase1"`` or ``"phase2"``).
        params: Phase-level params logged at the start of the run.

    Yields:
        The active nested MLflow ``Run``.
    """
    with mlflow.start_run(run_name=phase, nested=True) as run:
        mlflow.log_params(params)
        yield run


def log_training_curves(
    history: dict[str, list[float]],
    out_path: str = "artifacts/training_curves.png",
) -> None:
    """Plot loss and accuracy curves and log the PNG as an artifact.

    Args:
        history: Dict with ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc`` per-epoch lists.
        out_path: Where to write the PNG before logging it.
    """
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    ax_l.plot(history["train_loss"], label="train")
    ax_l.plot(history["val_loss"], label="val")
    ax_l.set_title("Loss")
    ax_l.legend()
    ax_l.set_xlabel("epoch")
    ax_a.plot(history["train_acc"], label="train")
    ax_a.plot(history["val_acc"], label="val")
    ax_a.set_title("Accuracy")
    ax_a.legend()
    ax_a.set_xlabel("epoch")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(out_path)


def log_test_artifacts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> None:
    """Log a confusion-matrix heatmap and per-class report for the test set.

    Call inside the parent run after final test-set evaluation.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Class names indexed by label, used for tick labels and the
            classification report.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(cm_path)

    report = classification_report(
        y_true, y_pred, target_names=list(class_names), digits=3
    )
    report_path = "artifacts/per_class_report.txt"
    Path(report_path).write_text(report)
    mlflow.log_artifact(report_path)


def log_model_to_registry(model: nn.Module) -> None:
    """Log the model via the pytorch flavor and register it as ``GreenVision``.

    Wraps the run-time call into one line so the orchestrator stays readable.
    Must be called inside an active run.

    Args:
        model: The trained model to log and register.
    """
    mlflow.pytorch.log_model(model, name="model", registered_model_name="GreenVision")
