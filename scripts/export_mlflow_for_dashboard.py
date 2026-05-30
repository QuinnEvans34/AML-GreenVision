"""Export training history + test-set diagnostics to web/public/training_data.json.

Phase 5 of the W10 build plan. The dashboard analytics page (Phase 6) reads
this static JSON instead of querying MLflow live — keeps the presentation
stable when the demo machine's network is unreliable.

Pulls from ./mlruns/:
  - Parent run for attempt_NNN (defaults to 002).
  - phase1 + phase2 child runs for per-epoch metric history.

Then re-runs the Production model against the held-out test set to
regenerate the confusion matrix and per-class precision/recall/F1.
This guarantees the numbers shown in the dashboard match exactly what
the Production model currently does — not a stale snapshot.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py
    PYTHONPATH=src .venv/bin/python scripts/export_mlflow_for_dashboard.py --attempt 002
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix

from greenvision.data.datasets import build_dataloaders


EXPERIMENT_NAME = "greenvision"
DEFAULT_OUTPUT = "web/public/training_data.json"
DEFAULT_DATA_ROOT = "data/raw/PlantVillage"
DEFAULT_TRACKING_URI = "file:./mlruns"
MODEL_URI = "models:/GreenVision/Production"


# ──────────────────────────────────────────────────────────────────────
# Display formatting — keep in sync with web/lib/format-class-name.ts
# ──────────────────────────────────────────────────────────────────────


def format_display(raw: str) -> str:
    """Convert a raw ImageFolder class string to a human label."""
    if raw == "Background_without_leaves":
        return "No leaf detected"
    crop_raw, _, condition_raw = raw.partition("___")
    crop = crop_raw.replace("_", " ").strip()
    if crop == "Pepper, bell":
        crop = "Bell pepper"
    if condition_raw == "healthy":
        return f"{crop} (healthy)"
    condition = condition_raw.replace("_", " ").strip()
    return f"{crop} — {condition}"


# ──────────────────────────────────────────────────────────────────────
# MLflow walking
# ──────────────────────────────────────────────────────────────────────


def find_attempt_runs(
    client: MlflowClient, attempt: str
) -> tuple[str, str | None, str | None]:
    """Locate the parent attempt run plus its phase1 / phase2 children.

    Args:
        client: An ``MlflowClient`` pointed at the project's tracking store.
        attempt: The attempt ID suffix, e.g. ``"002"``.

    Returns:
        Tuple ``(parent_run_id, phase1_run_id, phase2_run_id)``.
        ``phase1`` / ``phase2`` may be ``None`` if a phase was skipped.

    Raises:
        RuntimeError: If the experiment or the named parent run can't be found.
    """
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found in the tracking store."
        )

    target_name = f"attempt_{attempt}"
    candidates = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{target_name}'",
        max_results=5,
    )
    if not candidates:
        raise RuntimeError(
            f"Parent run '{target_name}' not found in experiment '{EXPERIMENT_NAME}'."
        )
    parent = candidates[0]
    parent_id = parent.info.run_id

    children = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
    )
    phase1 = next(
        (r.info.run_id for r in children if r.info.run_name == "phase1"),
        None,
    )
    phase2 = next(
        (r.info.run_id for r in children if r.info.run_name == "phase2"),
        None,
    )
    return parent_id, phase1, phase2


def collect_epoch_metrics(
    client: MlflowClient, run_id: str, phase_label: str
) -> list[dict[str, Any]]:
    """Pull per-epoch metric history for a child run.

    Returns one dict per epoch with ``epoch``, ``phase``, and any of the
    standard metric names that exist on the run.
    """
    metric_names = [
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "train_val_acc_gap",
        "lr_head",
        "lr_backbone",
    ]
    histories: dict[str, dict[int, float]] = {}
    for name in metric_names:
        history = client.get_metric_history(run_id, name)
        if history:
            histories[name] = {h.step: float(h.value) for h in history}

    epochs = sorted(set().union(*[h.keys() for h in histories.values()])) if histories else []
    return [
        {
            "epoch_in_phase": int(epoch),
            "phase": phase_label,
            **{
                metric: histories[metric].get(epoch)
                for metric in metric_names
                if metric in histories
            },
        }
        for epoch in epochs
    ]


# ──────────────────────────────────────────────────────────────────────
# Test-set re-evaluation
# ──────────────────────────────────────────────────────────────────────


def evaluate_production_on_test(
    data_root: str,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[list[int], list[int], list[str]]:
    """Load the Production model and run inference on the held-out test split."""
    model = mlflow.pytorch.load_model(MODEL_URI)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device).eval()

    _, _, test_loader, class_names = build_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            y_true.extend(int(v) for v in y.tolist())
            y_pred.extend(int(v) for v in preds)

    return y_true, y_pred, class_names


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt",
        default="002",
        help="Attempt ID suffix (default: 002 — our W9A1 winning run)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Same seed used during training so the test split is reproducible",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip the test-set re-evaluation (faster; confusion matrix will be empty)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # ── 1. Find the runs ──────────────────────────────────────────────
    print(f"→ Locating attempt_{args.attempt} run group…")
    parent_id, phase1_id, phase2_id = find_attempt_runs(client, args.attempt)
    print(f"  parent  : {parent_id}")
    print(f"  phase1  : {phase1_id}")
    print(f"  phase2  : {phase2_id}")

    # ── 2. Collect per-epoch metrics ──────────────────────────────────
    print("→ Pulling per-epoch metric history…")
    epoch_metrics: list[dict[str, Any]] = []
    global_epoch = 0
    if phase1_id is not None:
        phase1 = collect_epoch_metrics(client, phase1_id, "phase1")
        for m in phase1:
            epoch_metrics.append({**m, "epoch": global_epoch})
            global_epoch += 1
    if phase2_id is not None:
        phase2 = collect_epoch_metrics(client, phase2_id, "phase2")
        for m in phase2:
            epoch_metrics.append({**m, "epoch": global_epoch})
            global_epoch += 1
    print(f"  total epochs: {len(epoch_metrics)}")

    best = max(
        (m for m in epoch_metrics if m.get("val_acc") is not None),
        key=lambda m: m["val_acc"],
        default={},
    )
    best_val_acc = best.get("val_acc")
    best_epoch_global = best.get("epoch")
    if best_val_acc is not None:
        print(f"  best val_acc: {best_val_acc:.4f} @ epoch {best_epoch_global}")

    # Wall-clock training time from the parent run's start/end timestamps
    parent_run = client.get_run(parent_id)
    training_time_sec: float | None = None
    if parent_run.info.start_time and parent_run.info.end_time:
        training_time_sec = (
            parent_run.info.end_time - parent_run.info.start_time
        ) / 1000.0

    # ── 3. Test-set diagnostics ───────────────────────────────────────
    confusion: list[list[int]] = []
    per_class_metrics: list[dict[str, Any]] = []
    class_names: list[str] = []
    test_acc: float | None = None

    if not args.skip_test_eval:
        print("→ Loading Production model and re-running test-set inference…")
        y_true, y_pred, class_names = evaluate_production_on_test(
            args.data_root,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        test_acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        print(f"  test accuracy: {test_acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")

        labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion = cm.tolist()

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        for idx, name in enumerate(class_names):
            row = report.get(name, {})
            per_class_metrics.append(
                {
                    "index": idx,
                    "class_name": name,
                    "display_name": format_display(name),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "f1": float(row.get("f1-score", 0.0)),
                    "support": int(row.get("support", 0)),
                }
            )

    # ── 4. Assemble and write JSON ────────────────────────────────────
    output = {
        "_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "source_attempt": f"attempt_{args.attempt}",
            "tracking_uri": args.tracking_uri,
            "model_uri": MODEL_URI,
            "data_root": args.data_root,
            "seed": args.seed,
        },
        "best_run": {
            "attempt_id": args.attempt,
            "parent_run_id": parent_id,
            "phase1_run_id": phase1_id,
            "phase2_run_id": phase2_id,
            "best_val_acc": best_val_acc,
            "best_epoch_global": best_epoch_global,
            "test_acc": test_acc,
            "total_epochs": len(epoch_metrics),
            "training_time_seconds": training_time_sec,
            "num_classes": len(class_names) if class_names else None,
        },
        "epoch_metrics": epoch_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
        "class_names": class_names,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"✓ Wrote {output_path}")
    print(f"  epochs:           {len(epoch_metrics)}")
    print(f"  per-class metrics:{len(per_class_metrics)}")
    print(f"  confusion matrix: {len(confusion)}×{len(confusion[0]) if confusion else 0}")


if __name__ == "__main__":
    main()
