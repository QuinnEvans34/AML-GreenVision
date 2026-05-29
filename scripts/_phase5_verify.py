"""Phase 5 verification: exercise the MLflow nested-run + registry plumbing.

Creates one parent run with two nested children and registers a throwaway
GreenVision model. Run from the repo root:
    PYTHONPATH=. .venv/bin/python scripts/_phase5_verify.py
Then inspect with:
    mlflow ui --backend-store-uri file:./mlruns
"""

import mlflow
import torch.nn as nn

from greenvision.training.mlflow_utils import (
    init_mlflow,
    log_model_to_registry,
    parent_run,
    phase_run,
)

init_mlflow()

with parent_run("test", {"seed": 42, "batch_size": 64}):
    with phase_run("phase1", {"epochs": 3, "head_lr": 1e-3}):
        mlflow.log_metric("val_acc", 0.5, step=0)

    with phase_run("phase2", {"max_epochs": 20, "backbone_lr": 1e-4}):
        mlflow.log_metric("val_acc", 0.8, step=0)
        mlflow.log_metric("val_loss", 0.3, step=0)
        fake_model = nn.Linear(1280, 39)
        log_model_to_registry(fake_model)

print(
    "✓ Phase 5 verify complete — parent attempt_test with phase1 + phase2 nested runs."
)
