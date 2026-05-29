"""MLflow Model Registry helpers for promoting GreenVision to Production."""

from __future__ import annotations

import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

MODEL_NAME = "GreenVision"


def list_versions(client: MlflowClient | None = None) -> list[ModelVersion]:
    """Return every registered version of GreenVision, newest first.

    Args:
        client: Optional ``MlflowClient``; one is created if omitted.

    Returns:
        The registered model versions sorted by version number descending.
    """
    client = client or MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def promote_to_production(version: int, client: MlflowClient | None = None) -> None:
    """Move the given version of GreenVision to Production.

    Existing Production versions are automatically archived so we always have
    exactly one Production model — what the FastAPI lifespan loads.

    Args:
        version: The model version number to promote.
        client: Optional ``MlflowClient``; one is created if omitted.
    """
    client = client or MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )


def verify_production_loads() -> None:
    """Sanity-check that ``mlflow.pytorch.load_model`` works on Production.

    The serving layer depends on this; if it fails here, it'll fail there too.
    """
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
    print(f"✓ Production model loaded successfully — type: {type(model).__name__}")
