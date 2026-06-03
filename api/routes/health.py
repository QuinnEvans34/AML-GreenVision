"""GET /health — liveness + readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.inference import MLFLOW_TRACKING_URI, MODEL_URI
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    """Confirm the API is reachable and the model + KB are loaded."""
    state = request.app.state
    model_loaded = getattr(state, "model", None) is not None
    class_names = getattr(state, "class_names", None) or []
    metadata = getattr(state, "treatments_metadata", {}) or {}
    treatments = getattr(state, "treatments", {}) or {}

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        num_classes=len(class_names),
        model_version=MODEL_URI,
        tracking_uri=MLFLOW_TRACKING_URI,
        kb_version=metadata.get("version"),
        kb_entry_count=len(treatments) if treatments else None,
    )
