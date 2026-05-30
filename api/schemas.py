"""Pydantic v2 schemas for the GreenVision serving API.

Response shapes are documented in ``IMPLEMENTATION_GUIDE.md`` Decision 8.
The Treatment shape mirrors entries in ``data/treatments.json`` (Decision 9).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """Citation for a treatment recommendation."""

    name: str
    url: str


class Treatment(BaseModel):
    """Treatment KB entry for a single class.

    The schema is polymorphic by class type:

    - **Diseases** populate ``severity``, ``time_sensitivity``, ``action_steps``, ``sources``.
    - **Healthy** classes populate ``maintenance_tips`` and ``sources`` (no severity).
    - **Background** populates ``retake_guidance`` (no severity, no sources).
    """

    model_config = ConfigDict(extra="ignore")

    display_name: str = Field(..., description="Human-readable name for the class")
    is_healthy: bool
    is_background: bool
    severity: Optional[str] = Field(
        default=None, description="Disease severity tier: 'low' | 'medium' | 'high'"
    )
    summary: str
    time_sensitivity: Optional[str] = None
    action_steps: Optional[list[str]] = None
    maintenance_tips: Optional[list[str]] = None
    retake_guidance: Optional[list[str]] = None
    sources: list[Source] = Field(default_factory=list)


class TopKPrediction(BaseModel):
    """One entry in the top-k predictions list."""

    class_name: str = Field(..., description="Raw ImageFolder class name")
    display_name: str = Field(..., description="Human-readable label")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Full POST /predict response shape."""

    class_name: str = Field(..., description="Raw predicted class string")
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_healthy: bool
    is_background: bool
    top_k: list[TopKPrediction]
    treatment: Treatment
    warnings: list[str] = Field(default_factory=list)
    model_version: str = Field(..., description="MLflow model URI")
    inference_time_ms: float


class HealthResponse(BaseModel):
    """GET /health response shape."""

    status: str
    model_loaded: bool
    num_classes: int
    model_version: str
    tracking_uri: str
    kb_version: Optional[str] = None
    kb_entry_count: Optional[int] = None
