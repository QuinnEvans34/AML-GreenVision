"""Model loading and image inference helpers.

The lifespan in ``api.main`` calls ``load_production_model`` and
``load_class_names`` once at startup and stashes them on ``app.state``.
Per-request inference goes through ``predict_image`` which reuses the
same ``eval_tfms`` from training to guarantee train/inference parity
(``IMPLEMENTATION_GUIDE.md`` Decision 3, Decision 8).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from greenvision.data.transforms import eval_tfms

# ----------------------------------------------------------------------
# Configuration constants — keep aligned with IMPLEMENTATION_GUIDE
# ----------------------------------------------------------------------

MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_URI = "models:/GreenVision/Production"
CLASS_NAMES_PATH = Path("artifacts/checkpoints/class_names.json")

# Quality-check thresholds (Decision 8 + Decision 10)
BLUR_THRESHOLD = 100.0
MIN_IMAGE_DIM = 64
TOP_K_COUNT = 3

# Confidence bands (Decision 10)
HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.40


class PredictionResult(NamedTuple):
    """Internal result type returned by ``predict_image``."""

    predicted_class: str
    confidence: float
    top_k: list[tuple[str, float]]  # [(class_name, probability), ...]
    inference_ms: float
    warnings: list[str]


# ----------------------------------------------------------------------
# Loading helpers (called once at startup)
# ----------------------------------------------------------------------


def select_device() -> torch.device:
    """Pick MPS on Apple Silicon, CUDA if available, else CPU.

    Matches training-time device selection so the model behaves
    consistently between training and serving.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_production_model(device: torch.device) -> torch.nn.Module:
    """Load ``models:/GreenVision/Production`` from MLflow Registry.

    Args:
        device: Target device for the loaded model.

    Returns:
        The model in ``eval()`` mode, on the target device.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.to(device)
    model.eval()
    return model


def load_class_names() -> list[str]:
    """Load the 39 class names from the training-time artifact.

    The order here is the contract between model output indices and
    human-readable labels — never re-derive class names elsewhere.
    """
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names artifact missing at {CLASS_NAMES_PATH}. "
            "Train the model and ensure it logs class_names.json."
        )
    with CLASS_NAMES_PATH.open() as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Per-request inference
# ----------------------------------------------------------------------


def _laplacian_variance(pil_image: Image.Image) -> float:
    """Variance of the Laplacian of a grayscale image (blur estimator).

    Pure-numpy implementation; lower variance → more likely blurry.
    Uses ``np.roll`` to apply a 4-neighbor Laplacian kernel, then trims
    the border to avoid wrap-around artifacts.
    """
    gray = np.array(pil_image.convert("L"), dtype=np.float32)
    if gray.size < 9:  # too tiny for a 3x3 stencil
        return 0.0
    up = np.roll(gray, -1, axis=0)
    down = np.roll(gray, 1, axis=0)
    left = np.roll(gray, -1, axis=1)
    right = np.roll(gray, 1, axis=1)
    laplacian = up + down + left + right - 4.0 * gray
    return float(laplacian[1:-1, 1:-1].var())


def _quality_warnings(pil_image: Image.Image) -> list[str]:
    """Return non-fatal warnings about input image quality."""
    warnings: list[str] = []
    w, h = pil_image.size
    if min(w, h) < MIN_IMAGE_DIM:
        warnings.append(
            f"Image is very small ({w}×{h} px) — accuracy may be reduced"
        )
    blur = _laplacian_variance(pil_image)
    if blur < BLUR_THRESHOLD:
        warnings.append("Image appears blurry — please retake in better focus")
    return warnings


def predict_image(
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    pil_image: Image.Image,
) -> PredictionResult:
    """Run a single-image prediction through the loaded model.

    Args:
        model: The Production model from ``load_production_model``.
        class_names: Ordered list of class names from
            ``load_class_names`` (index → name).
        device: The device the model lives on.
        pil_image: Input image; will be converted to RGB.

    Returns:
        A ``PredictionResult`` with the predicted class, confidence,
        top-k alternatives, timing, and any quality warnings.
    """
    warnings = _quality_warnings(pil_image)

    t0 = time.perf_counter()
    tensor = eval_tfms(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    top_k_indices = np.argsort(probs)[::-1][:TOP_K_COUNT]
    top_k = [(class_names[int(i)], float(probs[int(i)])) for i in top_k_indices]
    predicted_class, confidence = top_k[0]

    # Confidence-based warnings (Decision 10)
    if confidence < MEDIUM_CONFIDENCE:
        warnings.append(
            f"Model confidence is low ({confidence:.0%}). "
            "This prediction should not be relied on alone — please retake "
            "the photo or consult an expert."
        )
    elif confidence < MODERATE_CONFIDENCE:
        warnings.append(
            f"Model confidence is moderate ({confidence:.0%}). "
            "Consider reviewing the alternative predictions below."
        )

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        top_k=top_k,
        inference_ms=elapsed_ms,
        warnings=warnings,
    )
