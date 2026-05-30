"""POST /predict — single-image disease diagnosis."""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from api.inference import MODEL_URI, predict_image
from api.schemas import PredictionResponse, TopKPrediction
from api.treatments import get_treatment

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB (Decision 8)


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    request: Request, file: UploadFile = File(...)
) -> PredictionResponse:
    """Diagnose a leaf image and return treatment recommendation.

    Workflow:
    1. Size check (reject > 10 MB, reject empty uploads).
    2. PIL decode (reject undecodable files).
    3. Run inference on the loaded Production model.
    4. Look up the treatment entry for the predicted class.
    5. Return the merged response.

    Args:
        request: The FastAPI request (used to read ``app.state``).
        file: The uploaded image.

    Returns:
        A ``PredictionResponse`` with prediction + treatment + warnings.
    """
    # ---- 1. Size + emptiness ----
    body = await file.read()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Empty file upload")
    if len(body) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
        )

    # ---- 2. Decode ----
    try:
        pil_image = Image.open(io.BytesIO(body))
        pil_image.load()  # force decode now so errors surface here
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image — please upload a JPG, PNG, or WebP file",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {exc}")

    # ---- 3. Inference ----
    state = request.app.state
    if getattr(state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = predict_image(
        state.model, state.class_names, state.device, pil_image
    )

    # ---- 4. Treatment merge ----
    try:
        treatment = get_treatment(state.treatments, result.predicted_class)
    except KeyError as exc:
        # Server-side inconsistency, not a client error
        raise HTTPException(status_code=500, detail=str(exc))

    top_k_response = [
        TopKPrediction(
            class_name=name,
            display_name=state.treatments[name].display_name,
            probability=prob,
        )
        for name, prob in result.top_k
    ]

    return PredictionResponse(
        class_name=result.predicted_class,
        display_name=treatment.display_name,
        confidence=result.confidence,
        is_healthy=treatment.is_healthy,
        is_background=treatment.is_background,
        top_k=top_k_response,
        treatment=treatment,
        warnings=result.warnings,
        model_version=MODEL_URI,
        inference_time_ms=round(result.inference_ms, 2),
    )
