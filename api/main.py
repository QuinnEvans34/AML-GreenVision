"""GreenVision FastAPI serving app.

Loads the Production model + class names + treatment KB once at startup
via the lifespan context, exposes ``/health`` and ``/predict``, and allows
CORS from the Next.js dev server.

Run:
    PYTHONPATH=src .venv/bin/uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.inference import (
    load_class_names,
    load_production_model,
    select_device,
)
from api.routes import health as health_routes
from api.routes import predict as predict_routes
from api.treatments import load_treatments


@asynccontextmanager
async def lifespan(app: FastAPI):
    """One-time startup: model, class names, treatment KB."""
    print("GreenVision API starting up…")

    device = select_device()
    print(f"  • Device:          {device}")

    print("  • Loading Production model from MLflow Registry…")
    model = load_production_model(device)
    print(f"    → {type(model).__name__} loaded on {device}")

    class_names = load_class_names()
    print(f"  • Class names:     {len(class_names)} classes")

    treatments, metadata = load_treatments()
    print(
        f"  • Treatment KB:    {len(treatments)} entries "
        f"(v{metadata.get('version', '?')})"
    )

    # Stash on app.state for route access
    app.state.model = model
    app.state.class_names = class_names
    app.state.treatments = treatments
    app.state.treatments_metadata = metadata
    app.state.device = device

    print("GreenVision API ready.\n")
    yield
    print("\nGreenVision API shutting down.")


app = FastAPI(
    title="GreenVision",
    description=(
        "Plant disease classifier serving API. Upload a leaf image to "
        "POST /predict and get back a diagnosis plus treatment recommendation. "
        "Powered by EfficientNet-B0 fine-tuned on PlantVillage, served from "
        "the MLflow Model Registry (Production stage)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for the Next.js dev server. The plan also uses next.config.js
# rewrites for same-origin requests; CORS is the belt-and-suspenders so
# direct fetches from :3000 work too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_routes.router)
app.include_router(predict_routes.router)
