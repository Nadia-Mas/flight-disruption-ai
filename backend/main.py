from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flightrescue import FlightRescueService  # noqa: E402

app = FastAPI(
    title="FlightRescue AI API",
    version="0.1.0",
    description="Research API for OGG flight-disruption risk and historical recovery context.",
)

allowed = os.getenv("FLIGHTRESCUE_CORS_ORIGINS", "https://nadia-mas.github.io,http://localhost:8000,http://127.0.0.1:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in allowed.split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = FlightRescueService(ROOT)


class FeatureRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Leakage-safe model features created by Notebook 04.")


class SimilarityRequest(BaseModel):
    standardized_event_vector: dict[str, float]
    k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "FlightRescue AI API",
        "status": service.status(),
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return service.status()


@app.post("/predict/features")
def predict_features(request: FeatureRequest) -> dict[str, Any]:
    try:
        return service.predict_features(request.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc


@app.post("/similar-events")
def similar_events(request: SimilarityRequest) -> dict[str, Any]:
    return {
        "events": service.similar_events_from_vector(
            request.standardized_event_vector,
            k=request.k,
        )
    }
