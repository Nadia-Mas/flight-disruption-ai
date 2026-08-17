from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flightrescue import FlightRescueService  # noqa: E402
from flightrescue.weather import NWSAirportWeather  # noqa: E402

app = FastAPI(
    title="FlightRescue AI API",
    version="0.4.0",
    description="FlightRescue research API with trained OGG inference, official NWS weather, historical airline evidence, and v3 dual-airport weather support for OGG/LAX/DFW.",
)

allowed = os.getenv(
    "FLIGHTRESCUE_CORS_ORIGINS",
    "https://nadia-mas.github.io,http://localhost:8000,http://127.0.0.1:8000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in allowed.split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = FlightRescueService(ROOT)
airport_weather = NWSAirportWeather()


class FeatureRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Leakage-safe model features created by the research pipeline.")


class ScenarioRequest(BaseModel):
    airline: str = Field(..., examples=["AA"])
    direction: Literal["arrival", "departure"] = "departure"
    other_airport: str = Field(..., min_length=3, max_length=4, examples=["LAX"])
    scheduled_local: str = Field(..., description="OGG-local scheduled date/time, e.g. 2026-08-17T14:30")
    distance_miles: float | None = Field(default=None, ge=0)
    event_type: Literal["auto", "normal", "rain", "wind", "flood", "tropical", "thunderstorm"] = "auto"
    temperature_f: float | None = None
    dewpoint_f: float | None = None
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    station_pressure_inhg: float | None = None
    sea_level_pressure_inhg: float | None = None
    visibility_miles: float | None = Field(default=None, ge=0)
    wind_direction_deg: float | None = Field(default=None, ge=0, le=360)
    wind_speed_mph: float | None = Field(default=None, ge=0)
    wind_gust_mph: float | None = Field(default=None, ge=0)
    precipitation_in: float | None = Field(default=None, ge=0)


class SimilarityRequest(BaseModel):
    standardized_event_vector: dict[str, float]
    k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root() -> dict[str, Any]:
    return {"name": "FlightRescue AI API", "status": service.status(), "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    status = service.status()
    status["dual_airport_weather"] = True
    status["v3_supported_airports"] = ["OGG", "LAX", "DFW"]
    return status


@app.get("/weather/ogg")
def weather_ogg(scheduled_local: str) -> dict[str, Any]:
    try:
        return service.nws_weather(scheduled_local)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"NWS lookup failed: {exc}") from exc


@app.get("/weather/airport/{airport}")
def weather_airport(airport: str, scheduled_local: str) -> dict[str, Any]:
    try:
        return airport_weather.forecast(airport, scheduled_local)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"NWS lookup failed: {exc}") from exc


@app.post("/predict/features")
def predict_features(request: FeatureRequest) -> dict[str, Any]:
    try:
        return service.predict_features(request.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/scenario")
def predict_scenario(request: ScenarioRequest) -> dict[str, Any]:
    try:
        return service.predict_scenario(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scenario prediction failed: {exc}") from exc


@app.get("/historical/context")
def historical_context(event_type: str = "normal", k: int = 5) -> dict[str, Any]:
    return {"events": service.historical_context(event_type=event_type, k=max(1, min(k, 20)))}


@app.post("/similar-events")
def similar_events(request: SimilarityRequest) -> dict[str, Any]:
    return {"events": service.similar_events_from_vector(request.standardized_event_vector, k=request.k)}
