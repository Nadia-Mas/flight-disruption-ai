from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

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
    version="0.3.0",
    description="Artifact-backed OGG flight-disruption risk with automatic NWS weather and historical recovery context.",
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

OGG_LAT = 20.8987
OGG_LON = -156.4305
OGG_TZ = ZoneInfo("Pacific/Honolulu")
NWS_USER_AGENT = "FlightRescueAI/0.3 (https://github.com/Nadia-Mas/flight-disruption-ai)"

WEATHER_PRESETS: dict[str, dict[str, float]] = {
    "normal": {},
    "rain": {
        "humidity_pct": 90.0, "visibility_miles": 6.0,
        "wind_speed_mph": 12.0, "wind_gust_mph": 20.0, "precipitation_in": 0.25,
    },
    "wind": {
        "visibility_miles": 10.0,
        "wind_speed_mph": 30.0, "wind_gust_mph": 45.0, "precipitation_in": 0.0,
    },
    "flood": {
        "humidity_pct": 95.0, "visibility_miles": 3.0,
        "wind_speed_mph": 18.0, "wind_gust_mph": 30.0, "precipitation_in": 0.75,
    },
    "tropical": {
        "humidity_pct": 95.0, "visibility_miles": 2.0,
        "wind_speed_mph": 45.0, "wind_gust_mph": 65.0, "precipitation_in": 0.75,
    },
    "thunderstorm": {
        "humidity_pct": 90.0, "visibility_miles": 4.0,
        "wind_speed_mph": 25.0, "wind_gust_mph": 45.0, "precipitation_in": 0.40,
    },
}

WEATHER_KEYS = [
    "temperature_f", "dewpoint_f", "humidity_pct",
    "station_pressure_inhg", "sea_level_pressure_inhg",
    "visibility_miles", "wind_direction_deg", "wind_speed_mph",
    "wind_gust_mph", "precipitation_in",
]


class FeatureRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Leakage-safe model features created by Notebook 04.")


class ScenarioRequest(BaseModel):
    airline: str = Field(..., examples=["HA"])
    direction: Literal["arrival", "departure"] = "departure"
    other_airport: str = Field(..., min_length=3, max_length=4, examples=["HNL"])
    scheduled_local: str = Field(..., description="OGG-local scheduled date/time, e.g. 2026-08-17T14:30")
    distance_miles: float | None = Field(default=None, ge=0)
    event_type: Literal["normal", "rain", "wind", "flood", "tropical", "thunderstorm"] = "normal"
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


def _nws_json(url: str) -> dict[str, Any]:
    req = Request(
        url,
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
    )
    with urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _number_from_qv(obj: Any) -> float | None:
    if isinstance(obj, dict) and isinstance(obj.get("value"), (int, float)):
        return float(obj["value"])
    return None


def _wind_mph(value: Any) -> float | None:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
    return max(nums) if nums else None


def _wind_direction_deg(value: Any) -> float | None:
    dirs = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return float(dirs[str(value).upper()]) if value and str(value).upper() in dirs else None


def nws_weather_for_time(scheduled_local: str) -> dict[str, Any]:
    try:
        target = datetime.fromisoformat(scheduled_local)
        if target.tzinfo is None:
            target = target.replace(tzinfo=OGG_TZ)
        else:
            target = target.astimezone(OGG_TZ)

        point = _nws_json(f"https://api.weather.gov/points/{OGG_LAT:.4f},{OGG_LON:.4f}")
        hourly_url = point.get("properties", {}).get("forecastHourly")
        if not hourly_url:
            raise RuntimeError("NWS point response did not include forecastHourly.")

        hourly = _nws_json(hourly_url)
        periods = hourly.get("properties", {}).get("periods", [])
        if not periods:
            raise RuntimeError("NWS hourly forecast returned no periods.")

        def delta_seconds(period: dict[str, Any]) -> float:
            return abs((datetime.fromisoformat(period["startTime"]) - target).total_seconds())

        period = min(periods, key=delta_seconds)
        delta_h = delta_seconds(period) / 3600.0
        if delta_h > 3:
            return {
                "available": False,
                "provider": "National Weather Service",
                "reason": "Requested flight time is outside the available NWS hourly forecast window.",
                "source_url": hourly_url,
            }

        dew_c = _number_from_qv(period.get("dewpoint"))
        pop = _number_from_qv(period.get("probabilityOfPrecipitation"))
        values = {
            "temperature_f": float(period["temperature"]) if isinstance(period.get("temperature"), (int, float)) else None,
            "dewpoint_f": dew_c * 9.0 / 5.0 + 32.0 if dew_c is not None else None,
            "humidity_pct": _number_from_qv(period.get("relativeHumidity")),
            "wind_speed_mph": _wind_mph(period.get("windSpeed")),
            "wind_direction_deg": _wind_direction_deg(period.get("windDirection")),
        }

        return {
            "available": True,
            "provider": "National Weather Service",
            "location": "Kahului Airport (OGG)",
            "forecast_time": period.get("startTime"),
            "short_forecast": period.get("shortForecast"),
            "precipitation_probability_pct": pop,
            "values": {k: v for k, v in values.items() if v is not None},
            "source_url": hourly_url,
        }
    except Exception as exc:
        return {"available": False, "provider": "National Weather Service", "reason": str(exc)}


def merge_weather(
    scenario: dict[str, Any],
    nws: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    event_type = str(scenario.get("event_type", "normal")).lower()
    preset = WEATHER_PRESETS.get(event_type, {})
    nws_values = nws.get("values", {}) if nws.get("available") else {}
    merged = dict(scenario)
    sources: dict[str, str] = {}

    for key in WEATHER_KEYS:
        user_value = scenario.get(key)
        if user_value is not None:
            merged[key] = user_value
            sources[key] = "user override"
            continue

        base = nws_values.get(key)
        preset_value = preset.get(key)

        if preset_value is not None:
            if base is None:
                value = preset_value
            elif key == "visibility_miles":
                value = min(float(base), float(preset_value))
            elif key in {"wind_speed_mph", "wind_gust_mph", "humidity_pct", "precipitation_in"}:
                value = max(float(base), float(preset_value))
            else:
                value = base
            merged[key] = value
            sources[key] = "NWS + scenario preset" if base is not None else "scenario preset"
        elif base is not None:
            merged[key] = base
            sources[key] = "NWS forecast"
        else:
            merged[key] = None
            sources[key] = "model imputation"

    return merged, sources


@app.get("/")
def root() -> dict[str, Any]:
    return {"name": "FlightRescue AI API", "status": service.status(), "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    status = service.status()
    status["weather_provider"] = "National Weather Service (api.weather.gov)"
    status["api_version"] = "0.3.0"
    return status


@app.get("/weather/forecast")
def weather_forecast(scheduled_local: str) -> dict[str, Any]:
    return nws_weather_for_time(scheduled_local)


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
        original = request.model_dump()
        nws = nws_weather_for_time(original["scheduled_local"])
        enriched, input_sources = merge_weather(original, nws)

        result = service.predict_scenario(enriched)

        raw_severe = float(result.get("severe_disruption_probability", 0.0))
        any_p = float(result.get("disruption_probability", 0.0))
        result["raw_severe_disruption_probability"] = raw_severe
        result["severe_disruption_probability"] = min(raw_severe, any_p)
        result["severe_disruption_flag"] = bool(
            result["severe_disruption_probability"]
            >= float(result.get("thresholds", {}).get("severe_disruption", 0.5))
        )

        result["confidence"] = "Experimental"
        result["weather"] = {
            "event_type": original.get("event_type", "normal"),
            "nws": nws,
            "model_inputs": {k: enriched.get(k) for k in WEATHER_KEYS},
            "input_sources": input_sources,
            "preset_applied": original.get("event_type", "normal") != "normal",
            "note": (
                "Official NWS forecast values are used when available. "
                "Hazard selections are what-if scenario floors for blank fields; "
                "explicit expert overrides always take precedence."
            ),
        }
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scenario prediction failed: {exc}") from exc


@app.get("/historical/context")
def historical_context(event_type: str = "normal", k: int = 5) -> dict[str, Any]:
    return {"events": service.historical_context(event_type=event_type, k=max(1, min(k, 20)))}


@app.post("/similar-events")
def similar_events(request: SimilarityRequest) -> dict[str, Any]:
    return {
        "events": service.similar_events_from_vector(
            request.standardized_event_vector,
            k=request.k,
        )
    }
