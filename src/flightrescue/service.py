from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ArtifactPaths:
    any_model: Path
    severe_model: Path
    metadata: Path
    similarity_index: Path


class FlightRescueService:
    """Artifact-backed inference and historical context for FlightRescue AI.

    The service never fabricates a trained-model result. If exported artifacts are
    unavailable, prediction endpoints return a service-unavailable error.
    """

    AIRLINE_CODES = {
        "american": "AA", "aa": "AA",
        "alaska": "AS", "as": "AS",
        "jetblue": "B6", "b6": "B6",
        "delta": "DL", "dl": "DL",
        "frontier": "F9", "f9": "F9",
        "hawaiian": "HA", "ha": "HA",
        "spirit": "NK", "nk": "NK",
        "united": "UA", "ua": "UA",
        "southwest": "WN", "wn": "WN",
    }

    EVENT_TERMS = {
        "normal": [],
        "rain": ["heavy rain", "flood", "flash flood"],
        "wind": ["high wind", "strong wind", "thunderstorm wind"],
        "flood": ["flash flood", "flood"],
        "tropical": ["tropical storm", "hurricane", "tropical depression"],
        "thunderstorm": ["thunderstorm", "lightning"],
    }

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.paths = ArtifactPaths(
            any_model=self.root / "models/any_disruption_model.joblib",
            severe_model=self.root / "models/severe_disruption_model.joblib",
            metadata=self.root / "models/flightrescue_inference_metadata.json",
            similarity_index=self.root / "data/processed/ogg_event_similarity_index_2020_2025.csv",
        )
        self.any_model = None
        self.severe_model = None
        self.metadata: dict[str, Any] = {}
        self.similarity_index: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        if self.paths.metadata.exists():
            self.metadata = json.loads(self.paths.metadata.read_text())
        if self.paths.any_model.exists():
            self.any_model = joblib.load(self.paths.any_model)
        if self.paths.severe_model.exists():
            self.severe_model = joblib.load(self.paths.severe_model)
        if self.paths.similarity_index.exists():
            self.similarity_index = pd.read_csv(self.paths.similarity_index, low_memory=False)

    @property
    def ready(self) -> bool:
        return self.any_model is not None and self.severe_model is not None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "any_model": self.paths.any_model.exists(),
            "severe_model": self.paths.severe_model.exists(),
            "metadata": self.paths.metadata.exists(),
            "similarity_index": self.paths.similarity_index.exists(),
            "version": self.metadata.get("artifact_version", "research-v1"),
        }

    @staticmethod
    def _risk_label(p: float) -> str:
        if p >= 0.75:
            return "very_high"
        if p >= 0.55:
            return "high"
        if p >= 0.35:
            return "moderate"
        return "low"

    def _complete_model_frame(self, supplied: dict[str, Any]) -> pd.DataFrame:
        cols = self.metadata.get("feature_columns", [])
        if not cols:
            # Exported sklearn pipelines still need the training feature names.
            raise RuntimeError("Inference metadata does not contain feature_columns.")
        row = {c: np.nan for c in cols}
        row.update({k: v for k, v in supplied.items() if k in row})
        return pd.DataFrame([row], columns=cols)

    def predict_features(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(
                "Model artifacts are not exported yet. Run the inference-artifact build workflow."
            )

        frame = self._complete_model_frame(features)
        p_any = float(self.any_model.predict_proba(frame)[:, 1][0])
        p_severe = float(self.severe_model.predict_proba(frame)[:, 1][0])

        any_threshold = float(self.metadata.get("any_disruption_threshold", 0.5))
        severe_threshold = float(self.metadata.get("severe_disruption_threshold", 0.5))

        return {
            "disruption_probability": p_any,
            "severe_disruption_probability": p_severe,
            "disruption_flag": bool(p_any >= any_threshold),
            "severe_disruption_flag": bool(p_severe >= severe_threshold),
            "risk_level": self._risk_label(p_any),
            "severe_risk_level": self._risk_label(p_severe),
            "thresholds": {
                "any_disruption": any_threshold,
                "severe_disruption": severe_threshold,
            },
            "model_status": "artifact_backed",
        }

    def scenario_features(self, scenario: dict[str, Any]) -> dict[str, Any]:
        dt = pd.Timestamp(scenario["scheduled_local"])
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
        hour_decimal = dt.hour + dt.minute / 60.0
        hhmm = dt.hour * 100 + dt.minute
        airline = str(scenario.get("airline", "")).strip().lower()
        airline_code = self.AIRLINE_CODES.get(airline, str(scenario.get("airline", "")).upper())

        f: dict[str, Any] = {
            "Reporting_Airline": airline_code,
            "direction": str(scenario.get("direction", "departure")).lower(),
            "other_airport": str(scenario.get("other_airport", "UNKNOWN")).upper(),
            "Distance": scenario.get("distance_miles"),
            # The public form provides the OGG-local scheduled clock time. The
            # unavailable counterpart is intentionally left for model imputation.
            "CRSDepTime": hhmm if str(scenario.get("direction", "departure")).lower() == "departure" else np.nan,
            "CRSArrTime": hhmm if str(scenario.get("direction", "departure")).lower() == "arrival" else np.nan,
            "sched_hour": dt.hour,
            "sched_dow": dt.dayofweek,
            "sched_month": dt.month,
            "sched_dayofyear": dt.dayofyear,
            "is_weekend": int(dt.dayofweek >= 5),
            "hour_sin": float(np.sin(2 * np.pi * hour_decimal / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour_decimal / 24)),
            "month_sin": float(np.sin(2 * np.pi * dt.month / 12)),
            "month_cos": float(np.cos(2 * np.pi * dt.month / 12)),
            "covid_era": int(dt.year in (2020, 2021)),
            "weather_age_min": 0.0,
            "HourlyDryBulbTemperature_num": scenario.get("temperature_f"),
            "HourlyDewPointTemperature_num": scenario.get("dewpoint_f"),
            "HourlyRelativeHumidity_num": scenario.get("humidity_pct"),
            "HourlyStationPressure_num": scenario.get("station_pressure_inhg"),
            "HourlySeaLevelPressure_num": scenario.get("sea_level_pressure_inhg"),
            "HourlyVisibility_num": scenario.get("visibility_miles"),
            "HourlyWindDirection_num": scenario.get("wind_direction_deg"),
            "HourlyWindSpeed_num": scenario.get("wind_speed_mph"),
            "HourlyWindGustSpeed_num": scenario.get("wind_gust_mph"),
            "HourlyPrecipitation_num": scenario.get("precipitation_in"),
        }
        return f

    def historical_context(self, event_type: str = "normal", k: int = 5) -> list[dict[str, Any]]:
        if self.similarity_index is None or self.similarity_index.empty:
            return []
        df = self.similarity_index.copy()
        terms = self.EVENT_TERMS.get(str(event_type).lower(), [str(event_type).lower()])
        if terms and "event_types" in df.columns:
            text = df["event_types"].fillna("").astype(str).str.lower()
            mask = np.logical_or.reduce([text.str.contains(t, regex=False).to_numpy() for t in terms])
            matched = df.loc[mask].copy()
            if not matched.empty:
                df = matched
        if "start_dt" in df.columns:
            df["_date"] = pd.to_datetime(df["start_dt"], errors="coerce")
            df = df.sort_values("_date", ascending=False)
        cols = [c for c in [
            "event_id", "start_dt", "end_dt", "event_types",
            "event_cancel_rate", "event_severe_rate", "recovery_hours_after_event",
        ] if c in df.columns]
        return df.head(k)[cols].replace({np.nan: None}).to_dict(orient="records")

    def predict_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        prediction = self.predict_features(self.scenario_features(scenario))
        events = self.historical_context(str(scenario.get("event_type", "normal")), k=5)
        recovery = [e.get("recovery_hours_after_event") for e in events if e.get("recovery_hours_after_event") is not None]
        cancel = [e.get("event_cancel_rate") for e in events if e.get("event_cancel_rate") is not None]
        severe = [e.get("event_severe_rate") for e in events if e.get("event_severe_rate") is not None]
        prediction["historical_context"] = events
        prediction["recovery"] = {
            "analogs_used": len(events),
            "median_hours": float(np.median(recovery)) if recovery else None,
            "range_hours": [float(np.min(recovery)), float(np.max(recovery))] if recovery else None,
            "median_cancel_rate": float(np.median(cancel)) if cancel else None,
            "median_severe_rate": float(np.median(severe)) if severe else None,
            "method": "event-type historical context; recovery is an operational proxy",
        }
        return prediction

    def similar_events_from_vector(self, vector: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        if self.similarity_index is None:
            return []
        z_cols = [c for c in self.similarity_index.columns if c.startswith("z__")]
        if not z_cols:
            return []
        query = np.array([[float(vector.get(c, 0.0)) for c in z_cols]])
        matrix = self.similarity_index[z_cols].fillna(0.0).to_numpy(dtype=float)
        scores = cosine_similarity(query, matrix).ravel()
        top = np.argsort(scores)[::-1][: max(1, min(k, len(scores)))]
        output_cols = [c for c in [
            "event_id", "start_dt", "end_dt", "event_types",
            "event_cancel_rate", "event_severe_rate", "recovery_hours_after_event",
        ] if c in self.similarity_index.columns]
        result = self.similarity_index.iloc[top][output_cols].copy()
        result.insert(1 if output_cols else 0, "similarity_score", scores[top])
        return result.replace({np.nan: None}).to_dict(orient="records")
