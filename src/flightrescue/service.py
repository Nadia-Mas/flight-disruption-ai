from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


HST = timezone(timedelta(hours=-10))
OGG_LAT = 20.8987
OGG_LON = -156.4305
NWS_USER_AGENT = "FlightRescueAI/0.3 (research prototype; github.com/Nadia-Mas/flight-disruption-ai)"


@dataclass
class ArtifactPaths:
    any_model: Path
    severe_model: Path
    metadata: Path
    similarity_index: Path
    airline_performance: Path


class FlightRescueService:
    """Artifact-backed OGG disruption inference with official NWS weather context."""

    AIRLINE_CODES = {
        "american": "AA", "american airlines": "AA", "aa": "AA",
        "alaska": "AS", "alaska airlines": "AS", "as": "AS",
        "jetblue": "B6", "b6": "B6",
        "delta": "DL", "delta air lines": "DL", "dl": "DL",
        "frontier": "F9", "frontier airlines": "F9", "f9": "F9",
        "hawaiian": "HA", "hawaiian airlines": "HA", "ha": "HA",
        "spirit": "NK", "spirit airlines": "NK", "nk": "NK",
        "united": "UA", "united airlines": "UA", "ua": "UA",
        "southwest": "WN", "southwest airlines": "WN", "wn": "WN",
    }

    AIRLINE_NAMES = {
        "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue",
        "DL": "Delta Air Lines", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines",
        "NK": "Spirit Airlines", "UA": "United Airlines", "WN": "Southwest Airlines",
    }

    EVENT_TERMS = {
        "normal": [],
        "rain": ["heavy rain", "flood", "flash flood", "rain"],
        "wind": ["high wind", "strong wind", "thunderstorm wind", "wind"],
        "flood": ["flash flood", "flood"],
        "tropical": ["tropical storm", "hurricane", "tropical depression"],
        "thunderstorm": ["thunderstorm", "lightning"],
    }

    SCENARIO_PRESETS = {
        "rain": {"humidity_pct": 90.0, "visibility_miles": 4.0, "wind_speed_mph": 15.0, "wind_gust_mph": 25.0, "precipitation_in": 0.40},
        "wind": {"visibility_miles": 8.0, "wind_speed_mph": 30.0, "wind_gust_mph": 45.0},
        "flood": {"humidity_pct": 95.0, "visibility_miles": 2.0, "wind_speed_mph": 20.0, "wind_gust_mph": 35.0, "precipitation_in": 1.00},
        "tropical": {"humidity_pct": 95.0, "visibility_miles": 2.0, "wind_speed_mph": 40.0, "wind_gust_mph": 60.0, "precipitation_in": 1.20},
        "thunderstorm": {"humidity_pct": 90.0, "visibility_miles": 3.0, "wind_speed_mph": 25.0, "wind_gust_mph": 45.0, "precipitation_in": 0.60},
    }

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.paths = ArtifactPaths(
            any_model=self.root / "models/any_disruption_model.joblib",
            severe_model=self.root / "models/severe_disruption_model.joblib",
            metadata=self.root / "models/flightrescue_inference_metadata.json",
            similarity_index=self.root / "data/processed/ogg_event_similarity_index_2020_2025.csv",
            airline_performance=self.root / "data/processed/ogg_event_airline_performance_2020_2025.csv",
        )
        self.any_model = None
        self.severe_model = None
        self.metadata: dict[str, Any] = {}
        self.similarity_index: pd.DataFrame | None = None
        self.airline_performance: pd.DataFrame | None = None
        self._nws_points: dict[str, str] | None = None
        self._weather_cache: dict[str, dict[str, Any]] = {}
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
        if self.paths.airline_performance.exists():
            self.airline_performance = pd.read_csv(self.paths.airline_performance, low_memory=False)

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
            "airline_event_performance": self.paths.airline_performance.exists(),
            "nws_weather": True,
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

    @staticmethod
    def _nws_get_json(url: str, timeout: int = 8) -> dict[str, Any]:
        req = Request(url, headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"})
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _scheduled_hst(value: str) -> datetime:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=HST)
        return dt.astimezone(HST)

    @staticmethod
    def _parse_iso_duration(value: str) -> timedelta:
        m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", value or "")
        if not m:
            return timedelta(hours=1)
        days, hours, minutes = (int(x or 0) for x in m.groups())
        return timedelta(days=days, hours=hours, minutes=minutes)

    @classmethod
    def _grid_value_at(cls, prop: dict[str, Any] | None, target_utc: datetime) -> float | None:
        if not prop:
            return None
        values = prop.get("values") or []
        nearest: tuple[float, float] | None = None
        for item in values:
            valid = str(item.get("validTime") or "")
            if "/" not in valid or item.get("value") is None:
                continue
            start_text, duration_text = valid.split("/", 1)
            try:
                start = datetime.fromisoformat(start_text.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                continue
            end = start + cls._parse_iso_duration(duration_text)
            val = float(item["value"])
            if start <= target_utc < end:
                return val
            delta = abs((start - target_utc).total_seconds())
            if nearest is None or delta < nearest[0]:
                nearest = (delta, val)
        return nearest[1] if nearest and nearest[0] <= 6 * 3600 else None

    @staticmethod
    def _compass_to_degrees(value: str | None) -> float | None:
        if not value:
            return None
        table = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
            "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
            "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        }
        return table.get(str(value).upper())

    @staticmethod
    def _mph_from_text(value: str | None) -> float | None:
        if not value:
            return None
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
        return float(np.mean(nums)) if nums else None

    @classmethod
    def _detect_event_type(cls, text: str | None, wind_speed: float | None = None, wind_gust: float | None = None, precip_prob: float | None = None) -> str:
        s = (text or "").lower()
        if any(k in s for k in ["hurricane", "tropical storm", "tropical depression"]):
            return "tropical"
        if any(k in s for k in ["thunderstorm", "thunderstorms", "lightning"]):
            return "thunderstorm"
        if any(k in s for k in ["flash flood", "flood"]):
            return "flood"
        if any(k in s for k in ["heavy rain", "rain", "showers"]):
            return "rain"
        if any(k in s for k in ["windy", "breezy", "gusty", "high wind"]) or (wind_gust or 0) >= 35 or (wind_speed or 0) >= 25:
            return "wind"
        if (precip_prob or 0) >= 60:
            return "rain"
        return "normal"

    def nws_weather(self, scheduled_local: str) -> dict[str, Any]:
        target_hst = self._scheduled_hst(scheduled_local)
        cache_key = target_hst.strftime("%Y-%m-%dT%H:00")
        if cache_key in self._weather_cache:
            return dict(self._weather_cache[cache_key])
        result: dict[str, Any] = {
            "available": False,
            "source": "National Weather Service (api.weather.gov)",
            "location": "Kahului Airport (OGG), Maui, HI",
            "requested_hst": target_hst.isoformat(),
        }
        try:
            if self._nws_points is None:
                points = self._nws_get_json(f"https://api.weather.gov/points/{OGG_LAT:.4f},{OGG_LON:.4f}")
                props = points.get("properties") or {}
                self._nws_points = {"forecastHourly": props.get("forecastHourly"), "forecastGridData": props.get("forecastGridData")}
            hourly_url = (self._nws_points or {}).get("forecastHourly")
            grid_url = (self._nws_points or {}).get("forecastGridData")
            if not hourly_url:
                raise RuntimeError("NWS point metadata did not provide an hourly forecast URL")
            hourly = self._nws_get_json(hourly_url)
            periods = ((hourly.get("properties") or {}).get("periods") or [])
            target_utc = target_hst.astimezone(timezone.utc)
            candidates: list[tuple[float, dict[str, Any]]] = []
            for period in periods:
                try:
                    start = datetime.fromisoformat(str(period.get("startTime")).replace("Z", "+00:00")).astimezone(timezone.utc)
                    end = datetime.fromisoformat(str(period.get("endTime")).replace("Z", "+00:00")).astimezone(timezone.utc)
                except (TypeError, ValueError):
                    continue
                if start <= target_utc < end:
                    candidates = [(0.0, period)]
                    break
                candidates.append((abs((start - target_utc).total_seconds()), period))
            if not candidates:
                result["reason"] = "NWS hourly forecast contains no periods"
                self._weather_cache[cache_key] = result
                return dict(result)
            delta, period = min(candidates, key=lambda x: x[0])
            if delta > 6 * 3600:
                result["reason"] = "Requested time is outside the current NWS hourly forecast window"
                self._weather_cache[cache_key] = result
                return dict(result)
            temp_f = float(period["temperature"]) if period.get("temperature") is not None else None
            humidity = ((period.get("relativeHumidity") or {}).get("value"))
            dew_c = ((period.get("dewpoint") or {}).get("value"))
            dew_f = (float(dew_c) * 9 / 5 + 32) if dew_c is not None else None
            wind_speed = self._mph_from_text(period.get("windSpeed"))
            wind_dir = self._compass_to_degrees(period.get("windDirection"))
            precip_prob = ((period.get("probabilityOfPrecipitation") or {}).get("value"))
            wind_gust = self._mph_from_text(period.get("windGust"))
            visibility_miles = None
            precip_in = None
            if grid_url:
                try:
                    grid = self._nws_get_json(grid_url)
                    gp = (grid.get("properties") or {})
                    def gv(name: str) -> float | None:
                        return self._grid_value_at(gp.get(name), target_utc)
                    grid_temp_c = gv("temperature")
                    grid_dew_c = gv("dewpoint")
                    grid_humidity = gv("relativeHumidity")
                    grid_wind_kmh = gv("windSpeed")
                    grid_gust_kmh = gv("windGust")
                    grid_wind_dir = gv("windDirection")
                    grid_vis_m = gv("visibility")
                    grid_qpf_mm = gv("quantitativePrecipitation")
                    if grid_temp_c is not None: temp_f = grid_temp_c * 9 / 5 + 32
                    if grid_dew_c is not None: dew_f = grid_dew_c * 9 / 5 + 32
                    if grid_humidity is not None: humidity = grid_humidity
                    if grid_wind_kmh is not None: wind_speed = grid_wind_kmh * 0.621371
                    if grid_gust_kmh is not None: wind_gust = grid_gust_kmh * 0.621371
                    if grid_wind_dir is not None: wind_dir = grid_wind_dir
                    if grid_vis_m is not None: visibility_miles = grid_vis_m / 1609.344
                    if grid_qpf_mm is not None: precip_in = grid_qpf_mm / 25.4
                except Exception:
                    pass
            short_forecast = str(period.get("shortForecast") or "")
            detected = self._detect_event_type(short_forecast, wind_speed, wind_gust, precip_prob)
            result.update({
                "available": True, "forecast_time": period.get("startTime"), "short_forecast": short_forecast,
                "temperature_f": temp_f, "dewpoint_f": dew_f,
                "humidity_pct": float(humidity) if humidity is not None else None,
                "visibility_miles": visibility_miles, "wind_direction_deg": wind_dir,
                "wind_speed_mph": wind_speed, "wind_gust_mph": wind_gust,
                "precipitation_in": precip_in,
                "precipitation_probability_pct": float(precip_prob) if precip_prob is not None else None,
                "detected_event_type": detected,
            })
        except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError) as exc:
            result["reason"] = f"NWS forecast unavailable: {exc}"
        except Exception as exc:
            result["reason"] = f"NWS forecast unavailable: {type(exc).__name__}"
        self._weather_cache[cache_key] = result
        return dict(result)

    def _complete_model_frame(self, supplied: dict[str, Any]) -> pd.DataFrame:
        cols = self.metadata.get("feature_columns", [])
        if not cols:
            raise RuntimeError("Inference metadata does not contain feature_columns.")
        row = {c: np.nan for c in cols}
        row.update({k: v for k, v in supplied.items() if k in row})
        return pd.DataFrame([row], columns=cols)

    def predict_features(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Model artifacts are not exported yet. Run the inference-artifact build workflow.")
        frame = self._complete_model_frame(features)
        p_any = float(self.any_model.predict_proba(frame)[:, 1][0])
        raw_severe = float(self.severe_model.predict_proba(frame)[:, 1][0])
        p_severe = min(raw_severe, p_any)
        any_threshold = float(self.metadata.get("any_disruption_threshold", 0.5))
        severe_threshold = float(self.metadata.get("severe_disruption_threshold", 0.5))
        return {
            "disruption_probability": p_any, "severe_disruption_probability": p_severe,
            "raw_severe_model_probability": raw_severe,
            "disruption_flag": bool(p_any >= any_threshold), "severe_disruption_flag": bool(p_severe >= severe_threshold),
            "risk_level": self._risk_label(p_any), "severe_risk_level": self._risk_label(p_severe),
            "thresholds": {"any_disruption": any_threshold, "severe_disruption": severe_threshold},
            "model_status": "artifact_backed", "confidence": "Experimental",
        }

    @staticmethod
    def _apply_stress_preset(base: dict[str, Any], event_type: str) -> tuple[dict[str, Any], bool]:
        preset = FlightRescueService.SCENARIO_PRESETS.get(event_type)
        if not preset:
            return base, False
        out = dict(base)
        for key, value in preset.items():
            current = out.get(key)
            if key == "visibility_miles":
                out[key] = value if current is None else min(float(current), value)
            elif key in {"humidity_pct", "wind_speed_mph", "wind_gust_mph", "precipitation_in"}:
                out[key] = value if current is None else max(float(current), value)
            else:
                out[key] = value if current is None else current
        return out, True

    def resolve_weather(self, scenario: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        nws = self.nws_weather(str(scenario["scheduled_local"]))
        weather = {
            "temperature_f": nws.get("temperature_f") if nws.get("available") else None,
            "dewpoint_f": nws.get("dewpoint_f") if nws.get("available") else None,
            "humidity_pct": nws.get("humidity_pct") if nws.get("available") else None,
            "station_pressure_inhg": None, "sea_level_pressure_inhg": None,
            "visibility_miles": nws.get("visibility_miles") if nws.get("available") else None,
            "wind_direction_deg": nws.get("wind_direction_deg") if nws.get("available") else None,
            "wind_speed_mph": nws.get("wind_speed_mph") if nws.get("available") else None,
            "wind_gust_mph": nws.get("wind_gust_mph") if nws.get("available") else None,
            "precipitation_in": nws.get("precipitation_in") if nws.get("available") else None,
        }
        for key in list(weather):
            if scenario.get(key) is not None:
                weather[key] = scenario.get(key)
        requested_event = str(scenario.get("event_type", "auto") or "auto").lower()
        effective_event = str(nws.get("detected_event_type", "normal")) if requested_event == "auto" else requested_event
        preset_applied = False
        if requested_event not in {"auto", "normal"}:
            weather, preset_applied = self._apply_stress_preset(weather, requested_event)
        context = {
            "nws": nws,
            "mode": "official_nws" if requested_event == "auto" else ("official_nws_plus_stress_test" if preset_applied else "official_nws"),
            "preset_applied": preset_applied, "requested_event_type": requested_event,
            "effective_event_type": effective_event, "model_weather_inputs": weather,
        }
        return weather, context, effective_event

    def scenario_features(self, scenario: dict[str, Any], weather: dict[str, Any] | None = None) -> dict[str, Any]:
        dt = pd.Timestamp(scenario["scheduled_local"])
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
        hour_decimal = dt.hour + dt.minute / 60.0
        hhmm = dt.hour * 100 + dt.minute
        airline = str(scenario.get("airline", "")).strip().lower()
        airline_code = self.AIRLINE_CODES.get(airline, str(scenario.get("airline", "")).upper())
        w = weather or scenario
        return {
            "Reporting_Airline": airline_code, "direction": str(scenario.get("direction", "departure")).lower(),
            "other_airport": str(scenario.get("other_airport", "UNKNOWN")).upper(), "Distance": scenario.get("distance_miles"),
            "CRSDepTime": hhmm if str(scenario.get("direction", "departure")).lower() == "departure" else np.nan,
            "CRSArrTime": hhmm if str(scenario.get("direction", "departure")).lower() == "arrival" else np.nan,
            "sched_hour": dt.hour, "sched_dow": dt.dayofweek, "sched_month": dt.month, "sched_dayofyear": dt.dayofyear,
            "is_weekend": int(dt.dayofweek >= 5), "hour_sin": float(np.sin(2 * np.pi * hour_decimal / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour_decimal / 24)), "month_sin": float(np.sin(2 * np.pi * dt.month / 12)),
            "month_cos": float(np.cos(2 * np.pi * dt.month / 12)), "covid_era": int(dt.year in (2020, 2021)),
            "weather_age_min": 0.0 if (weather or {}).get("temperature_f") is not None else np.nan,
            "HourlyDryBulbTemperature_num": w.get("temperature_f"), "HourlyDewPointTemperature_num": w.get("dewpoint_f"),
            "HourlyRelativeHumidity_num": w.get("humidity_pct"), "HourlyStationPressure_num": w.get("station_pressure_inhg"),
            "HourlySeaLevelPressure_num": w.get("sea_level_pressure_inhg"), "HourlyVisibility_num": w.get("visibility_miles"),
            "HourlyWindDirection_num": w.get("wind_direction_deg"), "HourlyWindSpeed_num": w.get("wind_speed_mph"),
            "HourlyWindGustSpeed_num": w.get("wind_gust_mph"), "HourlyPrecipitation_num": w.get("precipitation_in"),
        }

    def historical_context(self, event_type: str = "normal", k: int = 5) -> list[dict[str, Any]]:
        if self.similarity_index is None or self.similarity_index.empty:
            return []
        event_type = str(event_type).lower()
        if event_type in {"auto", "normal", "none", ""}:
            return []
        df = self.similarity_index.copy()
        terms = self.EVENT_TERMS.get(event_type, [event_type])
        if terms and "event_types" in df.columns:
            text = df["event_types"].fillna("").astype(str).str.lower()
            mask = np.logical_or.reduce([text.str.contains(t, regex=False).to_numpy() for t in terms])
            matched = df.loc[mask].copy()
            if matched.empty:
                return []
            df = matched
        if "start_dt" in df.columns:
            df["_date"] = pd.to_datetime(df["start_dt"], errors="coerce")
            df = df.sort_values("_date", ascending=False)
        cols = [c for c in ["event_id", "start_dt", "end_dt", "event_types", "event_cancel_rate", "event_severe_rate", "recovery_hours_after_event"] if c in df.columns]
        return df.head(max(1, min(k, 20)))[cols].replace({np.nan: None}).to_dict(orient="records")

    def airline_comparison(self, event_ids: list[str], selected_airline: str, direction: str) -> dict[str, Any]:
        if self.airline_performance is None or self.airline_performance.empty or not event_ids:
            return {"available": False, "rows": [], "selected_airline": selected_airline.upper(), "analogs_used": 0}
        df = self.airline_performance.copy()
        df = df[df["event_id"].astype(str).isin([str(x) for x in event_ids])]
        if "direction" in df.columns and direction:
            directed = df[df["direction"].astype(str).str.lower().eq(str(direction).lower())]
            if not directed.empty:
                df = directed
        if df.empty:
            return {"available": False, "rows": [], "selected_airline": selected_airline.upper(), "analogs_used": 0}
        count_cols = ["total_flights", "delayed_flights", "cancelled_flights", "severe_flights"]
        for c in count_cols:
            df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)
        grouped = df.groupby("airline", as_index=False)[count_cols].sum()
        grouped = grouped[grouped["total_flights"] > 0].copy()
        grouped["delay_rate"] = grouped["delayed_flights"] / grouped["total_flights"]
        grouped["cancellation_rate"] = grouped["cancelled_flights"] / grouped["total_flights"]
        grouped["severe_rate"] = grouped["severe_flights"] / grouped["total_flights"]
        tmp = df.copy()
        tmp["mean_delay_minutes"] = pd.to_numeric(tmp.get("mean_delay_minutes"), errors="coerce")
        tmp["weighted_delay"] = tmp["mean_delay_minutes"] * tmp["total_flights"]
        delay_agg = tmp.groupby("airline", as_index=False).agg(weighted_delay=("weighted_delay", "sum"), delay_weight=("total_flights", "sum"))
        delay_agg["mean_delay_minutes"] = delay_agg["weighted_delay"] / delay_agg["delay_weight"].replace(0, np.nan)
        grouped = grouped.merge(delay_agg[["airline", "mean_delay_minutes"]], on="airline", how="left")
        grouped["airline_name"] = grouped["airline"].map(self.AIRLINE_NAMES).fillna(grouped["airline"])
        selected = self.AIRLINE_CODES.get(str(selected_airline).lower(), str(selected_airline).upper())
        grouped["_selected"] = grouped["airline"].eq(selected)
        grouped = grouped.sort_values(["_selected", "cancellation_rate", "delay_rate"], ascending=[False, True, True])
        rows = grouped[["airline", "airline_name", "total_flights", "delay_rate", "cancellation_rate", "severe_rate", "mean_delay_minutes"]].replace({np.nan: None}).to_dict(orient="records")
        return {"available": True, "rows": rows, "selected_airline": selected, "analogs_used": int(df["event_id"].nunique()), "method": "BTS OGG flights during matched NOAA event windows; weighted across historical analogs"}

    def predict_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        weather, weather_context, effective_event = self.resolve_weather(scenario)
        prediction = self.predict_features(self.scenario_features(scenario, weather=weather))
        events = self.historical_context(effective_event, k=5)
        recovery = [e.get("recovery_hours_after_event") for e in events if e.get("recovery_hours_after_event") is not None]
        cancel = [e.get("event_cancel_rate") for e in events if e.get("event_cancel_rate") is not None]
        severe = [e.get("event_severe_rate") for e in events if e.get("event_severe_rate") is not None]
        event_ids = [str(e.get("event_id")) for e in events if e.get("event_id")]
        comparison = self.airline_comparison(event_ids, str(scenario.get("airline", "")), str(scenario.get("direction", "")))
        prediction.update({
            "weather": weather_context, "effective_event_type": effective_event, "historical_context": events,
            "airline_comparison": comparison,
            "recovery": {"analogs_used": len(events), "median_hours": float(np.median(recovery)) if recovery else None,
                         "range_hours": [float(np.min(recovery)), float(np.max(recovery))] if recovery else None,
                         "median_cancel_rate": float(np.median(cancel)) if cancel else None,
                         "median_severe_rate": float(np.median(severe)) if severe else None,
                         "method": "matched NOAA event-type context; recovery is an operational BTS proxy"},
            "prediction_basis": "trained OGG flight model using official NWS weather when available, plus separate historical airline analog evidence",
        })
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
        output_cols = [c for c in ["event_id", "start_dt", "end_dt", "event_types", "event_cancel_rate", "event_severe_rate", "recovery_hours_after_event"] if c in self.similarity_index.columns]
        result = self.similarity_index.iloc[top][output_cols].copy()
        result.insert(1 if output_cols else 0, "similarity_score", scores[top])
        return result.replace({np.nan: None}).to_dict(orient="records")
