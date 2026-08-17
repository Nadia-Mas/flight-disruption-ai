from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np

NWS_USER_AGENT = "FlightRescueAI/0.4 (research prototype; github.com/Nadia-Mas/flight-disruption-ai)"

AIRPORT_WEATHER = {
    "OGG": {"name": "Kahului Airport", "city": "Kahului / Maui, HI", "lat": 20.8987, "lon": -156.4305, "tz": "Pacific/Honolulu"},
    "LAX": {"name": "Los Angeles International Airport", "city": "Los Angeles, CA", "lat": 33.9416, "lon": -118.4085, "tz": "America/Los_Angeles"},
    "DFW": {"name": "Dallas Fort Worth International Airport", "city": "Dallas / Fort Worth, TX", "lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago"},
}


def _get_json(url: str, timeout: int = 8) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _duration(value: str) -> timedelta:
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", value or "")
    if not m:
        return timedelta(hours=1)
    d, h, minute = (int(x or 0) for x in m.groups())
    return timedelta(days=d, hours=h, minutes=minute)


def _grid_value(prop: dict[str, Any] | None, target_utc: datetime) -> float | None:
    if not prop:
        return None
    nearest = None
    for item in prop.get("values") or []:
        valid = str(item.get("validTime") or "")
        if "/" not in valid or item.get("value") is None:
            continue
        start_text, duration_text = valid.split("/", 1)
        try:
            start = datetime.fromisoformat(start_text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
        end = start + _duration(duration_text)
        value = float(item["value"])
        if start <= target_utc < end:
            return value
        delta = abs((start - target_utc).total_seconds())
        if nearest is None or delta < nearest[0]:
            nearest = (delta, value)
    return nearest[1] if nearest and nearest[0] <= 6 * 3600 else None


def _mph(text: str | None) -> float | None:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text or ""))]
    return float(np.mean(nums)) if nums else None


def _degrees(compass: str | None) -> float | None:
    values = {"N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,"SE":135,"SSE":157.5,"S":180,"SSW":202.5,"SW":225,"WSW":247.5,"W":270,"WNW":292.5,"NW":315,"NNW":337.5}
    return values.get(str(compass or "").upper())


def detect_event_type(text: str | None, wind_speed: float | None, wind_gust: float | None, precip_prob: float | None) -> str:
    s = str(text or "").lower()
    if any(k in s for k in ("hurricane", "tropical storm", "tropical depression")):
        return "tropical"
    if any(k in s for k in ("thunderstorm", "lightning")):
        return "thunderstorm"
    if any(k in s for k in ("flash flood", "flood")):
        return "flood"
    if any(k in s for k in ("heavy rain", "rain", "showers")) or (precip_prob or 0) >= 60:
        return "rain"
    if any(k in s for k in ("windy", "breezy", "gusty", "high wind")) or (wind_gust or 0) >= 35 or (wind_speed or 0) >= 25:
        return "wind"
    return "normal"


class NWSAirportWeather:
    def __init__(self):
        self._points: dict[str, dict[str, str | None]] = {}
        self._cache: dict[str, dict[str, Any]] = {}

    def forecast(self, airport: str, scheduled_local: str) -> dict[str, Any]:
        code = str(airport).upper()
        if code not in AIRPORT_WEATHER:
            raise ValueError(f"Unsupported weather airport: {code}")
        cfg = AIRPORT_WEATHER[code]
        tz = ZoneInfo(cfg["tz"])
        dt = datetime.fromisoformat(str(scheduled_local).replace("Z", "+00:00"))
        target = dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
        key = f"{code}:{target.strftime('%Y-%m-%dT%H:00%z')}"
        if key in self._cache:
            return dict(self._cache[key])
        result = {
            "available": False,
            "airport": code,
            "location": f"{cfg['name']} ({code}), {cfg['city']}",
            "timezone": cfg["tz"],
            "requested_local": target.isoformat(),
            "source": "National Weather Service (api.weather.gov)",
        }
        try:
            if code not in self._points:
                point = _get_json(f"https://api.weather.gov/points/{cfg['lat']:.4f},{cfg['lon']:.4f}")
                p = point.get("properties") or {}
                self._points[code] = {"hourly": p.get("forecastHourly"), "grid": p.get("forecastGridData")}
            urls = self._points[code]
            if not urls.get("hourly"):
                raise RuntimeError("NWS point lookup did not return an hourly forecast URL")
            hourly = _get_json(str(urls["hourly"]))
            periods = (hourly.get("properties") or {}).get("periods") or []
            target_utc = target.astimezone(timezone.utc)
            candidates = []
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
                result["reason"] = "No NWS hourly forecast period found"
                return result
            delta, period = min(candidates, key=lambda x: x[0])
            if delta > 6 * 3600:
                result["reason"] = "Requested time is outside the current NWS forecast window"
                return result
            temp_f = float(period["temperature"]) if period.get("temperature") is not None else None
            humidity = (period.get("relativeHumidity") or {}).get("value")
            dew_c = (period.get("dewpoint") or {}).get("value")
            dew_f = float(dew_c) * 9 / 5 + 32 if dew_c is not None else None
            wind_speed = _mph(period.get("windSpeed"))
            wind_gust = _mph(period.get("windGust"))
            wind_direction = _degrees(period.get("windDirection"))
            precip_prob = (period.get("probabilityOfPrecipitation") or {}).get("value")
            visibility_miles = precipitation_in = sea_level_pressure = station_pressure = None
            if urls.get("grid"):
                try:
                    grid = _get_json(str(urls["grid"]))
                    gp = grid.get("properties") or {}
                    gv = lambda n: _grid_value(gp.get(n), target_utc)
                    t = gv("temperature"); d = gv("dewpoint"); h = gv("relativeHumidity")
                    ws = gv("windSpeed"); wg = gv("windGust"); wd = gv("windDirection")
                    vis = gv("visibility"); qpf = gv("quantitativePrecipitation")
                    if t is not None: temp_f = t * 9 / 5 + 32
                    if d is not None: dew_f = d * 9 / 5 + 32
                    if h is not None: humidity = h
                    if ws is not None: wind_speed = ws * 0.621371
                    if wg is not None: wind_gust = wg * 0.621371
                    if wd is not None: wind_direction = wd
                    if vis is not None: visibility_miles = vis / 1609.344
                    if qpf is not None: precipitation_in = qpf / 25.4
                except Exception:
                    pass
            short = str(period.get("shortForecast") or "")
            result.update({
                "available": True,
                "forecast_time": period.get("startTime"),
                "short_forecast": short,
                "temperature_f": temp_f,
                "dewpoint_f": dew_f,
                "humidity_pct": float(humidity) if humidity is not None else None,
                "station_pressure_inhg": station_pressure,
                "sea_level_pressure_inhg": sea_level_pressure,
                "visibility_miles": visibility_miles,
                "wind_direction_deg": wind_direction,
                "wind_speed_mph": wind_speed,
                "wind_gust_mph": wind_gust,
                "precipitation_in": precipitation_in,
                "precipitation_probability_pct": float(precip_prob) if precip_prob is not None else None,
                "detected_event_type": detect_event_type(short, wind_speed, wind_gust, precip_prob),
            })
        except Exception as exc:
            result["reason"] = str(exc)
        self._cache[key] = dict(result)
        return result
