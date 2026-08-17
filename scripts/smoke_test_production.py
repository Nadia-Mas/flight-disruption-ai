from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import requests

API = "https://flight-disruption-ai.vercel.app"
HST = timezone(timedelta(hours=-10))


def get_json(path: str):
    r = requests.get(f"{API}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def post_json(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:1000]}")
    return r.json()


def main() -> int:
    target = (datetime.now(HST) + timedelta(days=1)).replace(minute=45, second=0, microsecond=0)
    if target.hour < 8:
        target = target.replace(hour=10)
    scheduled = target.strftime("%Y-%m-%dT%H:%M")

    health = get_json("/health")
    if not health.get("ready"):
        raise AssertionError(f"API not ready: {health}")

    weather = get_json(f"/weather/ogg?scheduled_local={scheduled}")
    if not weather.get("available"):
        raise AssertionError(f"NWS forecast unavailable for tomorrow smoke test: {weather}")

    base = {
        "airline": "AA",
        "direction": "departure",
        "other_airport": "HNL",
        "scheduled_local": scheduled,
        "distance_miles": 100,
    }

    scenarios = ["auto", "rain", "wind", "thunderstorm", "tropical"]
    results = {}
    for event_type in scenarios:
        payload = {**base, "event_type": event_type}
        body = post_json("/predict/scenario", payload)
        p_any = float(body["disruption_probability"])
        p_severe = float(body["severe_disruption_probability"])
        if not (0 <= p_any <= 1 and 0 <= p_severe <= 1):
            raise AssertionError(f"Probability out of bounds for {event_type}: {p_any}, {p_severe}")
        if p_severe > p_any + 1e-12:
            raise AssertionError(f"Severe exceeds any disruption for {event_type}: {p_severe} > {p_any}")
        if event_type != "auto" and not body.get("weather", {}).get("preset_applied"):
            raise AssertionError(f"Stress preset was not applied for {event_type}")
        results[event_type] = {
            "any": p_any,
            "severe": p_severe,
            "effective_event": body.get("effective_event_type"),
            "airline_comparison_available": body.get("airline_comparison", {}).get("available", False),
        }

    auto_risk = results["auto"]["any"]
    stress_moved = {k: abs(v["any"] - auto_risk) >= 0.005 for k, v in results.items() if k != "auto"}
    if not any(stress_moved.values()):
        raise AssertionError("None of the weather stress scenarios changed disruption risk by at least 0.5 percentage points")

    output = {
        "scheduled_hst": scheduled,
        "health": health,
        "nws": weather,
        "scenarios": results,
        "stress_scenarios_moved_risk": stress_moved,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise
