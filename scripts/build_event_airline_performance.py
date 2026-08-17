"""Build event-by-airline OGG performance summaries from public BTS data.

The output is compact and deployable. It contains only aggregate counts/rates,
not the large raw flight table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_phase1_data import download_bts  # noqa: E402

EVENT_INDEX = ROOT / "data/processed/ogg_event_similarity_index_2020_2025.csv"
OUT = ROOT / "data/processed/ogg_event_airline_performance_2020_2025.csv"


def hhmm_datetime(date_series: pd.Series, hhmm_series: pd.Series) -> pd.Series:
    base = pd.to_datetime(date_series, errors="coerce")
    t = pd.to_numeric(hhmm_series, errors="coerce")
    hour_raw = np.floor(t / 100)
    minute = t % 100
    valid = t.notna() & minute.between(0, 59) & hour_raw.between(0, 24)
    extra_day = hour_raw.eq(24).astype(float)
    hour = hour_raw.where(~hour_raw.eq(24), 0)
    result = base + pd.to_timedelta(hour.fillna(0), unit="h") + pd.to_timedelta(minute.fillna(0), unit="m") + pd.to_timedelta(extra_day.fillna(0), unit="D")
    return result.where(valid)


def build() -> pd.DataFrame:
    if not EVENT_INDEX.exists():
        raise FileNotFoundError(f"Missing {EVENT_INDEX}")

    flights = download_bts(2020, 2025)
    events = pd.read_csv(EVENT_INDEX, usecols=["event_id", "start_dt", "end_dt", "event_types"])
    events["start_dt"] = pd.to_datetime(events["start_dt"], errors="coerce")
    events["end_dt"] = pd.to_datetime(events["end_dt"], errors="coerce")
    events = events.dropna(subset=["start_dt", "end_dt"]).copy()

    flights["direction"] = np.where(flights["Origin"].eq("OGG"), "departure", "arrival")
    dep_dt = hhmm_datetime(flights["FlightDate"], flights["CRSDepTime"])
    arr_dt = hhmm_datetime(flights["FlightDate"], flights["CRSArrTime"])
    flights["ogg_sched_dt"] = np.where(flights["direction"].eq("departure"), dep_dt, arr_dt)
    flights["ogg_sched_dt"] = pd.to_datetime(flights["ogg_sched_dt"], errors="coerce")
    flights = flights.dropna(subset=["ogg_sched_dt", "Reporting_Airline"]).copy()

    flights["is_cancelled"] = pd.to_numeric(flights.get("Cancelled", 0), errors="coerce").fillna(0).gt(0).astype(int)
    dep_delay = pd.to_numeric(flights.get("DepDelayMinutes"), errors="coerce")
    arr_delay = pd.to_numeric(flights.get("ArrDelayMinutes"), errors="coerce")
    flights["delay_minutes"] = np.where(flights["direction"].eq("departure"), dep_delay, arr_delay)
    flights["is_delayed15"] = pd.Series(flights["delay_minutes"], index=flights.index).ge(15).astype(int)
    flights["is_severe"] = (flights["is_cancelled"].eq(1) | pd.Series(flights["delay_minutes"], index=flights.index).ge(120)).astype(int)

    chunks = []
    for ep in events.itertuples(index=False):
        mask = flights["ogg_sched_dt"].between(ep.start_dt, ep.end_dt, inclusive="both")
        sub = flights.loc[mask].copy()
        if sub.empty:
            continue
        agg = sub.groupby(["Reporting_Airline", "direction"], as_index=False).agg(
            total_flights=("Reporting_Airline", "size"),
            delayed_flights=("is_delayed15", "sum"),
            cancelled_flights=("is_cancelled", "sum"),
            severe_flights=("is_severe", "sum"),
            mean_delay_minutes=("delay_minutes", "mean"),
            median_delay_minutes=("delay_minutes", "median"),
        )
        agg.insert(0, "event_id", ep.event_id)
        agg.insert(1, "event_types", ep.event_types)
        agg = agg.rename(columns={"Reporting_Airline": "airline"})
        agg["delay_rate"] = agg["delayed_flights"] / agg["total_flights"]
        agg["cancellation_rate"] = agg["cancelled_flights"] / agg["total_flights"]
        agg["severe_rate"] = agg["severe_flights"] / agg["total_flights"]
        chunks.append(agg)

    if not chunks:
        raise RuntimeError("No event-airline rows were generated")
    out = pd.concat(chunks, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out):,} event-airline rows -> {OUT}")
    print(out.groupby("airline")["total_flights"].sum().sort_values(ascending=False).head(20))
    return out


if __name__ == "__main__":
    build()
