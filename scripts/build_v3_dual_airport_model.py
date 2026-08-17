"""Build FlightRescue v3 for OGG<->LAX and OGG<->DFW using weather at both airports.

Training policy
---------------
2020-2023: base model training
2024: probability calibration + threshold selection
2025: untouched temporal test

Weather source: NOAA/NCEI Local Climatological Data (LCD) at each airport.
Production forecasts are supplied separately from the National Weather Service.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "v3"

BTS_BASE = "https://transtats.bts.gov/PREZIP/"
BTS_NAME = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
LCD_BASE = "https://www.ncei.noaa.gov/data/local-climatological-data/access/"

AIRPORTS = {
    "OGG": {"lcd": "91190022516.csv", "tz": "Pacific/Honolulu"},
    "LAX": {"lcd": "72295023174.csv", "tz": "America/Los_Angeles"},
    "DFW": {"lcd": "72259003927.csv", "tz": "America/Chicago"},
}
TARGET_PAIRS = {frozenset(("OGG", "LAX")), frozenset(("OGG", "DFW"))}
WEATHER_COLUMNS = [
    "HourlyDryBulbTemperature", "HourlyDewPointTemperature", "HourlyRelativeHumidity",
    "HourlyStationPressure", "HourlySeaLevelPressure", "HourlyVisibility",
    "HourlyWindDirection", "HourlyWindSpeed", "HourlyWindGustSpeed", "HourlyPrecipitation",
]
BTS_KEEP = [
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate", "Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime", "DepDelayMinutes", "ArrDelayMinutes",
    "Cancelled", "Diverted", "CRSElapsedTime", "Distance",
]


def get(url: str, timeout: int = 120) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "FlightRescueAI-v3/0.1"})
    r.raise_for_status()
    return r


def numeric_weather(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "T": "0", "s": np.nan})
    s = s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")


def hhmm_minutes(v) -> float:
    try:
        x = int(float(v))
        return (x // 100) * 60 + (x % 100)
    except Exception:
        return np.nan


def load_bts() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            name = BTS_NAME.format(year=year, month=month)
            print("BTS", year, month)
            try:
                r = get(urljoin(BTS_BASE, name))
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue
                raise
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(csv_name) as fh:
                    df = pd.read_csv(fh, low_memory=False)
            cols = [c for c in BTS_KEEP if c in df.columns]
            df = df[cols].copy()
            keep = df.apply(lambda r: frozenset((str(r.Origin), str(r.Dest))) in TARGET_PAIRS, axis=1)
            frames.append(df.loc[keep])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RAW / "ogg_lax_dfw_flights_2020_2025.csv.gz", index=False, compression="gzip")
    return out


def load_lcd(code: str) -> pd.DataFrame:
    frames = []
    filename = AIRPORTS[code]["lcd"]
    for year in range(2020, 2026):
        url = f"{LCD_BASE}{year}/{filename}"
        print("LCD", code, year)
        r = get(url)
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        keep = ["DATE"] + [c for c in WEATHER_COLUMNS if c in df.columns]
        frames.append(df[keep].copy())
    w = pd.concat(frames, ignore_index=True)
    w["weather_time"] = pd.to_datetime(w["DATE"], errors="coerce").dt.tz_localize(None)
    for c in WEATHER_COLUMNS:
        if c not in w:
            w[c] = np.nan
        w[c] = numeric_weather(w[c])
    w = w.dropna(subset=["weather_time"]).sort_values("weather_time")
    w = w.groupby("weather_time", as_index=False)[WEATHER_COLUMNS].mean(numeric_only=True)
    return w


def scheduled_times(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["FlightDate"] = pd.to_datetime(f["FlightDate"], errors="coerce")
    dep_min = f["CRSDepTime"].map(hhmm_minutes)
    f["scheduled_origin_local"] = f["FlightDate"] + pd.to_timedelta(dep_min, unit="m")
    elapsed = pd.to_numeric(f["CRSElapsedTime"], errors="coerce")
    # For weather matching, derive destination local clock time from scheduled arrival HHMM.
    arr_min = f["CRSArrTime"].map(hhmm_minutes)
    arr_clock = f["FlightDate"] + pd.to_timedelta(arr_min, unit="m")
    crosses = arr_min < dep_min
    f["scheduled_destination_local"] = arr_clock + pd.to_timedelta(crosses.astype(int), unit="D")
    f["scheduled_elapsed_minutes"] = elapsed
    return f


def attach_weather(flights: pd.DataFrame, weather: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for (origin, dest), g in flights.groupby(["Origin", "Dest"], sort=False):
        g = g.copy()
        for role, code, time_col in [
            ("origin", origin, "scheduled_origin_local"),
            ("destination", dest, "scheduled_destination_local"),
        ]:
            left = g.sort_values(time_col)
            right = weather[code].sort_values("weather_time").rename(columns={c: f"{role}_{c}" for c in WEATHER_COLUMNS})
            g = pd.merge_asof(
                left,
                right,
                left_on=time_col,
                right_on="weather_time",
                direction="nearest",
                tolerance=pd.Timedelta("90min"),
            ).drop(columns=["weather_time"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    dep = x["scheduled_origin_local"]
    arr = x["scheduled_destination_local"]
    x["sched_dep_hour"] = dep.dt.hour + dep.dt.minute / 60
    x["sched_arr_hour"] = arr.dt.hour + arr.dt.minute / 60
    x["sched_dow"] = dep.dt.dayofweek
    x["sched_month"] = dep.dt.month
    x["is_weekend"] = (x["sched_dow"] >= 5).astype(int)
    x["dep_hour_sin"] = np.sin(2 * math.pi * x["sched_dep_hour"] / 24)
    x["dep_hour_cos"] = np.cos(2 * math.pi * x["sched_dep_hour"] / 24)
    dep_delay = pd.to_numeric(x["DepDelayMinutes"], errors="coerce").fillna(0)
    arr_delay = pd.to_numeric(x["ArrDelayMinutes"], errors="coerce").fillna(0)
    cancelled = pd.to_numeric(x["Cancelled"], errors="coerce").fillna(0).astype(int)
    diverted = pd.to_numeric(x["Diverted"], errors="coerce").fillna(0).astype(int)
    x["any_disruption"] = ((dep_delay >= 15) | (arr_delay >= 15) | (cancelled == 1) | (diverted == 1)).astype(int)
    x["severe_disruption"] = ((dep_delay >= 120) | (arr_delay >= 120) | (cancelled == 1) | (diverted == 1)).astype(int)
    return x


def make_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ], remainder="drop")
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def metrics(y, p, threshold):
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, p >= threshold, zero_division=0)),
        "positive_rate": float(np.mean(y)),
    }


def best_threshold(y, p):
    candidates = np.linspace(0.05, 0.75, 71)
    scores = [f1_score(y, p >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main():
    MODELS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    flights = scheduled_times(load_bts())
    weather = {code: load_lcd(code) for code in AIRPORTS}
    table = add_features(attach_weather(flights, weather))
    table.to_csv(PROCESSED / "v3_ogg_lax_dfw_modeling_table_2020_2025.csv.gz", index=False, compression="gzip")

    cat = ["Reporting_Airline", "Origin", "Dest"]
    num = ["Distance", "scheduled_elapsed_minutes", "sched_dep_hour", "sched_arr_hour", "sched_dow", "sched_month", "is_weekend", "dep_hour_sin", "dep_hour_cos"]
    num += [f"{role}_{c}" for role in ("origin", "destination") for c in WEATHER_COLUMNS]
    features = cat + num

    year = table["FlightDate"].dt.year
    train = table[year <= 2023].dropna(subset=["any_disruption", "severe_disruption"])
    val = table[year == 2024].dropna(subset=["any_disruption", "severe_disruption"])
    test = table[year == 2025].dropna(subset=["any_disruption", "severe_disruption"])

    any_base = make_pipeline(cat, num)
    any_base.fit(train[features], train["any_disruption"])
    any_val_raw = any_base.predict_proba(val[features])[:, 1]
    any_cal = IsotonicRegression(out_of_bounds="clip").fit(any_val_raw, val["any_disruption"])
    any_val = any_cal.predict(any_val_raw)
    any_test = any_cal.predict(any_base.predict_proba(test[features])[:, 1])
    any_thr = best_threshold(val["any_disruption"].to_numpy(), any_val)

    disrupted_train = train[train["any_disruption"] == 1]
    disrupted_val = val[val["any_disruption"] == 1]
    cond_base = make_pipeline(cat, num)
    cond_base.fit(disrupted_train[features], disrupted_train["severe_disruption"])
    cond_val_raw = cond_base.predict_proba(disrupted_val[features])[:, 1]
    cond_cal = IsotonicRegression(out_of_bounds="clip").fit(cond_val_raw, disrupted_val["severe_disruption"])

    cond_val_all = cond_cal.predict(cond_base.predict_proba(val[features])[:, 1])
    cond_test_all = cond_cal.predict(cond_base.predict_proba(test[features])[:, 1])
    severe_val = np.clip(any_val * cond_val_all, 0, any_val)
    severe_test = np.clip(any_test * cond_test_all, 0, any_test)
    severe_thr = best_threshold(val["severe_disruption"].to_numpy(), severe_val)

    from src.models.calibrated import CalibratedProbabilityModel
    any_model = CalibratedProbabilityModel(any_base, any_cal, features)
    cond_model = CalibratedProbabilityModel(cond_base, cond_cal, features)
    joblib.dump(any_model, MODELS / "v3_any_disruption_model.joblib", compress=3)
    joblib.dump(cond_model, MODELS / "v3_conditional_severe_model.joblib", compress=3)

    metadata = {
        "artifact_version": "research-v3-ogg-lax-dfw-dual-weather-calibrated",
        "supported_routes": ["OGG-LAX", "LAX-OGG", "OGG-DFW", "DFW-OGG"],
        "weather_source_training": "NOAA/NCEI Local Climatological Data at origin and destination",
        "weather_source_inference": "National Weather Service hourly forecast at origin and destination",
        "feature_columns": features,
        "categorical_columns": cat,
        "numeric_columns": num,
        "any_threshold": any_thr,
        "severe_threshold": severe_thr,
        "any_validation": metrics(val["any_disruption"].to_numpy(), any_val, any_thr),
        "any_test": metrics(test["any_disruption"].to_numpy(), any_test, any_thr),
        "severe_validation": metrics(val["severe_disruption"].to_numpy(), severe_val, severe_thr),
        "severe_test": metrics(test["severe_disruption"].to_numpy(), severe_test, severe_thr),
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "training_policy": "2020-2023 train; 2024 calibration/threshold selection; 2025 untouched temporal test",
    }
    (MODELS / "v3_inference_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
