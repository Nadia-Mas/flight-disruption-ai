"""Download and prepare Phase 1 data for the OGG flight-disruption project.

Sources
-------
1. U.S. DOT/BTS Reporting Carrier On-Time Performance (monthly ZIP files)
2. NOAA/NCEI Local Climatological Data for Kahului Airport

Raw nationwide BTS files are downloaded temporarily, filtered to flights touching
OGG, and discarded. This keeps local storage manageable.

Usage
-----
python scripts/download_phase1_data.py --start-year 2020 --end-year 2026

For 2026, BTS data currently run through May; unavailable future months are
skipped automatically.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BTS_DIR = ROOT / "data" / "raw" / "bts"
WEATHER_DIR = ROOT / "data" / "raw" / "weather"
PROCESSED_DIR = ROOT / "data" / "processed"

BTS_BASE = "https://transtats.bts.gov/PREZIP"
BTS_NAME = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
NOAA_LCD_BASE = "https://www.ncei.noaa.gov/data/local-climatological-data/access"
# Kahului Airport: WMO 911900 + WBAN 22516
NOAA_LCD_FILE = "91190022516.csv"

KEEP_BTS = [
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "Origin", "OriginCityName",
    "Dest", "DestCityName", "CRSDepTime", "DepTime", "DepDelayMinutes",
    "CRSArrTime", "ArrTime", "ArrDelayMinutes", "Cancelled",
    "CancellationCode", "Diverted", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]


def get(url: str, timeout: int = 120) -> requests.Response:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "flight-disruption-ai/0.1"})
    r.raise_for_status()
    return r


def download_bts(start_year: int, end_year: int) -> pd.DataFrame:
    BTS_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            name = BTS_NAME.format(year=year, month=month)
            url = f"{BTS_BASE}/{name}"
            print(f"BTS {year}-{month:02d} ...", end=" ")
            try:
                r = get(url)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    print("not available; skipped")
                    continue
                raise

            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    print("no CSV; skipped")
                    continue
                with zf.open(csv_names[0]) as fh:
                    df = pd.read_csv(fh, low_memory=False)

            # BTS files sometimes include a trailing unnamed column.
            available = [c for c in KEEP_BTS if c in df.columns]
            ogg = df.loc[(df["Origin"] == "OGG") | (df["Dest"] == "OGG"), available].copy()
            frames.append(ogg)
            print(f"{len(ogg):,} OGG flights")

    if not frames:
        raise RuntimeError("No BTS data were downloaded.")

    flights = pd.concat(frames, ignore_index=True)
    out = BTS_DIR / f"ogg_flights_{start_year}_{end_year}.csv.gz"
    flights.to_csv(out, index=False, compression="gzip")
    print(f"Saved {len(flights):,} OGG flight records -> {out}")
    return flights


def download_noaa_lcd(start_year: int, end_year: int) -> pd.DataFrame:
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        url = f"{NOAA_LCD_BASE}/{year}/{NOAA_LCD_FILE}"
        print(f"NOAA LCD {year} ...", end=" ")
        try:
            r = get(url)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                print("not available; skipped")
                continue
            raise
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        frames.append(df)
        print(f"{len(df):,} observations")

    if not frames:
        raise RuntimeError("No NOAA LCD data were downloaded.")

    weather = pd.concat(frames, ignore_index=True)
    out = WEATHER_DIR / f"ogg_lcd_{start_year}_{end_year}.csv.gz"
    weather.to_csv(out, index=False, compression="gzip")
    print(f"Saved {len(weather):,} weather observations -> {out}")
    return weather


def summarize(flights: pd.DataFrame, weather: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "flight_rows": len(flights),
        "flight_start": str(pd.to_datetime(flights["FlightDate"]).min().date()),
        "flight_end": str(pd.to_datetime(flights["FlightDate"]).max().date()),
        "weather_rows": len(weather),
        "weather_start": None,
        "weather_end": None,
    }
    if "DATE" in weather.columns:
        d = pd.to_datetime(weather["DATE"], errors="coerce")
        summary["weather_start"] = str(d.min())
        summary["weather_end"] = str(d.max())
    pd.Series(summary).to_json(PROCESSED_DIR / "phase1_download_summary.json", indent=2)
    print("Wrote data/processed/phase1_download_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    flights = download_bts(args.start_year, args.end_year)
    weather = download_noaa_lcd(args.start_year, args.end_year)
    summarize(flights, weather)


if __name__ == "__main__":
    main()
