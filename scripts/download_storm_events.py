"""Download NOAA Storm Events detail records and retain Hawaii events.

The script reads NOAA/NCEI's official bulk-download directory, selects the
latest Details file for each requested year, and filters records to HAWAII.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "incidents"
INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2026)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "flight-disruption-ai/0.1"}
    html = requests.get(INDEX_URL, headers=headers, timeout=60)
    html.raise_for_status()

    all_hi = []
    for year in range(args.start_year, args.end_year + 1):
        # Example: StormEvents_details-ftp_v1.0_d2024_c20250520.csv.gz
        pattern = rf'StormEvents_details-ftp_v[^"<> ]+_d{year}_c\d+\.csv\.gz'
        names = sorted(set(re.findall(pattern, html.text)))
        if not names:
            print(f"Storm Events {year}: not available; skipped")
            continue
        name = names[-1]
        url = INDEX_URL + name
        print(f"Storm Events {year}: {name}")
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)
        hi = df.loc[df["STATE"].astype(str).str.upper().eq("HAWAII")].copy()
        all_hi.append(hi)
        print(f"  retained {len(hi):,} Hawaii events")

    if not all_hi:
        raise RuntimeError("No Hawaii Storm Events records downloaded.")

    out_df = pd.concat(all_hi, ignore_index=True)
    out = OUT_DIR / f"hawaii_storm_events_{args.start_year}_{args.end_year}.csv.gz"
    out_df.to_csv(out, index=False, compression="gzip")
    print(f"Saved {len(out_df):,} Hawaii storm-event records -> {out}")


if __name__ == "__main__":
    main()
