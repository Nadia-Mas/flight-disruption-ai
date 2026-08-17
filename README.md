# Flight Disruption AI

An AI-driven platform for understanding and predicting weather-related flight disruptions and operational recovery.

## Core idea

Given an airport, airline, route, departure time, and current weather event, the system will estimate:

- probability of normal operation, severe delay, or cancellation;
- likely operational recovery time;
- historically similar weather/airport incidents;
- interpretable factors driving the prediction.

## First case study

**Kahului Airport (OGG), Maui, Hawaii**

The initial research phase integrates historical flight operations with airport weather observations and severe-weather event records, then uses those aligned data sources to develop disruption-prediction and recovery models.

## Web prototype

A GitHub Pages-compatible FlightRescue AI prototype now lives in `docs/`.

The public UI is designed as a research/product prototype and currently provides a scenario explorer plus project findings. Live trained-model probabilities will be connected through the future inference API; the static page intentionally does not present its illustrative scenario score as a real-time operational prediction.

Expected Pages URL after Pages is enabled for the repository:

`https://nadia-mas.github.io/flight-disruption-ai/`

## Research notebook pipeline

The notebook research phase now consists of nine stages:

1. `01_ogg_data_exploration.ipynb` — source loading and initial OGG exploration.
2. `02_ogg_phase1_eda.ipynb` — Phase 1 exploratory data analysis.
3. `03_build_modeling_table.ipynb` — flight + OGG weather + Hawaii storm-event alignment.
4. `04_validate_and_engineer_features.ipynb` — validation, leakage control, and feature engineering.
5. `05_baseline_disruption_models.ipynb` — multiclass baseline models.
6. `06_binary_and_severe_disruption_models.ipynb` — operational binary disruption/severe-disruption models.
7. `07_historical_event_recovery_dataset.ipynb` — hourly operations and BTS-derived recovery episodes.
8. `08_historical_similar_event_retrieval.ipynb` — historical analog retrieval and recovery evidence.
9. `09_flightrescue_inference_pipeline.ipynb` — integration layer for passenger-facing inference.

## Phase 1 data coverage

### BTS flight operations

- Airport focus: **OGG (Kahului Airport)**
- Date range available in the current download: **January 2020 through June 2026**
- July–December 2026 were not yet available from BTS at download time and were skipped.
- Total OGG flight records downloaded: **330,615**
- Local file: `data/raw/bts/ogg_flights_2020_2026.csv.gz`

### NOAA Local Climatological Data (LCD)

- Weather station: **Kahului Airport, Maui**
- Date range successfully downloaded: **2020 through 2025**
- Total weather observations downloaded: **66,074**
- Local file: `data/raw/weather/ogg_lcd_2020_2026.csv.gz`

### NOAA Storm Events

- Geographic scope: **State of Hawaii**
- Date range downloaded: **2020 through 2026**
- Total Hawaii storm-event records retained: **2,613**
- Local file: `data/raw/incidents/hawaii_storm_events_2020_2026.csv.gz`

LCD provides measured airport conditions around OGG at a particular time, while Storm Events provides broader recorded hazardous-weather context, including event type, start/end time, geography, magnitude when available, impacts, source, identifiers, and narrative descriptions.

## Phase 1 dataset alignment

The initial unit of analysis is **one scheduled flight touching OGG**:

```text
OGG flight at time t
        +
OGG airport weather around time t
        +
Hawaii severe-weather event context around time t
        ↓
model-ready flight-disruption observation
```

The project then adds a second event-centric representation for historical weather episodes, airport disruption, airline behavior, and operational recovery.

## Current raw-data summary

| Data source | Coverage | Records |
|---|---:|---:|
| BTS OGG flight operations | Jan 2020 – Jun 2026 | 330,615 flights |
| NOAA OGG LCD weather | 2020 – 2025 | 66,074 observations |
| NOAA Hawaii Storm Events | 2020 – 2026 | 2,613 events |

Raw downloaded datasets are intentionally excluded from Git history because they are generated/downloaded artifacts and can be large. They are recreated with the scripts in `scripts/`.

## Modeling components

1. **Flight Disruption Predictor** — flight-level risk estimation.
2. **Severe Disruption Predictor** — cancellation/severe-delay risk estimation.
3. **Recovery Intelligence** — BTS-derived operational recovery after event episodes.
4. **Historical Analogue Retrieval** — similar prior Hawaii weather/airport incidents.
5. **Inference Layer** — combines risk probabilities, historical evidence, recovery range, severity, and confidence.
6. **Web Interface** — GitHub Pages-compatible passenger-facing prototype.

## Project structure

- `data/raw/` — source datasets (not committed to Git)
- `data/processed/` — cleaned/model-ready datasets
- `notebooks/` — research experiments and pipeline notebooks 01–09
- `scripts/` — reproducible data-download scripts
- `src/preprocessing/` — cleaning and data integration
- `src/features/` — feature engineering
- `src/models/` — predictive and recovery models
- `src/retrieval/` — historical similarity engine
- `backend/` — FastAPI inference service target
- `frontend/` — application frontend development area
- `docs/` — GitHub Pages static prototype and project documentation
- `models/` — trained model artifacts
- `tests/` — automated tests

## Roadmap

**Completed research stages:** data acquisition, EDA, modeling-table construction, leakage-safe features, baseline classification, binary/severe risk models, recovery episodes, historical analog retrieval, and inference integration notebook.

**Current software stage:** package inference code and trained artifacts, expose a FastAPI endpoint, connect current weather/event inputs, then wire the API into the GitHub Pages UI.

## Status

The research notebook pipeline is complete through Notebook 09. A GitHub Pages-compatible FlightRescue AI web prototype has been added under `docs/`; live inference remains the next software-development step.
