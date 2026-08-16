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

## Phase 1 data coverage

The first downloaded dataset currently covers the following domains:

### BTS flight operations

- Airport focus: **OGG (Kahului Airport)**
- Date range available in the current download: **January 2020 through June 2026**
- July–December 2026 were not yet available from BTS at download time and were skipped.
- Total OGG flight records downloaded: **330,615**
- The records include flights arriving at or departing from OGG and contain operational variables such as scheduled and actual times, airline/carrier, origin, destination, delays, cancellation status, diversion status, and reported delay causes when available.
- Local file: `data/raw/bts/ogg_flights_2020_2026.csv.gz`

### NOAA Local Climatological Data (LCD)

- Weather station: **Kahului Airport, Maui**
- Date range successfully downloaded: **2020 through 2025**
- 2026 LCD data were not available from the source used by the downloader at the time of collection and were skipped.
- Total weather observations downloaded: **66,074**
- These observations provide the airport-level weather context used to align conditions with individual flights, including fields such as observation time, wind, gusts, visibility, precipitation, temperature, pressure, and reported weather conditions when present.
- Local file: `data/raw/weather/ogg_lcd_2020_2026.csv.gz`

### NOAA Storm Events

- Geographic scope: **State of Hawaii**
- Date range downloaded: **2020 through 2026**
- Total Hawaii storm-event records retained: **2,613**
- Local file: `data/raw/incidents/hawaii_storm_events_2020_2026.csv.gz`

The Storm Events dataset is different from the hourly airport weather data. LCD tells us the measured conditions around OGG at a particular time, while Storm Events tells us whether that period was part of a formally recorded hazardous-weather incident and describes the broader event.

Depending on the event record, Storm Events can provide information such as:

- event type, for example high wind, flash flood, heavy rain, thunderstorm wind, tropical-storm-related conditions, or other severe events;
- event start and end dates/times;
- affected county/zone or geographic area;
- event magnitude and magnitude type when reported;
- injuries and fatalities when reported;
- property and crop damage estimates when reported;
- source of the event report;
- episode/event identifiers that connect related observations;
- latitude/longitude or location information when available;
- narrative descriptions explaining what occurred and the impacts that were observed.

This source will let us create event-level features such as `storm_event_active`, `event_type`, `event_duration`, `event_severity`, and time relative to event start/end. It will also support the historical-analogue component of the project: when a new disruption occurs, the system can retrieve previous Hawaii events with similar weather, timing, airport impact, and flight outcomes and show how quickly operations recovered in those cases.

## Phase 1 dataset alignment

The intended unit of analysis is initially **one scheduled flight touching OGG**. For each flight we will align the flight record with the nearest relevant OGG weather observations and any severe-weather event active around that time.

Conceptually:

```text
OGG flight at time t
        +
OGG airport weather around time t
        +
Hawaii severe-weather event context around time t
        ↓
model-ready flight-disruption observation
```

The initial target classes are expected to distinguish normal operation, delay, severe delay, cancellation, and potentially diversion. Later phases will also model operational recovery time after a major disruption.

## Current raw-data summary

| Data source | Coverage | Records |
|---|---:|---:|
| BTS OGG flight operations | Jan 2020 – Jun 2026 | 330,615 flights |
| NOAA OGG LCD weather | 2020 – 2025 | 66,074 observations |
| NOAA Hawaii Storm Events | 2020 – 2026 | 2,613 events |

Raw downloaded datasets are intentionally excluded from Git history because they are generated/downloaded artifacts and can be large. They are recreated with the scripts in `scripts/`.

## Planned modeling components

1. **Flight Disruption Predictor** — classification of normal, delayed, severely delayed, cancelled, and potentially diverted flights.
2. **Recovery-Time Model** — estimation of how quickly airport/airline operations normalize after a disruption.
3. **Historical Analogue Retrieval** — retrieval of similar past weather and operational incidents.
4. **Explainable AI** — feature-level explanations for individual predictions.

## Project structure

- `data/raw/` — source datasets (not committed to Git)
- `data/processed/` — cleaned/model-ready datasets
- `data/incidents/` — derived historical incident records
- `notebooks/` — research experiments and exploratory analyses
- `scripts/` — reproducible data-download scripts
- `src/preprocessing/` — cleaning and data integration
- `src/features/` — feature engineering
- `src/models/` — predictive and recovery models
- `src/retrieval/` — historical similarity engine
- `backend/` — future FastAPI inference service
- `frontend/` — future GitHub Pages-compatible UI
- `models/` — trained model artifacts (not committed by default)
- `tests/` — automated tests

## Initial roadmap

**Phase 1:** OGG historical flight + weather + storm-event dataset and baseline analysis  
**Phase 2:** disruption prediction model  
**Phase 3:** operational recovery modeling  
**Phase 4:** historical-event similarity engine  
**Phase 5:** explainability and uncertainty  
**Phase 6:** public web interface

## Status

Phase 1 data acquisition is complete for the currently available source periods. The next step is exploratory analysis, temporal alignment, event labeling, and construction of the first model-ready OGG dataset.
