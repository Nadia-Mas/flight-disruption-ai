# FlightRescue AI — Flight Disruption Intelligence

FlightRescue AI is a research-driven passenger-facing platform for predicting flight disruption risk at **Kahului Airport (OGG), Maui, Hawaii**. It combines trained machine-learning models, official weather information, historical flight operations, and historical severe-weather analogs to help travelers understand disruption risk and how airlines have behaved under similar conditions.

## What FlightRescue does

Given an airline, origin, destination, and scheduled flight time, FlightRescue can provide:

- probability of any flight disruption;
- probability of severe disruption;
- official weather context for OGG;
- historically similar Hawaii weather/airport events;
- operational recovery context;
- airline-specific historical performance during matched events, including delay and cancellation behavior when the airline-event aggregate is available;
- comparison with other airlines operating at OGG under similar historical conditions.

## First case study

**Kahului Airport (OGG), Maui, Hawaii**

The current trained models are OGG-specific. The passenger interface accepts friendly origin and destination searches using airport code, airport name, or city, but the current research model requires OGG to be either the departure or arrival airport.

## Live application

The passenger-facing FlightRescue interface is deployed with GitHub Pages:

`https://nadia-mas.github.io/flight-disruption-ai/`

The production FastAPI inference service is deployed separately on Vercel. The web interface calls this API for live model inference and weather-aware scenario analysis.

## Weather intelligence

FlightRescue automatically retrieves official **National Weather Service (NWS)** forecast information for Kahului/OGG for the requested flight time when that forecast is available. Passengers therefore do not need to manually enter technical weather measurements.

The NWS conditions are converted into meteorological variables used by the trained inference pipeline, including temperature, humidity, wind, gusts, precipitation, and related weather context. Optional hazard presets in the interface are intended for hypothetical stress testing rather than as replacements for official observed/forecast weather.

## Historical airline evidence

FlightRescue combines weather similarity with historical OGG flight operations. A reproducible aggregation pipeline builds airline-level performance for historical NOAA event windows so matched historical events can be summarized using metrics such as:

- number of flights observed;
- percentage delayed by at least 15 minutes;
- cancellation percentage;
- severe-disruption percentage;
- mean delay time;
- comparison between the selected airline and other airlines operating at OGG.

These historical statistics are evidence from prior operations and are displayed separately from the machine-learning probability rather than being presented as guaranteed future outcomes.

## API usage and permission

The FlightRescue API is provided for **research, demonstration, and evaluation purposes**. Third-party use of the FlightRescue API — including integration into another application or website, automated access, redistribution, commercial use, or use of the API as part of another service — requires **prior permission from the project owner**.

To request permission to use or integrate the API, contact:

**Fatemeh (Nadia) Masoumi**  
**Email:** Fatemeh.masoumi.1994@gmail.com

Public availability of an API endpoint should not be interpreted as permission for unrestricted third-party use.

## Research notebook pipeline

The research pipeline consists of nine stages:

1. `01_ogg_data_exploration.ipynb` — source loading and initial OGG exploration.
2. `02_ogg_phase1_eda.ipynb` — Phase 1 exploratory data analysis.
3. `03_build_modeling_table.ipynb` — flight + OGG weather + Hawaii storm-event alignment.
4. `04_validate_and_engineer_features.ipynb` — validation, leakage control, and feature engineering.
5. `05_baseline_disruption_models.ipynb` — multiclass baseline models.
6. `06_binary_and_severe_disruption_models.ipynb` — operational binary disruption/severe-disruption models.
7. `07_historical_event_recovery_dataset.ipynb` — hourly operations and BTS-derived recovery episodes.
8. `08_historical_similar_event_retrieval.ipynb` — historical analog retrieval and recovery evidence.
9. `09_flightrescue_inference_pipeline.ipynb` — integration layer for passenger-facing inference.

## Data coverage

### BTS flight operations

- Airport focus: **OGG (Kahului Airport)**
- Current source download coverage: **January 2020 through June 2026**
- Total OGG flight records in the Phase 1 download: **330,615**
- Raw data are downloaded/generated artifacts and are not intended to be stored permanently in Git history.

### NOAA Local Climatological Data (LCD)

- Weather station: **Kahului Airport, Maui**
- Historical coverage used in the research pipeline: **2020 through 2025**
- Historical weather observations in the Phase 1 download: **66,074**

### NOAA Storm Events

- Geographic scope: **State of Hawaii**
- Historical source coverage: **2020 through 2026**
- Hawaii storm-event records retained in the Phase 1 download: **2,613**

LCD provides measured airport conditions around OGG at a particular time, while Storm Events provides broader hazardous-weather context, including event type, timing, geography, magnitude when available, impacts, source identifiers, and narrative descriptions.

## Dataset alignment

The primary flight-level unit of analysis is one scheduled flight touching OGG:

```text
OGG flight at time t
        +
OGG airport weather around time t
        +
Hawaii severe-weather event context around time t
        ↓
model-ready flight-disruption observation
```

For passenger-facing inference, the architecture extends this to:

```text
Passenger flight details
        ↓
Official NWS forecast at OGG
        ↓
Weather-aware ML inference
        +
Historical NOAA event similarity
        +
BTS airline performance during matched events
        ↓
FlightRescue risk + historical airline evidence
```

## Modeling components

1. **Flight Disruption Predictor** — flight-level probability of any disruption.
2. **Severe Disruption Predictor** — severe-delay/cancellation risk estimation.
3. **NWS Weather Integration** — official forecast context for the requested OGG flight time.
4. **Recovery Intelligence** — historical operational recovery after event episodes.
5. **Historical Analogue Retrieval** — similar prior Hawaii weather/airport incidents.
6. **Airline Event Performance** — BTS-derived airline delay/cancellation behavior during matched historical event windows.
7. **Inference Layer** — combines model probabilities, weather, historical evidence, recovery context, and model-maturity information.
8. **Passenger Web Interface** — GitHub Pages-compatible FlightRescue application.

## Project structure

- `api/` — Vercel/FastAPI entry point
- `backend/` — production inference service
- `data/raw/` — source datasets generated/downloaded as needed
- `data/processed/` — cleaned/model-ready and compact inference datasets
- `docs/` — public GitHub Pages passenger interface
- `models/` — trained inference artifacts
- `notebooks/` — research experiments and pipeline notebooks 01–09
- `scripts/` — reproducible data download and aggregation scripts
- `src/preprocessing/` — cleaning and data integration
- `src/features/` — feature engineering
- `src/models/` — predictive and recovery modeling code
- `src/retrieval/` — historical similarity engine
- `tests/` — automated tests

## Current status

The research pipeline through Notebook 09 is complete. Trained inference artifacts are deployed through a FastAPI backend on Vercel, and the passenger-facing application is connected to the production service. Official NWS forecast integration is implemented for OGG, and the historical airline-event aggregation pipeline extends the application with airline-specific delay and cancellation evidence under similar historical conditions.

## Research-use notice

FlightRescue is an experimental research system and is **not an official airline, airport, FAA, NOAA, or National Weather Service product**. Predictions and historical comparisons are informational and should not be treated as guarantees of flight status. Travelers should confirm operational decisions with their airline and official aviation/weather sources.
