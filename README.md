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

The initial research phase will integrate historical flight operations with severe-weather observations and event records, then establish baseline disruption-prediction models.

## Planned modeling components

1. **Flight Disruption Predictor** — classification of normal, delayed, severely delayed, and cancelled flights.
2. **Recovery-Time Model** — estimation of how quickly airport/airline operations normalize after a disruption.
3. **Historical Analogue Retrieval** — retrieval of similar past weather and operational incidents.
4. **Explainable AI** — feature-level explanations for individual predictions.

## Project structure

- `data/raw/` — source datasets (not committed to Git)
- `data/processed/` — cleaned/model-ready datasets
- `data/incidents/` — derived historical incident records
- `notebooks/` — research experiments and exploratory analyses
- `src/preprocessing/` — cleaning and data integration
- `src/features/` — feature engineering
- `src/models/` — predictive and recovery models
- `src/retrieval/` — historical similarity engine
- `backend/` — future FastAPI inference service
- `frontend/` — future GitHub Pages-compatible UI
- `models/` — trained model artifacts (not committed by default)
- `tests/` — automated tests

## Initial roadmap

**Phase 1:** OGG historical flight + weather dataset and baseline analysis  
**Phase 2:** disruption prediction model  
**Phase 3:** operational recovery modeling  
**Phase 4:** historical-event similarity engine  
**Phase 5:** explainability and uncertainty  
**Phase 6:** public web interface

## Status

Early research and prototyping.
