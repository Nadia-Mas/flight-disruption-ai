# FlightRescue AI — End-to-End Flight Disruption Intelligence

**FlightRescue AI** is an end-to-end data science and AI project built to help passengers understand flight-disruption risk during severe weather, beginning with **Kahului Airport (OGG), Maui, Hawaii**.

The project connects the full product and machine-learning lifecycle:

**public passenger interface → live weather data → feature engineering → trained AI models → historical-event retrieval → airline performance comparison → API inference → deployed web application.**

FlightRescue is not only a visualization dashboard. Behind the passenger-facing application is a reproducible data pipeline, historical aviation dataset, weather-integration layer, machine-learning inference service, model-validation workflow, historical similarity engine, and deployed FastAPI backend.

> **Current status:** FlightRescue is an experimental research prototype. The end-to-end product is deployed and functional, while the AI models are actively being improved, recalibrated, and expanded to use weather at both ends of selected routes.

---

## Why FlightRescue was built

The motivation became especially clear during **Hurricane / Tropical Storm Lala in August 2026**, when severe weather affected Hawaii, including Maui County, and air travel across the islands became uncertain. Travelers faced postponed flights, possible delays and cancellations, changing airline operations, airport disruptions, and rapidly changing weather conditions.

For a passenger trying to leave Hawaii, the practical question is not simply:

> **“Is there a storm?”**

The more useful questions are:

> **How likely is my airline to disrupt this flight?**  
> **What happened when similar conditions occurred before?**  
> **Did this airline usually delay flights or cancel them?**  
> **How did competing airlines perform during the same historical events?**  
> **How long did airport operations take to recover?**

FlightRescue was designed around those questions.

During Lala, Hawaii experienced major weather impacts, power outages, transportation disruptions, airport closures in parts of the state, and large numbers of postponed flights. Travelers were repeatedly advised to monitor airline operations as conditions changed. That real passenger uncertainty is the core problem this project is intended to address.

### The product idea

Instead of showing passengers only a weather forecast or only a flight-status page, FlightRescue attempts to combine:

1. **the passenger's actual route, airline, date, and time;**
2. **official National Weather Service forecast conditions;**
3. **historical BTS airline operations;**
4. **historical NOAA severe-weather events;**
5. **machine-learning disruption probabilities;** and
6. **evidence from how airlines behaved during comparable historical conditions.**

The result is a passenger-oriented decision-support system rather than a raw data-science notebook.

---

## Passenger persona

### Maya — a traveler trying to get home from Maui

**Persona:** Maya, 34, leisure traveler  
**Situation:** Her vacation in Maui is ending during a major Hawaii weather event.  
**Flight:** Kahului (OGG) to Los Angeles (LAX) or Dallas/Fort Worth (DFW).  
**Problem:** Her airline has already changed schedules several times and she does not know whether another delay or cancellation is likely.

Maya is not an aviation analyst. She may not know airport abbreviations, historical storm names, wind thresholds, visibility measurements, or how to interpret meteorological data.

She wants to enter:

- her airline;
- where she is flying from;
- where she is going;
- her flight date; and
- her scheduled time.

FlightRescue should handle the technical work behind the scenes.

### What Maya wants to know

A useful result for Maya should answer questions such as:

- **What is the estimated probability that my flight will be disrupted?**
- **What is the estimated severe-disruption risk?**
- **What does the official weather forecast look like at the relevant airports?**
- **Have similar weather events affected Maui before?**
- **During those events, what percentage of American / United / Hawaiian / Southwest / other flights were delayed?**
- **What percentage were canceled?**
- **How did my selected airline compare with other airlines?**
- **How long did operations typically take to recover?**

This passenger persona guides the product design: technical data should be converted into understandable evidence rather than requiring the passenger to interpret raw weather or aviation datasets.

---

## What FlightRescue does

Given an airline, route, scheduled date, and scheduled time, FlightRescue can provide:

- probability of **any flight disruption**;
- probability of **severe disruption**;
- official NWS forecast context;
- automatically derived weather features;
- historically similar Hawaii weather / airport events;
- historical recovery context;
- selected-airline delay and cancellation behavior during matched event windows;
- comparison with other airlines operating under similar historical OGG conditions;
- an accessible passenger-facing explanation of the result.

The application deliberately presents **model predictions** and **historical evidence** as separate concepts. Historical percentages are not disguised as model probabilities, and model probabilities are not presented as guarantees.

---

# End-to-end system architecture

FlightRescue covers the complete path from raw public data to a deployed passenger application.

```text
              PASSENGER
                  │
                  ▼
     Airline + route + date + time
                  │
                  ▼
      Passenger-friendly web interface
                  │
                  ▼
      Official NWS forecast retrieval
                  │
                  ▼
     Weather + flight feature engineering
                  │
          ┌───────┴────────┐
          ▼                ▼
  AI disruption model   Historical engine
          │                │
          │        NOAA severe-weather events
          │        BTS airline operations
          │        Similar-event retrieval
          │        Airline delay/cancel rates
          │                │
          └───────┬────────┘
                  ▼
         FastAPI inference service
                  │
                  ▼
       Passenger risk + evidence
                  │
                  ▼
        GitHub Pages application
```

## End-to-end technology stack

### Data engineering

- U.S. DOT / BTS Reporting Carrier On-Time Performance data
- NOAA Local Climatological Data
- NOAA Storm Events
- automated data downloads and filtering
- timestamp alignment
- route filtering
- airport-weather joins
- historical event windows
- airline-level aggregations

### Data science

- exploratory data analysis
- missing-value handling
- leakage control
- time-aware train / validation / test splits
- feature engineering
- weather-event representation
- historical-event similarity
- airline delay / cancellation metrics
- recovery episode analysis

### AI / machine learning

The backend contains trained models designed to estimate:

1. **Probability of any disruption** — delay, cancellation, diversion, or related operational disruption.
2. **Probability of severe disruption** — modeled using a two-stage severe-risk architecture.

The current production model uses airline, direction, route, schedule, calendar, and weather variables. A new model version is being developed to use **origin and destination weather separately** for a focused set of routes.

### Backend

- Python
- scikit-learn
- pandas / NumPy
- joblib model artifacts
- FastAPI
- REST inference endpoints
- NWS API integration
- CORS-protected public frontend access
- Vercel deployment

### Frontend

- passenger-oriented airport search by code, city, or airport name
- date and time selection
- automatic weather retrieval
- disruption-risk visualization
- severe-risk visualization
- historical analog explanation
- airline performance comparison
- responsive GitHub Pages deployment

### MLOps / reproducibility

- GitHub Actions model-building workflows
- automated artifact generation
- production smoke tests
- versioned inference metadata
- temporal held-out testing
- Vercel Git deployment
- GitHub Pages deployment

---

## Current geographic scope

The project began as an **OGG / Maui case study**.

The next focused model version intentionally limits expansion to:

- **OGG — Kahului Airport, Maui, Hawaii**
- **LAX — Los Angeles International Airport, California**
- **DFW — Dallas/Fort Worth International Airport, Texas**

The initial v3 route scope is:

- OGG → LAX
- LAX → OGG
- OGG → DFW
- DFW → OGG

Restricting the scope allows the model to be scientifically validated before expanding to hundreds of airports.

---

## Weather intelligence

FlightRescue uses the **National Weather Service API** rather than asking passengers to manually enter wind speed, humidity, precipitation, visibility, or other technical measurements.

The production system retrieves the forecast nearest the scheduled flight time and converts it into machine-learning features.

### Current production behavior

The deployed v2 model is primarily OGG-centric and evaluates official weather at Kahului Airport.

### v3 improvement in progress

FlightRescue v3 is being developed to use weather at **both ends of the flight**.

For example:

```text
OGG → LAX

OGG weather near scheduled departure
            +
LAX weather near scheduled arrival
            ↓
      dual-airport model
```

and:

```text
DFW → OGG

DFW weather near scheduled departure
            +
OGG weather near scheduled arrival
            ↓
      dual-airport model
```

Historical training data for v3 use NOAA/NCEI observations at the relevant airports, while production inference uses current NWS forecast data when available.

---

## Historical airline intelligence

Weather alone does not tell a passenger how a particular airline tends to respond operationally.

FlightRescue therefore builds airline-level statistics during historical severe-weather event windows.

For matched historical events the system can summarize:

- total flights observed;
- percentage delayed by at least 15 minutes;
- cancellation percentage;
- severe-disruption percentage;
- average delay;
- selected airline versus competing airlines.

The objective is to answer a question such as:

> **“During weather events similar to this one, how did American Airlines historically perform at OGG compared with other airlines?”**

This historical evidence complements the AI probability rather than replacing it.

---

## Model development and validation

FlightRescue should not be treated as a finished airline-grade forecasting model.

The engineering pipeline is end-to-end and deployed, but **model quality is still an active research area**.

Current work includes:

- probability calibration;
- reliability analysis;
- airline-sensitivity testing;
- route-direction sensitivity testing;
- weather monotonicity checks;
- improving rare severe-event prediction;
- dual-airport weather modeling;
- temporal validation on held-out years;
- comparison between predicted probabilities and real observed event frequencies.

### Why calibration matters

A model may correctly rank one flight as riskier than another while still producing probabilities that are too high or too low.

For a passenger-facing product, a displayed value such as **40%** should ideally mean that approximately 40 out of 100 comparable historical cases experienced disruption.

For that reason, FlightRescue is actively moving toward calibrated probability estimates rather than relying only on ranking metrics such as ROC-AUC.

### Current model maturity

**Experimental / research prototype**

The UI explicitly labels the model as experimental. Predictions are decision-support information, not operational guarantees.

---

## Live application

Passenger application:

`https://nadia-mas.github.io/flight-disruption-ai/`

The frontend is deployed through GitHub Pages and communicates with a separately deployed FastAPI inference service on Vercel.

---

## Research pipeline

The original OGG research workflow consists of nine notebook stages:

1. `01_ogg_data_exploration.ipynb` — source loading and initial OGG exploration.
2. `02_ogg_phase1_eda.ipynb` — exploratory data analysis.
3. `03_build_modeling_table.ipynb` — flight + OGG weather + Hawaii storm-event alignment.
4. `04_validate_and_engineer_features.ipynb` — validation, leakage control, and feature engineering.
5. `05_baseline_disruption_models.ipynb` — multiclass baseline models.
6. `06_binary_and_severe_disruption_models.ipynb` — binary and severe-disruption modeling.
7. `07_historical_event_recovery_dataset.ipynb` — hourly operations and recovery episodes.
8. `08_historical_similar_event_retrieval.ipynb` — historical analog retrieval.
9. `09_flightrescue_inference_pipeline.ipynb` — passenger-facing inference integration.

Additional production scripts extend the notebook research into deployable artifacts, airline-event aggregation, smoke testing, calibration, and dual-airport model development.

---

## Data coverage

### BTS flight operations

- Primary airport focus: **OGG (Kahului Airport)**
- Historical research period: **2020–2025**
- Extended source download coverage includes 2026 where data are available.
- Phase 1 OGG flight records: **330,615**

### NOAA Local Climatological Data

- Original station: **Kahului Airport (OGG)**
- v3 development adds **LAX** and **DFW** airport weather observations.
- Historical model-development period: **2020–2025**

### NOAA Storm Events

- Geographic focus: **Hawaii**
- Used to identify hazardous-weather episodes and historical analogs.

---

## Dataset alignment

### Original OGG model

```text
OGG flight at time t
        +
OGG weather near time t
        +
Hawaii severe-weather context
        ↓
model-ready observation
```

### Current end-to-end passenger system

```text
Passenger flight
        ↓
Official weather forecast
        ↓
Machine-learning prediction
        +
Historical event similarity
        +
Historical airline behavior
        ↓
Passenger-facing risk assessment
```

### v3 research direction

```text
Origin schedule + origin weather
              +
Destination schedule + destination weather
              +
Airline + route + calendar features
              ↓
Calibrated disruption model
              ↓
Historical airline/event evidence
              ↓
Passenger decision-support output
```

---

## Project structure

- `api/` — Vercel serverless entry point
- `backend/` — FastAPI application
- `data/raw/` — generated/downloaded source data
- `data/processed/` — model-ready tables and compact production evidence
- `docs/` — public passenger frontend
- `models/` — trained and versioned inference artifacts
- `notebooks/` — research notebooks 01–09
- `scripts/` — data download, model export, aggregation, smoke testing, and v3 training
- `src/flightrescue/` — inference, weather, and application services
- `src/features/` — feature engineering
- `src/models/` — predictive / calibration utilities
- `src/preprocessing/` — data integration
- `src/retrieval/` — historical similarity engine
- `tests/` — automated tests
- `.github/workflows/` — CI, model builds, smoke tests, and deployment workflows

---

## API usage and permission

The FlightRescue API is provided for **research, demonstration, and evaluation purposes**.

Third-party use of the API — including integration into another application or website, automated access, redistribution, commercial use, or use as part of another service — requires **prior permission from the project owner**.

To request permission:

**Fatemeh (Nadia) Masoumi**  
**Email:** Fatemeh.masoumi.1994@gmail.com

Public availability of an endpoint should **not** be interpreted as permission for unrestricted third-party API use.

---

## Current development status

### Completed / deployed

- end-to-end passenger application
- BTS / NOAA data pipelines
- OGG disruption model artifacts
- two-stage severe-disruption model
- FastAPI inference backend
- Vercel production deployment
- GitHub Pages passenger frontend
- NWS forecast integration
- historical event retrieval
- airline event-performance aggregation
- airline comparison UI
- automated production smoke testing

### Active AI research

- calibrated probabilities
- better severe-event prediction
- dual-airport weather features
- OGG ↔ LAX modeling
- OGG ↔ DFW modeling
- reliability / calibration curves
- controlled direction and airline audits
- improved interpretability and passenger explanations

The goal is not merely to make the interface work. The goal is to continue improving whether the **probabilities themselves are scientifically reliable and useful to passengers**.

---

## Lala 2026 as the motivating use case

FlightRescue was shaped by the type of uncertainty passengers experienced during **Hurricane / Tropical Storm Lala in Hawaii in August 2026**.

Lala brought strong winds, extreme rainfall, flooding, power outages, transportation disruption, and significant aviation uncertainty across the islands. Contemporary reports described airport closures in Hawaii, more than 190 postponed flights, and continuing warnings for passengers to confirm flight operations with their airlines as the storm moved through the state.

FlightRescue asks what a traveler would naturally want to know in that situation:

> **If weather like this happened before, what did the airlines actually do?**

That question connects the historical-data component with the predictive-AI component of the project.

---

## Research-use notice

FlightRescue is an experimental research system and is **not an official airline, airport, FAA, NOAA, National Weather Service, or U.S. DOT product**.

Predictions, historical comparisons, and recovery estimates are informational research outputs and should not be treated as guarantees of flight status. Travelers should always confirm operational decisions with their airline and official aviation / weather sources.

---

## Project vision

The long-term vision is a system that can answer:

> **“Given my flight, my airline, the weather at both airports, and what happened during comparable historical events, what disruption risk should I realistically prepare for?”**

FlightRescue turns that question into an end-to-end data science, machine-learning, API, and user-experience project.
