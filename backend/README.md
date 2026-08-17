# FlightRescue AI Backend

The backend exposes the artifact-backed inference layer used by the public frontend.

## Local run

```bash
pip install -r requirements.txt
python scripts/export_inference_artifacts.py
uvicorn backend.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /health` — reports whether trained model artifacts and the similarity index are available.
- `POST /predict/features` — predicts any-disruption and severe-disruption probabilities from the leakage-safe feature dictionary created by Notebook 04.
- `POST /similar-events` — retrieves historical analogs from a standardized event vector.

The API deliberately returns HTTP 503 when trained model artifacts are absent. It does not substitute a hand-written risk heuristic.

## Deployment

The backend is container-ready via `backend/Dockerfile`. GitHub Pages can host only the static frontend; this FastAPI service must run on a Python/container host. Once a public API URL is available, set it in `docs/config.js` so the GitHub Pages frontend can call it.
