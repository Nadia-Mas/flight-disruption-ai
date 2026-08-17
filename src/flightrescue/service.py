from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ArtifactPaths:
    any_model: Path
    severe_model: Path
    metadata: Path
    similarity_index: Path


class FlightRescueService:
    """Load exported Notebook-09 artifacts and serve deterministic inference.

    The service intentionally refuses to invent predictions when artifacts are
    missing. That keeps the public app from silently falling back to a heuristic.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.paths = ArtifactPaths(
            any_model=self.root / "models/any_disruption_model.joblib",
            severe_model=self.root / "models/severe_disruption_model.joblib",
            metadata=self.root / "models/flightrescue_inference_metadata.json",
            similarity_index=self.root / "data/processed/ogg_event_similarity_index_2020_2025.csv",
        )
        self.any_model = None
        self.severe_model = None
        self.metadata: dict[str, Any] = {}
        self.similarity_index: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        if self.paths.metadata.exists():
            self.metadata = json.loads(self.paths.metadata.read_text())
        if self.paths.any_model.exists():
            self.any_model = joblib.load(self.paths.any_model)
        if self.paths.severe_model.exists():
            self.severe_model = joblib.load(self.paths.severe_model)
        if self.paths.similarity_index.exists():
            self.similarity_index = pd.read_csv(self.paths.similarity_index, low_memory=False)

    @property
    def ready(self) -> bool:
        return self.any_model is not None and self.severe_model is not None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "any_model": self.paths.any_model.exists(),
            "severe_model": self.paths.severe_model.exists(),
            "metadata": self.paths.metadata.exists(),
            "similarity_index": self.paths.similarity_index.exists(),
        }

    @staticmethod
    def _risk_label(p: float) -> str:
        if p >= 0.75:
            return "very_high"
        if p >= 0.55:
            return "high"
        if p >= 0.35:
            return "moderate"
        return "low"

    def predict_features(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError(
                "Model artifacts are not exported yet. Run scripts/export_inference_artifacts.py "
                "inside the project Codespace after generating the processed feature table."
            )

        frame = pd.DataFrame([features])
        p_any = float(self.any_model.predict_proba(frame)[:, 1][0])
        p_severe = float(self.severe_model.predict_proba(frame)[:, 1][0])

        any_threshold = float(self.metadata.get("any_disruption_threshold", 0.5))
        severe_threshold = float(self.metadata.get("severe_disruption_threshold", 0.5))

        return {
            "disruption_probability": p_any,
            "severe_disruption_probability": p_severe,
            "disruption_flag": bool(p_any >= any_threshold),
            "severe_disruption_flag": bool(p_severe >= severe_threshold),
            "risk_level": self._risk_label(p_any),
            "severe_risk_level": self._risk_label(p_severe),
            "thresholds": {
                "any_disruption": any_threshold,
                "severe_disruption": severe_threshold,
            },
            "model_status": "artifact_backed",
        }

    def similar_events_from_vector(self, vector: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        if self.similarity_index is None:
            return []

        z_cols = [c for c in self.similarity_index.columns if c.startswith("z__")]
        if not z_cols:
            return []

        query = np.array([[float(vector.get(c, 0.0)) for c in z_cols]])
        matrix = self.similarity_index[z_cols].fillna(0.0).to_numpy(dtype=float)
        scores = cosine_similarity(query, matrix).ravel()
        top = np.argsort(scores)[::-1][: max(1, min(k, len(scores)))]

        output_cols = [
            c for c in [
                "event_id", "start_dt", "end_dt", "event_types",
                "event_cancel_rate", "event_severe_rate",
                "recovery_hours_after_event",
            ] if c in self.similarity_index.columns
        ]
        result = self.similarity_index.iloc[top][output_cols].copy()
        result.insert(1 if output_cols else 0, "similarity_score", scores[top])
        return result.replace({np.nan: None}).to_dict(orient="records")
