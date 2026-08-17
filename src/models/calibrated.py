from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalibratedProbabilityModel:
    """Small joblib-safe wrapper around a classifier plus 1-D probability calibrator."""

    estimator: Any
    calibrator: Any
    feature_columns: list[str]

    def _raw_probability(self, X):
        p = np.asarray(self.estimator.predict_proba(X), dtype=float)
        return p[:, 1]

    def predict_proba(self, X):
        raw = self._raw_probability(X)
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
