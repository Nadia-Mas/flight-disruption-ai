from __future__ import annotations

import numpy as np


class ConditionalSevereModel:
    """Compose P(any disruption) with P(severe | disruption).

    The wrapped estimators must expose ``predict_proba``. The returned positive
    probability is P(any) * P(severe | disruption), which is always bounded by
    the probability of any disruption.
    """

    def __init__(self, any_model, conditional_model):
        self.any_model = any_model
        self.conditional_model = conditional_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p_any = np.asarray(self.any_model.predict_proba(X))[:, 1]
        p_conditional = np.asarray(self.conditional_model.predict_proba(X))[:, 1]
        p_severe = np.clip(p_any * p_conditional, 0.0, 1.0)
        return np.column_stack([1.0 - p_severe, p_severe])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
