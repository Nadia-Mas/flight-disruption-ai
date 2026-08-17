from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "data/processed/ogg_model_features_v1.csv.gz"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Missing {FEATURE_FILE}. Run Notebooks 03-04 first in the Codespace."
    )

df = pd.read_csv(FEATURE_FILE, low_memory=False)
if "split" not in df.columns:
    raise KeyError("Expected temporal 'split' column from Notebook 04.")

train = df[df["split"].eq("train")].copy()
val = df[df["split"].eq("validation")].copy()

for part in [train, val]:
    part["target_any_disruption"] = (part["disruption_class"] != "normal").astype(int)
    part["target_severe"] = part["disruption_class"].isin(["severe_delay", "cancelled"]).astype(int)

exclude = {
    "disruption_class", "target_any_disruption", "target_severe", "split",
    "FlightDate", "ogg_sched_dt", "weather_dt", "Cancelled", "CancellationCode",
    "Diverted", "DepTime", "ArrTime", "DepDelay", "ArrDelay", "DepDelayMinutes",
    "ArrDelayMinutes", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
}
feature_cols = [c for c in train.columns if c not in exclude and not train[c].isna().all()]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

linear_preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

tree_preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), categorical_cols),
])


def make_logreg():
    return Pipeline([
        ("preprocess", linear_preprocess),
        ("model", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            solver="lbfgs", tol=1e-4,
        )),
    ])


def make_histgb():
    return Pipeline([
        ("preprocess", tree_preprocess),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            class_weight="balanced", random_state=42,
        )),
    ])


def choose_threshold(y, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.025):
        score = f1_score(y, prob >= t, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def select_and_fit(target: str):
    X_train = train[feature_cols]
    X_val = val[feature_cols]
    y_train = train[target]
    y_val = val[target]

    candidates = {
        "logistic_regression": make_logreg(),
        "hist_gradient_boosting": make_histgb(),
    }
    evaluated = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = choose_threshold(y_val, prob)
        evaluated.append((average_precision_score(y_val, prob), name, model, threshold, val_f1))

    evaluated.sort(reverse=True, key=lambda x: x[0])
    pr_auc, name, model, threshold, val_f1 = evaluated[0]

    # Refit the selected architecture on train + validation after model/threshold selection.
    trainval = pd.concat([train, val], ignore_index=True)
    y_trainval = trainval[target]
    final_model = make_logreg() if name == "logistic_regression" else make_histgb()
    final_model.fit(trainval[feature_cols], y_trainval)
    return final_model, name, threshold, float(pr_auc), float(val_f1)


any_model, any_name, any_threshold, any_pr, any_f1 = select_and_fit("target_any_disruption")
severe_model, severe_name, severe_threshold, severe_pr, severe_f1 = select_and_fit("target_severe")

joblib.dump(any_model, MODEL_DIR / "any_disruption_model.joblib")
joblib.dump(severe_model, MODEL_DIR / "severe_disruption_model.joblib")

metadata = {
    "feature_columns": feature_cols,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "any_disruption_model": any_name,
    "any_disruption_threshold": any_threshold,
    "any_validation_pr_auc": any_pr,
    "any_validation_f1": any_f1,
    "severe_disruption_model": severe_name,
    "severe_disruption_threshold": severe_threshold,
    "severe_validation_pr_auc": severe_pr,
    "severe_validation_f1": severe_f1,
    "training_policy": "model and threshold selected on 2024 validation; final estimator refit on 2020-2024",
}
(MODEL_DIR / "flightrescue_inference_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved inference artifacts to", MODEL_DIR)
print(json.dumps(metadata, indent=2))
