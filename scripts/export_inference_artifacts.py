from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flightrescue.composite import ConditionalSevereModel  # noqa: E402

FEATURE_FILE = ROOT / "data/processed/ogg_model_features_v1.csv.gz"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Missing {FEATURE_FILE}. Run Notebooks 03-04 first."
    )

df = pd.read_csv(FEATURE_FILE, low_memory=False)
if "split" not in df.columns:
    raise KeyError("Expected temporal 'split' column from Notebook 04.")

train = df[df["split"].eq("train")].copy()
val = df[df["split"].eq("validation")].copy()
test = df[df["split"].eq("test")].copy()

for part in [train, val, test]:
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


def candidate_models():
    return {
        "logistic_regression": make_logreg(),
        "hist_gradient_boosting": make_histgb(),
    }


def choose_threshold(y, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.025, 0.951, 0.025):
        score = f1_score(y, prob >= t, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def probability_metrics(y, prob, threshold):
    out = {
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "f1": float(f1_score(y, prob >= threshold, zero_division=0)),
    }
    if pd.Series(y).nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    else:
        out["roc_auc"] = None
    return out


def select_any_model():
    X_train = train[feature_cols]
    X_val = val[feature_cols]
    y_train = train["target_any_disruption"]
    y_val = val["target_any_disruption"]

    evaluated = []
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = choose_threshold(y_val, prob)
        evaluated.append({
            "name": name,
            "model": model,
            "prob": prob,
            "threshold": threshold,
            "pr_auc": float(average_precision_score(y_val, prob)),
            "f1": val_f1,
        })

    best = max(evaluated, key=lambda x: x["pr_auc"])
    trainval = pd.concat([train, val], ignore_index=True)
    final = make_logreg() if best["name"] == "logistic_regression" else make_histgb()
    final.fit(trainval[feature_cols], trainval["target_any_disruption"])
    return final, best


def select_conditional_severe_model(any_val_prob):
    disrupted_train = train[train["target_any_disruption"].eq(1)].copy()
    X_val = val[feature_cols]
    y_val_severe = val["target_severe"]

    evaluated = []
    for name, model in candidate_models().items():
        model.fit(disrupted_train[feature_cols], disrupted_train["target_severe"])
        p_cond = model.predict_proba(X_val)[:, 1]
        p_severe = np.clip(any_val_prob * p_cond, 0.0, 1.0)
        threshold, val_f1 = choose_threshold(y_val_severe, p_severe)
        evaluated.append({
            "name": name,
            "model": model,
            "conditional_prob": p_cond,
            "prob": p_severe,
            "threshold": threshold,
            "pr_auc": float(average_precision_score(y_val_severe, p_severe)),
            "f1": val_f1,
        })

    best = max(evaluated, key=lambda x: x["pr_auc"])
    trainval = pd.concat([train, val], ignore_index=True)
    disrupted_trainval = trainval[trainval["target_any_disruption"].eq(1)].copy()
    final_conditional = make_logreg() if best["name"] == "logistic_regression" else make_histgb()
    final_conditional.fit(disrupted_trainval[feature_cols], disrupted_trainval["target_severe"])
    return final_conditional, best, len(disrupted_train), len(disrupted_trainval)


any_model, any_best = select_any_model()
conditional_model, severe_best, n_cond_train, n_cond_trainval = select_conditional_severe_model(any_best["prob"])
severe_model = ConditionalSevereModel(any_model=any_model, conditional_model=conditional_model)

any_threshold = float(any_best["threshold"])
severe_threshold = float(severe_best["threshold"])

X_test = test[feature_cols]
y_test_any = test["target_any_disruption"]
y_test_severe = test["target_severe"]
test_prob_any = any_model.predict_proba(X_test)[:, 1]
test_prob_severe = severe_model.predict_proba(X_test)[:, 1]

any_val_metrics = probability_metrics(val["target_any_disruption"], any_best["prob"], any_threshold)
severe_val_metrics = probability_metrics(val["target_severe"], severe_best["prob"], severe_threshold)
any_test_metrics = probability_metrics(y_test_any, test_prob_any, any_threshold)
severe_test_metrics = probability_metrics(y_test_severe, test_prob_severe, severe_threshold)

joblib.dump(any_model, MODEL_DIR / "any_disruption_model.joblib")
joblib.dump(severe_model, MODEL_DIR / "severe_disruption_model.joblib")

metadata = {
    "artifact_version": "research-v2-two-stage-severe",
    "feature_columns": feature_cols,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "any_disruption_model": any_best["name"],
    "any_disruption_threshold": any_threshold,
    "any_validation_pr_auc": any_val_metrics["pr_auc"],
    "any_validation_f1": any_val_metrics["f1"],
    "any_validation_brier": any_val_metrics["brier"],
    "any_validation_roc_auc": any_val_metrics["roc_auc"],
    "any_test_pr_auc": any_test_metrics["pr_auc"],
    "any_test_f1": any_test_metrics["f1"],
    "any_test_brier": any_test_metrics["brier"],
    "any_test_roc_auc": any_test_metrics["roc_auc"],
    "severe_disruption_model": f"two_stage:{severe_best['name']}",
    "severe_probability_mode": "P(any disruption) * P(severe | disruption)",
    "severe_disruption_threshold": severe_threshold,
    "severe_validation_pr_auc": severe_val_metrics["pr_auc"],
    "severe_validation_f1": severe_val_metrics["f1"],
    "severe_validation_brier": severe_val_metrics["brier"],
    "severe_validation_roc_auc": severe_val_metrics["roc_auc"],
    "severe_test_pr_auc": severe_test_metrics["pr_auc"],
    "severe_test_f1": severe_test_metrics["f1"],
    "severe_test_brier": severe_test_metrics["brier"],
    "severe_test_roc_auc": severe_test_metrics["roc_auc"],
    "conditional_severe_training_rows_train": n_cond_train,
    "conditional_severe_training_rows_trainval": n_cond_trainval,
    "severe_positive_rate_validation": float(val["target_severe"].mean()),
    "severe_positive_rate_test": float(test["target_severe"].mean()),
    "training_policy": (
        "architectures and thresholds selected on 2024 validation; final estimators refit on 2020-2024; "
        "2025 test metrics are reported without model or threshold selection on test"
    ),
}
(MODEL_DIR / "flightrescue_inference_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Saved inference artifacts to", MODEL_DIR)
print(json.dumps(metadata, indent=2))
