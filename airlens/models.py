from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import OUTPUT_DIR, NUMERIC_AIR_FEATURES


def _make_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create one-hour-ahead targets using only previous/current observations.

    A row is retained only when its next available observation for the same station
    is exactly one clock hour later. Lag features are also checked against timestamps.
    """
    data = df.sort_values(["station", "timestamp"]).copy()
    grouped = data.groupby("station", sort=False)

    data["target_timestamp"] = grouped["timestamp"].shift(-1)
    data["target_next_pm25"] = grouped["PM2.5"].shift(-1)

    for lag in (1, 3, 6, 24):
        lag_value = grouped["PM2.5"].shift(lag)
        lag_time = grouped["timestamp"].shift(lag)
        expected_delta = pd.Timedelta(hours=lag)
        data[f"PM25_lag_{lag}h"] = lag_value.where(
            (data["timestamp"] - lag_time) == expected_delta
        )

    # Time features describe the current observation, available at prediction time.
    data["current_hour"] = data["timestamp"].dt.hour
    data["current_month"] = data["timestamp"].dt.month
    data["current_day_of_week"] = data["timestamp"].dt.dayofweek

    # Do not claim to forecast one hour ahead if the next row is several hours later.
    exact_next_hour = data["target_timestamp"].eq(data["timestamp"] + pd.Timedelta(hours=1))
    data = data.loc[exact_next_hour & data["target_next_pm25"].notna()].copy()

    numeric_features = [
        col for col in NUMERIC_AIR_FEATURES if col in data.columns
    ] + ["PM25_lag_1h", "PM25_lag_3h", "PM25_lag_6h", "PM25_lag_24h",
         "current_hour", "current_month", "current_day_of_week"]
    categorical_features = [c for c in ["station", "wd", "season"] if c in data.columns]
    return data, numeric_features, categorical_features


def _build_pipeline(numeric_features: list[str], categorical_features: list[str], model):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])


def _write_json(path: Path, obj: dict) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return str(value)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=convert)


def train_forecasting_models(df: pd.DataFrame, max_train_rows: int = 90000) -> dict:
    """Train chronologically split regression/classification forecasting models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    supervised, numeric_features, categorical_features = _make_supervised(df)
    if len(supervised) < 200:
        raise ValueError(
            f"Only {len(supervised)} consecutive next-hour rows are available. "
            "Check that the correct hourly PRSA station CSVs were loaded."
        )

    unique_target_times = np.sort(supervised["target_timestamp"].dropna().unique())
    cutoff_index = min(int(0.80 * (len(unique_target_times) - 1)), len(unique_target_times) - 1)
    cutoff = pd.Timestamp(unique_target_times[cutoff_index])
    train_all = supervised.loc[supervised["target_timestamp"] <= cutoff].copy()
    test = supervised.loc[supervised["target_timestamp"] > cutoff].copy()
    if len(train_all) < 100 or len(test) < 50:
        raise ValueError("Not enough train/test examples after chronological split.")

    # Threshold is calculated from training targets only; no test-target leakage.
    high_threshold = float(train_all["target_next_pm25"].quantile(0.75))
    train = train_all
    if len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=42).sort_values("target_timestamp")

    feature_columns = numeric_features + categorical_features
    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()
    # Convert pandas nullable string values (pd.NA) into np.nan for sklearn imputers.
    for column in categorical_features:
        X_train[column] = X_train[column].astype(object).where(X_train[column].notna(), np.nan)
        X_test[column] = X_test[column].astype(object).where(X_test[column].notna(), np.nan)
    y_train_reg = train["target_next_pm25"].astype(float)
    y_test_reg = test["target_next_pm25"].astype(float)

    # Regression: Random Forest, with training-only imputation/encoding.
    reg = _build_pipeline(
        numeric_features, categorical_features,
        RandomForestRegressor(
            n_estimators=80, max_depth=18, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        )
    )
    reg.fit(X_train, y_train_reg)
    reg_pred = reg.predict(X_test)

    reg_metrics = {
        "test_rows": int(len(test)),
        "mae": float(mean_absolute_error(y_test_reg, reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_reg, reg_pred))),
        "r2": float(r2_score(y_test_reg, reg_pred)),
        "baseline_persistence_mae": None,
        "baseline_persistence_rmse": None,
    }
    persistence_mask = test["PM2.5"].notna()
    if persistence_mask.any():
        persistence_pred = test.loc[persistence_mask, "PM2.5"].astype(float)
        persistence_actual = y_test_reg.loc[persistence_mask]
        reg_metrics["baseline_persistence_mae"] = float(mean_absolute_error(persistence_actual, persistence_pred))
        reg_metrics["baseline_persistence_rmse"] = float(np.sqrt(mean_squared_error(persistence_actual, persistence_pred)))

    # Classification: above the training-set 75th percentile = high next-hour PM2.5.
    y_train_cls = (y_train_reg > high_threshold).astype(int)
    y_test_cls = (y_test_reg > high_threshold).astype(int)
    clf = _build_pipeline(
        numeric_features, categorical_features,
        RandomForestClassifier(
            n_estimators=80, max_depth=18, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
    )
    clf.fit(X_train, y_train_cls)
    cls_pred = clf.predict(X_test)

    cls_metrics = {
        "test_rows": int(len(test)),
        "high_pm25_threshold_from_training_75th_percentile": high_threshold,
        "accuracy": float(accuracy_score(y_test_cls, cls_pred)),
        "precision": float(precision_score(y_test_cls, cls_pred, zero_division=0)),
        "recall": float(recall_score(y_test_cls, cls_pred, zero_division=0)),
        "f1": float(f1_score(y_test_cls, cls_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test_cls, cls_pred, labels=[0, 1], target_names=["Not high", "High"],
            zero_division=0, output_dict=True
        ),
    }

    # Baseline classifier predicts the most common training class.
    baseline_clf = DummyClassifier(strategy="most_frequent")
    baseline_clf.fit(np.zeros((len(y_train_cls), 1)), y_train_cls)
    baseline_cls_pred = baseline_clf.predict(np.zeros((len(y_test_cls), 1)))
    cls_metrics["baseline_most_frequent_accuracy"] = float(accuracy_score(y_test_cls, baseline_cls_pred))

    joblib.dump({"model": reg, "numeric_features": numeric_features,
                 "categorical_features": categorical_features}, OUTPUT_DIR / "pm25_next_hour_regressor.joblib")
    joblib.dump({"model": clf, "numeric_features": numeric_features,
                 "categorical_features": categorical_features,
                 "high_threshold": high_threshold}, OUTPUT_DIR / "pm25_high_pollution_classifier.joblib")

    # Prediction output supports the dashboard and report.
    prediction_frame = test[["timestamp", "target_timestamp", "station", "PM2.5", "target_next_pm25"]].copy()
    prediction_frame = prediction_frame.rename(columns={
        "PM2.5": "current_pm25", "target_next_pm25": "actual_next_pm25"
    })
    prediction_frame["predicted_next_pm25"] = reg_pred
    prediction_frame["actual_high_pollution"] = y_test_cls.to_numpy()
    prediction_frame["predicted_high_pollution"] = cls_pred
    prediction_frame.to_csv(OUTPUT_DIR / "forecast_predictions.csv", index=False)

    # Regression plot uses a deterministic subset to keep the graphic legible.
    plot_n = min(10000, len(y_test_reg))
    plot_idx = np.random.default_rng(42).choice(len(y_test_reg), size=plot_n, replace=False)
    actual_sample = y_test_reg.iloc[plot_idx].to_numpy()
    pred_sample = np.asarray(reg_pred)[plot_idx]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual_sample, pred_sample, alpha=0.25, s=12)
    upper = float(max(np.max(actual_sample), np.max(pred_sample)))
    ax.plot([0, upper], [0, upper], linestyle="--")
    ax.set_title("Next-Hour PM2.5: Actual vs Predicted")
    ax.set_xlabel("Actual PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_test_cls, cls_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not high", "High"], yticklabels=["Not high", "High"], ax=ax)
    ax.set_title("Next-Hour High-PM2.5 Classification")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {
        "project": "AirLens",
        "forecast_horizon": "one hour",
        "chronological_cutoff_target_timestamp": str(cutoff),
        "training_rows_used": int(len(train)),
        "training_rows_before_cap": int(len(train_all)),
        "test_rows": int(len(test)),
        "regression": reg_metrics,
        "classification": cls_metrics,
        "caveat": "Results are historical validation on Beijing data, not a live public-health warning system. Compare with baselines and avoid causal claims."
    }
    _write_json(OUTPUT_DIR / "model_metrics.json", metrics)

    print("\n--- NEXT-HOUR FORECAST RESULTS ---")
    print(f"Chronological cutoff (target timestamp): {cutoff}")
    print(f"Train rows used: {len(train):,}; test rows: {len(test):,}")
    print(f"Regression MAE: {reg_metrics['mae']:.3f}; RMSE: {reg_metrics['rmse']:.3f}; R²: {reg_metrics['r2']:.3f}")
    print(f"Persistence-baseline MAE: {reg_metrics['baseline_persistence_mae']}")
    print(f"Classification accuracy: {cls_metrics['accuracy']:.3f}; F1: {cls_metrics['f1']:.3f}")
    print("Model metrics saved to outputs/model_metrics.json")
    return metrics
