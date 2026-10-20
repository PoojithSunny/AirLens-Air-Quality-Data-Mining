from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, OUTPUT_DIR, NUMERIC_AIR_FEATURES


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean schema, timestamps, invalid sensor values and duplicate observations.

    Missing measurements are deliberately retained. Imputation for predictive models
    is fitted only on the training partition to avoid temporal leakage.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "station" not in df.columns:
        raise ValueError("Dataset is missing the 'station' column.")

    date_columns = {"year", "month", "day", "hour"}
    if not date_columns.issubset(df.columns):
        raise ValueError("Dataset must contain year, month, day, and hour columns.")

    # Convert expected measurement variables to numbers (bad text becomes missing).
    for column in NUMERIC_AIR_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    for column in ("year", "month", "day", "hour"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"],
         "day": df["day"], "hour": df["hour"]},
        errors="coerce"
    )
    df["station"] = df["station"].astype("string").str.strip()
    df.loc[df["station"].isin(["", "<NA>", "nan", "None"]), "station"] = pd.NA
    df = df.dropna(subset=["timestamp", "station"]).copy()

    before_dedupe = len(df)
    df = df.sort_values(["station", "timestamp"])
    df = df.drop_duplicates(subset=["station", "timestamp"], keep="first").copy()
    duplicates_removed = before_dedupe - len(df)

    # Negative concentrations and negative wind speed are invalid sensor readings.
    pollutant_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    for column in pollutant_columns:
        df.loc[df[column] < 0, column] = np.nan
    df.loc[df["WSPM"] < 0, "WSPM"] = np.nan

    df["year"] = df["timestamp"].dt.year.astype("int16")
    df["month"] = df["timestamp"].dt.month.astype("int8")
    df["day"] = df["timestamp"].dt.day.astype("int8")
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
    df["season"] = df["month"].map(_season)
    df["wd"] = df.get("wd", pd.Series(index=df.index, dtype="string")).astype("string")

    # Store a consistent, compact column order.
    keep = ["timestamp", "station", "year", "month", "day", "hour",
            "day_of_week", "season", "wd"] + NUMERIC_AIR_FEATURES
    df = df[keep].sort_values(["timestamp", "station"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "cleaned_observations.csv", index=False)

    missingness = df.isna().sum().rename("missing_count").to_frame()
    missingness["missing_percent"] = (missingness["missing_count"] / max(len(df), 1) * 100).round(3)
    missingness.sort_values("missing_percent", ascending=False).to_csv(
        OUTPUT_DIR / "missingness_report.csv"
    )

    preprocessing_summary = {
        "input_rows": int(len(raw)),
        "rows_after_timestamp_and_station_validation": int(before_dedupe),
        "duplicate_station_timestamp_rows_removed": int(duplicates_removed),
        "cleaned_rows": int(len(df)),
        "stations": sorted(df["station"].dropna().astype(str).unique().tolist()),
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
        "missing_values_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "note": "Missing sensor readings are retained. Predictive model imputation is fitted on training rows only."
    }
    with open(OUTPUT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing_summary, f, indent=2)

    print("\n--- PREPROCESSING SUMMARY ---")
    print(f"Input records: {len(raw):,}")
    print(f"Duplicate station/time records removed: {duplicates_removed:,}")
    print(f"Cleaned records: {len(df):,}")
    print(f"Stations: {df['station'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved: {PROCESSED_DIR / 'cleaned_observations.csv'}")
    return df
