from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR, NUMERIC_AIR_FEATURES, POLLUTANTS

sns.set_theme(style="whitegrid")


def exploratory_analysis(df: pd.DataFrame) -> None:
    """Generate descriptive statistics and reproducible EDA figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(df)),
        "stations": int(df["station"].nunique()),
        "timestamp_start": str(df["timestamp"].min()),
        "timestamp_end": str(df["timestamp"].max()),
        "mean_pm25": None if df["PM2.5"].notna().sum() == 0 else float(df["PM2.5"].mean()),
        "median_pm25": None if df["PM2.5"].notna().sum() == 0 else float(df["PM2.5"].median()),
        "mean_pm10": None if df["PM10"].notna().sum() == 0 else float(df["PM10"].mean()),
    }
    with open(OUTPUT_DIR / "dataset_profile.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # 1. Missing values
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    missing[missing > 0].plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing observations (%)")
    ax.set_xlabel("Column")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=150)
    plt.close(fig)

    # 2. Station-level PM2.5, sorted by mean concentration
    station_pm = df.groupby("station")["PM2.5"].mean().dropna().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    station_pm.plot(kind="barh", ax=ax)
    ax.set_title("Mean PM2.5 by Monitoring Station")
    ax.set_xlabel("Mean PM2.5 (µg/m³)")
    ax.set_ylabel("Station")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "station_pm25.png", dpi=150)
    plt.close(fig)

    # 3. Monthly series pooled across stations. Missing PM2.5 is ignored by mean().
    temp = df[["timestamp", "PM2.5"]].dropna().set_index("timestamp").sort_index()
    monthly = temp["PM2.5"].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly.plot(ax=ax, marker="o", markersize=3)
    ax.set_title("Monthly Mean PM2.5 Across Stations")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_pm25.png", dpi=150)
    plt.close(fig)
    monthly.rename("mean_pm25").to_csv(OUTPUT_DIR / "monthly_pm25_summary.csv")

    # 4. Hour of day by month heatmap
    heat_df = df.dropna(subset=["PM2.5"]).groupby(["month", "hour"])["PM2.5"].mean().unstack("hour")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_df, cmap="magma", ax=ax)
    ax.set_title("Mean PM2.5 by Month and Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "month_hour_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. Correlation heatmap for numeric observations
    numeric = df[NUMERIC_AIR_FEATURES].dropna(axis=1, how="all")
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Among Pollutants and Weather Variables")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print("\nEDA completed. Charts are saved in outputs/.")
    print("Dataset profile:", summary)


def run_clustering(df: pd.DataFrame, clusters: int = 4, sample_limit: int = 50000) -> pd.DataFrame:
    """Fit K-Means on a sample, then assign cluster IDs to every observation."""
    features = [col for col in NUMERIC_AIR_FEATURES if col in df.columns and df[col].notna().any()]
    if len(df) < clusters * 5:
        raise ValueError("Not enough observations for clustering. Check the input dataset.")
    fit_rows = df.sample(n=min(sample_limit, len(df)), random_state=42)
    X_fit = fit_rows[features]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=clusters, random_state=42, n_init=10, max_iter=300)),
    ])
    pipeline.fit(X_fit)

    # Predict labels for all records using the fitted sample model.
    labeled = df[["timestamp", "station", "PM2.5", "PM10", "TEMP", "WSPM", "season"]].copy()
    labeled["cluster"] = pipeline.predict(df[features])
    labeled.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    joblib_path = OUTPUT_DIR / "pollution_clusters.joblib"
    import joblib
    joblib.dump({"pipeline": pipeline, "features": features}, joblib_path)

    profile_columns = [c for c in ["PM2.5", "PM10", "NO2", "CO", "TEMP", "WSPM", "RAIN"] if c in df.columns]
    profiles = df[profile_columns].copy()
    profiles["cluster"] = labeled["cluster"].values
    profile = profiles.groupby("cluster").agg(["mean", "median", "count"]).round(2)
    profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv")

    compact = df[["PM2.5", "PM10", "TEMP", "WSPM"]].copy()
    compact["cluster"] = labeled["cluster"].values
    compact_profile = compact.groupby("cluster").mean(numeric_only=True).round(2)
    fig, ax = plt.subplots(figsize=(10, 6))
    compact_profile[[c for c in ["PM2.5", "PM10", "TEMP", "WSPM"] if c in compact_profile.columns]].plot(kind="bar", ax=ax)
    ax.set_title("Mean Conditions by K-Means Cluster")
    ax.set_xlabel("Cluster ID (arbitrary label)")
    ax.set_ylabel("Mean value in original units")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_profiles.png", dpi=150)
    plt.close(fig)

    # Human-readable ranking to avoid interpreting numeric cluster labels as an order.
    pm25_by_cluster = labeled.groupby("cluster")["PM2.5"].median().sort_values()
    names = {int(cluster_id): f"Regime {rank + 1}" for rank, cluster_id in enumerate(pm25_by_cluster.index)}
    with open(OUTPUT_DIR / "cluster_interpretation.txt", "w", encoding="utf-8") as handle:
        handle.write("Cluster IDs are arbitrary. Regime 1 has the lowest median PM2.5 among fitted clusters.\n\n")
        for cluster_id in pm25_by_cluster.index:
            handle.write(f"Cluster {cluster_id} -> {names[int(cluster_id)]}; median PM2.5={pm25_by_cluster.loc[cluster_id]:.2f}\n")
    print("K-Means clustering completed; see cluster_profiles.csv and cluster_interpretation.txt.")
    return labeled


def _tier(series: pd.Series, q1: float, q2: float, labels: tuple[str, str, str]) -> pd.Series:
    return pd.Series(np.select([series <= q1, series <= q2], [labels[0], labels[1]], default=labels[2]), index=series.index)


def run_association_mining(df: pd.DataFrame, sample_limit: int = 40000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine co-occurring pollution/weather categories with Apriori."""
    required = ["PM2.5", "PM10", "TEMP", "WSPM", "season", "station"]
    subset = df[required].copy()
    subset = subset.sample(n=min(sample_limit, len(subset)), random_state=42).reset_index(drop=True)

    def categorical_tier(col: str, labels: tuple[str, str, str]) -> pd.Series:
        valid = subset[col].dropna()
        if valid.empty:
            return pd.Series("Unknown", index=subset.index, dtype="string")
        q1, q2 = valid.quantile([0.33, 0.67]).tolist()
        values = _tier(subset[col], q1, q2, labels).astype("string")
        return values.fillna("Unknown")

    categories = pd.DataFrame(index=subset.index)
    categories["PM2.5"] = categorical_tier("PM2.5", ("Low", "Medium", "High"))
    categories["PM10"] = categorical_tier("PM10", ("Low", "Medium", "High"))
    categories["Temperature"] = categorical_tier("TEMP", ("Cool", "Mild", "Warm"))
    categories["Wind"] = categorical_tier("WSPM", ("Low", "Moderate", "High"))
    categories["Season"] = subset["season"].fillna("Unknown").astype("string")
    categories["Station"] = subset["station"].fillna("Unknown").astype("string")

    # Each row is a single station-hour transaction; each categorical attribute becomes one item.
    # Build a transaction matrix whose columns are clean, human-readable items.
    item_columns: dict[str, pd.Series] = {}
    for column in categories.columns:
        for value in categories[column].dropna().astype(str).unique():
            item_columns[f"{column}={value}"] = categories[column].astype(str).eq(value)
    basket = pd.DataFrame(item_columns, index=categories.index, dtype=bool)
    # Do not retain missing/invalid sensor transactions as if they were genuine patterns.
    valid_sensor_row = subset[["PM2.5", "PM10", "TEMP", "WSPM"]].notna().all(axis=1)
    basket = basket.loc[valid_sensor_row]
    if basket.empty:
        empty = pd.DataFrame(columns=["itemsets", "support"])
        empty.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)
        pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"]).to_csv(
            OUTPUT_DIR / "association_rules.csv", index=False
        )
        return empty, pd.DataFrame()

    frequent = apriori(basket, min_support=0.03, use_colnames=True, max_len=3, low_memory=True)
    frequent["itemsets_text"] = frequent["itemsets"].apply(lambda x: " AND ".join(sorted(x)))
    frequent.drop(columns=["itemsets"]).to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)

    if frequent.empty or frequent["itemsets"].map(len).max() < 2:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    else:
        rules = association_rules(frequent[["support", "itemsets"]], num_itemsets=len(basket), metric="confidence", min_threshold=0.45)
        if not rules.empty:
            rules["antecedents"] = rules["antecedents"].apply(lambda x: " AND ".join(sorted(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: " AND ".join(sorted(x)))
            rules = rules.sort_values(["lift", "confidence"], ascending=False)
            rules = rules[rules["lift"] > 1.0]
            cols = [c for c in ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"] if c in rules.columns]
            rules = rules[cols]
        else:
            rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
    print(f"Apriori completed: {len(frequent)} frequent itemsets, {len(rules)} positive-lift rules.")
    return frequent, rules
