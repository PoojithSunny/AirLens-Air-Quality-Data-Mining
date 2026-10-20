from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_observations.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(page_title="AirLens | Air Quality Data Mining", page_icon="🌫️", layout="wide")
st.title("🌫️ AirLens")
st.caption("Multi-station air pollution pattern mining and next-hour analytics")
st.markdown(
    "AirLens explores historical observations from the UCI Beijing Multi-Site Air Quality dataset. "
    "It is an educational analytics project, **not a live air-quality alert or health advisory system**."
)

if not DATA_PATH.exists():
    st.warning("Processed dataset not found yet. Follow README.md, place the official UCI ZIP in data/raw, then run `python run_pipeline.py`.")
    st.stop()

@st.cache_data(show_spinner="Loading cleaned observations...")
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    data["station"] = data["station"].astype(str)
    return data

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not read cleaned data: {exc}")
    st.stop()

stations = sorted(df["station"].dropna().unique().tolist())
with st.sidebar:
    st.header("Filters")
    selected_stations = st.multiselect("Monitoring stations", stations, default=stations[:min(3, len(stations))])
    pollutant_columns = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if c in df.columns]
    pollutant = st.selectbox("Pollutant", pollutant_columns, index=0 if "PM2.5" in pollutant_columns else 0)
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[df["station"].isin(selected_stations)].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[filtered["timestamp"].dt.date.between(start_date, end_date)]

if filtered.empty:
    st.info("No rows match these filters. Select at least one station and a wider date range.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(filtered):,}")
c2.metric("Stations selected", f"{filtered['station'].nunique():,}")
c3.metric(f"Mean {pollutant}", "—" if filtered[pollutant].notna().sum() == 0 else f"{filtered[pollutant].mean():,.2f}")
c4.metric(f"Median {pollutant}", "—" if filtered[pollutant].notna().sum() == 0 else f"{filtered[pollutant].median():,.2f}")

overview_tab, mining_tab, clusters_tab, forecast_tab, warehouse_tab = st.tabs([
    "Overview", "Association Mining", "Clusters", "Forecast Evaluation", "Data Warehouse"
])

with overview_tab:
    st.subheader(f"{pollutant} trends")
    trend = filtered.dropna(subset=[pollutant]).set_index("timestamp").groupby("station")[pollutant].resample("D").mean().reset_index()
    if not trend.empty:
        chart_data = trend.pivot(index="timestamp", columns="station", values=pollutant)
        st.line_chart(chart_data, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Mean by station")
        st.bar_chart(filtered.groupby("station")[pollutant].mean().sort_values())
    with right:
        st.subheader("Mean by hour of day")
        st.bar_chart(filtered.groupby("hour")[pollutant].mean().sort_index())
    st.subheader("Filtered records")
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "airlens_filtered_observations.csv", "text/csv")

with mining_tab:
    st.subheader("Apriori association rules")
    rules = load_csv(str(OUTPUT_DIR / "association_rules.csv"))
    frequent = load_csv(str(OUTPUT_DIR / "frequent_itemsets.csv"))
    if rules.empty:
        st.info("No rules were exported, or the pipeline has not yet created them. Run `python run_pipeline.py`.")
    else:
        if "lift" in rules.columns:
            min_lift = st.slider("Minimum lift", 1.0, max(3.0, float(rules["lift"].max())), 1.0, 0.1)
            rules_view = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)
        else:
            rules_view = rules
        st.write("Support = frequency of a pattern; confidence = conditional frequency; lift > 1 = positive association versus independence.")
        st.dataframe(rules_view.head(100), use_container_width=True)
    with st.expander("Frequent itemsets"):
        st.dataframe(frequent.head(100), use_container_width=True)

with clusters_tab:
    st.subheader("Pollution regimes")
    profiles = load_csv(str(OUTPUT_DIR / "cluster_profiles.csv"))
    assignments = load_csv(str(OUTPUT_DIR / "cluster_assignments.csv"))
    if profiles.empty or assignments.empty:
        st.info("Cluster outputs are missing. Run `python run_pipeline.py`.")
    else:
        st.caption("Cluster numbers are arbitrary; interpret them using the measured profiles rather than their IDs.")
        st.dataframe(profiles, use_container_width=True)
        assignments["timestamp"] = pd.to_datetime(assignments["timestamp"], errors="coerce")
        assignments = assignments[assignments["station"].isin(selected_stations)]
        if not assignments.empty:
            st.bar_chart(assignments["cluster"].value_counts().sort_index())
            st.dataframe(assignments.groupby("cluster").agg(
                observations=("station", "size"),
                median_pm25=("PM2.5", "median"),
                mean_pm10=("PM10", "mean"),
                mean_wind_speed=("WSPM", "mean")
            ).round(2), use_container_width=True)

with forecast_tab:
    st.subheader("One-hour-ahead PM2.5 model evaluation")
    metrics = load_json(str(OUTPUT_DIR / "model_metrics.json"))
    if not metrics:
        st.info("Forecast metrics are missing. Run `python run_pipeline.py`.")
    else:
        st.caption(f"Chronological cutoff: {metrics.get('chronological_cutoff_target_timestamp', 'n/a')}. The test period is later than the training period.")
        reg = metrics.get("regression", {})
        cls = metrics.get("classification", {})
        a, b, c, d = st.columns(4)
        a.metric("Regression MAE", f"{reg.get('mae', float('nan')):.2f} µg/m³")
        b.metric("Regression RMSE", f"{reg.get('rmse', float('nan')):.2f} µg/m³")
        c.metric("High-pollution F1", f"{cls.get('f1', float('nan')):.3f}")
        d.metric("Classification accuracy", f"{cls.get('accuracy', float('nan')):.3f}")
        st.write("Threshold used for the classification target (training 75th percentile):", cls.get("high_pm25_threshold_from_training_75th_percentile"))
        st.write("Persistence baseline MAE:", reg.get("baseline_persistence_mae"))
        st.write("A model should be compared with its baseline; higher accuracy alone does not guarantee a useful forecast.")
        st.image(str(OUTPUT_DIR / "forecast_actual_vs_predicted.png"), caption="Actual vs predicted next-hour PM2.5", use_container_width=True) if (OUTPUT_DIR / "forecast_actual_vs_predicted.png").exists() else None
        st.image(str(OUTPUT_DIR / "forecast_confusion_matrix.png"), caption="High PM2.5 classification confusion matrix", use_container_width=True) if (OUTPUT_DIR / "forecast_confusion_matrix.png").exists() else None
        pred = load_csv(str(OUTPUT_DIR / "forecast_predictions.csv"))
        if not pred.empty:
            st.dataframe(pred.head(1000), use_container_width=True)
            st.download_button("Download test predictions", pred.to_csv(index=False).encode("utf-8"), "airlens_test_predictions.csv", "text/csv")

with warehouse_tab:
    st.subheader("SQLite analytical warehouse")
    db_path = OUTPUT_DIR / "airlens_warehouse.db"
    if not db_path.exists():
        st.info("Warehouse missing. Run `python run_pipeline.py`.")
    else:
        st.success("Warehouse file found: outputs/airlens_warehouse.db")
        try:
            with sqlite3.connect(db_path) as conn:
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
                st.dataframe(tables, use_container_width=True)
                st.markdown("Example OLAP-style query: monthly mean PM2.5 by station")
                query = """
                SELECT s.station, t.year, t.month, AVG(f."PM2.5") AS mean_pm25, COUNT(*) AS observations
                FROM fact_air_quality f
                JOIN dim_station s ON f.station_key = s.station_key
                JOIN dim_time t ON f.time_key = t.time_key
                GROUP BY s.station, t.year, t.month
                ORDER BY t.year, t.month, s.station
                LIMIT 200
                """
                st.dataframe(pd.read_sql_query(query, conn), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not query the warehouse: {exc}")

st.divider()
st.caption("AirLens · Educational data-mining project · Source: UCI Beijing Multi-Site Air Quality dataset")
