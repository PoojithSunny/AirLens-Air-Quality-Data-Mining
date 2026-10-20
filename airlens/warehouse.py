from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR, NUMERIC_AIR_FEATURES


def build_warehouse(df: pd.DataFrame) -> Path:
    """Create a small star-schema-style SQLite warehouse for lab demonstration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / "airlens_warehouse.db"
    if db_path.exists():
        db_path.unlink()

    stations = pd.DataFrame({"station": sorted(df["station"].astype(str).unique())})
    stations.insert(0, "station_key", range(1, len(stations) + 1))

    times = df[["timestamp", "year", "month", "day", "hour", "day_of_week", "season"]].drop_duplicates("timestamp").copy()
    times = times.sort_values("timestamp").reset_index(drop=True)
    times.insert(0, "time_key", times["timestamp"].dt.strftime("%Y%m%d%H").astype("int64"))
    times["timestamp"] = times["timestamp"].astype(str)

    fact = df[["timestamp", "station"] + NUMERIC_AIR_FEATURES].copy()
    fact["timestamp"] = fact["timestamp"].astype(str)
    fact = fact.merge(stations, on="station", how="left")
    fact = fact.merge(times[["time_key", "timestamp"]], on="timestamp", how="left")
    fact = fact[["station_key", "time_key"] + NUMERIC_AIR_FEATURES]

    # Materialize an easy-to-query aggregate for OLAP-style station/month analysis.
    monthly = df.copy()
    monthly["year_month"] = monthly["timestamp"].dt.to_period("M").astype(str)
    summary = monthly.groupby(["station", "year_month"], as_index=False).agg(
        observation_count=("timestamp", "size"),
        mean_pm25=("PM2.5", "mean"),
        mean_pm10=("PM10", "mean"),
        mean_temperature=("TEMP", "mean"),
        mean_wind_speed=("WSPM", "mean")
    )

    with sqlite3.connect(db_path) as conn:
        stations.to_sql("dim_station", conn, index=False, if_exists="replace")
        times.to_sql("dim_time", conn, index=False, if_exists="replace", chunksize=5000)
        fact.to_sql("fact_air_quality", conn, index=False, if_exists="replace", chunksize=5000)
        summary.to_sql("station_month_summary", conn, index=False, if_exists="replace")
        conn.execute("CREATE INDEX idx_fact_station_time ON fact_air_quality(station_key, time_key)")
        conn.execute("CREATE INDEX idx_time_year_month ON dim_time(year, month)")
        conn.commit()

    print(f"SQLite warehouse created: {db_path}")
    print("Tables: dim_station, dim_time, fact_air_quality, station_month_summary")
    return db_path
