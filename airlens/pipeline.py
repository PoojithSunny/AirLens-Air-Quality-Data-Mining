from __future__ import annotations

import sys
import time
from pathlib import Path

from .analysis import exploratory_analysis, run_association_mining, run_clustering
from .config import OUTPUT_DIR, PROCESSED_DIR
from .data_io import load_raw_dataset
from .models import train_forecasting_models
from .preprocessing import preprocess
from .warehouse import build_warehouse


def main() -> int:
    started = time.time()
    print("=" * 72)
    print("AIRLENS: MULTI-STATION AIR-QUALITY DATA MINING PIPELINE")
    print("=" * 72)
    try:
        print("\n[1/6] Loading official PRSA dataset...")
        raw = load_raw_dataset()
        print(f"Loaded {len(raw):,} raw records from the data/raw folder.")

        print("\n[2/6] Preprocessing data...")
        clean = preprocess(raw)

        print("\n[3/6] Exploratory data analysis...")
        exploratory_analysis(clean)

        print("\n[4/6] Creating SQLite data warehouse...")
        build_warehouse(clean)

        print("\n[5/6] Running Apriori and K-Means...")
        run_association_mining(clean)
        run_clustering(clean)

        print("\n[6/6] Training chronological forecasting models...")
        train_forecasting_models(clean)

        print("\n" + "=" * 72)
        print("AIRLENS PIPELINE FINISHED")
        print(f"Elapsed time: {(time.time() - started) / 60:.1f} minutes")
        print(f"Processed data: {PROCESSED_DIR}")
        print(f"Results and charts: {OUTPUT_DIR}")
        print("Next step: start the dashboard with: python -m streamlit run app.py")
        print("=" * 72)
        return 0
    except FileNotFoundError as exc:
        print(f"\nDATA FILE NOT FOUND\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nPIPELINE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Scroll up to find the stage that failed. See README.md -> Troubleshooting.", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
