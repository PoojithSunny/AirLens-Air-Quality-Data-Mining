from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import RAW_DIR, EXPECTED_COLUMNS


def _read_candidate_csvs(paths: Iterable[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path, low_memory=False)
            frame.columns = [str(c).strip() for c in frame.columns]
            if EXPECTED_COLUMNS.issubset(set(frame.columns)):
                frame["_source_file"] = path.name
                frames.append(frame)
        except Exception as exc:  # show a useful error if files aren't valid CSVs
            print(f"Skipping {path.name}: {exc}")
    return frames


def load_raw_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load UCI PRSA CSV files from the supplied ZIP or extracted CSV files."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data folder does not exist: {raw_dir}\n"
            "Create data/raw and place the official UCI ZIP or extracted PRSA CSVs there."
        )

    frames: list[pd.DataFrame] = []
    # First read the official ZIP, if present. Read every matching CSV inside it.
    for zip_path in sorted(raw_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = [n for n in archive.namelist()
                         if n.lower().endswith(".csv") and not n.endswith("/")]
                for name in names:
                    try:
                        with archive.open(name) as handle:
                            frame = pd.read_csv(handle, low_memory=False)
                        frame.columns = [str(c).strip() for c in frame.columns]
                        if EXPECTED_COLUMNS.issubset(set(frame.columns)):
                            frame["_source_file"] = f"{zip_path.name}:{Path(name).name}"
                            frames.append(frame)
                    except Exception as exc:
                        print(f"Skipping {name} inside {zip_path.name}: {exc}")
        except zipfile.BadZipFile:
            print(f"Skipping invalid ZIP file: {zip_path}")

    # Also support CSV files that the user extracted manually.
    csv_paths = [p for p in raw_dir.rglob("*.csv") if p.is_file()]
    if csv_paths:
        frames.extend(_read_candidate_csvs(csv_paths))

    if not frames:
        raise FileNotFoundError(
            "No matching PRSA CSV data was found. Download the official UCI ZIP and "
            "place it in data/raw, or extract its station CSVs into data/raw.\n"
            "Expected columns include year, month, day, hour, station, PM2.5, PM10, TEMP, and WSPM."
        )

    data = pd.concat(frames, ignore_index=True)
    # If ZIP and extracted CSVs both exist, remove duplicated observations later by station/time.
    return data
