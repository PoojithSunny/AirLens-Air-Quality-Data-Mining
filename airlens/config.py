from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_AIR_FEATURES = POLLUTANTS + WEATHER
EXPECTED_COLUMNS = {
    "year", "month", "day", "hour", "station", "PM2.5", "PM10",
    "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN",
    "wd", "WSPM"
}
