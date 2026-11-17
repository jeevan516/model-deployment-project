"""Download and clean the January 2025 NYC Yellow Taxi data."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

TARGET = "duration"
MIN_DURATION, MAX_DURATION = 1, 60  # minutes; keeps 97.9% of January 2025
MIN_DISTANCE, MAX_DISTANCE = 0.0, 100.0  # miles; 0 and above 100 are recording errors


def download(year: int = 2025, month: int = 1) -> Path:
    """Download one month of Yellow Taxi data once and return the local path."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not path.exists():
        urlretrieve(DATA_URL.format(year=year, month=month), path)
    return path


def load(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or download())


def clean(df: pd.DataFrame, year: int = 2025, month: int = 1) -> pd.DataFrame:
    """Add the target and drop trips that cannot be real or are outside the month."""
    out = df.copy()
    out[TARGET] = (out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]).dt.total_seconds() / 60
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    keep = (
        out[TARGET].between(MIN_DURATION, MAX_DURATION)
        & (out["trip_distance"] > MIN_DISTANCE)
        & (out["trip_distance"] <= MAX_DISTANCE)
        & (out["tpep_pickup_datetime"] >= start)
        & (out["tpep_pickup_datetime"] < end)
    )
    return out.loc[keep].reset_index(drop=True)


if __name__ == "__main__":
    print(download())
