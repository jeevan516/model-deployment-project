"""Data checks: what is wrong in the raw file, and proof that the cleaned data is fit to train on.

    uv run python -m src.checks            # report for January 2025 (raw vs cleaned)

Each rule returns the rows that break it. On the raw file the counts explain why rows are
removed; on the cleaned data every rule the model depends on must have 0 failures, and
`assert_clean` (called before training) stops the pipeline if one does not.
"""

from __future__ import annotations

import json

import pandas as pd

from src import data

REQUIRED_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance"]
MIN_ZONE, MAX_ZONE = 1, 265  # TLC taxi zones 1-263, plus 264/265 for unknown and outside NYC


def _duration(df: pd.DataFrame) -> pd.Series:
    return (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60


def rules(year: int = 2025, month: int = 1) -> dict:
    """Rule name -> function returning a boolean Series that is True where a row BREAKS the rule."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return {
        "pickup or dropoff time missing": lambda d: d["tpep_pickup_datetime"].isna() | d["tpep_dropoff_datetime"].isna(),
        "pickup or dropoff zone missing": lambda d: d["PULocationID"].isna() | d["DOLocationID"].isna(),
        "zone outside 1-265": lambda d: ~d["PULocationID"].between(MIN_ZONE, MAX_ZONE) | ~d["DOLocationID"].between(MIN_ZONE, MAX_ZONE),
        "dropoff not after pickup": lambda d: _duration(d) <= 0,
        "duration under 1 minute": lambda d: _duration(d).between(0, 1, inclusive="neither"),
        "duration over 60 minutes": lambda d: _duration(d) > 60,
        "distance 0 or negative": lambda d: d["trip_distance"] <= 0,
        "distance over 100 miles": lambda d: d["trip_distance"] > 100,
        "pickup outside the month": lambda d: (d["tpep_pickup_datetime"] < start) | (d["tpep_pickup_datetime"] >= end),
    }


def report(df: pd.DataFrame, year: int = 2025, month: int = 1) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    rows = []
    for name, broken in rules(year, month).items():
        n = int(broken(df).sum())
        rows.append({"rule": name, "failing_rows": n, "failing_pct": round(100 * n / len(df), 3)})
    return pd.DataFrame(rows)


def assert_clean(df: pd.DataFrame, year: int = 2025, month: int = 1) -> None:
    """Stop training if the cleaned data breaks any rule or has nulls in a model column."""
    failures = report(df, year, month).query("failing_rows > 0")
    nulls = df[REQUIRED_COLUMNS].isna().sum()
    if not failures.empty or nulls.any():
        raise ValueError(f"cleaned data failed checks:\n{failures.to_string()}\nnulls:\n{nulls[nulls > 0]}")


def summary(df: pd.DataFrame) -> dict:
    """Distribution facts used to compare months (a simple drift check)."""
    duration = _duration(df)
    return {
        "rows": len(df),
        "duration_median": round(float(duration.median()), 2),
        "distance_median": round(float(df["trip_distance"].median()), 2),
        "rush_hour_share": round(float(df["tpep_pickup_datetime"].dt.hour.isin([7, 8, 9, 16, 17, 18]).mean()), 3),
        "weekend_share": round(float((df["tpep_pickup_datetime"].dt.day_of_week >= 5).mean()), 3),
        "airport_pickup_share": round(float(df["PULocationID"].isin([1, 132, 138]).mean()), 3),
    }


if __name__ == "__main__":
    raw = data.load()
    clean = data.clean(raw)
    print(f"raw rows {len(raw):,} -> clean rows {len(clean):,} ({len(clean) / len(raw):.1%} kept)")
    print("\nRaw January 2025:\n" + report(raw).to_string(index=False))
    print("\nCleaned January 2025:\n" + report(clean).to_string(index=False))
    assert_clean(clean)
    print("\nassert_clean: passed")
    print("\nsummary:", json.dumps(summary(clean)))
