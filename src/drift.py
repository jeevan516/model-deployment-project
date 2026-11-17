"""Answer 3b: a monthly drift check.

Compares each month with a reference month using the Population Stability Index (PSI) per
feature and for the target. Rule of thumb: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 retrain.

    uv run python -m src.drift                      # Nov 2024 - Feb 2025 against January 2025
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.months import load_month

WATCH, RETRAIN = 0.1, 0.25
NUMERIC = ["trip_distance", "duration"]
CATEGORICAL = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_weekday"]


def psi(reference: pd.Series, current: pd.Series, numeric: bool, bins: int = 10) -> float:
    """Population Stability Index between two samples (quantile bins for numbers, levels otherwise)."""
    if numeric:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = pd.cut(reference, edges).value_counts(normalize=True, sort=False)
        cur = pd.cut(current, edges).value_counts(normalize=True, sort=False)
    else:
        ref = reference.value_counts(normalize=True)
        cur = current.value_counts(normalize=True).reindex(ref.index, fill_value=0)
    ref, cur = ref.clip(lower=1e-4), cur.reindex(ref.index, fill_value=0).clip(lower=1e-4)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def status(value: float) -> str:
    return "retrain" if value > RETRAIN else "watch" if value > WATCH else "stable"


def report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    out = {c: round(psi(reference[c], current[c], True), 4) for c in NUMERIC}
    out |= {c: round(psi(reference[c], current[c], False), 4) for c in CATEGORICAL}
    worst = max(out.values())
    return {"psi": out, "max_psi": worst, "status": status(worst),
            "median_duration": round(float(current["duration"].median()), 2)}


def frame(year: int, month: int) -> pd.DataFrame:
    X, y, _ = load_month(year, month)
    return X.assign(duration=y.to_numpy())


def main() -> None:
    ref = frame(2025, 1)
    for y, m in [(2024, 11), (2024, 12), (2025, 2)]:
        print(json.dumps({"month": f"{y}-{m:02d}", "vs": "2025-01", **report(ref, frame(y, m))}))


if __name__ == "__main__":
    main()
