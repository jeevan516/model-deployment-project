"""Load cleaned months with features (and optional same-day context) for the follow-up experiments."""

from __future__ import annotations

import pandas as pd

from src import data, features
from src.live_context import add_context, live_traffic, weather

WEATHER_RANGE = ("2024-11-01", "2025-02-28")


def load_month(year: int, month: int, context: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, pickup time) for one cleaned month. context=True adds live traffic and weather."""
    df = data.clean(data.load(data.download(year, month)), year=year, month=month)
    X = features.build_features(df)
    if context:
        X = add_context(X, df["tpep_pickup_datetime"], live_traffic(df), weather(*WEATHER_RANGE))
    return X, df[data.TARGET], df["tpep_pickup_datetime"]
