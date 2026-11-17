"""A non-ML reference: predict from each route's typical speed.

For every pickup-dropoff pair seen at least `min_trips` times, store the median minutes per
mile; a new trip gets distance × that rate (or the overall median rate for unseen routes).
If machine learning cannot beat this, it is not worth deploying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class RouteSpeedBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, min_trips: int = 20):
        self.min_trips = min_trips

    def fit(self, X: pd.DataFrame, y):
        rate = np.asarray(y) / X["trip_distance"].clip(lower=0.1).to_numpy()
        frame = pd.DataFrame({"route": self._routes(X), "rate": rate})
        stats = frame.groupby("route")["rate"].agg(["median", "size"])
        self.route_rate_ = stats.loc[stats["size"] >= self.min_trips, "median"].to_dict()
        self.global_rate_ = float(np.median(rate))
        return self

    def predict(self, X: pd.DataFrame):
        rates = self._routes(X).map(self.route_rate_).fillna(self.global_rate_).to_numpy()
        return np.clip(X["trip_distance"].to_numpy() * rates, 1, 60)

    @staticmethod
    def _routes(X: pd.DataFrame) -> pd.Series:
        return X["PULocationID"].astype(str) + "_" + X["DOLocationID"].astype(str)
