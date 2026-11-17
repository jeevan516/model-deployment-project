"""Traffic-aware features for long and airport trips, wrapped around HistGradientBoosting.

The served model only knew distance, zones, hour and weekday, so it could not tell a fast
highway mile from a slow Midtown mile: 45-60 minute trips were underpredicted by ~8 minutes
and airport trips had twice the error of other trips (notebook 03).

`RouteAwareRegressor` takes the same five start-of-trip features as before (so the API does
not change) and adds, learned from the training trips only:

- route_rate:        median minutes per mile of the pickup->dropoff zone pair
- route_hour_rate:   the same, for the pickup's time-of-day band (night, morning rush, ...)
- zone_hour_rate:    median minutes per mile of all trips leaving the pickup zone in that hour
- route_trips:       how many training trips the route had (how much to trust the rates)
- route_expected:    distance x route_hour_rate, the "typical traffic" duration
- airport flags:     pickup at / dropoff at JFK, LaGuardia or Newark

Training rows get out-of-fold rates (K folds), so a trip never sees its own duration;
prediction uses rates from all training trips. Rare combinations fall back to wider ones
(route+band -> route -> pickup zone+hour -> overall), and rates are shrunk towards the
fallback by their trip count, so a route seen 3 times does not get an extreme rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

N_ZONES = 266  # TLC zone ids are 1..265
AIRPORTS = (1, 132, 138)  # Newark, JFK, LaGuardia
# Time-of-day bands: night, early, morning rush, midday, evening rush, evening
HOUR_BAND = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 0, 0])
N_BANDS = 6
EXTRA_FEATURES = ["route_rate", "route_hour_rate", "zone_hour_rate", "route_trips", "route_expected",
                  "pickup_airport", "dropoff_airport"]


def _rate(X: pd.DataFrame, y) -> np.ndarray:
    return np.asarray(y, dtype="float64") / X["trip_distance"].clip(lower=0.1).to_numpy()


def _keys(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pu = X["PULocationID"].to_numpy().clip(0, N_ZONES - 1)
    do = X["DOLocationID"].to_numpy().clip(0, N_ZONES - 1)
    hour = X["pickup_hour"].to_numpy().clip(0, 23)
    route = pu * N_ZONES + do
    return route, route * N_BANDS + HOUR_BAND[hour], pu * 24 + hour


def _group_stats(keys: np.ndarray, values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-key median and count as dense arrays (NaN where a key has no trips)."""
    frame = pd.DataFrame({"k": keys, "v": values}).groupby("k")["v"].agg(["median", "size"])
    median = np.full(size, np.nan)
    count = np.zeros(size)
    median[frame.index.to_numpy()] = frame["median"].to_numpy()
    count[frame.index.to_numpy()] = frame["size"].to_numpy()
    return median, count


class RouteStats:
    """Lookup tables of minutes-per-mile, fitted on training trips."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y) -> "RouteStats":
        rate = _rate(X, y)
        route, route_band, zone_hour = _keys(X)
        self.global_rate = float(np.median(rate))
        self.zone_hour, self.zone_hour_n = _group_stats(zone_hour, rate, N_ZONES * 24)
        self.route, self.route_n = _group_stats(route, rate, N_ZONES * N_ZONES)
        self.route_band, self.route_band_n = _group_stats(route_band, rate, N_ZONES * N_ZONES * N_BANDS)
        return self

    def _shrink(self, value, n, fallback):
        value = np.where(np.isnan(value), fallback, value)
        w = n / (n + self.smoothing)
        return w * value + (1 - w) * fallback

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        route, route_band, zone_hour = _keys(X)
        zone_hour_rate = self._shrink(self.zone_hour[zone_hour], self.zone_hour_n[zone_hour], self.global_rate)
        route_rate = self._shrink(self.route[route], self.route_n[route], zone_hour_rate)
        route_hour_rate = self._shrink(self.route_band[route_band], self.route_band_n[route_band], route_rate)
        out = X.copy()
        out["route_rate"] = route_rate
        out["route_hour_rate"] = route_hour_rate
        out["zone_hour_rate"] = zone_hour_rate
        out["route_trips"] = self.route_n[route]
        out["route_expected"] = (X["trip_distance"].to_numpy() * route_hour_rate).clip(0, 300)
        out["pickup_airport"] = X["PULocationID"].isin(AIRPORTS).astype("int8").to_numpy()
        out["dropoff_airport"] = X["DOLocationID"].isin(AIRPORTS).astype("int8").to_numpy()
        return out


class RouteAwareRegressor(BaseEstimator, RegressorMixin):
    """HistGradientBoosting on the five base features plus route/traffic statistics."""

    def __init__(self, n_folds: int = 5, smoothing: float = 20.0, max_iter: int = 500, learning_rate: float = 0.1,
                 max_leaf_nodes: int = 127, min_samples_leaf: int = 40, random_state: int = 42):
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype="float64")
        parts = []
        for fit_idx, out_idx in KFold(self.n_folds, shuffle=True, random_state=self.random_state).split(X):
            stats = RouteStats(self.smoothing).fit(X.iloc[fit_idx], y[fit_idx])
            parts.append(stats.transform(X.iloc[out_idx]))
        X_oof = pd.concat(parts).sort_index()
        self.stats_ = RouteStats(self.smoothing).fit(X, y)
        self.model_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
        ).fit(X_oof, y)
        self.feature_names_ = list(X_oof.columns)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self.stats_.transform(X.reset_index(drop=True))[self.feature_names_])
