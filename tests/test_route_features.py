"""Route and same-day context features: no leakage, sensible fallbacks."""

import numpy as np
import pandas as pd

from src.live_context import add_context, live_traffic
from src.route_features import RouteAwareRegressor, RouteStats


def _trips(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "PULocationID": rng.choice([132, 161, 236], n), "DOLocationID": rng.choice([161, 236, 138], n),
        "trip_distance": rng.uniform(0.5, 15, n), "pickup_hour": rng.integers(0, 24, n),
        "pickup_weekday": rng.integers(0, 7, n),
    })
    y = X.trip_distance * 4 + np.where(X.PULocationID == 132, 10, 0) + rng.normal(0, 1, n)
    return X, y


def test_unseen_route_falls_back_to_overall_rate():
    X, y = _trips()
    stats = RouteStats().fit(X, y)
    new = pd.DataFrame({"PULocationID": [7], "DOLocationID": [9], "trip_distance": [2.0],
                        "pickup_hour": [8], "pickup_weekday": [1]})
    out = stats.transform(new)
    assert out.loc[0, "route_trips"] == 0
    assert np.isclose(out.loc[0, "route_rate"], stats.global_rate)


def test_airport_flags():
    X, y = _trips()
    out = RouteStats().fit(X, y).transform(X)
    assert (out.pickup_airport == X.PULocationID.isin([1, 132, 138])).all()
    assert (out.dropoff_airport == X.DOLocationID.isin([1, 132, 138])).all()


def test_route_aware_learns_and_predicts_same_inputs_as_api():
    X, y = _trips()
    model = RouteAwareRegressor(max_iter=50).fit(X, y)
    pred = model.predict(X.head(5))
    assert pred.shape == (5,) and np.isfinite(pred).all()


def test_live_traffic_uses_only_trips_that_ended_before_the_pickup_hour():
    t0 = pd.Timestamp("2025-01-15 08:00")
    df = pd.DataFrame({
        "tpep_pickup_datetime": [t0 - pd.Timedelta(minutes=50), t0 + pd.Timedelta(minutes=5)],
        "tpep_dropoff_datetime": [t0 - pd.Timedelta(minutes=20), t0 + pd.Timedelta(minutes=40)],
        "duration": [30.0, 35.0], "trip_distance": [3.0, 7.0],
    })
    traffic = live_traffic(df)
    X = pd.DataFrame({"trip_distance": [7.0]})
    out = add_context(X, df["tpep_pickup_datetime"].iloc[[1]].reset_index(drop=True), traffic,
                      pd.DataFrame(columns=["temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_m", "wind_kmh"]))
    # the 08:05 trip sees only the trip that ended at 07:40 (10 min/mile), never its own
    assert out.loc[0, "live_rate_prev_hour"] == 10.0
    assert out.loc[0, "live_trips_prev_hour"] == 1
