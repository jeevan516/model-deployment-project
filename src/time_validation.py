"""Answer 1: validate by time, not at random.

A random 80/20 split inside January lets validation trips share hours (and weather) with
training trips, which flatters same-day features. Here the model trains on 1-24 January and
is validated on 25-31 January, then on February.

    uv run python -m src.time_validation
"""

from __future__ import annotations

import json

import mlflow
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src.compare import make_model
from src.live_context import CONTEXT_FEATURES
from src.months import load_month
from src.route_features import RouteAwareRegressor
from src.train import setup_mlflow

WEATHER_ONLY = [c for c in CONTEXT_FEATURES if not c.startswith("live_")]
BASE = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour", "pickup_weekday"]
VARIANTS = {
    "served (5 features)": (lambda: make_model("hgb"), BASE),
    "+ route speed by hour": (RouteAwareRegressor, BASE),
    "+ weather": (lambda: make_model("hgb"), BASE + WEATHER_ONLY),
    "+ route, live traffic, weather": (RouteAwareRegressor, BASE + CONTEXT_FEATURES),
}


def main() -> None:
    X, y, pickup = load_month(2025, 1, context=True)
    X_feb, y_feb, _ = load_month(2025, 2, context=True)
    early = (pickup < pd.Timestamp("2025-01-25")).to_numpy()
    print(f"train 1-24 Jan {early.sum():,} | validate 25-31 Jan {(~early).sum():,} | February {len(y_feb):,}")
    setup_mlflow()
    rows = []
    for name, (factory, cols) in VARIANTS.items():
        model = factory().fit(X.loc[early, cols], y[early])
        row = {"model": name,
               "rmse_late_jan": float(root_mean_squared_error(y[~early], model.predict(X.loc[~early, cols]))),
               "rmse_feb": float(root_mean_squared_error(y_feb, model.predict(X_feb[cols])))}
        with mlflow.start_run(run_name=f"time-split-{name.split()[0].strip('+')}-{len(cols)}"):
            mlflow.log_params({"split": "train 1-24 Jan, validate 25-31 Jan", "features": ",".join(cols)})
            mlflow.log_metrics({k: v for k, v in row.items() if k != "model"})
        rows.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))


if __name__ == "__main__":
    main()
