"""Answer 3a: train on several months, test on the next one.

Trains the served model (HistGradientBoosting, five features) on January 2025 only, then on
November 2024 - January 2025, and scores both on February 2025, a month neither has seen.

    uv run python -m src.multi_month
"""

from __future__ import annotations

import json
import time

import click

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.compare import make_model
from src.live_context import CONTEXT_FEATURES
from src.months import load_month
from src.train import setup_mlflow

TRAININGS = {"January only": [(2025, 1)], "Dec + Jan": [(2024, 12), (2025, 1)],
             "Nov + Dec + Jan": [(2024, 11), (2024, 12), (2025, 1)]}


LIVE = [c for c in CONTEXT_FEATURES if c.startswith("live_")]
BASE = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour", "pickup_weekday"]


@click.command()
@click.option("--live-traffic", is_flag=True, help="Add the live-traffic features (speed of trips in the previous hour).")
def main(live_traffic: bool) -> None:
    cols = BASE + LIVE if live_traffic else BASE
    months = {ym: (load_month(*ym, context=live_traffic)[0][cols], load_month(*ym)[1])
              for ym in {(2024, 11), (2024, 12), (2025, 1)}}
    X_feb, y_feb, _ = load_month(2025, 2, context=live_traffic)
    X_feb = X_feb[cols]
    setup_mlflow()
    for name, keys in TRAININGS.items():
        X = pd.concat([months[k][0] for k in keys], ignore_index=True)
        y = pd.concat([months[k][1] for k in keys], ignore_index=True)
        start = time.perf_counter()
        model = make_model("hgb").fit(X, y)
        pred = model.predict(X_feb)
        row = {"training": name + (" + live traffic" if live_traffic else ""), "train_rows": len(X), "fit_seconds": round(time.perf_counter() - start, 1),
               "rmse_feb": float(root_mean_squared_error(y_feb, pred)), "mae_feb": float(mean_absolute_error(y_feb, pred))}
        with mlflow.start_run(run_name=f"multi-month-{len(keys)}" + ("-live" if live_traffic else "")):
            mlflow.log_params({"training_months": ",".join(f"{a}-{b:02d}" for a, b in keys), "train_rows": len(X)})
            mlflow.log_metrics({"rmse_feb": row["rmse_feb"], "mae_feb": row["mae_feb"]})
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))


if __name__ == "__main__":
    main()
