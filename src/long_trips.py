"""Does knowing typical route traffic fix the long and airport trips?

Trains the served HistGradientBoosting and the RouteAwareRegressor (src/route_features.py) on
the same rows, scores both on the January validation set and on February 2025, and breaks
the error down by segment. Logs both to MLflow as long-trips-<name>.

    uv run python -m src.long_trips                       # all 2.67M training rows
    uv run python -m src.long_trips --train-size 500000   # quicker
    uv run python -m src.long_trips --save route_aware    # also save that model for the API

Segments by *actual* duration (45-60 min) always look underpredicted for any model, because
picking trips that turned out long selects the unlucky ones; segments by what is known at
pickup (airport, distance >= 10 miles, rush hour) are the fair test.
"""

from __future__ import annotations

import json
import time

import click
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from src import data, features, model_io
from src.compare import make_model, size_mb
from src.live_context import add_context, live_traffic, weather
from src.route_features import AIRPORTS, RouteAwareRegressor
from src.train import load_split, setup_mlflow


def make(name: str):
    base = name.removesuffix("+context")
    return RouteAwareRegressor() if base == "route_aware" else make_model(base)


def segment_table(X: pd.DataFrame, y, pred) -> pd.DataFrame:
    y = np.asarray(y)
    err = pred - y
    seg = {
        "all trips": np.ones(len(y), bool),
        "airport trips": (X.PULocationID.isin(AIRPORTS) | X.DOLocationID.isin(AIRPORTS)).to_numpy(),
        "distance >= 10 mi": (X.trip_distance >= 10).to_numpy(),
        "rush hour (7-9, 15-18 weekdays)": (X.pickup_hour.isin([7, 8, 9, 15, 16, 17, 18]) & (X.pickup_weekday < 5)).to_numpy(),
        "actual 45-60 min": (y >= 45),
        "actual 30-45 min": (y >= 30) & (y < 45),
    }
    return pd.DataFrame({
        name: {"trips": int(m.sum()), "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
               "mae": float(np.mean(np.abs(err[m]))), "bias": float(np.mean(err[m]))}
        for name, m in seg.items()
    }).T


@click.command()
@click.option("--models", default="hgb,route_aware,hgb+context,route_aware+context", show_default=True,
              help="+context adds live traffic and weather (src/live_context.py).")
@click.option("--train-size", type=int, default=0, show_default=True, help="0 = all 2.67 million rows.")
@click.option("--save", "save_name", default=None, help="Save this model for the API.")
def main(models: str, train_size: int, save_name: str | None) -> None:
    X_train, X_val, y_train, y_val, _ = load_split(train_size or None)
    feb = data.clean(data.load(data.download(2025, 2)), year=2025, month=2)
    X_feb, y_feb = features.build_features(feb), feb[data.TARGET]
    print(f"train {len(X_train):,} | January validation {len(X_val):,} | February {len(X_feb):,}")
    if "+context" in models:
        jan = data.clean(data.load())
        wx = weather()
        ctx = {
            "train": add_context(X_train, jan.loc[X_train.index, "tpep_pickup_datetime"], live_traffic(jan), wx),
            "val": add_context(X_val, jan.loc[X_val.index, "tpep_pickup_datetime"], live_traffic(jan), wx),
            "feb": add_context(X_feb, feb["tpep_pickup_datetime"], live_traffic(feb), wx),
        }

    setup_mlflow()
    results = {}
    for name in models.split(","):
        model = make(name)
        Xt, Xv, Xf = (ctx["train"], ctx["val"], ctx["feb"]) if name.endswith("+context") else (X_train, X_val, X_feb)
        with mlflow.start_run(run_name=f"long-trips-{name.replace('+', '-')}" + ("" if not train_size else f"-{train_size}")):
            start = time.perf_counter()
            model.fit(Xt, y_train)
            fit_s = time.perf_counter() - start
            pj, pf = model.predict(Xv), model.predict(Xf)
            tj, tf = segment_table(X_val, y_val, pj), segment_table(X_feb, y_feb, pf)
            metrics = {"jan_rmse": tj.loc["all trips", "rmse"], "feb_rmse": tf.loc["all trips", "rmse"],
                       "jan_mae": tj.loc["all trips", "mae"], "fit_seconds": fit_s, "model_mb": size_mb(model)}
            for seg in tj.index:
                key = seg.split(" (")[0].replace(" ", "_").replace(">=", "ge").replace("-", "_")
                metrics[f"jan_{key}_mae"] = tj.loc[seg, "mae"]
                metrics[f"feb_{key}_mae"] = tf.loc[seg, "mae"]
            mlflow.log_params({"approach": name, "train_rows": len(X_train)})
            mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id
        results[name] = (model, metrics, run_id)
        print(f"\n== {name}: fit {fit_s:.0f}s, {metrics['model_mb']:.1f} MB")
        print("January validation:\n" + tj.round(2).to_string())
        print("February 2025:\n" + tf.round(2).to_string())
        print(json.dumps({k: round(v, 3) for k, v in metrics.items()}))

    if save_name:
        model, metrics, run_id = results[save_name]
        path = model_io.save(model, metadata={
            "run_id": run_id, "run_name": f"long-trips-{save_name}", "train_rows": len(X_train),
            "rmse": metrics["jan_rmse"], "mae": metrics["jan_mae"], "feb_rmse": metrics["feb_rmse"],
            "features": features.FEATURES,
        })
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
