"""Compare model approaches on the same split, plus an out-of-time check on February 2025.

    uv run python -m src.compare
    uv run python -m src.compare --save hgb      # also save that model for the API
    uv run python -m src.compare --models hgb --train-size 0 --save hgb   # the served model: all rows

Approaches (all use the same five start-of-trip features):
- route_speed: no ML, median minutes-per-mile of each route × distance (src/baselines.py)
- linear: one-hot zones + distance/hour/weekday into Ridge (a simple reference)
- rf: the baseline RandomForestRegressor from src/train.py
- hgb: HistGradientBoostingRegressor (scikit-learn)
- xgb: XGBoost, the most widely used gradient-boosting library
- lgbm: LightGBM, gradient boosting with leaf-wise trees

Each model trains on the same 500,000 January rows, is scored on the full January validation
set and on all cleaned February 2025 trips, and is logged to MLflow as compare-<name>.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import time
from pathlib import Path

import click
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src import data, features, model_io
from src.baselines import RouteSpeedBaseline
from src.train import BASELINE_PARAMS, RANDOM_STATE, load_split, setup_mlflow


def make_model(name: str):
    if name == "route_speed":
        return RouteSpeedBaseline(min_trips=20)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=8, subsample=0.8,
                            colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=500, learning_rate=0.1, num_leaves=127, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    if name == "linear":
        zones = ColumnTransformer(
            [("zones", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["PULocationID", "DOLocationID"])],
            remainder="passthrough",
        )
        return make_pipeline(zones, Ridge(alpha=1.0))
    if name == "rf":
        return RandomForestRegressor(**BASELINE_PARAMS, n_jobs=-1, random_state=RANDOM_STATE)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, max_leaf_nodes=127,
                                             min_samples_leaf=40, random_state=RANDOM_STATE)
    raise ValueError(f"unknown model {name!r}; choose route_speed, linear, rf, hgb, xgb or lgbm")


def size_mb(model) -> float:
    """Size of the saved model: skops where it can serialise the model, else pickle (size only)."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return model_io.save(model, Path(tmp) / "m.skops").stat().st_size / 1e6
    except Exception:
        return len(pickle.dumps(model)) / 1e6


def scores(model, X, y, prefix: str) -> dict:
    start = time.perf_counter()
    pred = model.predict(X)
    ms_per_1k = (time.perf_counter() - start) / len(X) * 1000 * 1000
    return {f"{prefix}_rmse": float(root_mean_squared_error(y, pred)),
            f"{prefix}_mae": float(mean_absolute_error(y, pred)),
            f"{prefix}_ms_per_1k": ms_per_1k}


@click.command()
@click.option("--models", default="route_speed,linear,rf,hgb,xgb,lgbm", show_default=True)
@click.option("--train-size", type=int, default=500_000, show_default=True,
              help="Training rows sampled from the 80% split (0 = all 2.67 million).")
@click.option("--save", "save_name", default=None, help="Save this model (linear, rf or hgb) for the API.")
def main(models: str, train_size: int, save_name: str | None) -> None:
    X_train, X_val, y_train, y_val, _ = load_split(train_size or None)
    feb = data.clean(data.load(data.download(2025, 2)), year=2025, month=2)
    X_feb, y_feb = features.build_features(feb), feb[data.TARGET]
    print(f"train {len(X_train):,} | January validation {len(X_val):,} | February {len(X_feb):,}")

    setup_mlflow()
    rows, fitted = [], {}
    for name in models.split(","):
        model = make_model(name)
        suffix = "" if train_size == 500_000 else ("-all" if not train_size else f"-{train_size}")
        with mlflow.start_run(run_name=f"compare-{name}{suffix}"):
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            metrics = {**scores(model, X_val, y_val, "jan"), **scores(model, X_feb, y_feb, "feb"),
                       "fit_seconds": fit_seconds, "model_mb": size_mb(model)}
            metrics["rmse"] = metrics["jan_rmse"]
            mlflow.log_params({"approach": name, "estimator": type(model.steps[-1][1] if hasattr(model, "steps") else model).__name__,
                               "train_rows": len(X_train), "feb_rows": len(X_feb)})
            mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id
        fitted[name] = (model, metrics, run_id)
        rows.append({"model": name, **{k: round(v, 3) for k, v in metrics.items() if k != "rmse"}})
        print(json.dumps(rows[-1]))

    mean_rmse = float(root_mean_squared_error(y_feb, np.full(len(y_feb), y_train.mean())))
    print(f"February mean-duration baseline RMSE: {mean_rmse:.3f}")

    if save_name:
        model, metrics, run_id = fitted[save_name]
        path = model_io.save(model, metadata={
            "run_id": run_id, "run_name": f"compare-{save_name}{suffix}", "train_rows": len(X_train),
            "rmse": metrics["jan_rmse"],
            "mae": metrics["jan_mae"], "feb_rmse": metrics["feb_rmse"], "features": features.FEATURES,
        })
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
