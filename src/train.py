"""Train the trip-duration model and track the run in MLflow.

    uv run python -m src.train                       # baseline RandomForestRegressor
    uv run python -m src.train --params '{"max_depth": 25}' --run-name rf-custom

Runs go to mlflow.db (SQLite) with artifacts in ./mlruns. MLflow 3.17 no longer accepts a
plain ./mlruns folder as the tracking store. View the runs with:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
The fitted model is saved to artifacts/model.skops for the API, unless --no-save is given.
"""

from __future__ import annotations

import json
import os
import time

import click
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src import checks, data, features, model_io

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{data.PROJECT_ROOT / 'mlflow.db'}")
ARTIFACT_LOCATION = (data.PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT = "yellow-taxi-duration"
RANDOM_STATE = 42
BASELINE_PARAMS = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 10}


def load_split(train_size: int | None, random_state: int = RANDOM_STATE):
    """Clean data, 80/20 split, then optionally subsample the training part for speed."""
    df = data.clean(data.load())
    checks.assert_clean(df)  # stop here if the cleaned data breaks a rule (see src/checks.py)
    X, y = features.build_features(df), df[data.TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    if train_size and train_size < len(X_train):
        X_train = X_train.sample(n=train_size, random_state=random_state)
        y_train = y_train.loc[X_train.index]
    return X_train, X_val, y_train, y_val, len(df)


def setup_mlflow() -> None:
    """Point MLflow at the local SQLite store and keep artifacts in ./mlruns."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT)


def evaluate(model, X_val, y_val) -> dict:
    pred = model.predict(X_val)
    return {"rmse": float(root_mean_squared_error(y_val, pred)), "mae": float(mean_absolute_error(y_val, pred))}


def fit_and_log(params: dict, split, run_name: str, extra_tags: dict | None = None):
    """Fit one RandomForestRegressor, log params/metrics/model to MLflow, return model and metrics."""
    X_train, X_val, y_train, y_val, n_clean = split
    setup_mlflow()
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
    with mlflow.start_run(run_name=run_name) as run:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate(model, X_val, y_val)
        metrics["baseline_mean_rmse"] = float(root_mean_squared_error(y_val, np.full(len(y_val), y_train.mean())))
        metrics["fit_seconds"] = fit_seconds
        mlflow.log_params({**params, "features": ",".join(features.FEATURES),
                           "train_rows": len(X_train), "validation_rows": len(X_val),
                           "clean_rows": n_clean, "data": "yellow_tripdata_2025-01"})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_type": "RandomForestRegressor", **(extra_tags or {})})
        mlflow.sklearn.log_model(
            sk_model=model, name="model", serialization_format="skops",
            skops_trusted_types=model_io.TRUSTED_TYPES, input_example=X_val.head(3),
        )
        metrics["run_id"] = run.info.run_id
    return model, metrics


@click.command()
@click.option("--train-size", type=int, default=500_000, show_default=True,
              help="Training rows sampled from the 80% split (0 = all).")
@click.option("--params", default=json.dumps(BASELINE_PARAMS), show_default=True,
              help="RandomForestRegressor parameters as JSON.")
@click.option("--run-name", default="rf-baseline", show_default=True)
@click.option("--save/--no-save", default=True, show_default=True,
              help="Save the model to artifacts/model.skops for the API.")
def main(train_size: int, params: str, run_name: str, save: bool) -> None:
    split = load_split(train_size or None)
    model, metrics = fit_and_log(json.loads(params), split, run_name)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    if save:
        path = model_io.save(model, metadata={
            "run_id": metrics["run_id"], "run_name": run_name, "params": json.loads(params),
            "rmse": metrics["rmse"], "mae": metrics["mae"], "features": features.FEATURES,
        })
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
