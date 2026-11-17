"""Answer 4: a time range instead of one number.

Two extra HistGradientBoosting models predict the 10th and 90th percentile of the duration
(quantile loss), so the API can say "15-22 minutes". A good 80% range contains the real
duration for about 80% of trips; this script measures that on January and February,
overall and for long trips, and saves both models next to the served one.

    uv run python -m src.intervals                 # all 2.67M training rows
    uv run python -m src.intervals --train-size 500000
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src import data, features, model_io
from src.train import RANDOM_STATE, load_split, setup_mlflow

QUANTILES = {"low": 0.10, "high": 0.90}


def quantile_path(name: str, model_path: str | Path = model_io.DEFAULT_MODEL_PATH) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_{name}{model_path.suffix}")


def make_quantile_model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.1,
                                         max_leaf_nodes=63, min_samples_leaf=100, random_state=RANDOM_STATE)


def coverage(y, low, high) -> dict:
    y = np.asarray(y)
    inside = (y >= low) & (y <= high)
    long = y >= 45
    return {"coverage": float(inside.mean()), "width": float(np.median(high - low)),
            "coverage_long": float(inside[long].mean()), "width_long": float(np.median((high - low)[long]))}


@click.command()
@click.option("--train-size", type=int, default=0, show_default=True, help="0 = all 2.67M rows.")
@click.option("--save/--no-save", default=True, show_default=True)
def main(train_size: int, save: bool) -> None:
    X_train, X_val, y_train, y_val, _ = load_split(train_size or None)
    feb = data.clean(data.load(data.download(2025, 2)), year=2025, month=2)
    X_feb, y_feb = features.build_features(feb), feb[data.TARGET]
    models = {name: make_quantile_model(q).fit(X_train, y_train) for name, q in QUANTILES.items()}
    jan = coverage(y_val, *(models[n].predict(X_val) for n in ("low", "high")))
    febr = coverage(y_feb, *(models[n].predict(X_feb) for n in ("low", "high")))
    setup_mlflow()
    with mlflow.start_run(run_name="intervals-q10-q90"):
        mlflow.log_params({"quantiles": "0.10,0.90", "train_rows": len(X_train)})
        mlflow.log_metrics({f"jan_{k}": v for k, v in jan.items()} | {f"feb_{k}": v for k, v in febr.items()})
    print(json.dumps({"january": {k: round(v, 3) for k, v in jan.items()},
                      "february": {k: round(v, 3) for k, v in febr.items()}}))
    if save:
        for name, model in models.items():
            path = model_io.save(model, quantile_path(name), metadata={"quantile": QUANTILES[name], "train_rows": len(X_train),
                                                                       "jan_coverage": jan["coverage"], "feb_coverage": febr["coverage"]})
            print(f"saved {path}")


if __name__ == "__main__":
    main()
