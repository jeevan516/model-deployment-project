"""Stretch task: tune the RandomForestRegressor with Optuna and compare it with the baseline.

    uv run python -m src.tune --trials 20

Each trial trains on a 200,000-row sample and scores on a fixed 200,000-row slice of the
validation set (logged as nested MLflow runs). The best parameters are then retrained on the
same 500,000 training rows as the baseline and scored on the full validation set, so the two
final numbers are directly comparable. The tuned model is only saved for the API with --save.
"""

from __future__ import annotations

import json

import click
import mlflow
import optuna
from sklearn.ensemble import RandomForestRegressor

from src import model_io
from src.train import BASELINE_PARAMS, RANDOM_STATE, evaluate, fit_and_log, load_split, setup_mlflow


def objective_factory(X_tr, y_tr, X_va, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_categorical("max_features", [1.0, 0.8, 0.6]),
        }
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        rmse = evaluate(model, X_va, y_va)["rmse"]
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
        return rmse

    return objective


@click.command()
@click.option("--trials", default=20, show_default=True)
@click.option("--save/--no-save", default=False, show_default=True,
              help="Save the tuned model to artifacts/model.skops for the API.")
def main(trials: int, save: bool) -> None:
    split = load_split(500_000)
    X_train, X_val, y_train, y_val, _ = split
    X_tr, y_tr = X_train.sample(n=200_000, random_state=RANDOM_STATE), None
    y_tr = y_train.loc[X_tr.index]
    X_va, y_va = X_val.iloc[:200_000], y_val.iloc[:200_000]

    setup_mlflow()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name="optuna-search"):
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective_factory(X_tr, y_tr, X_va, y_va), n_trials=trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_search_rmse", study.best_value)
    print(f"best search RMSE {study.best_value:.4f} with {study.best_params}")

    tuned, metrics = fit_and_log(study.best_params, split, "rf-tuned", {"tuned_with": "optuna"})
    print(json.dumps({"tuned": study.best_params, **{k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in metrics.items()}}, indent=2))
    print(f"baseline params were {BASELINE_PARAMS}")
    if save:
        path = model_io.save(tuned, metadata={
            "run_id": metrics["run_id"], "run_name": "rf-tuned", "params": study.best_params,
            "rmse": metrics["rmse"], "mae": metrics["mae"],
        })
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
