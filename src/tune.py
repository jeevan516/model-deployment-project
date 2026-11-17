"""Stretch task: tune the RandomForest or HistGradientBoosting model with Optuna and compare it with the baseline.

    uv run python -m src.tune --trials 20                 # RandomForest
    uv run python -m src.tune --model hgb --trials 20     # HistGradientBoosting

Each trial trains on a 200,000-row sample and scores on a fixed 200,000-row slice of the
validation set (logged as nested MLflow runs). The best parameters are then retrained on the
same 500,000 training rows as the baseline and scored on the full validation set, so the two
final numbers are directly comparable. The tuned model is only saved for the API with --save.

Studies are stored in optuna.db, so you can explore them in Optuna's own dashboard:
    uv run optuna-dashboard sqlite:///optuna.db --port 8085      # http://127.0.0.1:8085
"""

from __future__ import annotations

import json

import click
import mlflow
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from src import model_io
from src.train import BASELINE_PARAMS, RANDOM_STATE, evaluate, fit_and_log, load_split, setup_mlflow


def suggest(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 1000, step=100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": trial.suggest_categorical("max_features", [1.0, 0.8, 0.6]),
    }


def build(model_name: str, params: dict):
    if model_name == "hgb":
        return HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    return RandomForestRegressor(**params, n_jobs=-1, random_state=RANDOM_STATE)


def objective_factory(X_tr, y_tr, X_va, y_va, model_name: str = "rf"):
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, model_name)
        model = build(model_name, params)
        model.fit(X_tr, y_tr)
        rmse = evaluate(model, X_va, y_va)["rmse"]
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
        return rmse

    return objective


def fit_hgb_and_log(params: dict, split):
    """Retrain the best HistGradientBoosting parameters on the full 500k rows and log them."""
    import time
    from src import features
    X_train, X_val, y_train, y_val, _ = split
    model = build("hgb", params)
    with mlflow.start_run(run_name="hgb-tuned") as run:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = {**evaluate(model, X_val, y_val), "fit_seconds": fit_seconds}
        mlflow.log_params({**params, "features": ",".join(features.FEATURES), "train_rows": len(X_train)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_type": "HistGradientBoostingRegressor", "tuned_with": "optuna"})
        mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="skops",
                                 skops_trusted_types=model_io.TRUSTED_TYPES, input_example=X_val.head(3))
        metrics["run_id"] = run.info.run_id
    return model, metrics


@click.command()
@click.option("--model", "model_name", type=click.Choice(["rf", "hgb"]), default="rf", show_default=True)
@click.option("--trials", default=20, show_default=True)
@click.option("--storage", default="sqlite:///optuna.db", show_default=True,
              help="Where Optuna keeps the study (open it with optuna-dashboard).")
@click.option("--save/--no-save", default=False, show_default=True,
              help="Save the tuned model to artifacts/model.skops for the API.")
def main(model_name: str, trials: int, storage: str, save: bool) -> None:
    split = load_split(500_000)
    X_train, X_val, y_train, y_val, _ = split
    X_tr, y_tr = X_train.sample(n=200_000, random_state=RANDOM_STATE), None
    y_tr = y_train.loc[X_tr.index]
    X_va, y_va = X_val.iloc[:200_000], y_val.iloc[:200_000]

    setup_mlflow()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name="optuna-search" if model_name == "rf" else f"optuna-search-{model_name}"):
        study_name = f"{model_name}-search"
        try:  # a re-run replaces the old study instead of adding to it
            optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.set_user_attr("rows", "200k train sample, 200k validation slice")
        study.optimize(objective_factory(X_tr, y_tr, X_va, y_va, model_name), n_trials=trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_search_rmse", study.best_value)
    print(f"best search RMSE {study.best_value:.4f} with {study.best_params}")

    if model_name == "rf":
        tuned, metrics = fit_and_log(study.best_params, split, "rf-tuned", {"tuned_with": "optuna"})
    else:
        tuned, metrics = fit_hgb_and_log(study.best_params, split)
    print(json.dumps({"tuned": study.best_params, **{k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in metrics.items()}}, indent=2))
    if model_name == "rf":
        print(f"baseline params were {BASELINE_PARAMS}")
    if save:
        path = model_io.save(tuned, metadata={
            "run_id": metrics["run_id"], "run_name": f"{model_name}-tuned", "params": study.best_params,
            "rmse": metrics["rmse"], "mae": metrics["mae"],
        })
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
