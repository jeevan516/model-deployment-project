"""Answer 5a: serve from the MLflow Model Registry by alias.

Logs the served model file (artifacts/model.skops) to MLflow, registers it as a new version of
"taxi-duration" and points the alias "champion" at it. The API loads
models:/taxi-duration@champion when MODEL_URI is set, so a new model goes live (or is rolled
back) by moving the alias, without rebuilding the container.

    uv run python -m src.registry                 # register artifacts/model.skops as champion
    uv run python -m src.registry --alias challenger
    MODEL_URI=models:/taxi-duration@champion uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import click
import mlflow
from mlflow import MlflowClient

from src import model_io
from src.train import setup_mlflow

MODEL_NAME = "taxi-duration"


def register(model_path=model_io.DEFAULT_MODEL_PATH, alias: str = "champion") -> str:
    """Register the model file as a new version and set `alias` on it. Returns the version."""
    model, meta = model_io.load(model_path), model_io.load_metadata(model_path)
    setup_mlflow()
    with mlflow.start_run(run_name=f"register-{alias}") as run:
        mlflow.log_params({"source_file": str(model_path), "source_run_id": meta.get("run_id") or "",
                           "train_rows": meta.get("train_rows") or 0})
        mlflow.log_metrics({k: meta[k] for k in ("rmse", "mae", "feb_rmse") if isinstance(meta.get(k), (int, float))})
        info = mlflow.sklearn.log_model(sk_model=model, name="model", serialization_format="skops",
                                        skops_trusted_types=model_io.TRUSTED_TYPES,
                                        registered_model_name=MODEL_NAME)
        mlflow.set_tag("alias", alias)
    version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"registered {MODEL_NAME} version {version} as @{alias} (run {run.info.run_id})")
    return version


def load(uri: str):
    """Load a registered model, e.g. models:/taxi-duration@champion."""
    setup_mlflow()
    return mlflow.sklearn.load_model(uri)


def describe(uri: str) -> dict:
    """Run id, name and validation RMSE behind models:/<name>@<alias>."""
    name, alias = uri.removeprefix("models:/").split("@")
    setup_mlflow()
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)
    return {"run_id": version.run_id, "run_name": f"{name} v{version.version} @{alias}",
            "rmse": run.data.metrics.get("rmse")}


@click.command()
@click.option("--alias", default="champion", show_default=True)
def main(alias: str) -> None:
    register(alias=alias)


if __name__ == "__main__":
    main()
