"""Train the served model (HistGradientBoosting) if MODEL_PATH does not exist yet.

Used by the Docker entrypoint so a fresh container works without a pre-trained file:
it downloads January 2025, trains on all 2.67 million training rows (TRAIN_SIZE to change)
and saves the model plus metadata. MLflow is not needed here.

    uv run python -m src.prepare_model
"""

from __future__ import annotations

import os
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src import features, model_io
from src.compare import make_model
from src.train import load_split


def prepare_ranges(path: Path) -> None:
    """With TRAIN_RANGES=1, also train the 10th/90th percentile models so /predict returns a range."""
    from src.intervals import QUANTILES, make_quantile_model, quantile_path
    if os.getenv("TRAIN_RANGES", "0") != "1" or all(quantile_path(n, path).exists() for n in QUANTILES):
        return
    print("training the 10th/90th percentile models for prediction ranges (2-4 minutes)")
    X_train, _, y_train, _, _ = load_split(int(os.getenv("TRAIN_SIZE", "0")) or None)
    for name, q in QUANTILES.items():
        model_io.save(make_quantile_model(q).fit(X_train, y_train), quantile_path(name, path), metadata={"quantile": q})
    print("saved range models")


def main() -> None:
    path = Path(os.getenv("MODEL_PATH", str(model_io.DEFAULT_MODEL_PATH)))
    if path.exists():
        print(f"model already at {path}")
        prepare_ranges(path)
        return
    print("no model yet: downloading January 2025 and training HistGradientBoosting on all rows (1-2 minutes)")
    X_train, X_val, y_train, y_val, _ = load_split(int(os.getenv("TRAIN_SIZE", "0")) or None)
    model = make_model("hgb").fit(X_train, y_train)
    pred = model.predict(X_val)
    rmse, mae = float(root_mean_squared_error(y_val, pred)), float(mean_absolute_error(y_val, pred))
    model_io.save(model, path, metadata={"run_id": None, "run_name": "hgb-container", "train_rows": len(X_train), "rmse": rmse,
                                         "mae": mae, "features": features.FEATURES})
    print(f"saved {path}: validation RMSE {rmse:.3f}, MAE {mae:.3f}")
    prepare_ranges(path)


if __name__ == "__main__":
    main()
