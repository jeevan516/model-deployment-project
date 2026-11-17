"""Prediction API: uv run uvicorn app.main:app --reload --port 8000

The model is loaded once and reused: from the MLflow registry when MODEL_URI is set
(e.g. models:/taxi-duration@champion), otherwise from MODEL_PATH (default artifacts/model.skops).
If the 10th/90th percentile models (src/intervals.py) sit next to MODEL_PATH, every prediction
also gets a range.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import Prediction, Trip
from src import model_io
from src.features import build_features

app = FastAPI(title="NYC Yellow Taxi Trip Duration", version="1.0.0")


def model_path() -> str:
    return os.getenv("MODEL_PATH", str(model_io.DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_model():
    uri = os.getenv("MODEL_URI")
    if uri:
        from src import registry
        return registry.load(uri), registry.describe(uri)
    try:
        return model_io.load(model_path()), model_io.load_metadata(model_path())
    except FileNotFoundError as exc:
        raise HTTPException(503, f"No model at {model_path()}. Train one: uv run python -m src.train") from exc


@lru_cache(maxsize=1)
def get_range_models():
    """(low, high) quantile models if both files exist, else None."""
    from src.intervals import quantile_path
    paths = [quantile_path(n, model_path()) for n in ("low", "high")]
    if not all(p.exists() for p in paths):
        return None
    return tuple(model_io.load(p) for p in paths)


def _predict(trips: list[Trip]) -> list[Prediction]:
    model, meta = get_model()
    raw = pd.DataFrame([t.model_dump() for t in trips])
    X = build_features(raw, pickup_column="pickup_datetime")
    mid = model.predict(X)
    ranges = get_range_models()
    low = high = [None] * len(mid)
    if ranges:
        # keep the range around the point prediction even where the two quantile models disagree with it
        low = [min(a, m) for a, m in zip(ranges[0].predict(X), mid)]
        high = [max(b, m) for b, m in zip(ranges[1].predict(X), mid)]
    return [Prediction(duration_minutes=round(float(m), 2), model_run_id=meta.get("run_id"),
                       low_minutes=None if lo is None else round(float(lo), 2),
                       high_minutes=None if hi is None else round(float(hi), 2))
            for m, lo, hi in zip(mid, low, high)]


@app.get("/health")
def health() -> dict:
    _, meta = get_model()
    return {"status": "ok", "model_run_id": meta.get("run_id"), "model_run_name": meta.get("run_name"),
            "validation_rmse": meta.get("rmse"), "ranges": get_range_models() is not None}


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip) -> Prediction:
    return _predict([trip])[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(trips: list[Trip]) -> list[Prediction]:
    if not trips:
        return []
    return _predict(trips)
