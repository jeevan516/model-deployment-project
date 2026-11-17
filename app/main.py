"""Prediction API: uv run uvicorn app.main:app --reload --port 8000

The model is loaded once (from MODEL_PATH, default artifacts/model.skops) and reused.
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
    try:
        return model_io.load(model_path()), model_io.load_metadata(model_path())
    except FileNotFoundError as exc:
        raise HTTPException(503, f"No model at {model_path()}. Train one: uv run python -m src.train") from exc


def _predict(trips: list[Trip]) -> list[float]:
    model, _ = get_model()
    raw = pd.DataFrame([t.model_dump() for t in trips])
    X = build_features(raw, pickup_column="pickup_datetime")
    return [round(float(p), 2) for p in model.predict(X)]


@app.get("/health")
def health() -> dict:
    _, meta = get_model()
    return {"status": "ok", "model_run_id": meta.get("run_id"), "model_run_name": meta.get("run_name"),
            "validation_rmse": meta.get("rmse")}


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip) -> Prediction:
    _, meta = get_model()
    return Prediction(duration_minutes=_predict([trip])[0], model_run_id=meta.get("run_id"))


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(trips: list[Trip]) -> list[Prediction]:
    if not trips:
        return []
    _, meta = get_model()
    return [Prediction(duration_minutes=d, model_run_id=meta.get("run_id")) for d in _predict(trips)]
