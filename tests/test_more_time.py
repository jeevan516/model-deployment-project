"""The 'more time' answers: ranges in the API, drift, monitoring, registry."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app, get_range_models
from src import model_io
from src.drift import psi, status
from src.intervals import coverage, make_quantile_model, quantile_path
from src.monitor import daily_report

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample-request.json").read_text())


def test_no_range_without_quantile_models(tiny_model_path):
    body = client.post("/predict", json=SAMPLE).json()
    assert body["low_minutes"] is None and body["high_minutes"] is None


def test_range_surrounds_the_prediction(tiny_model_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"PULocationID": rng.integers(1, 266, 400), "DOLocationID": rng.integers(1, 266, 400),
                      "trip_distance": rng.uniform(0.2, 20, 400), "pickup_hour": rng.integers(0, 24, 400),
                      "pickup_weekday": rng.integers(0, 7, 400)})
    y = 3 + 3 * X.trip_distance + rng.normal(0, 2, 400)
    for name, q in (("low", 0.1), ("high", 0.9)):
        model_io.save(make_quantile_model(q).set_params(max_iter=30, min_samples_leaf=5).fit(X, y),
                      quantile_path(name, tiny_model_path))
    get_range_models.cache_clear()
    body = client.post("/predict", json=SAMPLE).json()
    assert body["low_minutes"] <= body["duration_minutes"] <= body["high_minutes"]
    assert client.get("/health").json()["ranges"] is True


def test_coverage_counts_trips_inside_the_range():
    out = coverage([10, 20, 50], np.array([5, 25, 40]), np.array([15, 30, 55]))
    assert out["coverage"] == 2 / 3 and out["coverage_long"] == 1.0


def test_psi_is_zero_for_same_data_and_large_for_shift():
    a = pd.Series(np.random.default_rng(0).normal(10, 2, 5000))
    assert psi(a, a, numeric=True) < 0.01
    assert status(psi(a, a + 5, numeric=True)) == "retrain"


def test_daily_report_flags_bad_days():
    pickup = pd.Series(pd.to_datetime(["2025-02-01 08:00", "2025-02-01 09:00", "2025-02-02 08:00"]))
    report = daily_report(pickup, [10, 10, 10], [11, 9, 20], reference_mae=1.0)
    assert not report.loc[pd.Timestamp("2025-02-01").date(), "alert"]
    assert report.loc[pd.Timestamp("2025-02-02").date(), "alert"]


def test_registry_round_trip(tiny_model_path, tmp_path, monkeypatch):
    from src import registry, train
    monkeypatch.setattr(train, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train, "ARTIFACT_LOCATION", (tmp_path / "mlruns").as_uri())
    version = registry.register(tiny_model_path, alias="champion")
    model = registry.load(f"models:/{registry.MODEL_NAME}@champion")
    X = pd.DataFrame([{"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5, "pickup_hour": 8, "pickup_weekday": 2}])
    assert version == "1"
    assert registry.describe(f"models:/{registry.MODEL_NAME}@champion")["rmse"] == 1.0
    assert np.allclose(model.predict(X), model_io.load(tiny_model_path).predict(X))
