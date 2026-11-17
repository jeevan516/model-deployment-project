import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "sample-request.json").read_text())


def test_health_reports_the_model(tiny_model_path):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_run_id"] == "test-run"


def test_predict_returns_minutes_for_the_sample_request(tiny_model_path):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 1 < body["duration_minutes"] < 60
    assert body["model_run_id"] == "test-run"


def test_predict_batch_keeps_order(tiny_model_path):
    trips = [{**SAMPLE, "trip_distance": 1.0}, {**SAMPLE, "trip_distance": 15.0}]
    short, long = client.post("/predict_batch", json=trips).json()
    assert long["duration_minutes"] > short["duration_minutes"]
    assert client.post("/predict_batch", json=[]).json() == []


def test_invalid_input_is_rejected(tiny_model_path):
    for bad in ({**SAMPLE, "trip_distance": 0}, {**SAMPLE, "PULocationID": 999},
                {k: v for k, v in SAMPLE.items() if k != "pickup_datetime"}):
        assert client.post("/predict", json=bad).status_code == 422
