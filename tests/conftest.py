import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src import model_io
from src.features import FEATURES


@pytest.fixture
def raw_trips() -> pd.DataFrame:
    pickup = pd.to_datetime(["2025-01-15 08:30", "2025-01-15 09:00", "2025-01-31 23:50",
                             "2025-01-10 12:00", "2024-12-31 23:00", "2025-01-12 18:00"])
    minutes = [12, 0.5, 30, 75, 10, 20]  # 0.5 and 75 are outside 1-60
    return pd.DataFrame({
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + pd.to_timedelta(minutes, unit="min"),
        "PULocationID": [161, 161, 132, 1, 2, 236],
        "DOLocationID": [236, 237, 230, 2, 3, 161],
        "trip_distance": [2.5, 0.3, 18.0, 5.0, 1.0, 0.0],  # 0.0 is invalid
    })


@pytest.fixture
def tiny_model_path(tmp_path, monkeypatch):
    """A small forest trained on synthetic data where duration grows with distance."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "PULocationID": rng.integers(1, 266, n), "DOLocationID": rng.integers(1, 266, n),
        "trip_distance": rng.uniform(0.2, 20, n), "pickup_hour": rng.integers(0, 24, n),
        "pickup_weekday": rng.integers(0, 7, n),
    })[FEATURES]
    y = 3 + 3 * X["trip_distance"] + rng.normal(0, 1, n)
    model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=0).fit(X, y)
    path = model_io.save(model, tmp_path / "model.skops", metadata={"run_id": "test-run", "rmse": 1.0})
    monkeypatch.setenv("MODEL_PATH", str(path))
    from app import main
    main.get_model.cache_clear()
    yield path
    main.get_model.cache_clear()
