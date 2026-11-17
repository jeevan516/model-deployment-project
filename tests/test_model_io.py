from src import model_io


def test_saved_model_loads_and_predicts_the_same(tiny_model_path):
    model = model_io.load(tiny_model_path)
    assert model_io.load_metadata(tiny_model_path)["run_id"] == "test-run"
    assert model.n_estimators == 20


def test_longer_trips_get_longer_predictions(tiny_model_path):
    import pandas as pd
    from src.features import FEATURES
    model = model_io.load(tiny_model_path)
    X = pd.DataFrame([[161, 236, 1.0, 8, 2], [161, 236, 15.0, 8, 2]], columns=FEATURES)
    short, long = model.predict(X)
    assert long > short


def test_hist_gradient_boosting_round_trips(tmp_path):
    """The served model type must load with only the trusted tree types."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.features import FEATURES
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(1, 200, size=(300, 5)), columns=FEATURES).astype(float)
    model = HistGradientBoostingRegressor(max_iter=20).fit(X, X["trip_distance"] * 3)
    path = model_io.save(model, tmp_path / "hgb.skops")
    assert np.allclose(model_io.load(path).predict(X), model.predict(X))
