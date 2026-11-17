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
