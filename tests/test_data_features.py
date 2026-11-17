from src import data
from src.features import FEATURES, build_features


def test_clean_keeps_only_valid_january_trips(raw_trips):
    cleaned = data.clean(raw_trips)
    # dropped: 0.5 min, 75 min, December pickup, zero distance
    assert len(cleaned) == 2
    assert cleaned[data.TARGET].between(1, 60).all()
    assert (cleaned["trip_distance"] > 0).all()
    assert (cleaned["tpep_pickup_datetime"].dt.month == 1).all()


def test_target_is_duration_in_minutes(raw_trips):
    cleaned = data.clean(raw_trips)
    assert cleaned.loc[0, data.TARGET] == 12


def test_features_have_the_expected_columns_and_values(raw_trips):
    X = build_features(raw_trips)
    assert list(X.columns) == FEATURES
    first = X.iloc[0]
    assert (first.pickup_hour, first.pickup_weekday) == (8, 2)  # Wednesday 08:30
    assert first.trip_distance == 2.5


def test_features_contain_no_post_trip_information(raw_trips):
    assert not {"fare_amount", "tip_amount", "total_amount", "tpep_dropoff_datetime"} & set(FEATURES)
