import pytest

from src import checks, data


def test_raw_report_counts_each_problem(raw_trips):
    report = checks.report(raw_trips).set_index("rule")["failing_rows"]
    assert report["duration under 1 minute"] == 1
    assert report["duration over 60 minutes"] == 1
    assert report["distance 0 or negative"] == 1
    assert report["pickup outside the month"] == 1


def test_cleaned_data_passes_every_rule(raw_trips):
    cleaned = data.clean(raw_trips)
    assert checks.report(cleaned)["failing_rows"].sum() == 0
    checks.assert_clean(cleaned)


def test_assert_clean_stops_bad_data(raw_trips):
    with pytest.raises(ValueError, match="failed checks"):
        checks.assert_clean(raw_trips)


def test_missing_column_is_reported(raw_trips):
    with pytest.raises(ValueError, match="missing required columns"):
        checks.report(raw_trips.drop(columns="trip_distance"))
