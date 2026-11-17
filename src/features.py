"""Feature preparation shared by training and the API.

Only fields known when a trip starts are used: pickup and dropoff zone, distance and pickup
time. Fare, tip and total amount are known only after the trip, so they would leak the target.
"""

from __future__ import annotations

import pandas as pd

FEATURES = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour", "pickup_weekday"]


def build_features(df: pd.DataFrame, pickup_column: str = "tpep_pickup_datetime") -> pd.DataFrame:
    """Return the model's feature table from raw taxi columns."""
    pickup = pd.to_datetime(df[pickup_column])
    return pd.DataFrame(
        {
            "PULocationID": df["PULocationID"].astype("int32"),
            "DOLocationID": df["DOLocationID"].astype("int32"),
            "trip_distance": df["trip_distance"].astype("float64"),
            "pickup_hour": pickup.dt.hour.astype("int32"),
            "pickup_weekday": pickup.dt.dayofweek.astype("int32"),  # Monday = 0
        }
    )[FEATURES]
