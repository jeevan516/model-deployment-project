"""Request and response bodies for the prediction API."""

from datetime import datetime

from pydantic import BaseModel, Field


class Trip(BaseModel):
    """What is known when a yellow-taxi trip starts."""

    PULocationID: int = Field(ge=1, le=265, description="Pickup taxi zone (1-265)", examples=[161])
    DOLocationID: int = Field(ge=1, le=265, description="Dropoff taxi zone (1-265)", examples=[236])
    trip_distance: float = Field(gt=0, le=100, description="Trip distance in miles", examples=[2.5])
    pickup_datetime: datetime = Field(description="Pickup time", examples=["2025-01-15T08:30:00"])


class Prediction(BaseModel):
    duration_minutes: float = Field(description="Predicted trip duration in minutes")
    low_minutes: float | None = Field(default=None, description="10th percentile: 9 in 10 trips take longer")
    high_minutes: float | None = Field(default=None, description="90th percentile: 9 in 10 trips are faster")
    model_run_id: str | None = Field(default=None, description="MLflow run that produced the model")
