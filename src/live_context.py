"""Same-day context that the five base features cannot see: live traffic and weather.

Both are known at pickup time, in principle:

- live_rate_prev_hour: citywide median minutes per mile of the taxi trips that *ended* in the
  hour before this pickup (a dispatcher sees this from other cabs). A trip never contributes
  to its own value, because its own dropoff is after its own pickup.
- live_trips_prev_hour: how many trips ended in that hour (demand / congestion proxy).
- precipitation_mm, snowfall_cm, snow_depth_m, temperature_c, wind_kmh: hourly NYC weather
  (Open-Meteo archive, 40.75 N, 73.98 W, local time), cached in data/.

This is an offline experiment: the API would need a live traffic feed and a weather forecast
call to use it (see PROJECT-NOTES.md, Future engineering).
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.data import DATA_DIR

WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=40.75&longitude=-73.98"
               "&start_date={start}&end_date={end}&hourly=temperature_2m,precipitation,snowfall,snow_depth,"
               "wind_speed_10m&timezone=America%2FNew_York")
WEATHER_COLUMNS = {"temperature_2m": "temperature_c", "precipitation": "precipitation_mm",
                   "snowfall": "snowfall_cm", "snow_depth": "snow_depth_m", "wind_speed_10m": "wind_kmh"}
CONTEXT_FEATURES = ["live_rate_prev_hour", "live_trips_prev_hour", *WEATHER_COLUMNS.values()]


def weather(start: str = "2025-01-01", end: str = "2025-02-28") -> pd.DataFrame:
    """Hourly NYC weather indexed by local hour; downloaded once and cached."""
    path = DATA_DIR / f"weather_nyc_{start}_{end}.json"
    if not path.exists():
        with urlopen(WEATHER_URL.format(start=start, end=end), timeout=30) as resp:
            Path(path).write_bytes(resp.read())
    hourly = json.loads(Path(path).read_text())["hourly"]
    frame = pd.DataFrame(hourly).rename(columns=WEATHER_COLUMNS)
    return frame.assign(hour=pd.to_datetime(frame.pop("time"))).set_index("hour")


def live_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Per clock hour H: median minutes-per-mile and count of trips that ended during H-1."""
    rate = df["duration"] / df["trip_distance"].clip(lower=0.1)
    ended = df["tpep_dropoff_datetime"].dt.floor("h")
    stats = pd.DataFrame({"hour": ended, "rate": rate}).groupby("hour")["rate"].agg(["median", "size"])
    stats.index = stats.index + pd.Timedelta(hours=1)  # usable from the next hour on
    return stats.rename(columns={"median": "live_rate_prev_hour", "size": "live_trips_prev_hour"})


def add_context(X: pd.DataFrame, pickup: pd.Series, traffic: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Return X with the context columns, matched on the pickup's clock hour."""
    hour = pd.to_datetime(pickup).dt.floor("h").to_numpy()
    out = X.copy()
    t = traffic.reindex(hour)
    w = wx.reindex(hour)
    out["live_rate_prev_hour"] = t["live_rate_prev_hour"].to_numpy()
    out["live_trips_prev_hour"] = t["live_trips_prev_hour"].fillna(0).to_numpy()
    for col in WEATHER_COLUMNS.values():
        out[col] = w[col].to_numpy()
    return out
