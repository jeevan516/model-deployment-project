"""Answer 5b: monitor predictions against what actually happened.

Replays February 2025 through the served model as if it had been live, compares each day's
predictions with the real durations, and flags days whose MAE is more than 20% above the
January validation MAE (the level the model was accepted at). Writes reports/monitoring_2025-02.csv.

    uv run python -m src.monitor
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import data, features, model_io

ALERT_FACTOR = 1.2


def daily_report(pickup: pd.Series, y, pred, reference_mae: float) -> pd.DataFrame:
    err = np.asarray(pred) - np.asarray(y)
    frame = pd.DataFrame({"day": pd.to_datetime(pickup).dt.date.to_numpy(), "abs_err": np.abs(err), "err": err})
    out = frame.groupby("day").agg(trips=("err", "size"), mae=("abs_err", "mean"), bias=("err", "mean"))
    out["alert"] = out["mae"] > reference_mae * ALERT_FACTOR
    return out.round(3)


def main() -> None:
    model, meta = model_io.load(), model_io.load_metadata()
    feb = data.clean(data.load(data.download(2025, 2)), year=2025, month=2)
    pred = model.predict(features.build_features(feb))
    reference = float(meta.get("mae", 2.48))
    report = daily_report(feb["tpep_pickup_datetime"], feb[data.TARGET], pred, reference)
    out = data.PROJECT_ROOT / "reports" / "monitoring_2025-02.csv"
    out.parent.mkdir(exist_ok=True)
    report.to_csv(out)
    alerts = report[report["alert"]]
    print(json.dumps({"reference_mae": round(reference, 3), "threshold": round(reference * ALERT_FACTOR, 3),
                      "days": len(report), "alert_days": len(alerts),
                      "worst_day": str(report["mae"].idxmax()), "worst_mae": float(report["mae"].max()),
                      "best_mae": float(report["mae"].min()), "month_mae": round(float(np.mean(np.abs(pred - feb[data.TARGET]))), 3),
                      "alerts": [str(d) for d in alerts.index]}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
