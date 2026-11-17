# Model Deployment Project: project notes

_Last updated: 17 Nov 2026_

## Project
**Model Deployment Project.** Predict how long a NYC yellow-taxi trip will take, track the
experiments in MLflow and serve the model through a local FastAPI app.

- **Data:** NYC TLC Yellow Taxi trips, January 2025 (3,475,226 trips, 20 columns), downloaded
  from the TLC CloudFront URL into `data/`.
- **Target:** `duration` = dropoff − pickup in minutes.
- **Model:** scikit-learn `RandomForestRegressor`; stretch tasks: tests and Optuna tuning.

## Approach
1. **Explore** (`notebooks/01-explore-and-compare.ipynb`): duration ranges from −51,472 to
   5,626 minutes in the raw file; 124 negative and 20 over 24 hours.
2. **Clean** (`src/data.py`): keep 1-60 minutes, 0 < distance ≤ 100 miles, January pickups →
   3,339,084 trips (96.1%).
3. **Features** (`src/features.py`): pickup zone, dropoff zone, distance, pickup hour, weekday;
   only what is known when a trip starts (no fare or tip, which would leak the target).
4. **Train and track** (`src/train.py`): 80/20 split (seed 42), 500,000-row training sample,
   full validation set; params, RMSE, MAE and the model logged to MLflow (`mlflow.db` +
   `mlruns/`); model saved with skops to `artifacts/model.skops`.
5. **Tune** (`src/tune.py`): Optuna, 20 trials on 200,000 rows as nested MLflow runs; best
   parameters retrained on the same 500,000 rows and compared with the baseline.
6. **Serve** (`app/`): FastAPI `/health`, `/predict`, `/predict_batch`; model loaded once;
   Pydantic validation. Run with uvicorn; Dockerfile and compose file included.
7. **Test** (`tests/`): data cleaning, features, model save/load, API.

## Results
Real run on 17 Nov 2026 (Mac, uvicorn).

| Check | Result |
|---|---|
| Clean data | **3,339,084** of 3,475,226 trips kept; validation set **667,817** trips |
| Correlation with duration | trip_distance **0.80**; zones −0.12 / −0.10; hour and weekday ≈ 0 (linear) |
| Mean-duration baseline | RMSE **9.78** min |
| **Baseline RandomForest** (100 trees, depth 20, min leaf 10) | **RMSE 4.10**, MAE 2.76, fit 9.9 s, model **291 MB** (served) |
| Optuna (20 trials, 200k rows) | search RMSE 4.136 best, 4.333 worst; best: 100 trees, depth 26, min leaf 3, max_features 0.6 |
| Tuned RandomForest (same 500k rows) | RMSE **4.03**, MAE 2.71, fit 6.7 s, model **1,153 MB** |
| Tuning worth it? | No for serving: 1.7% (≈4 s) better for a 4× larger model |
| API `sample-request.json` (161 → 236, 2.5 mi, Wed 08:30) | **18.36 min** in 0.04 s |
| Same trip at 03:00 / 17:30 | 11.23 / 17.13 min; JFK → Times Square 17 mi at 17:30: 49.09 min |
| Invalid input | zone 999, distance 0, missing time → HTTP 422 |
| Tests | **10 / 10 pass** |
| MLflow | MLflow 3.17 rejects the README's `./mlruns` file store; switched to `sqlite:///mlflow.db` (UI checked) |
| Docker | Dockerfile and compose written; **not run**: Docker Desktop was not installed on the Mac on 17 Nov |

**Answers:** final RMSE **4.10 minutes** (served baseline). With more time: route speed by hour,
airport and weather features; gradient boosting for a smaller model; validate on February 2025;
load from the MLflow registry by alias; CI.

## Future engineering
- Features for the remaining error: route-level average speed by hour, airport flags, weather.
- Gradient boosting (LightGBM / HistGradientBoosting) for similar accuracy with a far smaller model.
- Out-of-time validation on February 2025.
- Load the model from the MLflow Model Registry by alias; CI for tests and the Docker image.
- Run and check the Docker path once Docker Desktop is back.

## Update log
- 17 Nov 2026: project built from the template: data, features, training with MLflow, Optuna tuning, FastAPI, tests, Docker files, notebook, README results; deck added.
