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
1. **Explore** (`notebooks/01-explore-and-compare.ipynb`): raw durations range from −51,472 to
   5,626 minutes; 124 negative and 20 over 24 hours.
2. **Clean** (`src/data.py`): keep 1-60 minutes, 0 < distance ≤ 100 miles, pickups in the month →
   3,339,084 January trips (96.1%).
3. **Features** (`src/features.py`): pickup zone, dropoff zone, distance, pickup hour, weekday;
   only what is known when a trip starts (no fare or tip, which would leak the target).
4. **Baseline** (`src/train.py`): the assignment's RandomForest on a 500,000-row sample of the 80%
   split (seed 42), scored on the full 20% validation set, logged to MLflow (`mlflow.db` + `mlruns/`).
5. **More approaches** (`src/compare.py`): Ridge with one-hot zones, RandomForest and
   HistGradientBoosting on the same rows, scored on January and on February 2025 (out-of-time).
6. **Tune** (`src/tune.py`): Optuna, 20 trials each for RandomForest and HistGradientBoosting.
7. **Serve** (`app/`): FastAPI `/health`, `/predict`, `/predict_batch`; serves the
   HistGradientBoosting model, loaded once; Pydantic validation.
8. **Docker** (`Dockerfile`, `docker-compose.yaml`, `portainer-windows.yaml`): the container trains
   the model on first start if none is mounted (`src/prepare_model.py`), so it runs anywhere.
9. **Test** (`tests/`): cleaning, features, model files (both tree types), API.

## Results
Real runs on 17 Nov 2026 (Mac). Validation: 667,817 January trips; out-of-time: 3,427,258 February trips.

| Model | RMSE Jan | RMSE Feb | MAE Jan | Size | Fit |
|---|---|---|---|---|---|
| Mean duration | 9.78 | 9.72 | | | |
| Ridge, one-hot zones | 5.47 | 5.61 | 3.92 | 0.1 MB | 0.3 s |
| RandomForest baseline (assignment) | 4.10 | 4.27 | 2.76 | 291 MB | 9.5 s |
| RandomForest + Optuna | 4.03 | | 2.71 | 1,153 MB | 6.7 s |
| **HistGradientBoosting (served)** | **3.78** | **3.96** | **2.53** | **8.6 MB** | 12 s |
| HistGradientBoosting + Optuna | 3.77 | | 2.52 | | 19 s |

| Check | Result |
|---|---|
| Correlation with duration | trip_distance **0.80**; zones −0.12 / −0.10; hour and weekday ≈ 0 (linear) |
| Optuna searches (200k rows) | RandomForest 4.136-4.333; HistGradientBoosting 3.866-4.015 |
| Tuning worth it? | No: RF +1.7% for 4× the size; HGB +0.01 min for 1,000 iterations |
| Out-of-time (February) | every model loses only 0.14-0.18 min RMSE |
| API in Docker (`sample-request.json`, 161 → 236, 2.5 mi, Wed 08:30) | **19.63 min** in 0.02 s; 03:00 → 12.65; 17:30 → 17.61; JFK → Times Square 50.42 |
| Container | healthy; about 220 MB memory; first start on an empty volume trains the model (RMSE 3.784) and serves it |
| Invalid input | zone 999, distance 0, missing time → HTTP 422 |
| Tests | **11 / 11 pass** |
| MLflow | MLflow 3.17 rejects the README's `./mlruns` file store; using `sqlite:///mlflow.db` |

**Answers:** final RMSE **3.78 minutes** (served HistGradientBoosting; 3.96 on February). The
assignment's RandomForest baseline reaches 4.10. With more time: route speed by hour, airport and
weather features; multi-month training with a drift check; serve from the MLflow registry; CI.

## Future engineering
- Features for the remaining error: route-level average speed by hour, airport flags, weather.
- Train on several months and validate on the next; monthly drift check on the API's inputs.
- Load the model from the MLflow Model Registry by alias; CI for tests and the Docker image.

## Update log
- 17 Nov 2026: project built from the template: data, features, training with MLflow, Optuna tuning, FastAPI, tests, Docker files, notebook, README results; deck added.
- 17 Nov 2026: more approaches (Ridge, HistGradientBoosting) with a February 2025 out-of-time check; HistGradientBoosting tuned with Optuna; API now serves HistGradientBoosting (RMSE 3.78, 8.6 MB); Docker run checked, container trains its model on first start; Portainer files for Mac and Windows; 11 tests; deck updated.
