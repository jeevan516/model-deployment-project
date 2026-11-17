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
1. **Explore** (notebook 01): raw durations range from −51,472 to 5,626 minutes.
2. **Check and clean** (`src/checks.py`, `src/data.py`, notebook 02): 9 rules with failure counts;
   keep 1-60 min, 0 < distance ≤ 100 miles, pickups in the month; `assert_clean` stops training
   if cleaned data breaks a rule.
3. **Features** (`src/features.py`): pickup zone, dropoff zone, distance, pickup hour, weekday;
   only start-of-trip information (no fare or tip: leakage).
4. **Baseline** (`src/train.py`): the assignment's RandomForest, logged to MLflow (`mlflow.db` + `mlruns/`).
5. **Six approaches** (`src/compare.py`): route-speed rule (no ML), Ridge, RandomForest,
   HistGradientBoosting, XGBoost, LightGBM; same 500k rows; scored on January and February 2025.
6. **Tune** (`src/tune.py`): Optuna, 20 trials each for RandomForest and HistGradientBoosting;
   studies stored in `optuna.db` for optuna-dashboard.
7. **Fit analysis** (notebook 03): train vs validation, learning curve, 5-fold CV, residuals by
   segment, permutation importance, February.
8. **Serve** (`app/`): HistGradientBoosting trained on all 2.67M training rows; FastAPI
   `/health`, `/predict`, `/predict_batch`; Docker container trains its model on first start if none is mounted.
9. **Test and CI**: 19 pytest tests; GitHub Actions runs them, builds the image and checks a fresh container.
10. **Long-trip experiment** (`src/long_trips.py`): route features learned out-of-fold from training trips
    (`src/route_features.py`: minutes per mile per zone pair and time-of-day band, pickup zone by hour,
    airport flags) and same-day context (`src/live_context.py`: citywide speed of trips that ended in the
    hour before pickup, hourly NYC weather from the Open-Meteo archive). Offline only; the API is unchanged.

## Results
Real runs on 17 Nov 2026 (Mac). Validation: 667,817 January trips; out-of-time: 3,427,258 February trips.

| Data step | Trips |
|---|---|
| Raw January | 3,475,226 |
| Duration 1-60 min | 3,403,248 (drops 38,060 under 1 min, 31,867 over 60) |
| Distance > 0 | 3,339,226 (drops 90,893: the biggest problem) |
| Distance ≤ 100, pickup in January | **3,339,084** (96.1%); all 9 rules then 0 failures |

| Model (500k rows unless noted) | RMSE Jan | RMSE Feb | Size |
|---|---|---|---|
| Mean duration | 9.78 | 9.72 | |
| Route speed rule (no ML) | 6.70 | 6.99 | 1.2 MB |
| Ridge, one-hot zones | 5.47 | 5.61 | 0.1 MB |
| RandomForest (assignment baseline) / + Optuna | 4.10 / 4.03 | 4.27 | 291 / 1,153 MB |
| XGBoost | 3.79 | 3.97 | 7.6 MB |
| HistGradientBoosting / + Optuna | 3.78 / 3.77 | 3.96 | 8.6 MB |
| LightGBM | 3.77 | 3.95 | 5.6 MB |
| **HistGradientBoosting, all 2.67M rows (served)** | **3.70** | **3.88** | **9.0 MB** |
| LightGBM, all rows | 3.69 | 3.88 | |

| Fit check (notebook 03) | Result |
|---|---|
| Train vs validation | linear 5.43/5.47 (underfits); forest 3.62/4.10 (gap 0.48); served 3.63/3.70 (gap 0.07) |
| Learning curve 25k → 500k | 4.22 → 3.78, still improving → train on all rows |
| 5-fold CV (200k) | 3.92 ± 0.03 |
| Residuals | mean +0.001 min; 56% within 2 min, 88% within 5 min |
| Weak spot | 45-60 min trips: MAE 8.8, underpredicted by about 8 min; airport trips MAE 4.1 vs 2.4 |
| Permutation importance | distance 10.1, hour 1.4, zones 0.8, weekday 0.5 |
| API (Docker) | sample request **19.39 min**; 03:00 → 12.35; JFK → Times Square 50.6; ~240 MB memory |
| Tests / CI | 19 / 19 locally (15 in the last CI run) |

Long-trip experiment, all 2,671,267 training rows (RMSE in minutes; MAE for segments):

| Model | RMSE Jan | RMSE Feb | Airport MAE Jan / Feb | 45-60 min bias Feb |
|---|---|---|---|---|
| Served HistGradientBoosting | 3.70 | 3.88 | 4.01 / 4.21 | −9.36 |
| + route speed by hour, airport flags | 3.66 | 3.84 | 3.99 / 4.20 | −9.20 |
| + live traffic and weather | **3.52** | **3.81** | 3.63 / 4.22 | −8.44 |

The January gain (−0.18) is inflated: validation trips share hours and weather with training trips.
On February the gain is −0.07. Long trips stay underpredicted by about 8 minutes.

**Answers:** final RMSE **3.70 minutes** (3.88 on February); traffic and weather features reach 3.81 on February offline. The RandomForest baseline reaches
4.10. Boosting libraries tie; tuning added ≤0.07, more training data added 0.08. With more time:
traffic features for long trips, multi-month training with drift checks, registry-based serving.

## Future engineering
- Validate by time (earlier days train, later days validate): a random split within one month inflated the
  same-day feature gain from 0.07 to 0.18.
- Live traffic feed and weather forecast in the API (worth 0.07 RMSE on February offline).
- Train on several months, validate on the next; monthly drift check (February 3.88 vs January 3.70).
- Prediction ranges (quantile models) for long trips, still underpredicted by about 8 minutes.
- Load the model from the MLflow Model Registry by alias; monitor predictions against actual durations.

## Update log
- 17 Nov 2026: project built from the template: data, features, training with MLflow, Optuna tuning, FastAPI, tests, Docker files, notebook, README results; deck added.
- 17 Nov 2026: more approaches (Ridge, HistGradientBoosting) with a February 2025 out-of-time check; HistGradientBoosting tuned with Optuna; API now serves HistGradientBoosting (RMSE 3.78, 8.6 MB); Docker run checked, container trains its model on first start; Portainer files for Mac and Windows; 11 tests; deck updated.
- 17 Nov 2026: GitHub Actions CI added and passing (tests, Docker build, first-start train-and-serve check).
- 17 Nov 2026: data checks (`src/checks.py`, 9 rules, training gate); XGBoost, LightGBM and a route-speed baseline; Optuna studies stored for optuna-dashboard; notebooks 02 (data) and 03 (fit); served model retrained on all rows (RMSE 3.70); 15 tests; deck updated.
- 17 Nov 2026: long-trip experiment (route features, live traffic, weather): February RMSE 3.88 → 3.81, January gain inflated by the random split; 4 new tests (19); deck redesigned (13 slides, native charts) with the experiment and an updated "more time" slide.
