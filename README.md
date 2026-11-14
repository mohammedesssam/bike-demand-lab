# Bike Demand Lab

## Repository packaging

The actual Python training, API and test source is browsable in this repository.
Original data, trained models, plots and execution evidence are supplied in a supplementary checksummed base64 archive because ordinary binary uploads failed. No external download or account is needed to restore it.

After cloning or downloading this repository, run:

```bash
python restore_artifacts.py
cd restored/bike-demand-lab
```

Then follow the original setup and test commands below from that restored directory. The archive SHA-256 is `304e3281a5ad20423b7e0dc5be5b09cde3ec7ea1cd05b4fe99e23ddcb54bdd0e`. Restoration only writes into a new `restored` directory and does not execute models or install anything. The source files in the restored package match the inspected source; its original README predates publication.

Repository publication is not cloud deployment. The app was exercised locally; no cloud-hosted prediction endpoint is claimed. GitHub Actions and Docker were not executed. This is newly AI-assisted educational work, not proof of prior independent applicant experience.


A small, reproducible, **AI-assisted educational project**: historical daily bike rental-count estimation from weather and calendar variables. Includes data validation, preprocessing, model comparison, a separate trained PyTorch network, untouched chronological evaluation, a FastAPI service and browser UI.

**Status:** trained and tested locally. Published here as source plus a restorable supplementary package; not hosted as a cloud service. Docker configuration is supplied but has not been built or run. This is not evidence of any applicant's prior independent experience. The applicant must understand, verify, adapt and accurately describe the work before using it in an application.

## The question and scope

Given a date and the day's weather, estimate the number of recorded Capital Bikeshare rentals in Washington, DC. Data covers 2011-2012. This is a weather-conditional historical regression demo, **not** validated next-day forecasting, unmet demand estimation, station availability, or a system for present-day Cairo. Observed same-day weather would not be known when scheduling bikes the night before. Operational forecasting would need forecast-weather inputs and a new backtest.

## Run the service

Tested with Python 3.10; package versions are pinned to those actually used. A fresh environment is recommended.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows alternative: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn bike_demand.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` locally. `/docs` has interactive API documentation, `/metrics` the complete measured report and `/health` readiness. Model artifacts are included, so serving does not need PyTorch or retraining. Never load unknown joblib/pickle artifacts: they can execute code.

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"day":"2012-10-15","temperature_normalized":0.45,"humidity_percent":65,"windspeed_normalized":0.18,"weather":1,"holiday":false}'
```

The date is deliberately limited to 2011-2012. `temperature_normalized` uses the original **daily UCI temp index**, not Celsius. Published hourly and daily temperature-normalization descriptions differ, so this demo does not invent a shared Celsius conversion. Humidity percent is divided by 100. Wind uses the source's normalized value (wind divided by 67). Calendar features are derived from date; no manual weekday mismatch is possible. Holiday means the source's US holiday indicator, not Egyptian holidays. Weather categories: 1 clear/partly cloudy; 2 mist/cloudy; 3 light rain/snow. Category 4 was not observed in the daily data and is rejected.

## Reproduce training and tests

```bash
python -m pip install -r requirements-dev.txt
# CPU PyTorch version used is in TORCH_VERSION.txt. Install separately to avoid GPU dependencies:
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m bike_demand.train
python -m pytest -q
```

If the displayed PyTorch install version differs from `TORCH_VERSION.txt`, use the numeric version in that file (omit its `+cpu` suffix). Results can differ slightly across library versions/architectures; seed 42 and deterministic CPU operations are set. Full retraining needs PyTorch, while runtime and default tests do not. Optional checkpoint test: `python -m bike_demand.check_torch`.

Training reads the included source CSV. It checks expected row count, unique dates, missing values and `cnt == casual + registered`. Numeric median imputation and standardization are fitted on **training data only**; categoricals use one-hot encoding. Missing inputs are rejected by the API; imputers make offline preprocessing explicit rather than changing the source file.

### Chronological evaluation

| Split | Dates | Rows | Purpose |
|---|---|---:|---|
| Training | 2011-01-01 to 2012-06-30 | 547 | Fit preprocessing and weights |
| Validation | 2012-07-01 to 2012-09-30 | 92 | Choose scikit-learn model and PyTorch checkpoint |
| Test | 2012-10-01 to 2012-12-31 | 92 | One final held-out comparison |

Nine input features: temperature, humidity, wind, linear time trend, sine/cosine month, weekday, holiday, weather category. `cnt` is the target. `casual` and `registered` sum exactly to it and are excluded. Row ID is also excluded. No target lags or future values are used.

Five scikit-learn candidates: Ridge alpha 1/10/100; histogram gradient boosting with 7/15 leaves, 150 iterations, learning rate 0.05 and L2 regularization 10. The minimum validation MAE among these selects the served estimator. The final model is intentionally **not refitted** on validation or test, keeping the reported deployed artifact identical to the evaluated artifact.

PyTorch comparison: one hidden layer of 24 ReLU units, Adam learning rate 0.01, weight decay 0.001; 600 full-batch epochs. Inputs and target are standardized with training-only statistics. Checkpoint epoch 213 has the lowest validation MAE. The network is a separate comparison, not the served model. Small tabular data does not automatically benefit from a neural network.

### Measured held-out results

| Model | MAE (rentals/day) | RMSE | R² |
|---|---:|---:|---:|
| Selected gradient boosting, 7 leaves | 1,033.87 | 1,258.90 | 0.5699 |
| PyTorch MLP | 1,080.16 | 1,303.20 | 0.5391 |
| Training-only weekday-mean baseline | 1,966.06 | 2,241.45 | -0.3636 |

The selected model reduces MAE by about 47.4% **relative to this simple static weekday baseline**, not every possible forecasting baseline. MAE is an average absolute count error, not "accuracy percent". R² = 0.570 does not mean 57% accuracy. The validation and test seasons differ; the test MAE is worse than validation MAE (663.87), a warning about generalization. These are one time window's results, not production guarantees.

See `reports/metrics.json`, row-level `reports/test_predictions.csv`, and the plot below. No model choice was made from test results. Next improvements would require a new validation protocol rather than repeatedly optimizing this already-seen test set.

Plot: `reports/test_predictions.png` inside the restored project package.

## Deployment

A real local HTTP service and browser flow were exercised. GitHub source publication is separate from cloud hosting. Cloud hosting remains undone and must not be described as completed deployment.

Docker configuration (not executed in this environment):
```bash
docker build -t bike-demand-lab .
docker run --rm -p 8000:8000 bike-demand-lab
```

For an approved container host: use the Dockerfile, set `PORT` if the host needs it, use `/health` as the health-check path, then open `/`, run `/predict`, and verify returned model and metrics against this report. Do not claim cloud deployment until those live checks pass. No hosting platform or cost is assumed. Public hosting needs HTTPS, request limits, monitoring and a review of costs; this demo has no authentication and accepts no personal data.

## Project map

- `data/day.csv`, `data/Readme.txt`: original UCI daily data and source description.
- `bike_demand/features.py`: shared training/serving contract.
- `bike_demand/train.py`: data checks, training, validation, test, plot and artifacts.
- `bike_demand/api.py`: validated FastAPI API and UI routes.
- `bike_demand/check_torch.py`: reload and verify the actual PyTorch checkpoint.
- `static/index.html`: responsive scenario UI.
- `tests/`: automated data, API, parity and metric tests.
- `reports/`: exact results, row predictions, plot and execution evidence.
- `artifacts/`: fitted scikit-learn and PyTorch models plus preprocessing.
- `EXPLAIN_AR.md`: short Egyptian Arabic explanation and practice questions.
- `ATTRIBUTION.md`, `CONTRIBUTIONS.md`: source licensing and honest work provenance.
- `.github/workflows/tests.yml`: proposed CI definition. Not yet executed on GitHub.

## Limits and next work

Only 731 rows from one city and two years. One holdout window; no uncertainty calibration. Abnormal events can cause large errors. Observational correlations are not causal effects. The trend is unsuitable for extrapolating years beyond training. A stronger future study would use rolling-origin validation, richer baselines, independent recent data and weather forecasts. User understanding and application eligibility are separate from code completion.
