I'm building a mini project for my Predictive Analytics course (CSE3141, Manipal University Jaipur). It will be evaluated in a viva, so every step must be explainable in plain language. Build it in phases. After EACH phase, stop, summarise what you did and what you found (with actual numbers from the data), and wait for my approval before continuing. Never invent or estimate results; only report numbers the code actually produced.

PROJECT: Forecasting Delhi's Air Quality Index (AQI) and predicting high-pollution days.

DATA: data/raw/city_day.csv (already downloaded and verified; source and CC0 license recorded in data/SOURCE.md). Do not modify the raw file. Do not generate or fill in synthetic data.

ENVIRONMENT: use the existing .venv (Python 3.12). Add packages to requirements.txt with pinned versions as you install them.

PROJECT STRUCTURE:
- data/raw/, data/processed/
- src/ (one module per phase: audit.py, clean.py, eda.py, forecast.py, classify.py)
- reports/figures/ (all plots saved as PNG, clear titles and axis labels)
- reports/results.md (metrics tables, filled only from real outputs)
- app.py (Streamlit)
- README.md
- Fixed random seed 42 everywhere. Comment code in plain English explaining WHY, not just what.
- Commit at the end of each phase with a clear message.

PHASE 1: DATA AUDIT (no changes to data)
- Load the CSV, confirm column names and dtypes.
- Filter City == "Delhi". Report row count, date range, missing % per column, duplicate dates, and any gaps in the daily date sequence.
- List the exact dates where AQI is missing.
- Stop and report.

PHASE 2: CLEANING
- Reindex to a continuous daily date range (confirm whether any dates were actually missing).
- Fill short AQI gaps (3 days or fewer) with time-based interpolation. Report any longer gaps and handle them explicitly, explaining the choice.
- For pollutant columns with heavy missingness, report the % and decide whether to drop, explaining why.
- Do NOT impute any pollutant column in Phase 2. Leave their NaNs in data/processed/delhi_daily.csv. Imputation happens later inside the Phase 5 pipeline, fit on training data only, to avoid leakage across the train/test split.
- Outliers: detect with IQR and box plots, but FLAG them, do not delete. Delhi smog spikes are real events, not errors. Explain this in the report.
- Save data/processed/delhi_daily.csv. Stop and report.

PHASE 3: EDA
- Plots: full AQI time series, monthly box plots (seasonality), histogram of AQI, correlation heatmap of pollutants, bar chart of AQI_Bucket counts.
- Summary stats: mean, median, std, variance of AQI by month and by year.
- Seasonal decomposition (trend, seasonality, residual) on the weekly series.
- Write 5 to 8 plain-language findings. Stop and report.

PHASE 4: FORECASTING (time series)
- Resample to weekly mean AQI.
- Split BY TIME: train = 2015-2018, test = 2019. Keep 2020 aside.
- Stationarity: ADF test, ACF and PACF plots, decide differencing.
- Models: (a) naive baseline (next week = this week), (b) seasonal naive (same week last year), (c) Holt-Winters exponential smoothing with seasonal_periods=52, (d) SARIMA with seasonal period 52, orders chosen by a small grid search on AIC (keep the grid small; report what was searched).
- Evaluate all four on 2019 with RMSE and MAE in one comparison table. Plot forecasts vs actuals.
- Residual diagnostics for SARIMA (Ljung-Box, residual plot).
- LOCKDOWN CASE STUDY: refit the best model on 2015-2019, forecast Jan-Jun 2020, plot against actual 2020 values, mark 25 March 2020 (lockdown start), and quantify the gap between forecast and actual after that date.
- Stop and report.

PHASE 5: CLASSIFICATION (will tomorrow be a bad air day?)
- Target: 1 if NEXT day's AQI > 300 (Very Poor or Severe), else 0.
- Features using ONLY information available today: today's pollutant levels, AQI lags (1, 2, 7 days), 7-day rolling mean AQI, month encoded as sin/cos, day of week. Explicitly check there is no leakage from future values.
- Split BY TIME: train = 2015-2018, test = 2019. Fit imputation and scaling on train only (use a sklearn Pipeline).
- Models: baseline (tomorrow's class = today's class), logistic regression (class_weight="balanced"), LDA, QDA.
- Tune logistic regression C with GridSearchCV using TimeSeriesSplit (not random k-fold), scoring F1.
- Report precision, recall, F1, AUC-ROC for all models in one table, plus confusion matrices and a combined ROC curve plot.
- Show the logistic regression coefficients and explain in plain words which features push the prediction up or down.
- Check class balance and comment on it. Stop and report.

PHASE 6: STREAMLIT APP (app.py)
- Page 1: overview (AQI trend chart and key EDA findings).
- Page 2: forecast (actual vs predicted, model comparison table).
- Page 3: "Tomorrow's risk": user enters today's pollutant values and recent AQI, app shows the predicted probability of a bad air day.
- Page 4: lockdown case study chart.
- Design rules: clean and minimal. No pill-shaped buttons, no purple gradients, no emoji icons, no em dashes in any text, no animations, no "Made with AI" tags, no made-up counters or metrics. Only display numbers loaded from real model outputs.
- Load saved models (joblib) rather than retraining on every page load.

PHASE 7: WRAP-UP
- README: problem statement, dataset and source (cite from data/SOURCE.md), method per phase, results tables (real numbers), limitations (data ends mid-2020, weekly averaging hides daily spikes, missing data handling), and how to run.
- reports/viva_notes.md: likely viva questions with short answers (why time-based split, why weekly resampling, why SARIMA over ARIMA, why F1 and AUC instead of accuracy, what LDA vs QDA assume, how grid search works, what overfitting would look like here).
