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
- Resample to weekly mean AQI. State which weekday weeks end on, how partial weeks at the start and end are handled, and how the occasional 53rd week affects the 52-week seasonal period.
- Split BY TIME: train = 2015-2018, test = weeks ending in 2019. Keep 2020 aside.
- Stationarity: ADF test, ACF and PACF plots, decide differencing.
- Models: (a) naive baseline, (b) seasonal naive (same week last year), (c) Holt-Winters exponential smoothing with seasonal_periods=52, (d) SARIMA with seasonal period 52, orders chosen by a small grid search on AIC (keep the grid small; report the grid searched, the chosen orders, AIC, and the fitting time).
- Holt-Winters: because AQI trends downward each year, compare (i) seasonal only, (ii) additive trend, (iii) damped additive trend. Report which wins and pick one for the main table.
- Evaluate at TWO horizons, same rules for every model:
  a. Full-year: fit on 2015-2018, forecast all 2019 weeks at once. Naive = last observed 2018 week repeated. Seasonal naive = same week of 2018.
  b. Rolling one-week-ahead: for each 2019 week, predict it using actual data up to the previous week, keeping the fitted parameters fixed (no refitting). Naive = previous week's actual.
  Report RMSE and MAE for all four models at both horizons in one table. Plot forecasts vs actuals.
- Residual diagnostics for SARIMA (Ljung-Box, residual plot).
- LOCKDOWN CASE STUDY, done fairly:
  - Refit the best model on 2015-2019 and forecast Jan to Jun 2020.
  - Report the forecast error separately for 2020-01-01 to 2020-03-24 (pre-lockdown, the model's normal error) and 2020-03-25 onward.
  - Lockdown effect estimate = the post-25-March gap MINUS the pre-lockdown error level. Explain in plain language that the raw gap overstates the effect because air was already improving year on year.
  - Mark 25 March 2020 on the plot. State clearly that this is an association, not proof of cause (weather also varies).
- Stop and report.

PHASE 5: CLASSIFICATION (will tomorrow be a bad air day?)
- Target: 1 if NEXT day's AQI > 300 (Very Poor or Severe), else 0.
- Features using ONLY information available today: today's pollutant levels, today's AQI (aqi_today), AQI lags (1, 2, 7 days before today), 7-day rolling mean AQI (ending today), month encoded as sin/cos, day of week (encoded as sin/cos). Explicitly check there is no leakage from future values.
- Split BY TIME: train = 2015-2018, test = 2019. Drop the 2018-12-31 training row (its label is from 2019). Fit imputation and scaling on train only (use a sklearn Pipeline).
- Models: baseline (tomorrow's class = today's class), logistic regression (class_weight="balanced"), LDA, QDA. If QDA fails or warns because of near-singular covariance, use a small reg_param and report the value and why.
- Tune logistic regression C with GridSearchCV using TimeSeriesSplit (not random k-fold), scoring F1.
- Keep the decision threshold at 0.5. If other thresholds are tested, choose only via TimeSeriesSplit CV on 2015-2018, never on 2019.
- Report precision, recall, F1, AUC-ROC for all models in one table, plus confusion matrices and a combined ROC curve plot.
- Change-day evaluation: identify 2019 days where tomorrow's status differs from today's, and report how many each model got right on those days only. The persistence baseline is wrong on every change day by definition.
- Show the logistic regression coefficients and explain in plain words which features push the prediction up or down. Explain PM2.5 and PM10 together (correlated, so individual coefficients are unstable; the L2 penalty helps).
- Check class balance and comment on it, including the distribution shift (training 43.7% bad days vs 26.0% in 2019) as a reason precision may drop on 2019.
- Save the final fitted pipeline with joblib for Phase 6.
- Stop and report.

PHASE 6: STREAMLIT APP (app.py)
- Data sources: the app only reads saved files (reports/outputs/ CSVs and JSON, models/logreg_pipeline.joblib, models/logreg_pipeline_meta.json). No retraining, no hardcoded numbers in app text; every number displayed comes from those files.
- Page 1: overview (AQI trend chart and key EDA findings).
- Page 2: forecast: both horizons (full-year and rolling) with the comparison table, and a one-paragraph plain explanation of why seasonal naive won the full-year horizon.
- Page 3: "Tomorrow's risk", two modes:
  a. Replay a real day (default): pick any 2019 date; the app fills in that day's real feature values, shows the predicted probability of a bad air day tomorrow, and what actually happened next day (AQI and bucket). Mark whether the date is a change day, with a quick way to jump to change days.
  b. Custom input: all features editable, pre-filled with a real 2019 day's values rather than zeros.
  Below the result, a short plain-language note that the model reads today's air, not tomorrow's weather, so it misses many sudden worsening days (with the real change-day numbers from the saved outputs).
- Page 4: lockdown case study: both models (seasonal naive and Holt-Winters), pre/post errors, the 2019 placebo, the 25 March marker, and the "association, not proof of cause" note.
- Design rules: clean and minimal. No pill-shaped buttons, no purple gradients, no emoji icons, no em dashes in any text, no animations, no "Made with AI" tags, no made-up counters or metrics. Only display numbers loaded from real model outputs. .streamlit/config.toml with a neutral, restrained theme (no purple). Charts consistent in style, readable axis labels and units. Plain, specific copy.
- Load saved models (joblib) rather than retraining on every page load.
- Verification: test every page with streamlit.testing.v1.AppTest, including replay of at least one change day and one normal day, and launch the app locally once to confirm it starts.

PHASE 7: WRAP-UP
- Reproducibility: run_all.py runs every phase in order from the raw CSV (downloading it with kagglehub if data/raw/city_day.csv is missing, and checking its SHA-256), then the app tests and an automated em/en dash check over every committed text file. Test it from a fresh clone in a temporary folder with a new virtual environment: all scripts run, the tests pass, and the saved model reproduces the same 2019 probabilities.
- README: problem statement, dataset and source (cite from data/SOURCE.md), setup (Python 3.12, pinned requirements), how to download the data, run_all, how to launch the app and run the tests, method per phase, results tables (real numbers), limitations, future work.
- Results and limitations, stated honestly with real numbers: seasonal naive won the full-year forecast and Holt-Winters the rolling one-week horizon (with why); F1 differences between logistic regression, LDA and QDA are too small to call a winner, while AUC is where logistic regression clearly beats the baseline; the models catch fewer than half of the 2019 change days (no weather data); the lockdown effect is an association and its size depends on the model; data ends mid-2020; weekly averaging hides daily spikes; missing data handling; Diwali dates come from secondary sources quoting DoPT orders; custom input in the app can produce physically impossible days.
- Future work (ideas, not results): meteorological data (wind speed, temperature, humidity, boundary layer height) as the main next step, and stubble-burning fire counts.
- reports/viva_notes.md: likely viva questions with short answers (why time-based split, why weekly resampling, why SARIMA over ARIMA, why F1 and AUC instead of accuracy, what LDA vs QDA assume, how grid search works, what overfitting would look like here), plus why seasonal naive beat SARIMA, why AIC cannot compare models with different differencing, how the lockdown placebo works, what a change day is and why it matters, why the "AQI 475 to 50" custom-input test is not meaningful, and how leakage was prevented at each phase.
