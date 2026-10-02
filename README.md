# Forecasting Delhi's Air Quality Index

Mini project for Predictive Analytics (CSE3141), Manipal University Jaipur.

**Problem.** Delhi has some of the worst air in the world, and it follows a strong seasonal pattern. This project asks two questions using real daily data from 2015 to mid-2020:

1. **Forecasting:** how well can simple time series models forecast Delhi's weekly average AQI a year ahead, and one week ahead?
2. **Classification:** using only what is known today, can we predict whether **tomorrow** will be a bad air day (AQI above 300, "Very Poor" or "Severe")?

A case study also asks how much cleaner the air was than expected after the COVID-19 lockdown began on 25 March 2020.

All numbers in this README come from the scripts in `src/` (see `reports/results.md` and the per-phase reports).

## Dataset

- **Air Quality Data in India (2015 to 2020)**, by Vopani (Rohan Rao), Kaggle dataset `rohanrao/air-quality-data-in-india`, version 12.
- Original source: Central Pollution Control Board (CPCB), Government of India.
- License: CC0: Public Domain (from the Kaggle API metadata).
- File used: `data/raw/city_day.csv`, 29,531 rows, 26 cities. Delhi: 2,009 days, 2015-01-01 to 2020-07-01.
- Downloaded on 2026-10-01 with the official `kagglehub` package. SHA-256 and full details: `data/SOURCE.md`.

Citation: Rao, R. (Vopani). *Air Quality Data in India (2015 to 2020)*. Kaggle, version 12. Data originally from the Central Pollution Control Board (CPCB), India. Accessed 1 October 2026.

## Setup

Requires **Python 3.12**. The package versions in `requirements.txt` are pinned. In particular, the saved model needs `scikit-learn==1.9.1` to load. (Python 3.14 was tried first but its virtual environment failed to install pip on this machine, and 3.12 has wider support for the scientific libraries.)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Get the data

`run_all.py` downloads it automatically if `data/raw/city_day.csv` is missing. To do it by hand:

```bash
.venv/bin/python -c "import kagglehub, shutil, os; p = kagglehub.dataset_download('rohanrao/air-quality-data-in-india/versions/12'); os.makedirs('data/raw', exist_ok=True); shutil.copy(os.path.join(p, 'city_day.csv'), 'data/raw/city_day.csv')"
```

The dataset is public, so no Kaggle login was needed when this was built. If kagglehub asks for credentials, put your Kaggle API token at `~/.kaggle/kaggle.json` (permissions 600). `run_all.py` checks the file's SHA-256 against `data/SOURCE.md` and stops if it differs.

## Run everything

```bash
.venv/bin/python run_all.py
```

This runs every phase in order from the raw CSV (audit, clean, EDA, forecast, classify), then the app tests and the dash check, and stops at the first failure. It takes about a minute; most of that is the SARIMA grid search.

## Launch the app

```bash
.venv/bin/streamlit run app.py
```

The app has four pages: Overview, Forecast, Tomorrow's risk and Lockdown case study. It only reads saved files in `reports/outputs/` and `models/`; it never retrains.

## Run the tests

```bash
.venv/bin/python tests/test_app.py
.venv/bin/python tests/test_no_dashes.py
```

These are 24 checks that use Streamlit's AppTest:
- every page loads with no errors and no em or en dashes;
- replaying a normal day and a change day shows the saved probability;
- the change-day jump works;
- the custom input works.

`tests/test_no_dashes.py` scans every committed text file (README, BRIEF, data/SOURCE.md, the reports, code and outputs) for em and en dashes, which the design rules forbid.

## Project structure

```
data/raw/            raw CSV (not in git)
data/processed/      cleaned Delhi daily file (not in git; rebuilt by src/clean.py)
data/SOURCE.md       dataset source, license, hash
src/audit.py         Phase 1: data audit (reads only)
src/clean.py         Phase 2: cleaning
src/eda.py           Phase 3: exploratory analysis
src/forecast.py      Phase 4: weekly forecasting and lockdown case study
src/classify.py      Phase 5: next-day bad-air-day classification
app.py               Phase 6: Streamlit app
tests/test_app.py    app tests
tests/test_no_dashes.py  em/en dash check over all committed text files
run_all.py           runs everything in order
models/              saved logistic regression pipeline and its metadata
reports/             per-phase reports, results.md, viva_notes.md, figures/, outputs/
BRIEF.md             the project brief, kept in line with the code
```

Random seed 42 is used everywhere.

## Method, phase by phase

**Phase 1: audit.** Delhi has 2,009 days with no missing calendar days and no duplicate dates. AQI is missing on 10 days (0.50%), in runs of at most 4 days. Xylene is missing on 38.88% of days.

**Phase 2: cleaning.**
- **Missing AQI:** the 10 days were filled by time-based interpolation and marked with `aqi_imputed`. All of them fall in 2016 and 2017, inside the training period.
- **AQI_Bucket:** recomputed for the filled days using the CPCB ranges. On the 1,999 days with a real AQI, those ranges match the dataset's own labels 1,999 of 1,999 times.
- **Xylene:** dropped. 78% of its missing days form one 606-day block (2016-01-02 to 2017-08-29).
- **Other pollutants:** not filled in here, to avoid leakage. They are filled inside the Phase 5 pipeline, using training data only.
- **Outliers:** the IQR rule flags 6 days, all in November (AQI 625 to 716). They are real smog episodes, so they are kept and marked with `aqi_outlier`, not deleted.

**Phase 3: EDA.**
- Descriptive statistics use all years. Anything used to choose model features (the correlations) uses 2015 to 2018 only.
- The seasonal decomposition is done on weekly means with a 52-week period.
- The five Diwali dates (2015 to 2019) are marked on the time series.

**Phase 4: forecasting.**
- **Data:** weekly mean AQI, with weeks ending on Sunday and partial weeks dropped. Train = weeks ending in 2015 to 2018 (208 weeks), test = weeks ending in 2019 (52 weeks).
- **Differencing:** chosen from ADF tests and the ACF: d = 1, D = 1.
- **Models:** naive, seasonal naive, Holt-Winters (three variants, chosen by AIC on training data), and SARIMA (a 36-model grid search on AIC).
- **Two horizons:**
  - Full year: fit on 2015 to 2018, then forecast all 52 weeks of 2019 at once.
  - Rolling one-week-ahead: each week is forecast from real data up to the previous week, with the parameters kept fixed.
- **Lockdown case study:** compares the forecast error before and after 25 March 2020. The same calculation is run on 2019 as a placebo.

**Phase 5: classification.**
- **Target:** 1 if tomorrow's AQI is above 300.
- **Features, all known by the end of today:** today's pollutant levels; today's AQI; AQI 1, 2 and 7 days earlier; the 7-day mean AQI; month and day of week as sin/cos.
- **Leakage checks:** features rebuilt from data cut off at each day matched in 100 of 100 random days, and the labels were checked on every row.
- **Split:** train = 2015-01-08 to 2018-12-30 (1,453 days), test = 2019 (365 days).
- **Models:** a persistence baseline (tomorrow = today), logistic regression (`class_weight="balanced"`, C tuned by TimeSeriesSplit cross-validation on F1), LDA and QDA. Filling in gaps and scaling happen inside sklearn Pipelines fitted on training data only.

**Phase 6: app.** Streamlit, reading only saved outputs.

## Results

### Weekly forecasts, 52 test weeks of 2019 (AQI points)

| Model | Full-year RMSE | Full-year MAE | Rolling 1-week RMSE | Rolling 1-week MAE |
|---|---|---|---|---|
| Naive | 252.8 | 231.6 | 71.7 | 53.1 |
| Seasonal naive | **67.9** | **53.6** | 67.9 | 53.6 |
| Holt-Winters (seasonal only) | 94.7 | 81.1 | **61.8** | **46.1** |
| SARIMA(0, 1, 1)x(1, 1, 1, 52) | 97.6 | 81.2 | 64.2 | 49.8 |

**Seasonal naive won the full-year horizon.** It simply copies the same week of 2018, the most recent year, so it starts from the most recent pollution level. Holt-Winters and SARIMA estimate their yearly pattern from all four training years, including the dirtier 2015 to 2017 (Holt-Winters' seasonal smoothing weight came out as 0, so its pattern never updates). With no new data for a whole year, they kept forecasting higher AQI than 2019 actually had. Averaged over the 52 test weeks, actual AQI was 231.6. Holt-Winters forecast 306.8 on average, SARIMA 306.4, and seasonal naive 248.4.

**Holt-Winters won the rolling one-week horizon.** Once the models see last week's real value, they correct their level every week. What then matters is modelling the short-term movement around the seasonal pattern, and Holt-Winters did that best.

The SARIMA residuals pass the Ljung-Box test at lags 10, 20 and 52 (p = 0.49, 0.32 and 0.66), so no pattern is left in them. Its MA(1) coefficient (-0.63) shows no sign of over-differencing.

### Lockdown case study (error = actual minus forecast; negative = cleaner than expected)

| Model | Role | Pre-lockdown mean error | Post-lockdown mean error | Effect (post minus pre) | 2019 placebo |
|---|---|---|---|---|---|
| Seasonal naive | main | -26.6 | -90.6 | **-64.0** | -6.0 |
| Holt-Winters (seasonal only) | robustness check | -17.7 | -93.1 | **-75.4** | -10.4 |

The raw post-lockdown gap overstates the effect, because early 2020 was already cleaner than the forecast before the lockdown. Subtracting the pre-lockdown error removes that. Side by side, the 2020 effect against the 2019 placebo is -64.0 vs -6.0 for seasonal naive and -75.4 vs -10.4 for Holt-Winters. 2019 had no lockdown, so the placebo shows how big this number is in a normal year. A ratio against a placebo this close to zero is unstable (a small change in the placebo would swing it a lot), so the absolute gap between effect and placebo is the fairer comparison. Both models point the same way; the exact size depends on the model.

### Tomorrow's bad air day, 365 test days of 2019 (threshold 0.5)

| Model | Precision | Recall | F1 | AUC-ROC | Wrong days (FP + FN) |
|---|---|---|---|---|---|
| Baseline (tomorrow = today) | 0.821 | 0.821 | 0.821 | 0.879 | 34 (17 + 17) |
| Logistic regression (C = 0.1) | 0.889 | 0.842 | 0.865 | 0.978 | 25 (10 + 15) |
| LDA | 0.865 | 0.874 | 0.869 | 0.974 | 25 (13 + 12) |
| QDA | 0.848 | 0.821 | 0.834 | 0.964 | 31 (14 + 17) |

The baseline's AUC comes from its 0/1 predictions (a single point on the ROC curve), not a full probability curve.

- **No clear winner on F1.** Logistic regression and LDA make exactly the same number of mistakes (25 of 365 days); they only split them differently between false alarms and misses. QDA makes 6 more. These differences are a handful of days in one test year and should not be called a win for any model.
- **AUC is where logistic regression clearly beats the baseline** (0.978 vs 0.879). Its probabilities rank bad days above other days far better than "tomorrow = today" can.
- **Class balance:** 43.50% of training days were followed by a bad day, but only 26.03% in 2019. At the 0.5 threshold this did not cause over-prediction: logistic regression predicted 90 bad days against 95 actual.

### Change days (2019 days where tomorrow's status differs from today's)

2019 has 34 change days: 17 turn bad, 17 turn better. The baseline gets all of them wrong by definition.

| Model | Correct of 34 | Turns bad (of 17) | Turns better (of 17) |
|---|---|---|---|
| Baseline | 0 | 0 | 0 |
| Logistic regression | 13 | 5 | 8 |
| LDA | 15 | 9 | 6 |
| QDA | 16 | 7 | 9 |

## Limitations

- **Sudden changes are mostly missed.** Every model gets fewer than half of the 2019 change days right. Logistic regression catches only 5 of the 17 days that suddenly turn bad. The dataset has no weather information (wind, temperature, humidity, mixing height), and weather is what usually drives these jumps. The models read today's air, not tomorrow's weather.
- **Small differences between classifiers.** Logistic regression, LDA and QDA are separated by a few days in a single test year. A different year could reorder them.
- **The lockdown effect is an association, not proof of cause.** Weather also varies from year to year and is not in the data. The size of the estimate depends on the model: -64.0 (seasonal naive) vs -75.4 (Holt-Winters), against placebos of -6.0 and -10.4.
- **Data ends on 1 July 2020.** 2020 has only January to June, and there is no data from after the lockdown period.
- **Weekly averaging hides daily spikes.** The forecasting models work on weekly means, so a single severe day (for example the day after Diwali) is smoothed out.
- **Missing data handling.**
  - 10 AQI days were interpolated.
  - Xylene was dropped, losing whatever it added.
  - Other pollutant gaps (up to 7.57% for SO2 in the training rows) were filled with the training median, which ignores the season.
- **Diwali dates** come from secondary websites quoting the official DoPT holiday orders, not from the dopt.gov.in PDFs themselves.
- **Custom input in the app can produce physically impossible days.** AQI is calculated from the pollutant values, so changing one input on its own (for example AQI today without PM2.5 and PM10) describes a day that cannot exist. The app says so on that page.
- **One city, one test year.** The results are for Delhi and 2019 only.

## Future work (ideas, not results)

- **Add meteorological data**: wind speed and direction, temperature, humidity, and planetary boundary layer height, for example from reanalysis datasets. This is the main next step, because weather is the likely cause of exactly the change days the current model misses.
- **Add stubble-burning fire counts** in Punjab and Haryana (for example satellite fire detections) as a feature for October and November.
- Tune the decision threshold with TimeSeriesSplit cross-validation on training data, if false alarms and misses have different costs.
- Test on more years once newer CPCB data is available, and on other cities in the same dataset.

## Reports

- `reports/phase1_audit.md` to `reports/phase5_classify.md`: full per-phase reports with every number.
- `reports/results.md`: metrics tables.
- `reports/viva_notes.md`: likely viva questions with short answers.
- `reports/figures/`: all plots.
