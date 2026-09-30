# Phase 5: Predicting tomorrow's bad air day (AQI > 300)

## 1. Features and leakage checks

Features for day t (all known by the end of day t):

- Today's pollutant levels: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene (Xylene was dropped in Phase 2).
- `aqi_today` (AQI of day t), `aqi_lag1`, `aqi_lag2`, `aqi_lag7` (AQI of days t-1, t-2, t-7), `aqi_roll7_mean` (mean AQI of days t-6 to t).
- Month as sin/cos and day of week as sin/cos (cyclic, so December sits next to January and Sunday next to Monday).

Leakage checks:

- Truncation test: for 100 random days (seed 42), features were recomputed from data that stops at that day and compared with the features from the full data. Mismatches: **0**. So no feature uses any day after t.
- Label test: target on day t equals (AQI on day t+1 > 300) for every row: **True**. The label is the only place a future value appears.
- Imputation and scaling are inside sklearn Pipelines, fitted on 2015-2018 rows only.
- Cross-validation uses TimeSeriesSplit with gap = 1, so a fold's last training row (whose label is the next day's AQI) never overlaps with the first validation day.
- The `aqi_outlier` flag from Phase 2 is not used as a feature (it was computed over all years).

## 2. Rows used

- The first 7 days (2015-01-01 to 2015-01-07) are dropped because `aqi_lag7` and the 7-day mean need 7 earlier days. They are dropped, not filled, because filling them would mean inventing AQI history.
- The 2018-12-31 row is dropped from training: its label is the AQI of 2019-01-01, a test-year value.
- Train: 1453 days, 2015-01-08 to 2018-12-30.
- Test: 365 days, 2019-01-01 to 2019-12-31. The last test row's label is the AQI of 2020-01-01; it is used only as that row's answer.
- Pollutant values missing in training rows (filled with the training median inside the pipeline): PM2.5 0.14%, PM10 5.30%, NO 0.14%, NO2 0.14%, NH3 0.62%, SO2 7.57%, O3 5.78%.
- Training labels built from interpolated AQI (Phase 2 `aqi_imputed`): 10 rows.

## 3. Class balance

| Set | Days | Bad days tomorrow (1) | Not bad (0) | % bad |
|---|---|---|---|---|
| Train 2015-2018 | 1453 | 632 | 821 | 43.50% |
| Test 2019 | 365 | 95 | 270 | 26.03% |

- The classes are moderately imbalanced: 43.5% bad days in training. Not extreme, but enough that a model can look accurate by leaning towards "not bad". That is why we report precision, recall, F1 and AUC rather than accuracy, and why logistic regression uses `class_weight="balanced"` (mistakes on the rarer class count more during training).
- **Distribution shift:** bad days fall from 43.5% in training to 26.0% in 2019, because Delhi's air improved. A model trained on the dirtier years may expect bad days more often than they happen in 2019 and raise more false alarms, which **may push precision down on 2019** even if the model has learnt the right patterns. The decision threshold is kept at 0.5; it was not tuned on 2019. Section 5 checks whether this happened.

## 4. Models

- **Baseline (persistence):** tomorrow's class = today's class (`aqi_today > 300`).
- **Logistic regression**, `class_weight="balanced"`, L2 penalty (default). C searched over [0.001, 0.01, 0.1, 1, 10, 100] with GridSearchCV, TimeSeriesSplit(5 folds, gap 1), scoring F1, on 2015-2018 only.

| C | Mean CV F1 | Std |
|---|---|---|
| 0.001 | 0.8556 | 0.0348 |
| 0.01 | 0.8625 | 0.0330 |
| 0.1 | 0.8676 | 0.0289 |
| 1.0 | 0.8580 | 0.0294 |
| 10.0 | 0.8468 | 0.0350 |
| 100.0 | 0.8463 | 0.0352 |

  Chosen C = **0.1**. Smaller C means a stronger penalty that pulls coefficients towards zero.
- **LDA:** warnings when fitting: none.
- **QDA:** fitted without warnings, so no regularisation was needed (reg_param = 0).

## 5. Results on 2019 (365 days, threshold 0.5)

| Model | Precision | Recall | F1 | AUC-ROC | Accuracy (context only) |
|---|---|---|---|---|---|
| Baseline (persistence) | 0.821 | 0.821 | 0.821 | 0.879 | 0.907 |
| Logistic regression | 0.889 | 0.842 | 0.865 | 0.978 | 0.932 |
| LDA | 0.865 | 0.874 | 0.869 | 0.974 | 0.932 |
| QDA | 0.848 | 0.821 | 0.834 | 0.964 | 0.915 |

The baseline's AUC is computed from its 0/1 predictions, so it is a single operating point, not a full curve.

Distribution shift check: 2019 had 95 bad days. Number of days each model predicted as bad: Baseline (persistence) 95, Logistic regression 90, LDA 96, QDA 92. The largest over-prediction is +1 days (a negative number means under-prediction), so at the 0.5 threshold the shift did not produce a visible wave of false alarms in 2019. A likely reason: today's pollutant and AQI values already reflect the cleaner air, so the inputs shift along with the outcome.

Confusion matrices (rows = actual, columns = predicted):

| Model | TN (correct not bad) | FP (false alarm) | FN (missed bad day) | TP (caught bad day) |
|---|---|---|---|---|
| Baseline (persistence) | 253 | 17 | 17 | 78 |
| Logistic regression | 260 | 10 | 15 | 80 |
| LDA | 257 | 13 | 12 | 83 |
| QDA | 256 | 14 | 17 | 78 |

Figures: `phase5_confusion_matrices.png`, `phase5_roc_curves.png`.

## 6. Change days in 2019 (tomorrow's status differs from today's)

There are **34 change days** in 2019: 17 where tomorrow turns bad (today not bad, tomorrow bad) and 17 where tomorrow turns better (today bad, tomorrow not bad).

**The persistence baseline is wrong on every change day by definition**, because it always predicts that tomorrow will be the same as today. Change days are therefore where a model shows value beyond simply copying today.

| Model | Correct on change days | Correct: turns bad | Correct: turns better |
|---|---|---|---|
| Baseline (persistence) | 0 of 34 | 0 of 17 | 0 of 17 |
| Logistic regression | 13 of 34 | 5 of 17 | 8 of 17 |
| LDA | 15 of 34 | 9 of 17 | 6 of 17 |
| QDA | 16 of 34 | 7 of 17 | 9 of 17 |

## 7. Logistic regression coefficients

Features are standardised, so each coefficient is the change in the log-odds of a bad day tomorrow when that feature rises by one standard deviation (training data) and the others stay the same. Positive pushes the prediction towards "bad", negative pushes it towards "not bad". Intercept: -0.1127.

| Feature | Coefficient | Odds multiplier per 1 std |
|---|---|---|
| PM2.5 | +1.1245 | 3.079 |
| PM10 | +0.9296 | 2.534 |
| CO | +0.6559 | 1.927 |
| aqi_today | +0.4388 | 1.551 |
| NO | +0.4064 | 1.501 |
| NO2 | +0.3176 | 1.374 |
| aqi_lag7 | +0.2563 | 1.292 |
| O3 | +0.2154 | 1.240 |
| Benzene | +0.2137 | 1.238 |
| month_cos | +0.2020 | 1.224 |
| aqi_roll7_mean | +0.1903 | 1.210 |
| dow_sin | +0.1592 | 1.173 |
| aqi_lag1 | +0.1573 | 1.170 |
| aqi_lag2 | +0.1430 | 1.154 |
| SO2 | +0.0667 | 1.069 |
| NH3 | -0.0255 | 0.975 |
| dow_cos | -0.0302 | 0.970 |
| NOx | -0.0597 | 0.942 |
| Toluene | -0.2716 | 0.762 |
| month_sin | -0.2909 | 0.748 |

**In plain words:**

- **PM2.5 and PM10 together:** their coefficients are +1.125 and +0.930, a combined +2.054. They should be read as one "particulate matter" signal, not separately: in training they are correlated at r = 0.83, so the model can shift weight from one to the other with almost no change in its predictions. Their individual values (and even their signs) are therefore unstable; their combined effect is more trustworthy. The default L2 penalty helps here: it discourages large opposite-signed coefficients on correlated features and shares the weight between them.
- **The AQI history block** (`aqi_today`, lags 1, 2, 7 and the 7-day mean) has the same issue: these five are correlated with each other at r = 0.68 or more in training. Their combined coefficient is +1.186. Read together: when recent AQI has been high, tomorrow is more likely to be bad.
- Strongest pushes towards "bad": `PM2.5` (+1.125), `PM10` (+0.930), `CO` (+0.656).
- Strongest pushes towards "not bad": `month_sin` (-0.291), `Toluene` (-0.272), `NOx` (-0.060).
- Month and day-of-week terms describe the season and weekly cycle; their sin and cos parts only make sense together (for example, the month pair together places each month on the yearly circle).

Figure: `phase5_logreg_coefficients.png`.

## 8. Saved model

- `models/logreg_pipeline.joblib`: the tuned logistic regression pipeline (imputer, scaler, model), fitted on 2015-2018. Reloading it reproduces the 2019 probabilities exactly.
- `models/logreg_pipeline_meta.json`: feature order, C, thresholds, training period, 2019 metrics and scikit-learn version.

