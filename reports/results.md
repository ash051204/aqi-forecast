# Results

All numbers are produced by the scripts in `src/`.

<!-- phase4:start -->
## Phase 4: Forecasting weekly mean AQI (test = 52 weeks of 2019)

| Model | Full-year RMSE | Full-year MAE | Rolling 1-week RMSE | Rolling 1-week MAE |
|---|---|---|---|---|
| Naive | 252.8 | 231.6 | 71.7 | 53.1 |
| Seasonal naive | 67.9 | 53.6 | 67.9 | 53.6 |
| Holt-Winters (seasonal only) | 94.7 | 81.1 | 61.8 | 46.1 |
| SARIMA(0, 1, 1)x(1, 1, 1, 52) | 97.6 | 81.2 | 64.2 | 49.8 |

Lockdown case study. Sign convention: error = actual minus forecast (negative = cleaner than expected).

| Model | Period | Weeks | Mean error (actual - forecast) | MAE |
|---|---|---|---|---|
| Seasonal naive | Pre-lockdown | 12 | -26.6 | 48.3 |
| Seasonal naive | Post-lockdown | 13 | -90.6 | 94.0 |
| Holt-Winters (seasonal only) | Pre-lockdown | 12 | -17.7 | 32.4 |
| Holt-Winters (seasonal only) | Post-lockdown | 13 | -93.1 | 93.1 |

| Model | Role | 2020 effect (post minus pre mean error) | 2019 placebo |
|---|---|---|---|
| Seasonal naive | main | -64.0 | -6.0 |
| Holt-Winters (seasonal only) | robustness check | -75.4 | -10.4 |
<!-- phase4:end -->

<!-- phase5:start -->
## Phase 5: Tomorrow's bad air day (AQI > 300), test = 2019 (365 days)

| Model | Precision | Recall | F1 | AUC-ROC | Accuracy (context only) |
|---|---|---|---|---|---|
| Baseline (persistence) | 0.821 | 0.821 | 0.821 | 0.879 | 0.907 |
| Logistic regression | 0.889 | 0.842 | 0.865 | 0.978 | 0.932 |
| LDA | 0.865 | 0.874 | 0.869 | 0.974 | 0.932 |
| QDA | 0.848 | 0.821 | 0.834 | 0.964 | 0.915 |

Change days in 2019: 34 (17 turn bad, 17 turn better).

| Model | Correct on change days | Turns bad | Turns better |
|---|---|---|---|
| Baseline (persistence) | 0 of 34 | 0 of 17 | 0 of 17 |
| Logistic regression | 13 of 34 | 5 of 17 | 8 of 17 |
| LDA | 15 of 34 | 9 of 17 | 6 of 17 |
| QDA | 16 of 34 | 7 of 17 | 9 of 17 |

Logistic regression C = 0.1; QDA reg_param = 0.0.
<!-- phase5:end -->
