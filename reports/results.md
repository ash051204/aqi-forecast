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

Lockdown case study (Seasonal naive, refit on 2015-2019):

| Period | Weeks | Mean error (actual - forecast) | MAE |
|---|---|---|---|
| Pre-lockdown | 12 | -26.6 | 48.3 |
| Post-lockdown | 13 | -90.6 | 94.0 |

Estimated lockdown effect (post minus pre mean error): -64.0 AQI points.
<!-- phase4:end -->
