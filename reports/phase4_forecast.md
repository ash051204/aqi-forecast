# Phase 4: Forecasting weekly mean AQI (Delhi)

## 1. Weekly resampling

- Each week runs Monday to Sunday and is labelled by its **Sunday** (pandas rule `W-SUN`). The weekly value is the mean of the 7 daily AQI values.
- Partial weeks: 2 were found and dropped: week ending 2015-01-04 has 4 days; week ending 2020-07-05 has 3 days. The data starts on a Thursday (2015-01-01) and ends on a Wednesday (2020-07-01).
- Result: 286 full weeks, ending 2015-01-11 to 2020-06-28.
- Weeks per year (by the Sunday the week ends on): 2015: 51, 2016: 52, 2017: 53, 2018: 52, 2019: 52, 2020: 26.
- **The 53rd week.** A year has 52 weeks plus 1 or 2 days, so every 5 or 6 years a calendar year contains 53 Sundays. Here that is 2017 (53 weeks). The models use a fixed 52-week season, so "the same week last year" means exactly 364 days earlier. That keeps the seasonal pattern slowly sliding against the calendar, by about 1.25 days per year, or about 5 days over the 4 training years. This is less than one week, while Delhi's high-pollution season lasts about 8 to 10 weeks, so the mismatch is small. We do not delete or merge the 53rd week, because that would break the equal spacing that time series models need.

Split by time: train = weeks ending in 2015-2018 (208 weeks, 2015-01-11 to 2018-12-30); test = weeks ending in 2019 (52 weeks, 2019-01-06 to 2019-12-29); held back = weeks ending in 2020 (26 weeks, 2020-01-05 to 2020-06-28).
Note: the first test week (ending 2019-01-06) includes 2018-12-31 because it is a Monday of that week.

## 2. Stationarity (training weeks only)

ADF test: the null hypothesis is that the series has a unit root (is non-stationary). A p-value below 0.05 means we can treat the series as stationary.

| Series | Weeks | ADF statistic | p-value | Stationary at 5%? |
|---|---|---|---|---|
| Level (no differencing) | 208 | -4.172 | 0.0007 | yes |
| First difference (d=1) | 207 | -4.044 | 0.0012 | yes |
| Seasonal difference (D=1, lag 52) | 156 | -2.374 | 0.1492 | no |
| Seasonal + first difference (D=1, d=1) | 155 | -4.847 | 0.0000 | yes |

Autocorrelation of the training weeks at lag 1: 0.841; at lag 52: 0.731.

Figure: `phase4_acf_pacf.png`.

**Differencing decision: d = 1, D = 1.**

- The level series passes the ADF test (p = 0.0007), but the ADF test looks for a random-walk type of drift, not a yearly cycle. The ACF shows a strong repeat at lag 52 (0.731) and Phase 3 found a seasonal strength of 0.83, so the yearly pattern has to be removed: **D = 1**.
- After seasonal differencing the ADF test no longer rejects a unit root (p = 0.1492). In plain words: once the yearly cycle is taken out, what is left is the slow year-on-year decline, which wanders rather than returning to a fixed mean. Adding a first difference fixes this (p = 0.0000): **d = 1**.
- Cost: seasonal plus first differencing uses up the first 53 training weeks, leaving 155 weeks to estimate the ARMA part. This is why the grid is kept small.

## 3. Holt-Winters variants (additive seasonality, 52-week season)

| Variant | AIC (train) | Optimiser converged | Full-year RMSE | Full-year MAE | Rolling RMSE | Rolling MAE | Fit time (s) |
|---|---|---|---|---|---|---|---|
| HW seasonal only | 1644.4 | yes | 94.7 | 81.1 | 61.8 | 46.1 | 0.05 |
| HW additive trend | 1648.3 | no (warning) | 103.4 | 89.6 | 61.9 | 46.2 | 0.11 |
| HW damped additive trend | 1648.5 | no (warning) | 90.1 | 76.3 | 61.4 | 45.6 | 0.11 |

- Chosen for the main table: **HW seasonal only** (lowest AIC on the training weeks). The choice is made on training data so the 2019 test weeks stay unseen.
- On the 2019 test weeks the lowest full-year RMSE is HW damped additive trend and the lowest rolling RMSE is HW damped additive trend. AIC and test performance do not fully agree; this is reported, not used to change the choice.
- Fitted smoothing parameters of the chosen variant: alpha (level) = 0.366, gamma (season) = 0.000.

## 4. SARIMA grid search (d = 1, D = 1, season = 52)

Grid: p in [0, 1, 2], q in [0, 1, 2], P in [0, 1], Q in [0, 1], giving 36 models. Fitted on the 208 training weeks, ranked by AIC. Total grid time: 49.6 s.
Models that failed to converge (excluded): 5.

Top 10 by AIC:

| Rank | (p,d,q) | (P,D,Q,s) | AIC | Converged | Fit time (s) |
|---|---|---|---|---|---|
| 1 | (0, 1, 1) | (1, 1, 1, 52) | 1702.9 | yes | 1.5 |
| 2 | (0, 1, 2) | (1, 1, 1, 52) | 1704.8 | yes | 1.7 |
| 3 | (1, 1, 1) | (1, 1, 1, 52) | 1704.8 | yes | 1.6 |
| 4 | (2, 1, 1) | (1, 1, 1, 52) | 1705.3 | yes | 3.6 |
| 5 | (2, 1, 0) | (1, 1, 1, 52) | 1705.3 | yes | 1.5 |
| 6 | (1, 1, 2) | (1, 1, 1, 52) | 1706.4 | no | 3.8 |
| 7 | (2, 1, 2) | (1, 1, 1, 52) | 1707.1 | no | 3.9 |
| 8 | (0, 1, 1) | (1, 1, 0, 52) | 1707.6 | yes | 0.6 |
| 9 | (0, 1, 1) | (0, 1, 1, 52) | 1708.5 | yes | 1.0 |
| 10 | (2, 1, 0) | (1, 1, 0, 52) | 1709.3 | yes | 0.8 |

**Chosen: SARIMA(0, 1, 1)x(1, 1, 1, 52), AIC = 1702.9.** Refit time for the chosen model: 1.5 s.

Estimated coefficients:

| Parameter | Estimate | Std error | p-value |
|---|---|---|---|
| ma.L1 | -0.6279 | 0.0727 | 0.0000 |
| ar.S.L52 | -0.3477 | 0.1833 | 0.0578 |
| ma.S.L52 | -0.6850 | 0.6387 | 0.2835 |
| sigma2 | 2259.4366 | 979.1547 | 0.0210 |

## 5. Model comparison on the 52 test weeks of 2019

Full-year = fitted on 2015-2018 and forecast all 2019 weeks in one go (up to 52 weeks ahead). Rolling 1-week = each 2019 week forecast from real data up to the previous week, parameters kept fixed (no refitting). Units: AQI points.

| Model | Full-year RMSE | Full-year MAE | Rolling 1-week RMSE | Rolling 1-week MAE |
|---|---|---|---|---|
| Naive | 252.8 | 231.6 | 71.7 | 53.1 |
| Seasonal naive | 67.9 | 53.6 | 67.9 | 53.6 |
| Holt-Winters (seasonal only) | 94.7 | 81.1 | 61.8 | 46.1 |
| SARIMA(0, 1, 1)x(1, 1, 1, 52) | 97.6 | 81.2 | 64.2 | 49.8 |

Lowest full-year RMSE: **Seasonal naive**. Lowest rolling RMSE: **Holt-Winters (seasonal only)**.
The seasonal naive scores are identical at both horizons by construction: its forecast for any 2019 week is the value 52 weeks earlier, which is always a 2018 week already known at the end of 2018, so knowing the 2019 weeks in between changes nothing.

Figures: `phase4_forecasts_full_year.png`, `phase4_forecasts_rolling.png`.

## 6. SARIMA residual diagnostics (training weeks)

Residuals used: 155 (the first 53 are skipped because the differencing needs 53 weeks to start). Mean -2.34, std 58.07.

Ljung-Box test: the null hypothesis is that the residuals have no autocorrelation (they are white noise). A p-value above 0.05 means no evidence of leftover pattern.

| Lags tested | Q statistic | p-value | Leftover autocorrelation at 5%? |
|---|---|---|---|
| 10 | 9.46 | 0.4895 | no |
| 20 | 22.31 | 0.3242 | no |
| 52 | 47.27 | 0.6601 | no |

Figure: `phase4_sarima_residuals.png`.

## 7. Lockdown case study (January to June 2020)

Model: **Seasonal naive**, the best full-year model on 2019 (lowest full-year RMSE). This model has no parameters to estimate, so "refitting" on 2015-2019 simply means the forecast is built from the data up to the end of 2019: each 2020 week gets the value of the same week of 2019 (52 weeks earlier). Forecast: 26 weeks ending 2020-01-05 to 2020-06-28.

- Pre-lockdown weeks: ending 2020-01-05 to 2020-03-22 (12 weeks; the first one includes 30 and 31 December 2019).
- Post-lockdown weeks: ending 2020-04-05 to 2020-06-28 (13 weeks).
- Left out: week ending 2020-03-29 (23 to 29 March), which mixes 2 days before and 5 days after the start of the lockdown.

Error = actual minus forecast. Negative means the air was cleaner than the model expected.

| Period | Weeks | Mean actual | Mean forecast | Mean error | MAE | RMSE | Mean % error |
|---|---|---|---|---|---|---|---|
| Pre-lockdown (1 Jan to 22 Mar) | 12 | 256.1 | 282.8 | -26.6 | 48.3 | 55.0 | -7.6% |
| Post-lockdown (from 30 Mar) | 13 | 127.2 | 217.9 | -90.6 | 94.0 | 105.7 | -38.9% |

**Estimated lockdown effect = post-lockdown mean error minus pre-lockdown mean error = -90.6 - (-26.6) = -64.0 AQI points** (in percentage terms -38.9% - (-7.6%) = -31.3 percentage points).

Context from 2019, a year without a lockdown: the same model, forecasting 2019 from the end of 2018, had a mean error of -11.5 for weeks ending up to 24 Mar 2019 (12 weeks) and -17.4 for weeks starting 25 Mar to the end of June 2019 (14 weeks), a post-minus-pre difference of -6.0. So in a normal year the same calculation gives -6.0, against -64.0 in 2020.

**How to read this, in plain language:**

- The raw post-lockdown gap overstates the lockdown effect. Delhi's air was already getting cleaner year on year (mean daily AQI 297 in 2015, 232 in 2019), and early 2020 was already cleaner than the model expected before any lockdown. Subtracting the pre-lockdown error removes that "already improving" part, so what remains is the extra drop that lines up in time with the lockdown.
- The pre-lockdown weeks are 1 to 12 weeks ahead of the forecast origin, while the post-lockdown weeks are 14 to 26 weeks ahead. For the seasonal naive model this matters less, because each forecast is just the same week of the year before and does not get less reliable further out; it does mean the comparison relies on 2019 being a typical year.
- **This is an association, not proof of cause.** Weather (rain, wind, temperature) also changes from year to year and affects AQI, and this model has no weather data. The estimate says how much cleaner the air was than expected after 25 March; it cannot prove the lockdown alone caused all of it.

Figure: `phase4_lockdown_case_study.png`.

