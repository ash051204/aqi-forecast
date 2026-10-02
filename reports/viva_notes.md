# Viva notes

Short, plain-language answers. Every number comes from the scripts in `src/` (see the per-phase reports).

## Data and splitting

**Why a time-based split instead of a random split?**
We want to know how well the model predicts the future from the past, because that is how it would be used. A random split would put some 2019 days into training, so the model would learn from days that come after the ones it is tested on. Neighbouring days are also very similar (today's and tomorrow's AQI correlate at 0.90), so random splitting would make the test look far easier than it really is. We trained on 2015 to 2018 and tested on 2019.

**Why resample to weekly averages for forecasting?**
- **Less noise:** daily AQI jumps around a lot, and weekly means show the seasonal pattern more clearly.
- **Practical season length:** a yearly season is 52 steps at weekly frequency but 365 at daily frequency, which SARIMA handles badly.
- **The cost:** single-day spikes are smoothed out. That is listed as a limitation.
- **Partial weeks:** weeks end on Sunday. The partial first week (4 days) and last week (3 days) were dropped.

**How was missing data handled?**
- **AQI:** 10 missing days, all in 2016 and 2017, filled by time-based interpolation and marked with `aqi_imputed`.
- **Xylene:** dropped. It was missing on 38.88% of days, mostly in one 606-day block, so filling it would mean inventing a year and a half of readings.
- **Other pollutants:** gaps filled with the training median inside the Phase 5 pipeline, never using 2019.

**Why keep the outliers?**
The IQR rule flagged 6 days, all in November (AQI 625 to 716). These are real smog episodes, not sensor errors, and they are exactly the days a warning system cares about. They are marked, not deleted.

## Forecasting

**Why SARIMA and not plain ARIMA?**
Delhi's AQI has a strong yearly cycle: the seasonal strength is 0.83, and the lag-52 autocorrelation of the training weeks is 0.731. Plain ARIMA only links a week to the few weeks just before it, so it cannot capture "this week looks like the same week last year". SARIMA adds seasonal terms at lag 52 and seasonal differencing (D = 1), which model that yearly repeat directly.

**Why did seasonal naive beat SARIMA on the full-year forecast?**
- **The numbers:** seasonal naive's full-year RMSE was 67.9, against 97.6 for SARIMA and 94.7 for Holt-Winters.
- **What seasonal naive does:** it copies the same week of 2018, the most recent year.
- **What went wrong for the fitted models:** they estimate their yearly pattern from all of 2015 to 2018, including the dirtier early years. Over the 52 test weeks, actual AQI averaged 231.6, while SARIMA forecast 306.4 and Holt-Winters 306.8. Seasonal naive forecast 248.4.
- **Why the simple model wins:** with a whole year to forecast and no new data, starting from the most recent level mattered more than a clever model.
- **The rolling test:** with one-week-ahead forecasts the models can correct themselves each week, and Holt-Winters did best (RMSE 61.8).

**Why can AIC not compare models with different differencing?**
AIC compares how well models explain the *same* data. Differencing changes the data. A model with d = 1 is fitted to the week-to-week changes, while a model with d = 0 is fitted to the levels. Their likelihoods are about different series, so comparing their AIC values is like comparing marks from two different exams. We fixed d = 1 and D = 1 from the ADF tests and the ACF first, and only compared p, q, P and Q by AIC. The planned check for d = 0 uses a validation year (2018) instead of AIC. It was not needed, because the MA(1) coefficient was -0.63, not close to -1.

**How does the grid search work?**
Grid search tries every combination from a small list of settings and keeps the best by a chosen score.
- **SARIMA:** p and q from 0 to 2, P and Q 0 or 1. That is 36 models scored by AIC on the training weeks; 5 did not converge and were excluded. The best was SARIMA(0,1,1)x(1,1,1,52), AIC 1702.9.
- **Logistic regression:** GridSearchCV tried C = 0.001, 0.01, 0.1, 1, 10, 100, scored each by F1 with TimeSeriesSplit (5 folds, gap 1) on 2015 to 2018, and chose C = 0.1 (CV F1 0.8676).

**Why TimeSeriesSplit instead of normal k-fold?**
Each fold trains on earlier days and validates on later days, which is the same "past predicts future" rule as the main split. Normal k-fold would validate on days that come before some of its training days. The gap of 1 day stops the last training row, whose label is the next day's AQI, from overlapping the first validation day.

## Classification

**Why F1 and AUC instead of accuracy?**
Only 26.03% of 2019 days were followed by a bad day. A model that always says "not bad" would score 74.0% accuracy (270 of 365) and never catch a single bad day.
- **F1** balances precision (how many alarms were real) and recall (how many bad days were caught).
- **AUC** measures how well the predicted probabilities rank bad days above other days, whatever threshold is used.

**What do LDA and QDA assume?**
Both assume that, within each class (bad and not bad), the features follow a multivariate normal distribution.
- **LDA** also assumes both classes share the same covariance matrix. This gives a straight-line (linear) boundary.
- **QDA** lets each class have its own covariance matrix. This gives a curved (quadratic) boundary, but it needs more parameters, so it can overfit more easily.

In our data QDA fitted without warnings, so no regularisation was needed.

**Which classifier won?**
None clearly on F1. Logistic regression (F1 0.865) and LDA (F1 0.869) both made exactly 25 mistakes out of 365 days; they just split them differently between false alarms and misses. QDA made 31 and the baseline 34. The clear gain is AUC: logistic regression 0.978 against the baseline's 0.879.

**What is a change day, and why does it matter?**
A change day is a day where tomorrow's status (bad or not bad) differs from today's. 2019 had 34: 17 turning bad and 17 turning better. The "tomorrow = today" baseline is wrong on every one of them by definition, so these are the only days where a model can show real value beyond copying today. Logistic regression got 13 of 34 right, and only 5 of the 17 that turned bad. Those sudden worsenings usually come from weather changes, which the dataset does not include.

**Why is the "AQI 475 to 50" custom-input test not meaningful?**
On 2019-01-01 the model gave 99.9%. Changing only "AQI today" from 475 to 50 in the app's custom-input mode moved it to 99.4% (the `tests/test_app.py` custom-input check reproduces this). But AQI is calculated from the pollutant values, so a day with AQI 50 and very high PM2.5 and PM10 cannot exist. The model was asked about an impossible day, and its answer tells us nothing about which input matters more. That test only proved the Predict button works. The app now warns about this on the custom-input page.

**What do the logistic regression coefficients say?**
- **PM2.5 and PM10:** read together, because they are correlated (r = 0.83 in the training rows). This lets the model shift weight between them without changing its predictions, so their individual coefficients are unstable. Their combined coefficient is +2.054: more particulate matter today means a higher chance of a bad day tomorrow.
- **The five AQI-history features:** a second correlated block, with a combined coefficient of +1.186.
- **The L2 penalty (C = 0.1):** keeps these correlated coefficients from growing large with opposite signs.

## Overfitting, leakage and the lockdown

**What would overfitting look like here?**
- A model that fits the training years very well but does worse on 2019.
- In the C search, weaker regularisation scored worse in cross-validation: CV F1 was 0.8676 at C = 0.1 but 0.8463 at C = 100. The model with more freedom learnt training noise.
- In forecasting, the fitted models (Holt-Winters, SARIMA) lost to the simple seasonal naive on the full 2019 year, because their yearly pattern was tied to the older, dirtier training years. That is the same problem in a milder form: learning the past too specifically.
- In the SARIMA grid, the largest models (p = 2, q = 2) did not improve AIC and some failed to converge.

**How was leakage prevented at each phase?**
- **Phase 2:** only AQI was filled, using the days on either side of each gap; all gaps were in 2016 and 2017. Pollutants were not filled here. The outlier flag uses all years, so it is never a model input.
- **Phase 3:** correlations used to choose features were computed on 2015 to 2018 only. The full-period heatmap is labelled "descriptive only".
- **Phase 4:**
  - Stationarity tests, model choices (Holt-Winters variant by AIC, SARIMA orders by AIC) and the over-differencing check all used training data only.
  - Rolling forecasts used only data up to the previous week, with parameters fixed.
  - 2019 was used once, to pick the best model for the lockdown study, before 2020 was touched.
- **Phase 5:**
  - Features were rebuilt from data cut off at each day, and they matched in 100 of 100 random days.
  - The label is the only place a future value appears.
  - The 2018-12-31 row was dropped because its label is from 2019.
  - Filling in gaps and scaling happen inside a Pipeline fitted on training rows.
  - Cross-validation is TimeSeriesSplit with a 1-day gap.
  - The threshold stayed at 0.5 and was not tuned on 2019.

**How does the lockdown placebo work?**
We compare how wrong the forecast was before and after 25 March 2020. Error is actual minus forecast. With seasonal naive the mean error was -26.6 before and -90.6 after, so the lockdown effect estimate is -90.6 - (-26.6) = -64.0 AQI points. Subtracting the pre-lockdown error removes the part of the gap that comes from air already improving year on year.

The placebo runs the identical calculation on 2019, a year with no lockdown, and gets -6.0. So in a normal year this number is small: -64.0 in 2020 against -6.0 in 2019.

The robustness check with Holt-Winters gives -75.4 against a placebo of -10.4. Ratios against a placebo this close to zero are unstable (a small change in the placebo would swing them a lot), so compare the absolute gap between effect and placebo instead. Both models agree on the direction; the size depends on the model.

It is still an association, not proof of cause, because weather also changes between years and is not in the data.
