# Phase 3: Exploratory Data Analysis (Delhi)

Data: `data/processed/delhi_daily.csv`, 2009 days, 2015-01-01 to 2020-07-01.
Descriptive results use all years. Results used to choose model features use 2015-2018 only (marked).

## 1. AQI time series and Diwali

Figure: `reports/figures/phase3_aqi_timeseries.png` (Oct-Nov shaded, Diwali marked with dashed lines).

Diwali dates were verified against Government of India DoPT holiday orders for Central Government offices in Delhi:

| Diwali | Source | AQI on Diwali | AQI next day | Max AQI in next 7 days | Oct-Nov max AQI (date) |
|---|---|---|---|---|---|
| 2015-11-11 | DoPT OM F.No.12/5/2014-JCA-2 (6 Jun 2014) | 431 | 421 | 454 | 483 (2015-11-07) |
| 2016-10-30 | DoPT OM F.No.12/7/2015-JCA-2 (11 Jun 2015) | 522 | 520 | 676 | 716 (2016-11-07) |
| 2017-10-19 | DoPT OM F.No.12/8/2016-JCA-2 (24 Jun 2016) | 336 | 497 | 497 | 677 (2017-11-09) |
| 2018-11-07 | DoPT OM F.No.12/3/2017-JCA-2 (14 Jun 2017) | 317 | 452 | 487 | 487 (2018-11-09) |
| 2019-10-27 | DoPT OM F.No.12/2/2018-JCA-2 (11 Jul 2018) | 354 | 417 | 659 | 659 (2019-11-03) |

## 2. Distribution of daily AQI (all years)

Figures: `phase3_monthly_boxplots.png`, `phase3_aqi_histogram.png`.

Mean 258.6, median 257.0, std 119.9, variance 14375.5, min 29.0, max 716.0, skewness 0.32.

## 3. AQI summary statistics by month (all years)

| Month | Days | Mean | Median | Std | Variance |
|---|---|---|---|---|---|
| Jan | 186 | 360.4 | 358.0 | 80.9 | 6550.3 |
| Feb | 170 | 298.8 | 312.5 | 69.5 | 4825.5 |
| Mar | 186 | 222.6 | 217.0 | 76.3 | 5818.1 |
| Apr | 180 | 248.4 | 252.5 | 89.8 | 8063.0 |
| May | 186 | 258.7 | 254.5 | 93.5 | 8738.6 |
| Jun | 180 | 199.0 | 184.5 | 92.6 | 8570.6 |
| Jul | 156 | 137.8 | 130.0 | 64.2 | 4122.3 |
| Aug | 155 | 131.1 | 109.0 | 63.0 | 3974.2 |
| Sep | 150 | 157.4 | 136.5 | 70.5 | 4968.9 |
| Oct | 155 | 312.3 | 317.0 | 102.1 | 10431.6 |
| Nov | 150 | 402.0 | 383.5 | 107.2 | 11495.4 |
| Dec | 155 | 371.2 | 379.0 | 77.7 | 6042.5 |

## 4. AQI summary statistics by year (all years; 2020 covers 1 Jan to 1 Jul only)

| Year | Days | Mean | Median | Std | Variance |
|---|---|---|---|---|---|
| 2015 | 365 | 297.0 | 303.0 | 81.9 | 6700.2 |
| 2016 | 366 | 301.0 | 292.5 | 123.1 | 15163.7 |
| 2017 | 365 | 252.2 | 274.0 | 138.9 | 19300.8 |
| 2018 | 365 | 249.2 | 243.0 | 114.3 | 13060.7 |
| 2019 | 365 | 232.1 | 209.0 | 117.6 | 13829.2 |
| 2020 | 183 | 181.7 | 148.0 | 90.6 | 8211.0 |

## 5. AQI category counts (all years)

Figure: `phase3_bucket_counts.png`.

| Category | Days | % |
|---|---|---|
| Good | 23 | 1.14% |
| Satisfactory | 164 | 8.16% |
| Moderate | 521 | 25.93% |
| Poor | 542 | 26.98% |
| Very Poor | 520 | 25.88% |
| Severe | 239 | 11.90% |

## 6. Class balance preview for Phase 5 (AQI > 300)

| Period | Days | Days with AQI > 300 | % | Day pairs | Tomorrow's status = today's |
|---|---|---|---|---|---|
| 2015 | 365 | 183 | 50.14% | 365 | 79.73% |
| 2016 | 366 | 173 | 47.27% | 366 | 86.34% |
| 2017 | 365 | 163 | 44.66% | 365 | 88.49% |
| 2018 | 365 | 120 | 32.88% | 365 | 86.85% |
| 2019 | 365 | 95 | 26.03% | 365 | 90.68% |
| 2020 | 183 | 25 | 13.66% | 182 | 91.76% |
| Train 2015-2018 | 1461 | 639 | 43.74% | 1460 | 85.34% |
| Test 2019 | 365 | 95 | 26.03% | 365 | 90.68% |
| Overall | 2009 | 759 | 37.78% | 2008 | 86.90% |

Day pairs are (today, tomorrow). Each year's pairs start on that year's dates, so the pair 31 Dec to 1 Jan counts in the earlier year. The final day (2020-07-01) has no tomorrow and is excluded. The training row stops at the pair 2018-12-30 to 2018-12-31 so that no 2019 value is used.
Status changes over all years: 131 days go from not-bad to bad, 132 go from bad to not-bad.

## 7. Correlations

Figures: `phase3_corr_full_descriptive.png` (2015-2020, description only) and `phase3_corr_train_2015_2018.png` (training period, used for feature decisions).

### 7a. Training period 2015-2018 (for feature decisions)

| Pollutant | r with same-day AQI | r with next-day AQI |
|---|---|---|
| PM10 | 0.86 | 0.83 |
| PM2.5 | 0.86 | 0.84 |
| Benzene | 0.65 | 0.63 |
| NO2 | 0.62 | 0.62 |
| NO | 0.61 | 0.61 |
| NOx | 0.49 | 0.49 |
| NH3 | 0.47 | 0.45 |
| SO2 | 0.43 | 0.43 |
| O3 | 0.34 | 0.33 |
| Toluene | 0.30 | 0.29 |
| CO | 0.25 | 0.24 |

Same-day AQI vs next-day AQI (lag-1 autocorrelation) in 2015-2018: 0.90

Pollutant pairs with |r| >= 0.8 in 2015-2018 (strongly overlapping information):

| Pair | r |
|---|---|
| PM2.5 and PM10 | 0.84 |

### 7b. Full period 2015-2020 (description only)

| Pollutant | r with AQI (2015-2020) | r with AQI (2015-2018) | Difference |
|---|---|---|---|
| PM10 | 0.88 | 0.86 | +0.02 |
| PM2.5 | 0.88 | 0.86 | +0.02 |
| Benzene | 0.67 | 0.65 | +0.02 |
| NO2 | 0.66 | 0.62 | +0.04 |
| NO | 0.64 | 0.61 | +0.03 |
| NOx | 0.56 | 0.49 | +0.07 |
| NH3 | 0.52 | 0.47 | +0.05 |
| SO2 | 0.42 | 0.43 | -0.02 |
| O3 | 0.33 | 0.34 | -0.01 |
| Toluene | 0.28 | 0.30 | -0.01 |
| CO | 0.28 | 0.25 | +0.04 |

## 8. Seasonal decomposition (weekly mean AQI, additive, period 52)

Figure: `phase3_seasonal_decomposition.png`.

Weekly series: 288 weeks, 2015-01-04 to 2020-07-05 (week-ending Sundays).
Seasonal component: highest +197.2 (week ending 15 Nov), lowest -156.7 (week ending 02 Aug), swing 353.9 AQI points.
Seasonal strength, 1 - Var(residual) / Var(seasonal + residual): 0.83 (0 = no seasonality, 1 = all variation after removing trend is seasonal).
Trend defined from 2015-07-05 to 2020-01-05 (26 weeks are lost at each end by the centred 52-week moving average).

| Year | Mean of trend component |
|---|---|
| 2015 | 300.5 |
| 2016 | 295.0 |
| 2017 | 258.7 |
| 2018 | 245.5 |
| 2019 | 229.2 |
| 2020 | 200.6 |

Residual: std 46.2, largest positive 184.1 (week ending 2018-06-17), largest negative -146.2 (week ending 2017-06-18).

## 9. Findings in plain language

1. **Delhi's air follows a strong yearly cycle.** The monthly mean AQI is highest in November (402) and lowest in August (131), about 3.1 times higher. The decomposition gives a seasonal strength of 0.83, so once the trend is removed most of what is left is the repeating yearly pattern. This is why the forecasting models in Phase 4 need a seasonal part.
2. **Winter is the bad-air season, the monsoon is the clean season.** The months from October to January all have mean AQI above 300 (Oct 312, Nov 402, Dec 371, Jan 360), while July to September sit between 131 and 157. Monsoon rain washes particles out of the air; in winter, cold still air traps them near the ground.
3. **Diwali falls inside the worst period, but it is not always the single worst day.** AQI the day after Diwali was higher than on Diwali itself in 3 of 5 years. In 2 of 5 years the highest Oct-Nov AQI came within 7 days after Diwali (counting Diwali itself). The table in section 1 lists the exact values. Diwali overlaps with crop-stubble burning and falling temperatures, so this data alone cannot separate the effect of fireworks from those other causes.
4. **Bad-air days are common, not rare.** 37.8% of all days had AQI above 300. The share varies by year from 13.7% (2020) to 50.1% (2015). 2020 is low partly because it only covers January to June, which misses the Oct-Nov season. The classes are imbalanced but not extreme, which matters for choosing metrics in Phase 5.
5. **Tomorrow usually looks like today.** In the training years, tomorrow's bad/not-bad status matched today's on 85.3% of days, and the lag-1 correlation of AQI is 0.90. A "tomorrow = today" baseline will therefore be hard to beat on accuracy; the real test for Phase 5 models is the days when the status changes.
6. **Particulate matter drives the AQI.** In the training period the two pollutants most correlated with AQI are PM10 (r = 0.865) and PM2.5 (r = 0.865), well ahead of Benzene (r = 0.648). They are also correlated with each other (r = 0.84). This fits how AQI works: it is set by the worst pollutant, which in Delhi is usually PM2.5 or PM10.
7. **Some pollutants carry overlapping information.** 1 pollutant pair(s) have |r| of 0.8 or more in 2015-2018 (section 7a). Correlated inputs can make logistic regression coefficients unstable, so this is a point to watch when interpreting coefficients in Phase 5.
8. **The first half of 2020 was cleaner than earlier years.** The mean AQI for January to June was 182 in 2020, compared with 246 to 306 for the same months in 2015-2019. Phase 4 will test how much of this coincides with the 25 March 2020 lockdown.

