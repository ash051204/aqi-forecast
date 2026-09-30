# Phase 2: Cleaning (Delhi)

## 1. Reindex to a continuous daily range

Rows before: 2009, rows after: 2009, rows added: 0
Range: 2015-01-01 to 2020-07-01

## 2. CPCB AQI bucket ranges used

| AQI range | Bucket |
|---|---|
| 0 to 50 | Good |
| 51 to 100 | Satisfactory |
| 101 to 200 | Moderate |
| 201 to 300 | Poor |
| 301 to 400 | Very Poor |
| above 400 | Severe |

Fractional values are assigned by their upper bound (for example 100.4 counts as above 100, so Moderate).
Check against the dataset's own labels on rows with real AQI: 1999 of 1999 agree.

## 3. AQI interpolation

Missing AQI days filled: 10
All filled dates between 2016-07-24 and 2017-08-29; all inside the 2015-2018 training period: True
AQI missing after interpolation: 0

**Why interpolation is acceptable here:** there are only 10 missing days (0.50%), the longest gap is 4 days, and all of them fall in the 2015-2018 training period. Each fill uses only the real values on the day before and the day after the gap, so nothing from the 2019 test year is used. The filled rows are marked with `aqi_imputed = True`, so they can be traced or excluded later.

| Date | Interpolated AQI | Bucket |
|---|---|---|
| 2016-07-24 | 179.0 | Moderate |
| 2017-06-23 | 117.5 | Moderate |
| 2017-08-13 | 49.7 | Good |
| 2017-08-14 | 54.3 | Satisfactory |
| 2017-08-22 | 47.3 | Good |
| 2017-08-23 | 52.7 | Satisfactory |
| 2017-08-26 | 97.8 | Satisfactory |
| 2017-08-27 | 91.6 | Satisfactory |
| 2017-08-28 | 85.4 | Satisfactory |
| 2017-08-29 | 79.2 | Satisfactory |

## 4. Xylene missingness over time

Total missing: 781 of 2009 days (38.88%)

| Year | Days | Missing | Missing % |
|---|---|---|---|
| 2015 | 365 | 152 | 41.64% |
| 2016 | 366 | 365 | 99.73% |
| 2017 | 365 | 264 | 72.33% |
| 2018 | 365 | 0 | 0.00% |
| 2019 | 365 | 0 | 0.00% |
| 2020 | 183 | 0 | 0.00% |

Number of separate missing runs: 6
Longest consecutive missing run: 606 days (2016-01-02 to 2017-08-29)
Share of all missing Xylene days inside that longest run: 77.59%
First and last date with a Xylene reading: 2015-01-01 to 2020-07-01
Missing share inside the 2015-2018 training period: 781 of 1461 days (53.46%)
Correlation with Toluene on days Xylene was recorded: 0.18; with Benzene: 0.34

**Decision: drop Xylene.** The missing days are not scattered: 78% of them sit in one unbroken block of 606 days (2016-01-02 to 2017-08-29), so the sensor was effectively offline for most of 2016 and 2017. Within the training years 53% of Xylene values are missing. Filling a gap that long would mean inventing more than a year and a half of readings, and the model would then learn from made-up values rather than measurements. The trade-off: Xylene is only weakly correlated with Toluene and Benzene on the days it was recorded (numbers above), so those two do not fully stand in for it and some information is lost. We accept that loss because keeping a column that is mostly invented in the training years is worse.

## 5. Remaining pollutant columns (left with NaNs, not imputed)

| Column | Missing count | Missing % |
|---|---|---|
| PM2.5 | 2 | 0.10% |
| PM10 | 77 | 3.83% |
| NO | 2 | 0.10% |
| NO2 | 2 | 0.10% |
| NOx | 0 | 0.00% |
| NH3 | 9 | 0.45% |
| CO | 0 | 0.00% |
| SO2 | 110 | 5.48% |
| O3 | 84 | 4.18% |
| Benzene | 0 | 0.00% |
| Toluene | 0 | 0.00% |

## 6. AQI outliers (IQR rule, flagged not deleted)

Q1 = 159.00, Q3 = 345.00, IQR = 186.00
Lower fence = -120.00, upper fence = 624.00
Flagged days: 6 (0.30%); below lower fence: 0, above upper fence: 6
AQI range of flagged days: 625.0 to 716.0

Flagged days by calendar month:

| Month | Flagged days |
|---|---|
| Nov | 6 |

Flagged days by year:

| Year | Flagged days |
|---|---|
| 2016 | 3 |
| 2017 | 2 |
| 2019 | 1 |

**Why these days are flagged, not deleted:** every flagged day is in November, the month when crop-stubble burning, Diwali fireworks, cold still air and low wind trap pollution over Delhi. These are real, recorded smog episodes, not sensor errors. They are the most important days for a public-health forecast and exactly the kind of day the Phase 5 classifier must learn to predict. Deleting them would make Delhi's air look cleaner than it really is. The `aqi_outlier` column keeps them identifiable. It is computed over 2015-2020, so it is for description only and must not be used as a model feature.

IQR outlier counts per pollutant (context only, nothing flagged or removed):

| Column | Outliers | % of non-missing |
|---|---|---|
| PM2.5 | 73 | 3.64% |
| PM10 | 19 | 0.98% |
| NO | 133 | 6.63% |
| NO2 | 51 | 2.54% |
| NOx | 79 | 3.93% |
| NH3 | 81 | 4.05% |
| CO | 185 | 9.21% |
| SO2 | 64 | 3.37% |
| O3 | 98 | 5.09% |
| Benzene | 84 | 4.18% |
| Toluene | 99 | 4.93% |

## 7. Output

Saved `data/processed/delhi_daily.csv`: 2009 rows, 17 columns
Columns: Date, City, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, AQI, AQI_Bucket, aqi_imputed, aqi_outlier

