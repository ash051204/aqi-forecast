# Phase 1: Data Audit (Delhi)

Source file: `data/raw/city_day.csv` (2,574,056 bytes)
Whole file: 29,531 rows, 16 columns, 26 cities

## 1. Columns and data types (whole file, as loaded)

Columns match expected list: True

| Column | dtype |
|---|---|
| City | str |
| Date | str |
| PM2.5 | float64 |
| PM10 | float64 |
| NO | float64 |
| NO2 | float64 |
| NOx | float64 |
| NH3 | float64 |
| CO | float64 |
| SO2 | float64 |
| O3 | float64 |
| Benzene | float64 |
| Toluene | float64 |
| Xylene | float64 |
| AQI | float64 |
| AQI_Bucket | str |

Date is stored as text; values that fail to parse as YYYY-MM-DD: 0

## 2. Delhi subset

Rows: 2,009
Date range: 2015-01-01 to 2020-07-01

## 3. Missing values per column (Delhi)

| Column | Missing count | Missing % |
|---|---|---|
| City | 0 | 0.00% |
| Date | 0 | 0.00% |
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
| Xylene | 781 | 38.88% |
| AQI | 10 | 0.50% |
| AQI_Bucket | 10 | 0.50% |

## 4. Duplicate dates

Duplicate Date values: 0

## 5. Gaps in the daily date sequence

Days in full calendar range: 2,009
Days present in data: 2,009
Calendar days with no row at all: 0

## 6. Dates where AQI is missing

Count: 10

- 2016-07-24
- 2017-06-23
- 2017-08-13
- 2017-08-14
- 2017-08-22
- 2017-08-23
- 2017-08-26
- 2017-08-27
- 2017-08-28
- 2017-08-29

Consecutive runs of missing AQI (start, end, length in days):

| Start | End | Days |
|---|---|---|
| 2016-07-24 | 2016-07-24 | 1 |
| 2017-06-23 | 2017-06-23 | 1 |
| 2017-08-13 | 2017-08-14 | 2 |
| 2017-08-22 | 2017-08-23 | 2 |
| 2017-08-26 | 2017-08-29 | 4 |

Rows where AQI and AQI_Bucket disagree on being missing: 0

