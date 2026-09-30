"""Phase 1: data audit for Delhi.

This script only READS the raw file and reports what is in it. It never
changes or saves the data, because we want an honest picture of the raw
data before we decide how to clean it in Phase 2.

Run from the project root:
    .venv/bin/python src/audit.py
The report is printed and also saved to reports/phase1_audit.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed for consistency across the project (nothing random happens here,
# but every module sets it so the rule is simple to explain).
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "city_day.csv"
REPORT_PATH = ROOT / "reports" / "phase1_audit.md"

EXPECTED_COLUMNS = [
    "City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI", "AQI_Bucket",
]


def missing_runs(is_missing: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Group consecutive missing days into (start, end, length) runs.

    Why: Phase 2 treats short gaps (3 days or fewer) differently from long
    ones, so it helps to know the run lengths now, not just the dates.
    """
    runs = []
    # A new run starts whenever the missing flag changes value.
    run_id = (is_missing != is_missing.shift()).cumsum()
    for _, block in is_missing.groupby(run_id):
        if block.iloc[0]:
            runs.append((block.index[0], block.index[-1], len(block)))
    return runs


def main() -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    # Read Date as text first so we can check that every value parses as a
    # real date, instead of letting pandas silently coerce bad values.
    df = pd.read_csv(RAW_PATH)
    out("# Phase 1: Data Audit (Delhi)")
    out()
    out(f"Source file: `data/raw/city_day.csv` ({RAW_PATH.stat().st_size:,} bytes)")
    out(f"Whole file: {len(df):,} rows, {df.shape[1]} columns, {df['City'].nunique()} cities")
    out()

    # 1. Column names and dtypes
    out("## 1. Columns and data types (whole file, as loaded)")
    out()
    out(f"Columns match expected list: {list(df.columns) == EXPECTED_COLUMNS}")
    out()
    out("| Column | dtype |")
    out("|---|---|")
    for col, dtype in df.dtypes.items():
        out(f"| {col} | {dtype} |")
    out()
    parsed = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    out(f"Date is stored as text; values that fail to parse as YYYY-MM-DD: {parsed.isna().sum()}")
    out()

    # 2. Delhi subset
    delhi = df[df["City"] == "Delhi"].copy()
    delhi["Date"] = pd.to_datetime(delhi["Date"], format="%Y-%m-%d")
    delhi = delhi.sort_values("Date")

    out("## 2. Delhi subset")
    out()
    out(f"Rows: {len(delhi):,}")
    out(f"Date range: {delhi['Date'].min().date()} to {delhi['Date'].max().date()}")
    out()

    # 3. Missing % per column
    out("## 3. Missing values per column (Delhi)")
    out()
    out("| Column | Missing count | Missing % |")
    out("|---|---|---|")
    for col in delhi.columns:
        n = delhi[col].isna().sum()
        out(f"| {col} | {n} | {100 * n / len(delhi):.2f}% |")
    out()

    # 4. Duplicate dates: two rows for the same day would double-count that
    # day in any average, so we must know if they exist.
    dup_count = delhi["Date"].duplicated().sum()
    out("## 4. Duplicate dates")
    out()
    out(f"Duplicate Date values: {dup_count}")
    if dup_count:
        out(f"Duplicated dates: {sorted(delhi.loc[delhi['Date'].duplicated(keep=False), 'Date'].dt.date.unique())}")
    out()

    # 5. Gaps in the daily calendar: time series models assume one row per
    # day, so any missing calendar day would need to be added in Phase 2.
    full_range = pd.date_range(delhi["Date"].min(), delhi["Date"].max(), freq="D")
    absent = full_range.difference(delhi["Date"])
    out("## 5. Gaps in the daily date sequence")
    out()
    out(f"Days in full calendar range: {len(full_range):,}")
    out(f"Days present in data: {delhi['Date'].nunique():,}")
    out(f"Calendar days with no row at all: {len(absent)}")
    if len(absent):
        out(f"Absent dates: {[d.date().isoformat() for d in absent]}")
    out()

    # 6. Dates where AQI is missing (row exists but AQI is empty)
    aqi = delhi.set_index("Date")["AQI"]
    missing_aqi = aqi[aqi.isna()].index
    out("## 6. Dates where AQI is missing")
    out()
    out(f"Count: {len(missing_aqi)}")
    out()
    for d in missing_aqi:
        out(f"- {d.date().isoformat()}")
    out()
    out("Consecutive runs of missing AQI (start, end, length in days):")
    out()
    out("| Start | End | Days |")
    out("|---|---|---|")
    for start, end, length in missing_runs(aqi.isna()):
        out(f"| {start.date()} | {end.date()} | {length} |")
    out()

    # Extra check: AQI_Bucket should be empty exactly when AQI is empty,
    # because the bucket is derived from the AQI value.
    mismatch = (delhi["AQI"].isna() != delhi["AQI_Bucket"].isna()).sum()
    out(f"Rows where AQI and AQI_Bucket disagree on being missing: {mismatch}")
    out()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved report to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
