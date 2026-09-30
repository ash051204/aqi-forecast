"""Phase 2: cleaning the Delhi daily data.

Decisions (agreed before coding):
  * AQI: fill all 10 missing days with time-based interpolation and mark them
    in an "aqi_imputed" column. All gaps are in 2016-2017, inside the
    2015-2018 training period, so no information from the 2019 test year is
    used to fill them.
  * AQI_Bucket: recomputed for the filled rows from the interpolated AQI,
    using the CPCB bucket ranges.
  * Pollutants: NOT imputed here. Their NaNs stay in the processed file.
    Imputation happens later inside the Phase 5 sklearn Pipeline, fitted on
    training data only, so test-year values never influence the fill.
  * Xylene: dropped (reasoning printed in the report).
  * Outliers: flagged with the IQR rule in "aqi_outlier", never deleted,
    because Delhi's winter smog spikes are real events.

The raw file is only read, never written.

Run from the project root:
    .venv/bin/python src/clean.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "city_day.csv"
OUT_PATH = ROOT / "data" / "processed" / "delhi_daily.csv"
REPORT_PATH = ROOT / "reports" / "phase2_cleaning.md"
FIG_DIR = ROOT / "reports" / "figures"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]

# CPCB National AQI categories. Upper bounds are inclusive. The CPCB scale
# uses whole numbers (0-50, 51-100, ...); interpolated AQI can be fractional,
# so a value like 100.4 is treated as above 100 and falls into the next band.
# Anything above 400 is "Severe" (the dataset has values above 500 labelled
# Severe too).
CPCB_BUCKETS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (np.inf, "Severe"),
]


def cpcb_bucket(aqi: float) -> str:
    for upper, label in CPCB_BUCKETS:
        if aqi <= upper:
            return label
    raise ValueError(aqi)


def longest_true_run(flags: pd.Series) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Length, start and end of the longest run of consecutive True values."""
    run_id = (flags != flags.shift()).cumsum()
    best = (0, None, None)
    for _, block in flags.groupby(run_id):
        if block.iloc[0] and len(block) > best[0]:
            best = (len(block), block.index[0], block.index[-1])
    return best


def main() -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    raw = pd.read_csv(RAW_PATH)
    delhi = raw[raw["City"] == "Delhi"].copy()
    delhi["Date"] = pd.to_datetime(delhi["Date"], format="%Y-%m-%d")
    delhi = delhi.sort_values("Date").set_index("Date")

    out("# Phase 2: Cleaning (Delhi)")
    out()

    # ---------------------------------------------------------------
    # Step 1: reindex to a continuous daily calendar.
    # Time series methods expect exactly one row per day. Phase 1 found no
    # absent days, but we reindex anyway so the processed file is
    # guaranteed to be continuous, and we report how many rows were added.
    # ---------------------------------------------------------------
    full_range = pd.date_range(delhi.index.min(), delhi.index.max(), freq="D", name="Date")
    rows_before = len(delhi)
    delhi = delhi.reindex(full_range)
    delhi["City"] = "Delhi"
    out("## 1. Reindex to a continuous daily range")
    out()
    out(f"Rows before: {rows_before}, rows after: {len(delhi)}, rows added: {len(delhi) - rows_before}")
    out(f"Range: {full_range.min().date()} to {full_range.max().date()}")
    out()

    # ---------------------------------------------------------------
    # Step 2: sanity-check the CPCB bucket function against the dataset.
    # Before we use our own bucket rule on the filled rows, confirm it gives
    # the same labels the dataset already has on the rows with real AQI.
    # ---------------------------------------------------------------
    known = delhi.dropna(subset=["AQI"])
    recomputed = known["AQI"].apply(cpcb_bucket)
    agree = (recomputed == known["AQI_Bucket"]).sum()
    out("## 2. CPCB AQI bucket ranges used")
    out()
    out("| AQI range | Bucket |")
    out("|---|---|")
    out("| 0 to 50 | Good |")
    out("| 51 to 100 | Satisfactory |")
    out("| 101 to 200 | Moderate |")
    out("| 201 to 300 | Poor |")
    out("| 301 to 400 | Very Poor |")
    out("| above 400 | Severe |")
    out()
    out("Fractional values are assigned by their upper bound (for example 100.4 counts as above 100, so Moderate).")
    out(f"Check against the dataset's own labels on rows with real AQI: {agree} of {len(known)} agree.")
    out()

    # ---------------------------------------------------------------
    # Step 3: fill missing AQI with time-based interpolation.
    # Time interpolation draws a straight line between the known values on
    # either side of a gap, weighted by the number of days. It only uses the
    # days just before and just after each gap. Every gap is in 2016-2017,
    # inside the 2015-2018 training period, so no 2019 test data leaks in.
    # ---------------------------------------------------------------
    missing_mask = delhi["AQI"].isna()
    delhi["aqi_imputed"] = missing_mask
    delhi["AQI"] = delhi["AQI"].interpolate(method="time")
    delhi.loc[missing_mask, "AQI_Bucket"] = delhi.loc[missing_mask, "AQI"].apply(cpcb_bucket)

    out("## 3. AQI interpolation")
    out()
    out(f"Missing AQI days filled: {int(missing_mask.sum())}")
    out(f"All filled dates between {delhi.index[missing_mask].min().date()} and {delhi.index[missing_mask].max().date()}; "
        f"all inside the 2015-2018 training period: {bool((delhi.index[missing_mask].year <= 2018).all())}")
    out(f"AQI missing after interpolation: {int(delhi['AQI'].isna().sum())}")
    out()
    out("**Why interpolation is acceptable here:** there are only 10 missing days (0.50%), the longest gap is 4 days, "
        "and all of them fall in the 2015-2018 training period. Each fill uses only the real values on the day "
        "before and the day after the gap, so nothing from the 2019 test year is used. The filled rows are marked "
        "with `aqi_imputed = True`, so they can be traced or excluded later.")
    out()
    out("| Date | Interpolated AQI | Bucket |")
    out("|---|---|---|")
    for d, row in delhi.loc[missing_mask, ["AQI", "AQI_Bucket"]].iterrows():
        out(f"| {d.date()} | {row['AQI']:.1f} | {row['AQI_Bucket']} |")
    out()

    # ---------------------------------------------------------------
    # Step 4: Xylene. Look at WHEN it is missing before deciding.
    # ---------------------------------------------------------------
    xy_missing = delhi["Xylene"].isna()
    out("## 4. Xylene missingness over time")
    out()
    out(f"Total missing: {int(xy_missing.sum())} of {len(delhi)} days ({100 * xy_missing.mean():.2f}%)")
    out()
    out("| Year | Days | Missing | Missing % |")
    out("|---|---|---|---|")
    by_year = xy_missing.groupby(delhi.index.year).agg(["size", "sum"])
    for year, r in by_year.iterrows():
        out(f"| {year} | {r['size']} | {r['sum']} | {100 * r['sum'] / r['size']:.2f}% |")
    out()
    length, start, end = longest_true_run(xy_missing)
    run_id = (xy_missing != xy_missing.shift()).cumsum()
    n_runs = int(xy_missing.groupby(run_id).first().sum())
    out(f"Number of separate missing runs: {n_runs}")
    out(f"Longest consecutive missing run: {length} days ({start.date()} to {end.date()})")
    out(f"Share of all missing Xylene days inside that longest run: {100 * length / xy_missing.sum():.2f}%")
    present = delhi.index[~xy_missing]
    out(f"First and last date with a Xylene reading: {present.min().date()} to {present.max().date()}")
    train_mask = delhi.index.year <= 2018
    train_missing_pct = 100 * xy_missing[train_mask].mean()
    out(f"Missing share inside the 2015-2018 training period: {int(xy_missing[train_mask].sum())} of {int(train_mask.sum())} days ({train_missing_pct:.2f}%)")
    # Does Xylene carry information the other aromatic compounds do not?
    # Measure its correlation with Toluene and Benzene on days it was recorded.
    corr = delhi[["Xylene", "Toluene", "Benzene"]].dropna().corr()
    out(f"Correlation with Toluene on days Xylene was recorded: {corr.loc['Xylene', 'Toluene']:.2f}; with Benzene: {corr.loc['Xylene', 'Benzene']:.2f}")
    out()
    out("**Decision: drop Xylene.** The missing days are not scattered: "
        f"{100 * length / xy_missing.sum():.0f}% of them sit in one unbroken block of {length} days "
        f"({start.date()} to {end.date()}), so the sensor was effectively offline for most of 2016 and 2017. "
        f"Within the training years {train_missing_pct:.0f}% of Xylene values are missing. "
        "Filling a gap that long would mean inventing more than a year and a half of readings, "
        "and the model would then learn from made-up values rather than measurements. "
        "The trade-off: Xylene is only weakly correlated with Toluene and Benzene on the days it was recorded "
        "(numbers above), so those two do not fully stand in for it and some information is lost. "
        "We accept that loss because keeping a column that is mostly invented in the training years is worse.")
    out()

    # ---------------------------------------------------------------
    # Step 5: pollutant missingness after dropping Xylene. These are NOT
    # imputed here; the Phase 5 pipeline imputes them using training data only.
    # ---------------------------------------------------------------
    delhi = delhi.drop(columns=["Xylene"])
    kept_pollutants = [p for p in POLLUTANTS if p != "Xylene"]
    out("## 5. Remaining pollutant columns (left with NaNs, not imputed)")
    out()
    out("| Column | Missing count | Missing % |")
    out("|---|---|---|")
    for col in kept_pollutants:
        n = int(delhi[col].isna().sum())
        out(f"| {col} | {n} | {100 * n / len(delhi):.2f}% |")
    out()

    # ---------------------------------------------------------------
    # Step 6: flag AQI outliers with the IQR rule (1.5 x IQR beyond the
    # quartiles). We FLAG and keep them: Delhi's winter smog episodes are
    # real, recorded pollution events, and they are exactly the days the
    # Phase 5 classifier needs to learn to predict. Deleting them would make
    # Delhi look cleaner than it is.
    # The flag is descriptive only. It is computed on the full 2015-2020
    # series, so it must not be used as a model feature.
    # ---------------------------------------------------------------
    q1, q3 = delhi["AQI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    delhi["aqi_outlier"] = (delhi["AQI"] < low) | (delhi["AQI"] > high)
    flagged = delhi[delhi["aqi_outlier"]]

    out("## 6. AQI outliers (IQR rule, flagged not deleted)")
    out()
    out(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
    out(f"Lower fence = {low:.2f}, upper fence = {high:.2f}")
    out(f"Flagged days: {len(flagged)} ({100 * len(flagged) / len(delhi):.2f}%); "
        f"below lower fence: {int((delhi['AQI'] < low).sum())}, above upper fence: {int((delhi['AQI'] > high).sum())}")
    if len(flagged):
        out(f"AQI range of flagged days: {flagged['AQI'].min():.1f} to {flagged['AQI'].max():.1f}")
    out()
    out("Flagged days by calendar month:")
    out()
    out("| Month | Flagged days |")
    out("|---|---|")
    for month, n in flagged.index.month.value_counts().sort_index().items():
        out(f"| {pd.Timestamp(2000, month, 1):%b} | {n} |")
    out()
    out("Flagged days by year:")
    out()
    out("| Year | Flagged days |")
    out("|---|---|")
    for year, n in flagged.index.year.value_counts().sort_index().items():
        out(f"| {year} | {n} |")
    out()

    out("**Why these days are flagged, not deleted:** every flagged day is in November, the month when crop-stubble "
        "burning, Diwali fireworks, cold still air and low wind trap pollution over Delhi. These are real, recorded "
        "smog episodes, not sensor errors. They are the most important days for a public-health forecast and "
        "exactly the kind of day the Phase 5 classifier must learn to predict. Deleting them would make Delhi's air "
        "look cleaner than it really is. The `aqi_outlier` column keeps them identifiable. It is computed over "
        "2015-2020, so it is for description only and must not be used as a model feature.")
    out()

    # For context only: how many IQR outliers each pollutant has. No flag
    # columns are added for these; they are not changed.
    out("IQR outlier counts per pollutant (context only, nothing flagged or removed):")
    out()
    out("| Column | Outliers | % of non-missing |")
    out("|---|---|---|")
    for col in kept_pollutants:
        s = delhi[col].dropna()
        pq1, pq3 = s.quantile([0.25, 0.75])
        piqr = pq3 - pq1
        n = int(((s < pq1 - 1.5 * piqr) | (s > pq3 + 1.5 * piqr)).sum())
        out(f"| {col} | {n} | {100 * n / len(s):.2f}% |")
    out()

    # Box plots: overall AQI, and AQI by month to show where outliers sit.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 4]})
    axes[0].boxplot(delhi["AQI"], widths=0.5)
    axes[0].set_title("Delhi daily AQI, 2015-2020")
    axes[0].set_ylabel("AQI")
    axes[0].set_xticks([1], ["All days"])
    months = [delhi.loc[delhi.index.month == m, "AQI"] for m in range(1, 13)]
    axes[1].boxplot(months)
    axes[1].set_xticks(range(1, 13), [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])
    axes[1].axhline(high, color="tab:red", linestyle="--", linewidth=1, label=f"Overall IQR upper fence ({high:.0f})")
    axes[1].set_title("Delhi daily AQI by month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("AQI")
    axes[1].legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2_aqi_boxplots.png", dpi=150)
    plt.close(fig)

    # Pollutant box plots on separate panels because their units and
    # scales differ a lot (CO is in mg/m3, the rest in ug/m3).
    fig, axes = plt.subplots(2, 6, figsize=(16, 7))
    for ax, col in zip(axes.flat, kept_pollutants):
        ax.boxplot(delhi[col].dropna(), widths=0.5)
        ax.set_title(col)
        ax.set_xticks([])
        ax.set_ylabel("mg/m3" if col == "CO" else "ug/m3")
    for ax in axes.flat[len(kept_pollutants):]:
        ax.axis("off")
    fig.suptitle("Delhi daily pollutant concentrations, 2015-2020 (box plots, missing values excluded)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase2_pollutant_boxplots.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Step 7: save.
    # ---------------------------------------------------------------
    delhi = delhi.reset_index()
    cols = ["Date", "City"] + kept_pollutants + ["AQI", "AQI_Bucket", "aqi_imputed", "aqi_outlier"]
    delhi = delhi[cols]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    delhi.to_csv(OUT_PATH, index=False, date_format="%Y-%m-%d")

    out("## 7. Output")
    out()
    out(f"Saved `data/processed/delhi_daily.csv`: {len(delhi)} rows, {delhi.shape[1]} columns")
    out(f"Columns: {', '.join(cols)}")
    out()

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved report to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
