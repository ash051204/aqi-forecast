"""Phase 3: exploratory data analysis (EDA) of Delhi's daily AQI.

Leakage rule used in this file:
  * Descriptive plots and statistics may use every year (2015 to mid-2020),
    because they only describe the data and do not train anything.
  * Anything that will be used to CHOOSE model features (the correlation
    analysis) is also computed on 2015-2018 only, the training period. If we
    looked at 2019 to pick features, the test set would no longer be a fair,
    unseen check.

Run from the project root:
    .venv/bin/python src/eda.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "delhi_daily.csv"
REPORT_PATH = ROOT / "reports" / "phase3_eda.md"
FIG_DIR = ROOT / "reports" / "figures"
OUT_DIR = ROOT / "reports" / "outputs"

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene"]
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BAD_AIR_THRESHOLD = 300  # AQI above 300 = Very Poor or Severe (the Phase 5 target)
TRAIN_END = "2018-12-31"

# Diwali dates, checked against Government of India DoPT holiday orders
# (the official list of Central Government holidays for Delhi/New Delhi).
# Each date is listed with the order it came from so it can be traced.
DIWALI = {
    "2015-11-11": "DoPT OM F.No.12/5/2014-JCA-2 (6 Jun 2014)",
    "2016-10-30": "DoPT OM F.No.12/7/2015-JCA-2 (11 Jun 2015)",
    "2017-10-19": "DoPT OM F.No.12/8/2016-JCA-2 (24 Jun 2016)",
    "2018-11-07": "DoPT OM F.No.12/3/2017-JCA-2 (14 Jun 2017)",
    "2019-10-27": "DoPT OM F.No.12/2/2018-JCA-2 (11 Jul 2018)",
}

MONTH_NAMES = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]


def stats_table(groups: pd.core.groupby.SeriesGroupBy, label: str, names=None) -> list[str]:
    """Markdown table of mean, median, std and variance for each group."""
    s = groups.agg(["count", "mean", "median", "std", "var"])
    rows = [f"| {label} | Days | Mean | Median | Std | Variance |", "|---|---|---|---|---|---|"]
    for key, r in s.iterrows():
        name = names[key - 1] if names else key
        rows.append(f"| {name} | {int(r['count'])} | {r['mean']:.1f} | {r['median']:.1f} | {r['std']:.1f} | {r['var']:.1f} |")
    return rows


def corr_heatmap(corr: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
    aqi = df["AQI"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    out("# Phase 3: Exploratory Data Analysis (Delhi)")
    out()
    out(f"Data: `data/processed/delhi_daily.csv`, {len(df)} days, {df.index.min().date()} to {df.index.max().date()}.")
    out("Descriptive results use all years. Results used to choose model features use 2015-2018 only (marked).")
    out()

    # ---------------------------------------------------------------
    # Figure 1: full AQI time series with Diwali and the Oct-Nov season.
    # A 30-day rolling mean is drawn on top because daily values jump
    # around a lot and the rolling mean makes the yearly cycle easier to see.
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for year in range(2015, 2020):
        ax.axvspan(pd.Timestamp(year, 10, 1), pd.Timestamp(year, 11, 30), color="tab:orange", alpha=0.12,
                   label="Oct-Nov season" if year == 2015 else None)
    ax.plot(aqi.index, aqi, color="tab:blue", linewidth=0.6, alpha=0.6, label="Daily AQI")
    ax.plot(aqi.index, aqi.rolling(30, center=True).mean(), color="black", linewidth=1.5, label="30-day rolling mean")
    ax.axhline(BAD_AIR_THRESHOLD, color="tab:red", linestyle=":", linewidth=1, label="AQI 300 (Very Poor threshold)")
    for i, d in enumerate(DIWALI):
        ax.axvline(pd.Timestamp(d), color="tab:purple", linestyle="--", linewidth=1,
                   label="Diwali" if i == 0 else None)
    ax.set_title("Delhi daily AQI, January 2015 to July 2020")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    ax.set_xlim(aqi.index.min(), aqi.index.max())
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_aqi_timeseries.png", dpi=150)
    plt.close(fig)

    # AQI around each Diwali: the day itself, the next day, and the highest
    # value in the 7 days after, to see whether Diwali lines up with peaks.
    out("## 1. AQI time series and Diwali")
    out()
    out("Figure: `reports/figures/phase3_aqi_timeseries.png` (Oct-Nov shaded, Diwali marked with dashed lines).")
    out()
    out("Diwali dates were verified against Government of India DoPT holiday orders for Central Government offices in Delhi:")
    out()
    out("| Diwali | Source | AQI on Diwali | AQI next day | Max AQI in next 7 days | Oct-Nov max AQI (date) |")
    out("|---|---|---|---|---|---|")
    for d, src in DIWALI.items():
        t = pd.Timestamp(d)
        week = aqi[t + pd.Timedelta(days=1): t + pd.Timedelta(days=7)]
        season = aqi[f"{t.year}-10-01": f"{t.year}-11-30"]
        out(f"| {d} | {src} | {aqi[t]:.0f} | {aqi[t + pd.Timedelta(days=1)]:.0f} | {week.max():.0f} | "
            f"{season.max():.0f} ({season.idxmax().date()}) |")
    out()

    # ---------------------------------------------------------------
    # Figure 2: monthly box plots to show seasonality.
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([aqi[aqi.index.month == m] for m in range(1, 13)])
    ax.set_xticks(range(1, 13), MONTH_NAMES)
    ax.axhline(BAD_AIR_THRESHOLD, color="tab:red", linestyle=":", linewidth=1, label="AQI 300")
    ax.set_title("Delhi daily AQI by calendar month, 2015-2020")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_monthly_boxplots.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Figure 3: histogram of daily AQI with the CPCB band edges drawn in.
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(aqi, bins=50, color="tab:blue", edgecolor="white")
    for edge in [50, 100, 200, 300, 400]:
        ax.axvline(edge, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Distribution of Delhi daily AQI, 2015-2020 (dashed lines = CPCB band edges)")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Number of days")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_aqi_histogram.png", dpi=150)
    plt.close(fig)

    out("## 2. Distribution of daily AQI (all years)")
    out()
    out("Figures: `phase3_monthly_boxplots.png`, `phase3_aqi_histogram.png`.")
    out()
    out(f"Mean {aqi.mean():.1f}, median {aqi.median():.1f}, std {aqi.std():.1f}, variance {aqi.var():.1f}, "
        f"min {aqi.min():.1f}, max {aqi.max():.1f}, skewness {aqi.skew():.2f}.")
    out()

    # ---------------------------------------------------------------
    # Summary statistics by month and by year (descriptive, all years).
    # ---------------------------------------------------------------
    out("## 3. AQI summary statistics by month (all years)")
    out()
    for row in stats_table(aqi.groupby(aqi.index.month), "Month", MONTH_NAMES):
        out(row)
    out()
    out("## 4. AQI summary statistics by year (all years; 2020 covers 1 Jan to 1 Jul only)")
    out()
    for row in stats_table(aqi.groupby(aqi.index.year), "Year"):
        out(row)
    out()

    # ---------------------------------------------------------------
    # Figure 5: AQI_Bucket counts in CPCB order.
    # ---------------------------------------------------------------
    counts = df["AQI_Bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color="tab:blue")
    ax.bar_label(bars, labels=[f"{v} ({100 * v / counts.sum():.1f}%)" for v in counts.values], fontsize=8)
    ax.set_title("Delhi days per CPCB AQI category, 2015-2020")
    ax.set_xlabel("AQI category")
    ax.set_ylabel("Number of days")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_bucket_counts.png", dpi=150)
    plt.close(fig)

    out("## 5. AQI category counts (all years)")
    out()
    out("Figure: `phase3_bucket_counts.png`.")
    out()
    out("| Category | Days | % |")
    out("|---|---|---|")
    for name, n in counts.items():
        out(f"| {name} | {n} | {100 * n / counts.sum():.2f}% |")
    out()

    # ---------------------------------------------------------------
    # Class balance preview for Phase 5.
    # "Bad day" = AQI > 300. Persistence agreement = share of days where
    # tomorrow's bad/not-bad status is the same as today's; this is how often
    # the Phase 5 baseline ("tomorrow = today") would be right.
    # ---------------------------------------------------------------
    bad = (aqi > BAD_AIR_THRESHOLD).astype(int)
    tomorrow = bad.shift(-1)
    pairs = pd.DataFrame({"today": bad, "tomorrow": tomorrow}).dropna()  # last day has no tomorrow
    same = pairs["today"] == pairs["tomorrow"]

    out("## 6. Class balance preview for Phase 5 (AQI > 300)")
    out()
    out("| Period | Days | Days with AQI > 300 | % | Day pairs | Tomorrow's status = today's |")
    out("|---|---|---|---|---|---|")
    periods = [(str(y), bad.index.year == y, pairs.index.year == y) for y in range(2015, 2021)]
    periods += [
        # Stop the training pairs at 2018-12-30 so the last pair's "tomorrow"
        # is still in 2018; the pair 2018-12-31 to 2019-01-01 would peek at test data.
        ("Train 2015-2018", bad.index <= TRAIN_END, pairs.index < TRAIN_END),
        ("Test 2019", bad.index.year == 2019, pairs.index.year == 2019),
        ("Overall", np.ones(len(bad), bool), np.ones(len(pairs), bool)),
    ]
    for name, m_bad, m_pair in periods:
        b = bad[m_bad]
        s = same[m_pair]
        out(f"| {name} | {len(b)} | {int(b.sum())} | {100 * b.mean():.2f}% | {len(s)} | {100 * s.mean():.2f}% |")
    out()
    out("Day pairs are (today, tomorrow). Each year's pairs start on that year's dates, so the pair 31 Dec to 1 Jan "
        "counts in the earlier year. The final day (2020-07-01) has no tomorrow and is excluded. "
        "The training row stops at the pair 2018-12-30 to 2018-12-31 so that no 2019 value is used.")
    # How often does the status CHANGE, split by direction? The baseline
    # misses exactly these days, so this tells us where it will fail.
    up = int(((pairs["today"] == 0) & (pairs["tomorrow"] == 1)).sum())
    down = int(((pairs["today"] == 1) & (pairs["tomorrow"] == 0)).sum())
    out(f"Status changes over all years: {up} days go from not-bad to bad, {down} go from bad to not-bad.")
    out()

    # ---------------------------------------------------------------
    # Correlation heatmaps.
    # (a) Full period: for description only.
    # (b) 2015-2018 only: the one used for feature decisions.
    # Pearson correlation on pairwise complete rows (pollutant NaNs are
    # not imputed, so each pair uses the days where both are recorded).
    # ---------------------------------------------------------------
    cols = POLLUTANTS + ["AQI"]
    corr_full = df[cols].corr()
    train = df.loc[:TRAIN_END]
    corr_train = train[cols].corr()
    corr_heatmap(corr_full, "Correlation of pollutants and AQI, 2015-2020\n(full period, DESCRIPTIVE ONLY)",
                 FIG_DIR / "phase3_corr_full_descriptive.png")
    corr_heatmap(corr_train, "Correlation of pollutants and AQI, 2015-2018\n(TRAINING PERIOD, used for feature decisions)",
                 FIG_DIR / "phase3_corr_train_2015_2018.png")

    out("## 7. Correlations")
    out()
    out("Figures: `phase3_corr_full_descriptive.png` (2015-2020, description only) and "
        "`phase3_corr_train_2015_2018.png` (training period, used for feature decisions).")
    out()
    out("### 7a. Training period 2015-2018 (for feature decisions)")
    out()
    # Also correlate each pollutant with NEXT day's AQI, since that is what
    # the Phase 5 classifier predicts. shift(-1) is computed inside the
    # training slice, so the 2018-12-31 row has no next-day value and is
    # dropped rather than peeking at 2019-01-01.
    next_aqi = train["AQI"].shift(-1)
    out("| Pollutant | r with same-day AQI | r with next-day AQI |")
    out("|---|---|---|")
    order = corr_train["AQI"].drop("AQI").sort_values(ascending=False).index
    for p in order:
        out(f"| {p} | {corr_train.loc[p, 'AQI']:.2f} | {train[p].corr(next_aqi):.2f} |")
    out()
    out(f"Same-day AQI vs next-day AQI (lag-1 autocorrelation) in 2015-2018: {train['AQI'].corr(next_aqi):.2f}")
    out()
    out("Pollutant pairs with |r| >= 0.8 in 2015-2018 (strongly overlapping information):")
    out()
    out("| Pair | r |")
    out("|---|---|")
    strong = []
    for i, a in enumerate(POLLUTANTS):
        for b in POLLUTANTS[i + 1:]:
            if abs(corr_train.loc[a, b]) >= 0.8:
                strong.append((a, b, corr_train.loc[a, b]))
    for a, b, r in sorted(strong, key=lambda x: -abs(x[2])):
        out(f"| {a} and {b} | {r:.2f} |")
    if not strong:
        out("| none | |")
    out()
    out("### 7b. Full period 2015-2020 (description only)")
    out()
    out("| Pollutant | r with AQI (2015-2020) | r with AQI (2015-2018) | Difference |")
    out("|---|---|---|---|")
    for p in order:
        out(f"| {p} | {corr_full.loc[p, 'AQI']:.2f} | {corr_train.loc[p, 'AQI']:.2f} | "
            f"{corr_full.loc[p, 'AQI'] - corr_train.loc[p, 'AQI']:+.2f} |")
    out()

    # ---------------------------------------------------------------
    # Seasonal decomposition of the WEEKLY mean AQI.
    # Weekly averaging smooths out day-to-day noise; a period of 52 weeks
    # captures the yearly cycle. Additive model: AQI = trend + season + residual.
    # Weeks end on Sunday (pandas default "W"). The first and last weeks may
    # be partial because the data starts on a Thursday and ends on a Wednesday.
    # ---------------------------------------------------------------
    weekly = aqi.resample("W").mean()
    dec = seasonal_decompose(weekly, model="additive", period=52)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    parts = [(weekly, "Observed weekly mean AQI"), (dec.trend, "Trend"),
             (dec.seasonal, "Seasonal component (repeats every 52 weeks)"), (dec.resid, "Residual")]
    for ax, (series, name) in zip(axes, parts):
        if name == "Residual":
            ax.scatter(series.index, series, s=6, color="tab:blue")
            ax.axhline(0, color="grey", linewidth=0.8)
        else:
            ax.plot(series.index, series, color="tab:blue")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("AQI")
    axes[-1].set_xlabel("Week")
    fig.suptitle("Seasonal decomposition of Delhi weekly mean AQI (additive, period 52 weeks)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase3_seasonal_decomposition.png", dpi=150)
    plt.close(fig)

    seasonal_one_year = dec.seasonal.iloc[:52]
    trend = dec.trend.dropna()
    resid = dec.resid.dropna()
    # Share of variation explained: compare the spread of each component
    # with the spread of the detrended series (seasonal + residual).
    detrended = (dec.seasonal + dec.resid).dropna()
    strength_seasonal = max(0.0, 1 - resid.var() / detrended.var())
    trend_by_year = trend.groupby(trend.index.year).mean()

    out("## 8. Seasonal decomposition (weekly mean AQI, additive, period 52)")
    out()
    out("Figure: `phase3_seasonal_decomposition.png`.")
    out()
    out(f"Weekly series: {len(weekly)} weeks, {weekly.index.min().date()} to {weekly.index.max().date()} (week-ending Sundays).")
    peak_week = seasonal_one_year.idxmax()
    low_week = seasonal_one_year.idxmin()
    out(f"Seasonal component: highest {seasonal_one_year.max():+.1f} (week ending {peak_week:%d %b}), "
        f"lowest {seasonal_one_year.min():+.1f} (week ending {low_week:%d %b}), "
        f"swing {seasonal_one_year.max() - seasonal_one_year.min():.1f} AQI points.")
    out(f"Seasonal strength, 1 - Var(residual) / Var(seasonal + residual): {strength_seasonal:.2f} "
        "(0 = no seasonality, 1 = all variation after removing trend is seasonal).")
    out(f"Trend defined from {trend.index.min().date()} to {trend.index.max().date()} "
        "(26 weeks are lost at each end by the centred 52-week moving average).")
    out()
    out("| Year | Mean of trend component |")
    out("|---|---|")
    for y, v in trend_by_year.items():
        out(f"| {y} | {v:.1f} |")
    out()
    out(f"Residual: std {resid.std():.1f}, largest positive {resid.max():.1f} (week ending {resid.idxmax().date()}), "
        f"largest negative {resid.min():.1f} (week ending {resid.idxmin().date()}).")
    out()

    # Numbers the findings below rely on.
    monthly_mean = aqi.groupby(aqi.index.month).mean()
    yearly_bad = bad.groupby(bad.index.year).mean() * 100
    train_same = 100 * same[pairs.index < TRAIN_END].mean()
    full_months = aqi.loc["2015-01-01":"2019-12-31"]
    jan_jun = aqi[aqi.index.month <= 6]
    jan_jun_mean = jan_jun.groupby(jan_jun.index.year).mean()
    top = corr_train["AQI"].drop("AQI").sort_values(ascending=False)
    diwali_peak_hits = sum(
        aqi[pd.Timestamp(d): pd.Timestamp(d) + pd.Timedelta(days=7)].max()
        >= aqi[f"{d[:4]}-10-01": f"{d[:4]}-11-30"].max() for d in DIWALI
    )

    out("## 9. Findings in plain language")
    out()
    out(f"1. **Delhi's air follows a strong yearly cycle.** The monthly mean AQI is highest in November "
        f"({monthly_mean[11]:.0f}) and lowest in August ({monthly_mean[8]:.0f}), about "
        f"{monthly_mean[11] / monthly_mean[8]:.1f} times higher. The decomposition gives a seasonal strength of "
        f"{strength_seasonal:.2f}, so once the trend is removed most of what is left is the repeating yearly pattern. "
        "This is why the forecasting models in Phase 4 need a seasonal part.")
    out(f"2. **Winter is the bad-air season, the monsoon is the clean season.** The months from October to January all "
        f"have mean AQI above 300 ({', '.join(f'{MONTH_NAMES[m - 1]} {monthly_mean[m]:.0f}' for m in [10, 11, 12, 1])}), while "
        f"July to September sit between {monthly_mean[[7, 8, 9]].min():.0f} and {monthly_mean[[7, 8, 9]].max():.0f}. "
        "Monsoon rain washes particles out of the air; in winter, cold still air traps them near the ground.")
    next_day_higher = sum(aqi[pd.Timestamp(d) + pd.Timedelta(days=1)] > aqi[pd.Timestamp(d)] for d in DIWALI)
    out(f"3. **Diwali falls inside the worst period, but it is not always the single worst day.** "
        f"AQI the day after Diwali was higher than on Diwali itself in {next_day_higher} of 5 years. In {diwali_peak_hits} of 5 years "
        "the highest Oct-Nov AQI came within 7 days after Diwali (counting Diwali itself). The table in section 1 lists "
        "the exact values. Diwali overlaps with crop-stubble burning and falling temperatures, so this data alone "
        "cannot separate the effect of fireworks from those other causes.")
    out(f"4. **Bad-air days are common, not rare.** {100 * bad.mean():.1f}% of all days had AQI above 300. The share "
        f"varies by year from {yearly_bad.min():.1f}% ({yearly_bad.idxmin()}) to {yearly_bad.max():.1f}% "
        f"({yearly_bad.idxmax()}). 2020 is low partly because it only covers January to June, which misses the "
        "Oct-Nov season. The classes are imbalanced but not extreme, which matters for choosing metrics in Phase 5.")
    out(f"5. **Tomorrow usually looks like today.** In the training years, tomorrow's bad/not-bad status matched "
        f"today's on {train_same:.1f}% of days, and the lag-1 correlation of AQI is "
        f"{train['AQI'].corr(next_aqi):.2f}. A \"tomorrow = today\" baseline will therefore be hard to beat on "
        "accuracy; the real test for Phase 5 models is the days when the status changes.")
    out(f"6. **Particulate matter drives the AQI.** In the training period the two pollutants most correlated with AQI "
        f"are {top.index[0]} (r = {top.iloc[0]:.3f}) and {top.index[1]} (r = {top.iloc[1]:.3f}), well ahead of "
        f"{top.index[2]} (r = {top.iloc[2]:.3f}). They are also correlated with each other "
        f"(r = {corr_train.loc['PM2.5', 'PM10']:.2f}). "
        f"This fits how AQI works: it is set by the worst pollutant, which in Delhi is usually PM2.5 or PM10.")
    out(f"7. **Some pollutants carry overlapping information.** {len(strong)} pollutant pair(s) have |r| of 0.8 or more "
        "in 2015-2018 (section 7a). Correlated inputs can make logistic regression coefficients unstable, "
        "so this is a point to watch when interpreting coefficients in Phase 5.")
    out(f"8. **The first half of 2020 was cleaner than earlier years.** The mean AQI for January to June was "
        f"{jan_jun_mean[2020]:.0f} in 2020, compared with {jan_jun_mean.loc[2015:2019].min():.0f} to "
        f"{jan_jun_mean.loc[2015:2019].max():.0f} for the same months in 2015-2019. Phase 4 will test how much of this "
        "coincides with the 25 March 2020 lockdown.")
    out()

    text = "\n".join(lines) + "\n"
    assert "\u2014" not in text and "\u2013" not in text, "report contains an em or en dash"
    REPORT_PATH.write_text(text)

    # Files for the Streamlit app, which only reads saved outputs.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[["AQI", "AQI_Bucket"]].to_csv(OUT_DIR / "phase3_daily_aqi.csv", date_format="%Y-%m-%d")
    findings = [l.split(". ", 1)[1] for l in lines[lines.index("## 9. Findings in plain language"):]
                if l[:1].isdigit() and ". " in l]
    (OUT_DIR / "phase3_findings.json").write_text(json.dumps(
        dict(findings=findings, start=str(df.index.min().date()), end=str(df.index.max().date()),
             n_days=len(df), bad_air_threshold=BAD_AIR_THRESHOLD), indent=2))
    print(f"\nSaved report to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
