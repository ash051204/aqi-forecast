"""Phase 4: forecasting Delhi's weekly mean AQI.

Plan:
  * Weekly mean AQI (weeks end on Sunday; partial weeks dropped).
  * Train = weeks ending 2015-2018, test = weeks ending 2019, 2020 held back.
  * Stationarity checks on the training weeks only (ADF, ACF, PACF).
  * Four models: naive, seasonal naive, Holt-Winters, SARIMA.
  * Two ways of testing each model on 2019:
      (a) full-year: fit on 2015-2018, forecast all 52 weeks of 2019 at once.
      (b) rolling one-week-ahead: forecast each 2019 week using real data up
          to the week before, with the model's parameters kept fixed.
  * Lockdown case study on 2020 with the best full-year model.

Run from the project root:
    .venv/bin/python src/forecast.py
"""

import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "delhi_daily.csv"
REPORT_PATH = ROOT / "reports" / "phase4_forecast.md"
RESULTS_PATH = ROOT / "reports" / "results.md"
FIG_DIR = ROOT / "reports" / "figures"
OUT_DIR = ROOT / "reports" / "outputs"

SEASON = 52
LOCKDOWN = pd.Timestamp("2020-03-25")

# SARIMA search space. d and D are fixed by the stationarity checks (AIC
# cannot compare models with different differencing, because differencing
# changes the data the likelihood is computed on). Only p, q, P, Q are searched.
GRID_P = [0, 1, 2]
GRID_Q = [0, 1, 2]
GRID_SP = [0, 1]
GRID_SQ = [0, 1]

MODEL_NAMES = ["Naive", "Seasonal naive", "Holt-Winters", "SARIMA"]


def rmse(actual, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def build_weekly(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Weekly mean AQI with weeks ending on Sunday. Returns (weekly, day counts)."""
    groups = daily.resample("W-SUN")
    return groups.mean(), groups.count()


# ---------------------------------------------------------------------
# Holt-Winters helpers
# ---------------------------------------------------------------------
HW_VARIANTS = {
    "HW seasonal only": dict(trend=None, damped_trend=False),
    "HW additive trend": dict(trend="add", damped_trend=False),
    "HW damped additive trend": dict(trend="add", damped_trend=True),
}


def fit_hw(y: pd.Series, spec: dict):
    """Fit Holt-Winters and report whether the optimiser converged."""
    model = ExponentialSmoothing(y, seasonal="add", seasonal_periods=SEASON, **spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = model.fit()
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return res, converged


def hw_rolling(fitted, y_all: pd.Series, spec: dict, test_index) -> pd.Series:
    """One-week-ahead Holt-Winters forecasts with parameters FIXED.

    We rebuild the same model on train + test data and lock every parameter
    (smoothing weights, damping, starting level/trend/season) to the values
    learnt on the training data. Running it over the longer series then
    updates the level and season week by week using real data, without
    re-estimating anything. Its fitted value for a 2019 week is exactly the
    forecast made at the end of the previous week.
    """
    p = fitted.params
    fixed = {"smoothing_level": p["smoothing_level"], "smoothing_seasonal": p["smoothing_seasonal"],
             "initial_level": p["initial_level"]}
    if spec["trend"]:
        fixed["smoothing_trend"] = p["smoothing_trend"]
        fixed["initial_trend"] = p["initial_trend"]
    if spec["damped_trend"]:
        fixed["damping_trend"] = p["damping_trend"]
    for i, s in enumerate(p["initial_seasons"]):
        fixed[f"initial_seasonal.{i}"] = s
    model = ExponentialSmoothing(y_all, seasonal="add", seasonal_periods=SEASON, **spec)
    # With every parameter fixed statsmodels warns that there is nothing to
    # estimate; that is exactly what we want here, so the warning is silenced.
    with model.fix_params(fixed), warnings.catch_warnings():
        warnings.simplefilter("ignore", EstimationWarning)
        rerun = model.fit()
    # Safety check: on the training weeks the rerun must match the original
    # fit exactly, otherwise the parameters were not really held fixed.
    n_train = len(fitted.fittedvalues)
    assert np.allclose(rerun.fittedvalues.iloc[:n_train], fitted.fittedvalues), "HW parameters were not fixed"
    return rerun.fittedvalues.loc[test_index]


def fit_sarima(y: pd.Series, order, sorder):
    """Fit SARIMA and report whether the optimiser converged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = SARIMAX(y, order=order, seasonal_order=sorder).fit(disp=False)
    conv_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged = bool(res.mle_retvals.get("converged", True)) and not conv_warning
    return res, converged


def main() -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")["AQI"]

    out("# Phase 4: Forecasting weekly mean AQI (Delhi)")
    out()

    # =================================================================
    # 1. Weekly resampling
    # =================================================================
    weekly_all, counts = build_weekly(daily)
    partial = counts[counts < 7]
    # A week with only 3 or 4 days is not comparable with a full week (its
    # mean covers a different mix of weekdays), so partial weeks are dropped
    # rather than kept as if they were full weeks.
    weekly = weekly_all[counts == 7].asfreq("W-SUN")
    assert weekly.notna().all(), "gap inside the weekly series"

    out("## 1. Weekly resampling")
    out()
    out("- Each week runs Monday to Sunday and is labelled by its **Sunday** (pandas rule `W-SUN`). "
        "The weekly value is the mean of the 7 daily AQI values.")
    out(f"- Partial weeks: {len(partial)} were found and dropped: "
        + "; ".join(f"week ending {d.date()} has {n} days" for d, n in partial.items())
        + ". The data starts on a Thursday (2015-01-01) and ends on a Wednesday (2020-07-01).")
    out(f"- Result: {len(weekly)} full weeks, ending {weekly.index.min().date()} to {weekly.index.max().date()}.")
    per_year = weekly.groupby(weekly.index.year).size()
    out("- Weeks per year (by the Sunday the week ends on): "
        + ", ".join(f"{y}: {n}" for y, n in per_year.items()) + ".")
    drift_days = 4 * 365.25 - 4 * SEASON * 7
    out(f"- **The 53rd week.** A year has 52 weeks plus 1 or 2 days, so every 5 or 6 years a calendar year contains "
        f"53 Sundays. Here that is 2017 ({per_year.get(2017)} weeks). The models use a fixed 52-week season, so "
        "\"the same week last year\" means exactly 364 days earlier. That keeps the seasonal pattern slowly sliding "
        f"against the calendar, by about 1.25 days per year, or about {drift_days:.0f} days over the 4 training years. "
        "This is less than one week, while Delhi's high-pollution season lasts about 8 to 10 weeks, so the "
        "mismatch is small. We do not delete or merge the 53rd week, because that would break the equal spacing "
        "that time series models need.")
    out()

    train = weekly[weekly.index.year <= 2018]
    test = weekly[weekly.index.year == 2019]
    hold = weekly[weekly.index.year == 2020]
    y_to_2019 = weekly[weekly.index.year <= 2019]
    out(f"Split by time: train = weeks ending in 2015-2018 ({len(train)} weeks, {train.index.min().date()} to "
        f"{train.index.max().date()}); test = weeks ending in 2019 ({len(test)} weeks, {test.index.min().date()} to "
        f"{test.index.max().date()}); held back = weeks ending in 2020 ({len(hold)} weeks, {hold.index.min().date()} "
        f"to {hold.index.max().date()}).")
    out(f"Note: the first test week (ending {test.index[0].date()}) includes 2018-12-31 because it is a Monday of that week.")
    out()

    # =================================================================
    # 2. Stationarity (training weeks only)
    # =================================================================
    out("## 2. Stationarity (training weeks only)")
    out()
    out("ADF test: the null hypothesis is that the series has a unit root (is non-stationary). "
        "A p-value below 0.05 means we can treat the series as stationary.")
    out()
    out("| Series | Weeks | ADF statistic | p-value | Stationary at 5%? |")
    out("|---|---|---|---|---|")
    transforms = {
        "Level (no differencing)": train,
        "First difference (d=1)": train.diff().dropna(),
        "Seasonal difference (D=1, lag 52)": train.diff(SEASON).dropna(),
        "Seasonal + first difference (D=1, d=1)": train.diff(SEASON).diff().dropna(),
    }
    adf = {}
    for name, s in transforms.items():
        stat, p = adfuller(s, autolag="AIC", result_object=False)[:2]
        adf[name] = p
        out(f"| {name} | {len(s)} | {stat:.3f} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    out()
    lag52 = train.autocorr(SEASON)
    out(f"Autocorrelation of the training weeks at lag 1: {train.autocorr(1):.3f}; at lag 52: {lag52:.3f}.")
    out()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(train, lags=60, ax=axes[0, 0], title="ACF, weekly AQI level (train 2015-2018)")
    plot_pacf(train, lags=60, ax=axes[0, 1], method="ywm", title="PACF, weekly AQI level (train 2015-2018)")
    sd = train.diff(SEASON).diff().dropna()
    plot_acf(sd, lags=60, ax=axes[1, 0], title="ACF after seasonal (52) and first differencing")
    plot_pacf(sd, lags=60, ax=axes[1, 1], method="ywm", title="PACF after seasonal (52) and first differencing")
    for ax in axes.flat:
        ax.set_xlabel("Lag (weeks)")
        ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase4_acf_pacf.png", dpi=150)
    plt.close(fig)

    out("Figure: `phase4_acf_pacf.png`.")
    out()
    out("**Differencing decision: d = 1, D = 1.**")
    out()
    out(f"- The level series passes the ADF test (p = {adf['Level (no differencing)']:.4f}), but the ADF test looks for "
        f"a random-walk type of drift, not a yearly cycle. The ACF shows a strong repeat at lag 52 ({lag52:.3f}) and "
        "Phase 3 found a seasonal strength of 0.83, so the yearly pattern has to be removed: **D = 1**.")
    out(f"- After seasonal differencing the ADF test no longer rejects a unit root "
        f"(p = {adf['Seasonal difference (D=1, lag 52)']:.4f}). In plain words: once the yearly cycle is taken out, "
        "what is left is the slow year-on-year decline, which wanders rather than returning to a fixed mean. "
        f"Adding a first difference fixes this (p = {adf['Seasonal + first difference (D=1, d=1)']:.4f}): **d = 1**.")
    out(f"- Cost: seasonal plus first differencing uses up the first {SEASON + 1} training weeks, leaving "
        f"{len(sd)} weeks to estimate the ARMA part. This is why the grid is kept small.")
    out()

    # =================================================================
    # 3. Models, full-year horizon (fit on 2015-2018, forecast all of 2019)
    # =================================================================
    h = len(test)
    full = pd.DataFrame(index=test.index)
    full["Actual"] = test
    # Naive: repeat the last 2018 week for every 2019 week.
    full["Naive"] = train.iloc[-1]
    # Seasonal naive: each 2019 week gets the value from 52 weeks earlier (2018).
    full["Seasonal naive"] = train.iloc[-SEASON:].to_numpy()

    roll = pd.DataFrame(index=test.index)
    roll["Actual"] = test
    # Rolling naive: last week's real value.
    roll["Naive"] = y_to_2019.shift(1).loc[test.index]
    # Rolling seasonal naive: 52 weeks back always lands in 2018, which is
    # already known, so it is the same forecast as the full-year version.
    roll["Seasonal naive"] = y_to_2019.shift(SEASON).loc[test.index]

    # --- Holt-Winters: three variants ---
    hw_rows = []
    hw_fits = {}
    for name, spec in HW_VARIANTS.items():
        t0 = time.perf_counter()
        fitted, conv = fit_hw(train, spec)
        secs = time.perf_counter() - t0
        hw_fits[name] = fitted
        fc_full = fitted.forecast(h)
        fc_roll = hw_rolling(fitted, y_to_2019, spec, test.index)
        hw_rows.append(dict(name=name, aic=fitted.aic, secs=secs, converged=conv,
                            full_rmse=rmse(test, fc_full), full_mae=mae(test, fc_full),
                            roll_rmse=rmse(test, fc_roll), roll_mae=mae(test, fc_roll),
                            fc_full=fc_full, fc_roll=fc_roll, params=fitted.params))
    # Pick the variant by AIC on the TRAINING data, so that the 2019 test
    # weeks are not used to make choices. Test scores are shown alongside
    # so the reader can see whether AIC and test performance agree.
    # Only converged fits are eligible, for the same reason as SARIMA below.
    hw_best = min((r for r in hw_rows if r["converged"]), key=lambda r: r["aic"])
    full["Holt-Winters"] = hw_best["fc_full"].to_numpy()
    roll["Holt-Winters"] = hw_best["fc_roll"].to_numpy()

    out("## 3. Holt-Winters variants (additive seasonality, 52-week season)")
    out()
    out("| Variant | AIC (train) | Optimiser converged | Full-year RMSE | Full-year MAE | Rolling RMSE | Rolling MAE | Fit time (s) |")
    out("|---|---|---|---|---|---|---|---|")
    for r in hw_rows:
        out(f"| {r['name']} | {r['aic']:.1f} | {'yes' if r['converged'] else 'no (warning)'} | {r['full_rmse']:.1f} | "
            f"{r['full_mae']:.1f} | {r['roll_rmse']:.1f} | {r['roll_mae']:.1f} | {r['secs']:.2f} |")
    out()
    best_full_hw = min(hw_rows, key=lambda r: r["full_rmse"])
    best_roll_hw = min(hw_rows, key=lambda r: r["roll_rmse"])
    out(f"- Chosen for the main table: **{hw_best['name']}** (lowest AIC on the training weeks). "
        "The choice is made on training data so the 2019 test weeks stay unseen.")
    out(f"- On the 2019 test weeks the lowest full-year RMSE is {best_full_hw['name']} and the lowest rolling RMSE is "
        f"{best_roll_hw['name']}"
        + (", so AIC and test performance agree." if best_full_hw['name'] == hw_best['name'] == best_roll_hw['name']
           else ". AIC and test performance do not fully agree; this is reported, not used to change the choice."))
    p = hw_best["params"]
    desc = f"alpha (level) = {p['smoothing_level']:.3f}, gamma (season) = {p['smoothing_seasonal']:.3f}"
    if HW_VARIANTS[hw_best["name"]]["trend"]:
        desc += f", beta (trend) = {p['smoothing_trend']:.3f}"
    if HW_VARIANTS[hw_best["name"]]["damped_trend"]:
        desc += f", phi (damping) = {p['damping_trend']:.3f}"
    out(f"- Fitted smoothing parameters of the chosen variant: {desc}.")
    out()

    # --- SARIMA grid search ---
    grid_rows = []
    t_grid = time.perf_counter()
    for p_, q_, P_, Q_ in product(GRID_P, GRID_Q, GRID_SP, GRID_SQ):
        order, sorder = (p_, 1, q_), (P_, 1, Q_, SEASON)
        t0 = time.perf_counter()
        try:
            res, conv = fit_sarima(train, order, sorder)
            grid_rows.append(dict(order=order, sorder=sorder, aic=res.aic, converged=conv,
                                  secs=time.perf_counter() - t0))
        except Exception as e:  # record failures instead of hiding them
            grid_rows.append(dict(order=order, sorder=sorder, aic=np.nan, converged=False,
                                  secs=time.perf_counter() - t0, error=str(e)))
    grid_secs = time.perf_counter() - t_grid
    grid = pd.DataFrame(grid_rows)
    # Only models whose optimiser converged are eligible; an unconverged
    # fit's AIC is not trustworthy.
    eligible = grid[grid["converged"] & grid["aic"].notna()].sort_values("aic")
    best = eligible.iloc[0]
    t0 = time.perf_counter()
    sarima, _ = fit_sarima(train, best["order"], best["sorder"])
    sarima_secs = time.perf_counter() - t0
    full["SARIMA"] = sarima.forecast(h).to_numpy()
    # Rolling: append 2019 to the model WITHOUT refitting; the Kalman filter
    # then produces each week's forecast from data up to the week before.
    appended = sarima.append(test, refit=False)
    assert np.allclose(appended.params, sarima.params)
    roll["SARIMA"] = appended.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False).predicted_mean.to_numpy()

    out("## 4. SARIMA grid search (d = 1, D = 1, season = 52)")
    out()
    out(f"Grid: p in {GRID_P}, q in {GRID_Q}, P in {GRID_SP}, Q in {GRID_SQ}, giving {len(grid)} models. "
        f"Fitted on the {len(train)} training weeks, ranked by AIC. Total grid time: {grid_secs:.1f} s.")
    out(f"Models that failed to converge (excluded): {int((~grid['converged']).sum())}.")
    out()
    out("Top 10 by AIC:")
    out()
    out("| Rank | (p,d,q) | (P,D,Q,s) | AIC | Converged | Fit time (s) |")
    out("|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(grid.sort_values("aic").head(10).iterrows(), 1):
        out(f"| {i} | {r['order']} | {r['sorder']} | {r['aic']:.1f} | {'yes' if r['converged'] else 'no'} | {r['secs']:.1f} |")
    out()
    out(f"**Chosen: SARIMA{best['order']}x{best['sorder']}, AIC = {best['aic']:.1f}.** "
        f"Refit time for the chosen model: {sarima_secs:.1f} s.")
    out()
    out("Estimated coefficients:")
    out()
    out("| Parameter | Estimate | Std error | p-value |")
    out("|---|---|---|---|")
    for name in sarima.params.index:
        out(f"| {name} | {sarima.params[name]:.4f} | {sarima.bse[name]:.4f} | {sarima.pvalues[name]:.4f} |")
    out()
    grid.assign(order=grid["order"].astype(str), sorder=grid["sorder"].astype(str)).to_csv(
        OUT_DIR / "phase4_sarima_grid.csv", index=False)

    # =================================================================
    # 5. Comparison table
    # =================================================================
    metrics = []
    for m in MODEL_NAMES:
        metrics.append(dict(Model=m, full_rmse=rmse(test, full[m]), full_mae=mae(test, full[m]),
                            roll_rmse=rmse(test, roll[m]), roll_mae=mae(test, roll[m])))
    metrics = pd.DataFrame(metrics)
    labels = {"Holt-Winters": f"Holt-Winters ({hw_best['name'].removeprefix('HW ')})",
              "SARIMA": f"SARIMA{best['order']}x{best['sorder']}"}

    table = ["| Model | Full-year RMSE | Full-year MAE | Rolling 1-week RMSE | Rolling 1-week MAE |",
             "|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        table.append(f"| {labels.get(r['Model'], r['Model'])} | {r['full_rmse']:.1f} | {r['full_mae']:.1f} | "
                     f"{r['roll_rmse']:.1f} | {r['roll_mae']:.1f} |")

    out("## 5. Model comparison on the 52 test weeks of 2019")
    out()
    out("Full-year = fitted on 2015-2018 and forecast all 2019 weeks in one go (up to 52 weeks ahead). "
        "Rolling 1-week = each 2019 week forecast from real data up to the previous week, parameters kept fixed "
        "(no refitting). Units: AQI points.")
    out()
    for row in table:
        out(row)
    out()
    best_full = metrics.loc[metrics["full_rmse"].idxmin(), "Model"]
    best_roll = metrics.loc[metrics["roll_rmse"].idxmin(), "Model"]
    out(f"Lowest full-year RMSE: **{labels.get(best_full, best_full)}**. Lowest rolling RMSE: **{labels.get(best_roll, best_roll)}**.")
    out("The seasonal naive scores are identical at both horizons by construction: its forecast for any 2019 week is "
        "the value 52 weeks earlier, which is always a 2018 week already known at the end of 2018, so knowing the "
        "2019 weeks in between changes nothing.")
    out()
    metrics.to_csv(OUT_DIR / "phase4_metrics.csv", index=False)
    full.to_csv(OUT_DIR / "phase4_forecasts_full_year_2019.csv", date_format="%Y-%m-%d")
    roll.to_csv(OUT_DIR / "phase4_forecasts_rolling_2019.csv", date_format="%Y-%m-%d")

    colors = {"Naive": "tab:grey", "Seasonal naive": "tab:green", "Holt-Winters": "tab:orange", "SARIMA": "tab:red"}
    for df_, title, fname in [
        (full, "2019 weekly AQI: full-year forecasts made at the end of 2018", "phase4_forecasts_full_year.png"),
        (roll, "2019 weekly AQI: rolling one-week-ahead forecasts (parameters fixed)", "phase4_forecasts_rolling.png"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(train.index[-26:], train.iloc[-26:], color="black", linewidth=1, alpha=0.5, label="Actual (end of training)")
        ax.plot(df_.index, df_["Actual"], color="black", linewidth=2, label="Actual 2019")
        for m in MODEL_NAMES:
            ax.plot(df_.index, df_[m], color=colors[m], linewidth=1.3, linestyle="--", label=labels.get(m, m))
        ax.axvline(test.index[0] - pd.Timedelta(days=3.5), color="grey", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Week ending (Sunday)")
        ax.set_ylabel("Weekly mean AQI")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)
    out("Figures: `phase4_forecasts_full_year.png`, `phase4_forecasts_rolling.png`.")
    out()

    # =================================================================
    # 6. SARIMA residual diagnostics
    # =================================================================
    # The first d + D*52 = 53 residuals come from the start-up of the
    # differencing and are not real one-step errors, so they are skipped.
    burn = 1 + SEASON
    resid = sarima.resid.iloc[burn:]
    lb = acorr_ljungbox(resid, lags=[10, 20, 52], return_df=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(resid.index, resid, color="tab:blue", linewidth=1)
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_title("SARIMA residuals over time (train)")
    axes[0].set_xlabel("Week ending")
    axes[0].set_ylabel("Residual (AQI)")
    plot_acf(resid, lags=52, ax=axes[1], title="ACF of SARIMA residuals")
    axes[1].set_xlabel("Lag (weeks)")
    axes[1].set_ylabel("Correlation")
    axes[2].hist(resid, bins=25, color="tab:blue", edgecolor="white")
    axes[2].set_title("Distribution of SARIMA residuals")
    axes[2].set_xlabel("Residual (AQI)")
    axes[2].set_ylabel("Number of weeks")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase4_sarima_residuals.png", dpi=150)
    plt.close(fig)

    out("## 6. SARIMA residual diagnostics (training weeks)")
    out()
    out(f"Residuals used: {len(resid)} (the first {burn} are skipped because the differencing needs {burn} weeks to start). "
        f"Mean {resid.mean():.2f}, std {resid.std():.2f}.")
    out()
    out("Ljung-Box test: the null hypothesis is that the residuals have no autocorrelation (they are white noise). "
        "A p-value above 0.05 means no evidence of leftover pattern.")
    out()
    out("| Lags tested | Q statistic | p-value | Leftover autocorrelation at 5%? |")
    out("|---|---|---|---|")
    for lag, r in lb.iterrows():
        out(f"| {lag} | {r['lb_stat']:.2f} | {r['lb_pvalue']:.4f} | {'yes' if r['lb_pvalue'] < 0.05 else 'no'} |")
    out()
    out("Figure: `phase4_sarima_residuals.png`.")
    out()

    # =================================================================
    # 7. Lockdown case study
    # =================================================================
    # The best model is chosen by FULL-YEAR RMSE, because the lockdown test
    # is also a many-weeks-ahead forecast (Jan to Jun 2020 made at the end of
    # 2019). 2019 was used to pick the model; 2020 has not been touched yet.
    lock_model = best_full
    h2 = len(hold)
    t0 = time.perf_counter()
    lower = upper = None
    if lock_model == "Naive":
        fc = pd.Series(y_to_2019.iloc[-1], index=hold.index)
    elif lock_model == "Seasonal naive":
        fc = pd.Series(y_to_2019.iloc[-SEASON:].iloc[:h2].to_numpy(), index=hold.index)
    elif lock_model == "Holt-Winters":
        fc = fit_hw(y_to_2019, HW_VARIANTS[hw_best["name"]])[0].forecast(h2)
    else:
        refit, _ = fit_sarima(y_to_2019, best["order"], best["sorder"])
        pred = refit.get_forecast(h2)
        fc = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)
        lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    lock_secs = time.perf_counter() - t0

    lock = pd.DataFrame({"Actual": hold, "Forecast": fc.to_numpy()}, index=hold.index)
    lock["Error"] = lock["Actual"] - lock["Forecast"]  # negative = cleaner than forecast
    lock["Pct error"] = 100 * lock["Error"] / lock["Forecast"]
    week_start = lock.index - pd.Timedelta(days=6)
    # Pre-lockdown weeks end on or before 22 March. The week of 23 to 29 March
    # contains 2 days before and 5 days after the lockdown began, so it is
    # left out of both groups instead of being forced into one.
    pre_mask = lock.index < LOCKDOWN
    post_mask = week_start >= LOCKDOWN
    straddle = lock.index[~pre_mask & ~post_mask]
    lock["Period"] = np.where(pre_mask, "pre", np.where(post_mask, "post", "straddle"))
    lock.to_csv(OUT_DIR / "phase4_lockdown_2020.csv", date_format="%Y-%m-%d")

    def summary(mask):
        s = lock[mask]
        return dict(n=len(s), me=s["Error"].mean(), mae=s["Error"].abs().mean(),
                    rmse=float(np.sqrt((s["Error"] ** 2).mean())), mpe=s["Pct error"].mean(),
                    actual=s["Actual"].mean(), forecast=s["Forecast"].mean())

    pre, post = summary(pre_mask), summary(post_mask)
    effect = post["me"] - pre["me"]
    effect_pct = post["mpe"] - pre["mpe"]

    # Fairness check: the same model, fitted on 2015-2018, forecasting 2019
    # (a year with no lockdown). Its signed error for the same calendar
    # weeks shows what "normal" long-horizon error looks like in spring.
    f19 = full[lock_model]
    err19 = test - f19
    pre19 = err19[(err19.index.month <= 3) & (err19.index < "2019-03-25")]
    post19 = err19[(err19.index - pd.Timedelta(days=6) >= "2019-03-25") & (err19.index <= "2019-06-30")]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(y_to_2019.index[-26:], y_to_2019.iloc[-26:], color="black", linewidth=1, alpha=0.5, label="Actual (late 2019)")
    ax.plot(lock.index, lock["Actual"], color="black", linewidth=2, marker="o", markersize=3, label="Actual 2020")
    ax.plot(lock.index, lock["Forecast"], color="tab:red", linewidth=1.8, linestyle="--",
            label=f"Forecast from end of 2019: {labels.get(lock_model, lock_model)}")
    if lower is not None:
        ax.fill_between(lock.index, lower, upper, color="tab:red", alpha=0.12, label="95% prediction interval")
    ax.axvline(LOCKDOWN, color="tab:blue", linewidth=1.5, label="Lockdown start, 25 March 2020")
    ax.axvspan(LOCKDOWN, lock.index.max(), color="tab:blue", alpha=0.06)
    ax.set_title("Delhi weekly AQI in 2020: forecast vs actual around the 25 March lockdown")
    ax.set_xlabel("Week ending (Sunday)")
    ax.set_ylabel("Weekly mean AQI")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase4_lockdown_case_study.png", dpi=150)
    plt.close(fig)

    out("## 7. Lockdown case study (January to June 2020)")
    out()
    if lock_model in ("Naive", "Seasonal naive"):
        how = (f"This model has no parameters to estimate, so \"refitting\" on 2015-2019 simply means the forecast "
               f"is built from the data up to the end of 2019"
               + (": each 2020 week gets the value of the same week of 2019 (52 weeks earlier)." if lock_model == "Seasonal naive"
                  else ": every 2020 week gets the last 2019 week's value."))
    else:
        how = (f"Refit on all {len(y_to_2019)} weeks ending 2015-2019 (same specification, parameters re-estimated) "
               f"in {lock_secs:.1f} s.")
    out(f"Model: **{labels.get(lock_model, lock_model)}**, the best full-year model on 2019 (lowest full-year RMSE). "
        f"{how} Forecast: {h2} weeks ending {hold.index.min().date()} to {hold.index.max().date()}.")
    out()
    out(f"- Pre-lockdown weeks: ending {lock.index[pre_mask].min().date()} to {lock.index[pre_mask].max().date()} "
        f"({pre['n']} weeks; the first one includes 30 and 31 December 2019).")
    out(f"- Post-lockdown weeks: ending {lock.index[post_mask].min().date()} to {lock.index[post_mask].max().date()} ({post['n']} weeks).")
    out(f"- Left out: week ending {', '.join(str(d.date()) for d in straddle)} (23 to 29 March), which mixes 2 days before "
        "and 5 days after the start of the lockdown.")
    out()
    out("Error = actual minus forecast. Negative means the air was cleaner than the model expected.")
    out()
    out("| Period | Weeks | Mean actual | Mean forecast | Mean error | MAE | RMSE | Mean % error |")
    out("|---|---|---|---|---|---|---|---|")
    for name, s in [("Pre-lockdown (1 Jan to 22 Mar)", pre), ("Post-lockdown (from 30 Mar)", post)]:
        out(f"| {name} | {s['n']} | {s['actual']:.1f} | {s['forecast']:.1f} | {s['me']:+.1f} | {s['mae']:.1f} | "
            f"{s['rmse']:.1f} | {s['mpe']:+.1f}% |")
    out()
    out(f"**Estimated lockdown effect = post-lockdown mean error minus pre-lockdown mean error = "
        f"{post['me']:+.1f} - ({pre['me']:+.1f}) = {effect:+.1f} AQI points** "
        f"(in percentage terms {post['mpe']:+.1f}% - ({pre['mpe']:+.1f}%) = {effect_pct:+.1f} percentage points).")
    out()
    out(f"Context from 2019, a year without a lockdown: the same model, forecasting 2019 from the end of 2018, had a "
        f"mean error of {pre19.mean():+.1f} for weeks ending up to 24 Mar 2019 ({len(pre19)} weeks) and "
        f"{post19.mean():+.1f} for weeks starting 25 Mar to the end of June 2019 ({len(post19)} weeks), a post-minus-pre "
        f"difference of {post19.mean() - pre19.mean():+.1f}. So in a normal year the same calculation gives "
        f"{post19.mean() - pre19.mean():+.1f}, against {effect:+.1f} in 2020.")
    out()
    out("**How to read this, in plain language:**")
    out()
    yearly = daily[daily.index.year <= 2019].groupby(daily.index.year[daily.index.year <= 2019]).mean()
    out(f"- The raw post-lockdown gap overstates the lockdown effect. Delhi's air was already getting cleaner "
        f"year on year (mean daily AQI {yearly[2015]:.0f} in 2015, {yearly[2019]:.0f} in 2019), and early 2020 was already "
        "cleaner than the model expected before any lockdown. Subtracting the pre-lockdown error removes that "
        "\"already improving\" part, so what remains is the extra drop that lines up in time with the lockdown.")
    steps = pd.Series(np.arange(1, h2 + 1), index=lock.index)
    out(f"- The pre-lockdown weeks are {steps[pre_mask].min()} to {steps[pre_mask].max()} weeks ahead of the forecast "
        f"origin, while the post-lockdown weeks are {steps[post_mask].min()} to {steps[post_mask].max()} weeks ahead. "
        + ("For the seasonal naive model this matters less, because each forecast is just the same week of the year "
           "before and does not get less reliable further out; it does mean the comparison relies on 2019 being a "
           "typical year." if lock_model == "Seasonal naive" else
           "Forecast errors usually grow with distance, so the adjustment is not perfect."))
    out("- **This is an association, not proof of cause.** Weather (rain, wind, temperature) also changes from year "
        "to year and affects AQI, and this model has no weather data. The estimate says how much cleaner the air "
        "was than expected after 25 March; it cannot prove the lockdown alone caused all of it.")
    out()
    out("Figure: `phase4_lockdown_case_study.png`.")
    out()

    text = "\n".join(lines) + "\n"
    assert "—" not in text and "–" not in text, "report contains an em or en dash"
    REPORT_PATH.write_text(text)

    # Metrics tables for reports/results.md. Each phase owns one marked
    # section so reruns replace it instead of appending duplicates.
    section = ["<!-- phase4:start -->", "## Phase 4: Forecasting weekly mean AQI (test = 52 weeks of 2019)", ""]
    section += table
    section += ["", f"Lockdown case study ({labels.get(lock_model, lock_model)}, refit on 2015-2019):", "",
                "| Period | Weeks | Mean error (actual - forecast) | MAE |", "|---|---|---|---|",
                f"| Pre-lockdown | {pre['n']} | {pre['me']:+.1f} | {pre['mae']:.1f} |",
                f"| Post-lockdown | {post['n']} | {post['me']:+.1f} | {post['mae']:.1f} |",
                "", f"Estimated lockdown effect (post minus pre mean error): {effect:+.1f} AQI points.",
                "<!-- phase4:end -->"]
    block = "\n".join(section)
    existing = RESULTS_PATH.read_text() if RESULTS_PATH.exists() else "# Results\n\nAll numbers are produced by the scripts in `src/`.\n"
    if "<!-- phase4:start -->" in existing:
        head, rest = existing.split("<!-- phase4:start -->", 1)
        existing = head + block + rest.split("<!-- phase4:end -->", 1)[1]
    else:
        existing = existing.rstrip() + "\n\n" + block + "\n"
    RESULTS_PATH.write_text(existing)
    print(f"\nSaved {REPORT_PATH.relative_to(ROOT)} and {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
