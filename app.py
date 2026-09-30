"""Streamlit app: Delhi AQI forecasting and bad-air-day prediction.

The app only READS files saved by the phase scripts:
  reports/outputs/*.csv, reports/outputs/*.json, models/logreg_pipeline.joblib
  and models/logreg_pipeline_meta.json. Nothing is trained here, and every
number shown comes from those files (or is calculated from them on the page).

Run from the project root:
    .venv/bin/streamlit run app.py
"""

import json
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.classify import calendar_features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "outputs"
MODELS = ROOT / "models"

# Chart colours: categorical slots checked with a colour-blindness validator.
# Actual values are near-black, the naive baseline is a muted grey reference
# line, and the three real models take the first three categorical slots.
INK = "#0b0b0b"
MUTED = "#8a8984"
COLORS = {"Seasonal naive": "#2a78d6", "Holt-Winters": "#eb6834", "SARIMA": "#199e70", "Naive": MUTED}
AXIS = dict(labelFontSize=12, titleFontSize=13, gridColor="#e6e6e3", domainColor="#8a8984", tickColor="#8a8984")

# Smoothing window for the overview chart (a display choice, not a result).
ROLL_DAYS = 30

PAGES = ["Overview", "Forecast", "Tomorrow's risk", "Lockdown case study"]


# ---------------------------------------------------------------------
# Loading saved outputs (cached so pages switch quickly)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(name: str, date_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(OUT / name)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELS / "logreg_pipeline.joblib")


def source_line() -> str:
    """Dataset citation, read from data/SOURCE.md."""
    text = (ROOT / "data" / "SOURCE.md").read_text()
    # The design rules forbid em and en dashes, so ranges like "2015-2020"
    # written with a dash in the source file are shown with "to".
    text = re.sub(r"\s*[\u2013\u2014]\s*", " to ", text)
    fields = dict(re.findall(r"- \*\*(.+?):\*\* (.+)", text))
    return (f"Data: {fields.get('Dataset', '')}, {fields.get('Author', '')}. "
            f"Original source: {fields.get('Original source', '')}. License: {fields.get('License', '').split(' (')[0]}.")


def style(chart: alt.Chart) -> alt.Chart:
    """Shared chart styling so every chart looks the same."""
    return (chart.configure_axis(**AXIS)
            .configure_legend(labelFontSize=12, titleFontSize=12, orient="top", direction="horizontal", labelLimit=0)
            .configure_view(strokeWidth=0))


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}"


def feature_label(name: str) -> str:
    """Readable names for model features, derived from the feature names."""
    if name == "aqi_today":
        return "AQI today"
    m = re.fullmatch(r"aqi_lag(\d+)", name)
    if m:
        n = int(m.group(1))
        return f"AQI {n} day{'s' if n > 1 else ''} before today"
    m = re.fullmatch(r"aqi_roll(\d+)_mean", name)
    if m:
        return f"Mean AQI over the {m.group(1)} days ending today"
    return name


# ---------------------------------------------------------------------
# Page 1: overview
# ---------------------------------------------------------------------
def page_overview() -> None:
    info = load_json(OUT / "phase3_findings.json")
    meta = load_json(MODELS / "logreg_pipeline_meta.json")
    daily = load_csv("phase3_daily_aqi.csv", "Date")
    thr = meta["bad_day_aqi_threshold"]

    st.title("Delhi air quality")
    lock_year = pd.Timestamp(load_json(OUT / "phase4_summary.json")["lockdown_date"]).year
    st.write(f"Daily AQI for Delhi from {info['start']} to {info['end']} ({info['n_days']:,} days). "
             "This app shows the exploratory findings, weekly AQI forecasts, a next-day bad-air-day model, and "
             f"a case study of the {lock_year} lockdown.")

    roll_name = f"{ROLL_DAYS}-day mean"
    daily[roll_name] = daily["AQI"].rolling(ROLL_DAYS, center=True).mean()
    long = daily.melt(id_vars=["Date", "AQI_Bucket"], value_vars=["AQI", roll_name],
                      var_name="Series", value_name="Value").dropna(subset=["Value"])
    long["Series"] = long["Series"].replace({"AQI": "Daily AQI"})
    color = alt.Scale(domain=["Daily AQI", roll_name], range=["#9ec5f4", INK])
    lines = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
        x=alt.X("Date:T", title="Date", axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("Value:Q", title="AQI (CPCB index)"),
        color=alt.Color("Series:N", scale=color, title=None),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".0f"),
                 alt.Tooltip("AQI_Bucket:N", title="Category")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(color="#e34948", strokeDash=[4, 4]).encode(y="y:Q")
    st.subheader("Daily AQI")
    st.altair_chart(style((lines + rule).properties(height=340)), width="stretch")
    st.caption(f"Dashed red line: AQI {thr}. Days above it are Very Poor or Severe (the bad air days predicted "
               "on the Tomorrow's risk page).")
    with st.expander("Data table"):
        st.dataframe(daily[["Date", "AQI", "AQI_Bucket"]].rename(columns={"AQI_Bucket": "Category"}),
                     hide_index=True, width="stretch")

    st.subheader("Key findings")
    for i, f in enumerate(info["findings"], 1):
        st.markdown(f"{i}. {f}")
    st.caption("From the exploratory analysis (reports/phase3_eda.md). " + source_line())


# ---------------------------------------------------------------------
# Page 2: forecast
# ---------------------------------------------------------------------
def forecast_chart(df: pd.DataFrame, labels: dict) -> alt.Chart:
    models = ["Naive", "Seasonal naive", "Holt-Winters", "SARIMA"]
    long = df.melt(id_vars="Date", value_vars=["Actual"] + models, var_name="Series", value_name="AQI")
    names = {"Actual": "Actual"} | {m: labels[m] for m in models}
    long["Series"] = long["Series"].map(names)
    domain = [names[k] for k in ["Actual"] + models]
    rng = [INK] + [COLORS[m] for m in models]
    base = alt.Chart(long).encode(
        x=alt.X("Date:T", title="Week ending (Sunday)"),
        y=alt.Y("AQI:Q", title="Weekly mean AQI"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), title=None, sort=domain),
        tooltip=[alt.Tooltip("Date:T", title="Week ending"), "Series:N", alt.Tooltip("AQI:Q", format=".1f")],
    )
    # The naive baseline is drawn dashed as a reference line; both layers
    # share one colour scale, so a single legend covers every series.
    naive = base.transform_filter(alt.datum.Series == names["Naive"]).mark_line(strokeWidth=1.5, strokeDash=[4, 3])
    others = base.transform_filter(alt.datum.Series != names["Naive"]).mark_line(strokeWidth=2)
    return (naive + others).properties(height=360)


def page_forecast() -> None:
    summ = load_json(OUT / "phase4_summary.json")
    labels = summ["labels"]
    metrics = load_csv("phase4_metrics.csv")
    full = load_csv("phase4_forecasts_full_year_2019.csv", "Date")
    roll = load_csv("phase4_forecasts_rolling_2019.csv", "Date")

    st.title("Weekly AQI forecast")
    st.write(f"Models were fitted on the weeks ending {summ['train_weeks'][0]} to {summ['train_weeks'][1]} "
             f"and tested on the {len(full)} weeks ending {summ['test_weeks'][0]} to {summ['test_weeks'][1]}.")

    horizon = st.radio("Forecast horizon", ["Full year", "Rolling one week ahead"], horizontal=True, key="horizon")
    if horizon == "Full year":
        st.caption("Each model forecasts every week of the test year at once, using only data up to the end of training.")
        st.altair_chart(style(forecast_chart(full, labels)), width="stretch")
    else:
        st.caption("Each week is forecast from real data up to the week before. Model parameters stay as fitted on "
                   "the training weeks; nothing is refitted.")
        st.altair_chart(style(forecast_chart(roll, labels)), width="stretch")

    st.subheader("Model comparison (AQI points, lower is better)")
    table = metrics.assign(Model=metrics["Model"].map(labels)).rename(columns={
        "full_rmse": "Full-year RMSE", "full_mae": "Full-year MAE",
        "roll_rmse": "Rolling RMSE", "roll_mae": "Rolling MAE"})
    st.dataframe(table.style.format(precision=1), hide_index=True, width="stretch")

    # Why seasonal naive won the full-year horizon, from the saved forecasts.
    actual_mean = full["Actual"].mean()
    means = {m: full[m].mean() for m in ["Seasonal naive", "Holt-Winters", "SARIMA"]}
    fitted_over = all(means[m] > actual_mean for m in ["Holt-Winters", "SARIMA"])
    st.subheader("Why seasonal naive wins the full-year horizon")
    st.write(
        f"Over the test weeks the actual weekly AQI averaged {fmt(actual_mean)}. "
        f"{labels['Holt-Winters']} and {labels['SARIMA']} forecast an average of {fmt(means['Holt-Winters'])} and "
        f"{fmt(means['SARIMA'])}"
        + (": both expected dirtier air than there was, because they estimate their yearly pattern from all the "
           "training years, including the earlier, more polluted ones. " if fitted_over else ". ")
        + f"Seasonal naive only copies the same week of the last training year (average {fmt(means['Seasonal naive'])}), "
        "so it starts from the most recent level instead of a long-run one. With a whole year to forecast and no "
        f"new data, that matters more than modelling skill. In the rolling test the models see last week's real "
        f"value and can correct their level, and there {labels[summ['best_rolling']]} has the lowest error."
    )
    with st.expander("Forecast data tables"):
        st.write("Full year")
        st.dataframe(full.style.format(precision=1), hide_index=True, width="stretch")
        st.write("Rolling one week ahead")
        st.dataframe(roll.style.format(precision=1), hide_index=True, width="stretch")


# ---------------------------------------------------------------------
# Page 3: tomorrow's risk
# ---------------------------------------------------------------------
def show_result(prob: float, meta: dict) -> None:
    thr = meta["decision_threshold"]
    verdict = "Bad air day" if prob >= thr else "Not a bad air day"
    c1, c2 = st.columns(2)
    c1.metric("Probability of a bad air day tomorrow", f"{prob:.1%}")
    c2.metric(f"Model decision for tomorrow (says bad at {thr:.0%} or more)", verdict)


def change_day_note(meta: dict) -> None:
    changes = load_csv("phase5_change_days.csv").set_index("Model")
    replay = load_csv("phase5_replay_2019.csv", "Date")
    lr = changes.loc["Logistic regression"]
    base = changes.loc["Baseline (persistence)"]
    n_all = int(replay["change_day"].sum())
    n_bad = int((replay["change_type"] == "turns bad").sum())
    n_good = int((replay["change_type"] == "turns better").sum())
    st.info(
        "This model reads today's air, not tomorrow's weather. It works best when tomorrow looks like today and "
        f"misses many sudden changes. In the test year there were {n_all} change days. The model got "
        f"{int(lr['all'])} of them right: {int(lr['bad'])} of the {n_bad} days that suddenly turned bad, and "
        f"{int(lr['good'])} of the {n_good} days that turned better. The \"tomorrow = today\" baseline gets "
        f"{int(base['all'])} of them right by definition."
    )


def page_risk() -> None:
    meta = load_json(MODELS / "logreg_pipeline_meta.json")
    model = load_model()
    replay = load_csv("phase5_replay_2019.csv", "Date").set_index("Date")
    features = meta["features"]

    st.title("Tomorrow's risk")
    st.write(f"Predicted probability that tomorrow's AQI will be above {meta['bad_day_aqi_threshold']} "
             f"(Very Poor or Severe). Logistic regression trained on {meta['trained_on']}.")

    mode = st.radio("Mode", ["Replay a real day", "Custom input"], horizontal=True, key="risk_mode")
    dates = replay.index
    if "replay_date" not in st.session_state:
        st.session_state.replay_date = dates.min().date()

    if mode == "Replay a real day":
        change_days = replay[replay["change_day"]]
        options = ["Choose a change day"] + [f"{d.date()} ({t})" for d, t in change_days["change_type"].items()]

        def jump():
            choice = st.session_state.jump_to
            if choice != options[0]:
                st.session_state.replay_date = pd.Timestamp(choice.split(" ")[0]).date()

        c1, c2 = st.columns(2)
        c1.date_input("Date (today)", min_value=dates.min().date(), max_value=dates.max().date(), key="replay_date")
        c2.selectbox(f"Jump to a change day ({len(change_days)} in the test year)", options, key="jump_to", on_change=jump)

        day = pd.Timestamp(st.session_state.replay_date)
        row = replay.loc[day]
        prob = float(model.predict_proba(row[features].to_frame().T.astype(float))[:, 1][0])
        st.subheader(f"Today: {day.date()} ({day.day_name()})")
        st.write(f"AQI today: {fmt(row['aqi_today'], 0)} ({row['aqi_bucket_today']}).")
        show_result(prob, meta)

        predicted_bad = prob >= meta["decision_threshold"]
        actual_bad = bool(row["actual_tomorrow_bad"])
        st.subheader("What actually happened tomorrow")
        st.write(f"AQI on {(day + pd.Timedelta(days=1)).date()}: {fmt(row['next_day_aqi'], 0)} ({row['next_day_bucket']}). "
                 + ("The model was right." if predicted_bad == actual_bad else "The model was wrong."))
        if row["change_day"]:
            st.warning(f"This is a change day: tomorrow {row['change_type']} compared with today. "
                       "The \"tomorrow = today\" baseline is wrong on this day.")
        else:
            st.write("This is not a change day: tomorrow's category (bad or not bad) is the same as today's.")
        with st.expander("Feature values used for this day"):
            st.dataframe(pd.DataFrame({"Feature": [feature_label(f) for f in features],
                                       "Value": [row[f] for f in features]}),
                         hide_index=True, width="stretch")
    else:
        prefill = st.date_input("Pre-fill with a real day's values", value=dates.min().date(),
                                min_value=dates.min().date(), max_value=dates.max().date(), key="prefill_date")
        row = replay.loc[pd.Timestamp(prefill)]
        medians = meta["training_medians"]
        values = {}
        with st.form("custom_form"):
            day = st.date_input("Date for today (sets month and day of week)", value=prefill, key=f"custom_date_{prefill}")
            st.write("AQI history")
            aqi_feats = [f for f in features if f.startswith("aqi_")]
            cols = st.columns(len(aqi_feats))
            for c, f in zip(cols, aqi_feats):
                values[f] = c.number_input(feature_label(f), min_value=0.0, value=float(row[f]), step=1.0,
                                           key=f"{f}_{prefill}")
            st.write("Today's pollutant levels (CO in mg/m3, others in ug/m3). Leave blank if not measured; "
                     "blank values are filled with the training median.")
            pollutants = [f for f in features if not f.startswith("aqi_") and f not in
                          ("month_sin", "month_cos", "dow_sin", "dow_cos")]
            cols = st.columns(4)
            for i, f in enumerate(pollutants):
                v = row[f]
                values[f] = cols[i % 4].number_input(
                    f, min_value=0.0, value=None if pd.isna(v) else float(v), step=1.0, format="%.2f",
                    key=f"{f}_{prefill}", help=f"Training median: {fmt(medians[f], 2)}")
            submitted = st.form_submit_button("Predict", key="predict")
        st.info("AQI is calculated from the pollutant values, so the AQI inputs and the pollutant inputs are tied "
                "together in real data. Changing one alone (for example AQI today without changing PM2.5 and PM10) "
                "creates a day that cannot happen in reality, and the prediction for such a day is not meaningful. "
                "For a realistic what-if, change the related values together.")
        # Inside a form, edits only take effect when "Predict" is pressed, so
        # the result below always matches the values last submitted (the
        # pre-filled real day until then).
        if submitted:
            st.caption("Prediction updated with the values above.")
        cal = calendar_features(pd.DatetimeIndex([pd.Timestamp(day)])).iloc[0]
        x = pd.DataFrame([{f: (values[f] if f in values else cal[f]) for f in features}], dtype=float)
        prob = float(model.predict_proba(x)[:, 1][0])
        show_result(prob, meta)
    change_day_note(meta)


# ---------------------------------------------------------------------
# Page 4: lockdown case study
# ---------------------------------------------------------------------
def page_lockdown() -> None:
    summ = load_json(OUT / "phase4_summary.json")
    labels = summ["labels"]
    lock = load_csv("phase4_lockdown_2020.csv", "Date")
    full = load_csv("phase4_forecasts_full_year_2019.csv", "Date")
    lock_date = pd.Timestamp(summ["lockdown_date"])
    main, models = summ["lockdown_main"], summ["lockdown_models"]
    res = summ["lockdown_results"]

    st.title("Lockdown case study")
    st.write(f"Forecasts made at the end of the test year for the weeks ending {lock['Date'].min().date()} to "
             f"{lock['Date'].max().date()}, compared with what actually happened. The lockdown began on "
             f"{lock_date:%d %B %Y}. Main model: {labels[main]}. Robustness check: "
             f"{', '.join(labels[m] for m in models if m != main)}, refitted on all years up to the end of the test year.")

    context = full[["Date", "Actual"]].tail(len(lock)).assign(Series="Actual (previous year)")
    parts = [context.rename(columns={"Actual": "AQI"}),
             lock[["Date", "Actual"]].rename(columns={"Actual": "AQI"}).assign(Series="Actual")]
    for m in models:
        role = "main" if m == main else "robustness check"
        parts.append(lock[["Date", f"Forecast: {m}"]].rename(columns={f"Forecast: {m}": "AQI"})
                     .assign(Series=f"Forecast: {labels[m]} ({role})"))
    long = pd.concat(parts)
    domain = ["Actual (previous year)", "Actual"] + [f"Forecast: {labels[m]} ({'main' if m == main else 'robustness check'})" for m in models]
    rng = [MUTED, INK] + [COLORS[m] for m in models]
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=20)).encode(
        x=alt.X("Date:T", title="Week ending (Sunday)"),
        y=alt.Y("AQI:Q", title="Weekly mean AQI"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), title=None, sort=domain),
        tooltip=[alt.Tooltip("Date:T", title="Week ending"), "Series:N", alt.Tooltip("AQI:Q", format=".1f")],
    )
    band = alt.Chart(pd.DataFrame({"start": [lock_date], "end": [lock["Date"].max()]})).mark_rect(
        color="#2a78d6", opacity=0.07).encode(x="start:T", x2="end:T")
    rule = alt.Chart(pd.DataFrame({"d": [lock_date]})).mark_rule(color="#1c5cab", strokeWidth=2).encode(x="d:T")
    text = alt.Chart(pd.DataFrame({"d": [lock_date], "t": [f"Lockdown starts {lock_date:%d %b %Y}"]})).mark_text(
        align="left", dx=5, dy=8, color="#1c5cab", fontSize=12).encode(x="d:T", y=alt.value(0), text="t:N")
    st.altair_chart(style((band + lines + rule + text).properties(height=380)), width="stretch")

    st.write("**Sign convention: error = actual minus forecast.** A negative error means the air was cleaner than "
             "the model expected.")
    rows = []
    for m in models:
        for period, key in [("Pre-lockdown", "pre"), ("Post-lockdown", "post")]:
            s = res[m][key]
            rows.append({"Model": labels[m], "Period": period, "Weeks": s["n"], "Mean actual": s["actual"],
                         "Mean forecast": s["forecast"], "Mean error (actual - forecast)": s["me"], "MAE": s["mae"]})
    st.dataframe(pd.DataFrame(rows).style.format(precision=1), hide_index=True, width="stretch")
    st.caption(f"Pre-lockdown weeks end {summ['lockdown_pre_weeks'][0]} to {summ['lockdown_pre_weeks'][1]}; "
               f"post-lockdown weeks end {summ['lockdown_post_weeks'][0]} to {summ['lockdown_post_weeks'][1]}. "
               f"The week ending {', '.join(summ['lockdown_excluded_weeks'])} is left out because it contains days "
               "from before and after the lockdown began.")

    st.subheader("Estimated lockdown effect")
    eff = pd.DataFrame([{"Model": labels[m], "Role": "main" if m == main else "robustness check",
                         "Effect (post minus pre mean error)": res[m]["effect"],
                         "Placebo: same calculation, previous year": res[m]["placebo"],
                         "Effect minus placebo": res[m]["effect"] - res[m]["placebo"]} for m in models])
    st.dataframe(eff.style.format(precision=1), hide_index=True, width="stretch")
    r = res[main]
    st.write(
        f"With {labels[main]}, the air after the lockdown began was {fmt(-r['post']['me'])} AQI points cleaner than "
        f"forecast on average. But before the lockdown it was already {fmt(-r['pre']['me'])} points cleaner than "
        "forecast, because Delhi's air had been improving year on year. The raw gap therefore overstates the "
        f"lockdown effect; subtracting the pre-lockdown error leaves {fmt(r['effect'])} AQI points. In the previous "
        f"year, with no lockdown, the same calculation gives {fmt(r['placebo'])}."
    )
    st.warning("This is an association, not proof of cause. Weather (rain, wind, temperature) also changes from "
               "year to year and affects AQI, and these models have no weather data.")


# ---------------------------------------------------------------------
st.set_page_config(page_title="Delhi AQI", layout="wide")
# Hide the animated "running" indicator (no animations in this app).
st.markdown("<style>[data-testid='stStatusWidget']{display:none}</style>", unsafe_allow_html=True)
page = st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.caption("All numbers on these pages are read from the saved outputs in reports/outputs and models/.")
{"Overview": page_overview, "Forecast": page_forecast, "Tomorrow's risk": page_risk,
 "Lockdown case study": page_lockdown}[page]()
