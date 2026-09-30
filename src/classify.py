"""Phase 5: will tomorrow be a bad air day in Delhi?

Target: 1 if TOMORROW's AQI is above 300 (Very Poor or Severe), else 0.
Every feature uses only information that is available by the end of today.

Split by time: train = 2015-2018, test = 2019. Imputation and scaling are
inside sklearn Pipelines, so they are fitted on training rows only.

Run from the project root:
    .venv/bin/python src/classify.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "delhi_daily.csv"
REPORT_PATH = ROOT / "reports" / "phase5_classify.md"
RESULTS_PATH = ROOT / "reports" / "results.md"
FIG_DIR = ROOT / "reports" / "figures"
OUT_DIR = ROOT / "reports" / "outputs"
MODEL_DIR = ROOT / "models"

THRESHOLD_AQI = 300
DECISION_THRESHOLD = 0.5  # kept at the default; not tuned on 2019
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_YEAR = 2019

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
AQI_FEATURES = ["aqi_today", "aqi_lag1", "aqi_lag2", "aqi_lag7", "aqi_roll7_mean"]
CALENDAR_FEATURES = ["month_sin", "month_cos", "dow_sin", "dow_cos"]
FEATURES = POLLUTANTS + AQI_FEATURES + CALENDAR_FEATURES

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
QDA_REG_GRID = [0.001, 0.01, 0.05, 0.1]
CV = TimeSeriesSplit(n_splits=5, gap=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for each day t, using only days up to and including t.

    - Pollutants: today's measured values (NaNs left for the pipeline).
    - aqi_today: today's AQI. It is published at the end of today, so it is
      known when we forecast tomorrow. It is also what the baseline uses.
    - aqi_lag1, aqi_lag2, aqi_lag7: AQI 1, 2 and 7 days before today.
    - aqi_roll7_mean: mean AQI of the 7 days ending today (t-6 to t).
      rolling() only looks backwards, so no future day is included.
    - month and day of week as sin/cos: December (12) and January (1) are
      neighbours in the calendar but far apart as numbers; putting them on a
      circle keeps neighbours close. Same idea for Sunday and Monday.
    """
    f = pd.DataFrame(index=df.index)
    for p in POLLUTANTS:
        f[p] = df[p]
    f["aqi_today"] = df["AQI"]
    f["aqi_lag1"] = df["AQI"].shift(1)
    f["aqi_lag2"] = df["AQI"].shift(2)
    f["aqi_lag7"] = df["AQI"].shift(7)
    f["aqi_roll7_mean"] = df["AQI"].rolling(7).mean()
    month = df.index.month
    dow = df.index.dayofweek  # Monday = 0
    f["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return f


def build_target(df: pd.DataFrame) -> pd.Series:
    # shift(-1) moves tomorrow's AQI onto today's row. This is the ONLY place
    # a future value is used, and only as the label, never as a feature.
    tomorrow = df["AQI"].shift(-1)
    return (tomorrow > THRESHOLD_AQI).astype(float).where(tomorrow.notna())


def make_pipeline(model) -> Pipeline:
    # The imputer fills pollutant gaps with the training median and the
    # scaler puts features on the same scale. Inside a Pipeline both are
    # fitted only on the rows passed to fit(), i.e. training rows (and inside
    # CV, only the training part of each fold).
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


def fit_with_warnings(pipe, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(X, y)
    return [f"{w.category.__name__}: {w.message}" for w in caught]


def main() -> None:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    for d in (FIG_DIR, OUT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date")
    X_all = build_features(df)
    y_all = build_target(df)

    out("# Phase 5: Predicting tomorrow's bad air day (AQI > 300)")
    out()

    # =================================================================
    # 1. Leakage checks
    # =================================================================
    # Check 1: recompute the features for sample days using ONLY data up to
    # that day. If any feature secretly used a later day, the truncated
    # version would differ from the full version.
    rng = np.random.default_rng(SEED)
    sample_days = rng.choice(X_all.index[7:], size=100, replace=False)
    mismatches = 0
    for t in sample_days:
        truncated = build_features(df.loc[:t]).iloc[-1]
        full_row = X_all.loc[t]
        same = np.isclose(truncated.to_numpy(float), full_row.to_numpy(float), equal_nan=True).all()
        mismatches += int(not same)
    # Check 2: the label for day t must equal "AQI on day t+1 > 300".
    label_ok = bool(((df["AQI"].shift(-1) > THRESHOLD_AQI).astype(float)[:-1] == y_all[:-1]).all())

    out("## 1. Features and leakage checks")
    out()
    out("Features for day t (all known by the end of day t):")
    out()
    out(f"- Today's pollutant levels: {', '.join(POLLUTANTS)} (Xylene was dropped in Phase 2).")
    out("- `aqi_today` (AQI of day t), `aqi_lag1`, `aqi_lag2`, `aqi_lag7` (AQI of days t-1, t-2, t-7), "
        "`aqi_roll7_mean` (mean AQI of days t-6 to t).")
    out("- Month as sin/cos and day of week as sin/cos (cyclic, so December sits next to January and Sunday next to Monday).")
    out()
    out("Leakage checks:")
    out()
    out(f"- Truncation test: for 100 random days (seed {SEED}), features were recomputed from data that stops at that "
        f"day and compared with the features from the full data. Mismatches: **{mismatches}**. "
        "So no feature uses any day after t.")
    out(f"- Label test: target on day t equals (AQI on day t+1 > {THRESHOLD_AQI}) for every row: **{label_ok}**. "
        "The label is the only place a future value appears.")
    out("- Imputation and scaling are inside sklearn Pipelines, fitted on 2015-2018 rows only.")
    out("- Cross-validation uses TimeSeriesSplit with gap = 1, so a fold's last training row (whose label is the next "
        "day's AQI) never overlaps with the first validation day.")
    out("- The `aqi_outlier` flag from Phase 2 is not used as a feature (it was computed over all years).")
    out()

    # =================================================================
    # 2. Train / test rows
    # =================================================================
    data = X_all.join(y_all.rename("target"))
    first_complete = X_all[AQI_FEATURES].dropna().index.min()
    data = data.loc[first_complete:]
    train = data[data.index < TRAIN_END]  # stops at 2018-12-30; see below
    test = data[data.index.year == TEST_YEAR]
    X_train, y_train = train[FEATURES], train["target"].astype(int)
    X_test, y_test = test[FEATURES], test["target"].astype(int)
    assert y_train.notna().all() and y_test.notna().all()

    out("## 2. Rows used")
    out()
    out(f"- The first {X_all.index.get_loc(first_complete)} days (2015-01-01 to "
        f"{(first_complete - pd.Timedelta(days=1)).date()}) are dropped because `aqi_lag7` and the 7-day mean need 7 earlier days. "
        "They are dropped, not filled, because filling them would mean inventing AQI history.")
    out(f"- The 2018-12-31 row is dropped from training: its label is the AQI of 2019-01-01, a test-year value.")
    out(f"- Train: {len(train)} days, {train.index.min().date()} to {train.index.max().date()}.")
    out(f"- Test: {len(test)} days, {test.index.min().date()} to {test.index.max().date()}. "
        "The last test row's label is the AQI of 2020-01-01; it is used only as that row's answer.")
    miss = X_train[POLLUTANTS].isna().mean() * 100
    out("- Pollutant values missing in training rows (filled with the training median inside the pipeline): "
        + ", ".join(f"{p} {v:.2f}%" for p, v in miss[miss > 0].items()) + ".")
    out(f"- Training labels built from interpolated AQI (Phase 2 `aqi_imputed`): "
        f"{int(df['aqi_imputed'].shift(-1).reindex(train.index).fillna(False).astype(bool).sum())} rows.")
    out()

    # =================================================================
    # 3. Class balance
    # =================================================================
    tr_rate, te_rate = y_train.mean() * 100, y_test.mean() * 100
    out("## 3. Class balance")
    out()
    out("| Set | Days | Bad days tomorrow (1) | Not bad (0) | % bad |")
    out("|---|---|---|---|---|")
    out(f"| Train 2015-2018 | {len(y_train)} | {int(y_train.sum())} | {int((1 - y_train).sum())} | {tr_rate:.2f}% |")
    out(f"| Test 2019 | {len(y_test)} | {int(y_test.sum())} | {int((1 - y_test).sum())} | {te_rate:.2f}% |")
    out()
    out(f"- The classes are moderately imbalanced: {tr_rate:.1f}% bad days in training. Not extreme, but enough that a "
        "model can look accurate by leaning towards \"not bad\". That is why we report precision, recall, F1 and AUC "
        "rather than accuracy, and why logistic regression uses `class_weight=\"balanced\"` (mistakes on the rarer "
        "class count more during training).")
    out(f"- **Distribution shift:** bad days fall from {tr_rate:.1f}% in training to {te_rate:.1f}% in 2019, because "
        "Delhi's air improved. A model trained on the dirtier years may expect bad days more often than they happen "
        "in 2019 and raise more false alarms, which **may push precision down on 2019** even if the model has learnt "
        f"the right patterns. The decision threshold is kept at {DECISION_THRESHOLD}; it was not tuned on 2019. "
        "Section 5 checks whether this happened.")
    out()

    # =================================================================
    # 4. Models
    # =================================================================
    # Baseline: tomorrow's class = today's class.
    base_pred = (X_test["aqi_today"] > THRESHOLD_AQI).astype(int)

    # Logistic regression, C tuned by TimeSeriesSplit CV on training rows.
    logreg = make_pipeline(LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED))
    search = GridSearchCV(logreg, {"model__C": C_GRID}, scoring="f1", cv=CV)
    with warnings.catch_warnings(record=True) as lr_warn:
        warnings.simplefilter("always")
        search.fit(X_train, y_train)
    lr_best = search.best_estimator_
    best_C = search.best_params_["model__C"]

    # LDA
    lda = make_pipeline(LinearDiscriminantAnalysis())
    lda_warn = fit_with_warnings(lda, X_train, y_train)

    # QDA: first without regularisation. QDA estimates a separate covariance
    # matrix per class; strongly correlated features (PM2.5 and PM10, and the
    # AQI lag features) can make those matrices nearly singular.
    qda0 = make_pipeline(QuadraticDiscriminantAnalysis())
    qda0_warn = fit_with_warnings(qda0, X_train, y_train)
    qda_reg = 0.0
    qda_cv = None
    if qda0_warn:
        # Choose the smallest-effective reg_param by TimeSeriesSplit CV F1 on
        # 2015-2018. reg_param shrinks each class covariance towards the
        # identity, which makes it invertible.
        qda_cv = {}
        for r in QDA_REG_GRID:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qda_cv[r] = cross_val_score(make_pipeline(QuadraticDiscriminantAnalysis(reg_param=r)),
                                            X_train, y_train, cv=CV, scoring="f1").mean()
        qda_reg = max(qda_cv, key=qda_cv.get)
    qda = make_pipeline(QuadraticDiscriminantAnalysis(reg_param=qda_reg))
    qda_warn = fit_with_warnings(qda, X_train, y_train)

    out("## 4. Models")
    out()
    out(f"- **Baseline (persistence):** tomorrow's class = today's class (`aqi_today > {THRESHOLD_AQI}`).")
    out(f"- **Logistic regression**, `class_weight=\"balanced\"`, L2 penalty (default). C searched over {C_GRID} with "
        f"GridSearchCV, TimeSeriesSplit(5 folds, gap 1), scoring F1, on 2015-2018 only.")
    out()
    out("| C | Mean CV F1 | Std |")
    out("|---|---|---|")
    cvres = pd.DataFrame(search.cv_results_)
    for _, r in cvres.iterrows():
        out(f"| {r['param_model__C']} | {r['mean_test_score']:.4f} | {r['std_test_score']:.4f} |")
    out()
    out(f"  Chosen C = **{best_C}**. Smaller C means a stronger penalty that pulls coefficients towards zero.")
    if lr_warn:
        out(f"  Warnings during the search: {len(lr_warn)} ({sorted(set(w.category.__name__ for w in lr_warn))}).")
    out(f"- **LDA:** warnings when fitting: {lda_warn if lda_warn else 'none'}.")
    if qda0_warn:
        out(f"- **QDA:** without regularisation it warned: `{qda0_warn[0]}`"
            + (f" (and {len(qda0_warn) - 1} more)" if len(qda0_warn) > 1 else "") + ". "
            "This happens because QDA inverts each class's covariance matrix, and strongly correlated inputs "
            "(PM2.5 with PM10, and the AQI lag features with each other) make that matrix close to singular, so the "
            "inverse is numerically unstable. We therefore use a small `reg_param`, which blends each covariance "
            "matrix with the identity matrix so it can be inverted safely.")
        out()
        out("  | reg_param | Mean CV F1 (TimeSeriesSplit, 2015-2018) |")
        out("  |---|---|")
        for r, v in qda_cv.items():
            out(f"  | {r} | {v:.4f} |")
        out()
        out(f"  Chosen reg_param = **{qda_reg}** (best CV F1 on training data). "
            f"Warnings after regularisation: {qda_warn if qda_warn else 'none'}.")
    else:
        out(f"- **QDA:** fitted without warnings, so no regularisation was needed (reg_param = 0).")
    out()

    # =================================================================
    # 5. Evaluation on 2019
    # =================================================================
    probs = {
        "Logistic regression": lr_best.predict_proba(X_test)[:, 1],
        "LDA": lda.predict_proba(X_test)[:, 1],
        "QDA": qda.predict_proba(X_test)[:, 1],
    }
    preds = {"Baseline (persistence)": base_pred.to_numpy()}
    for m, p in probs.items():
        preds[m] = (p >= DECISION_THRESHOLD).astype(int)
    # The baseline only gives 0/1, so its "AUC" is computed from those hard
    # labels (a single point on the ROC curve joined to the corners).
    scores = {"Baseline (persistence)": base_pred.to_numpy().astype(float), **probs}

    rows = []
    for m in preds:
        cm = confusion_matrix(y_test, preds[m], labels=[0, 1])
        rows.append(dict(Model=m, Precision=precision_score(y_test, preds[m], zero_division=0),
                         Recall=recall_score(y_test, preds[m]), F1=f1_score(y_test, preds[m]),
                         AUC=roc_auc_score(y_test, scores[m]), Accuracy=accuracy_score(y_test, preds[m]),
                         TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1]))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "phase5_metrics.csv", index=False)

    table = ["| Model | Precision | Recall | F1 | AUC-ROC | Accuracy (context only) |", "|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        table.append(f"| {r['Model']} | {r['Precision']:.3f} | {r['Recall']:.3f} | {r['F1']:.3f} | {r['AUC']:.3f} | {r['Accuracy']:.3f} |")

    out(f"## 5. Results on 2019 ({len(y_test)} days, threshold {DECISION_THRESHOLD})")
    out()
    for row in table:
        out(row)
    out()
    out("The baseline's AUC is computed from its 0/1 predictions, so it is a single operating point, not a full curve.")
    out()
    out(f"Distribution shift check: 2019 had {int(y_test.sum())} bad days. Number of days each model predicted as bad: "
        + ", ".join(f"{m} {int(preds[m].sum())}" for m in preds) + ". "
        + (f"The largest over-prediction is {max(int(preds[m].sum()) - int(y_test.sum()) for m in probs):+d} days "
           f"(a negative number means under-prediction), so at the 0.5 threshold the shift did not produce a visible "
           "wave of false alarms in 2019. A likely reason: today's pollutant and AQI values already reflect the "
           "cleaner air, so the inputs shift along with the outcome."))
    out()
    out("Confusion matrices (rows = actual, columns = predicted):")
    out()
    out("| Model | TN (correct not bad) | FP (false alarm) | FN (missed bad day) | TP (caught bad day) |")
    out("|---|---|---|---|---|")
    for _, r in metrics.iterrows():
        out(f"| {r['Model']} | {r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} |")
    out()

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, m in zip(axes, preds):
        cm = confusion_matrix(y_test, preds[m], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Not bad", "Bad"])
        ax.set_yticks([0, 1], ["Not bad", "Bad"])
        ax.set_xlabel("Predicted tomorrow")
        ax.set_ylabel("Actual tomorrow")
        ax.set_title(m, fontsize=10)
    fig.suptitle("Confusion matrices on 2019 test days (bad = next-day AQI above 300)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for m, color in zip(scores, ["tab:grey", "tab:blue", "tab:green", "tab:red"]):
        fpr, tpr, _ = roc_curve(y_test, scores[m])
        auc = metrics.set_index("Model").loc[m, "AUC"]
        ax.plot(fpr, tpr, color=color, marker="o" if m.startswith("Baseline") else None,
                label=f"{m} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="Random guessing")
    ax.set_xlabel("False positive rate (false alarms / all not-bad days)")
    ax.set_ylabel("True positive rate (caught / all bad days)")
    ax.set_title("ROC curves on 2019 test days")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_roc_curves.png", dpi=150)
    plt.close(fig)
    out("Figures: `phase5_confusion_matrices.png`, `phase5_roc_curves.png`.")
    out()

    # =================================================================
    # 6. Change days
    # =================================================================
    today_bad = (X_test["aqi_today"] > THRESHOLD_AQI).astype(int)
    change = today_bad != y_test
    to_bad = change & (y_test == 1)
    to_good = change & (y_test == 0)
    out("## 6. Change days in 2019 (tomorrow's status differs from today's)")
    out()
    out(f"There are **{int(change.sum())} change days** in 2019: {int(to_bad.sum())} where tomorrow turns bad "
        f"(today not bad, tomorrow bad) and {int(to_good.sum())} where tomorrow turns better (today bad, tomorrow not bad).")
    out()
    out("**The persistence baseline is wrong on every change day by definition**, because it always predicts that "
        "tomorrow will be the same as today. Change days are therefore where a model shows value beyond simply "
        "copying today.")
    out()
    out("| Model | Correct on change days | Correct: turns bad | Correct: turns better |")
    out("|---|---|---|---|")
    change_rows = []
    for m in preds:
        p = pd.Series(preds[m], index=y_test.index)
        c_all = int((p[change] == y_test[change]).sum())
        c_bad = int((p[to_bad] == 1).sum())
        c_good = int((p[to_good] == 0).sum())
        change_rows.append(dict(Model=m, all=c_all, bad=c_bad, good=c_good))
        out(f"| {m} | {c_all} of {int(change.sum())} | {c_bad} of {int(to_bad.sum())} | {c_good} of {int(to_good.sum())} |")
    out()
    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(OUT_DIR / "phase5_change_days.csv", index=False)
    pd.DataFrame({"Date": y_test.index, "aqi_today": X_test["aqi_today"].to_numpy(), "actual_tomorrow_bad": y_test.to_numpy(),
                  "change_day": change.to_numpy(), **{f"pred_{m}": preds[m] for m in preds},
                  **{f"prob_{m}": probs[m] for m in probs}}).to_csv(
        OUT_DIR / "phase5_test_predictions_2019.csv", index=False, date_format="%Y-%m-%d")

    # =================================================================
    # 7. Logistic regression coefficients
    # =================================================================
    coefs = pd.Series(lr_best.named_steps["model"].coef_[0], index=FEATURES).sort_values()
    intercept = lr_best.named_steps["model"].intercept_[0]
    coefs.to_csv(OUT_DIR / "phase5_logreg_coefficients.csv", header=["coefficient"])
    corr_pm = X_train["PM2.5"].corr(X_train["PM10"])
    aqi_block = X_train[AQI_FEATURES].corr()
    aqi_min_corr = aqi_block.where(~np.eye(len(AQI_FEATURES), dtype=bool)).min().min()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(coefs.index, coefs.values, color=["tab:red" if v > 0 else "tab:blue" for v in coefs.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Logistic regression coefficients (standardised features, C = {best_C})")
    ax.set_xlabel("Coefficient (change in log-odds of a bad day tomorrow per 1 std increase)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_logreg_coefficients.png", dpi=150)
    plt.close(fig)

    out("## 7. Logistic regression coefficients")
    out()
    out("Features are standardised, so each coefficient is the change in the log-odds of a bad day tomorrow when that "
        "feature rises by one standard deviation (training data) and the others stay the same. Positive pushes the "
        "prediction towards \"bad\", negative pushes it towards \"not bad\". "
        f"Intercept: {intercept:.4f}.")
    out()
    out("| Feature | Coefficient | Odds multiplier per 1 std |")
    out("|---|---|---|")
    for f_, v in coefs.sort_values(ascending=False).items():
        out(f"| {f_} | {v:+.4f} | {np.exp(v):.3f} |")
    out()
    pm_sum = coefs["PM2.5"] + coefs["PM10"]
    aqi_sum = coefs[AQI_FEATURES].sum()
    top_pos = coefs.sort_values(ascending=False).head(3)
    top_neg = coefs.sort_values().head(3)
    out("**In plain words:**")
    out()
    out(f"- **PM2.5 and PM10 together:** their coefficients are {coefs['PM2.5']:+.3f} and {coefs['PM10']:+.3f}, a combined "
        f"{pm_sum:+.3f}. They should be read as one \"particulate matter\" signal, not separately: in training they are "
        f"correlated at r = {corr_pm:.2f}, so the model can shift weight from one to the other with almost no change in "
        "its predictions. Their individual values (and even their signs) are therefore unstable; their combined effect "
        "is more trustworthy. The default L2 penalty helps here: it discourages large opposite-signed coefficients on "
        "correlated features and shares the weight between them.")
    out(f"- **The AQI history block** (`aqi_today`, lags 1, 2, 7 and the 7-day mean) has the same issue: these five are "
        f"correlated with each other at r = {aqi_min_corr:.2f} or more in training. Their combined coefficient is "
        f"{aqi_sum:+.3f}. Read together: when recent AQI has been high, tomorrow is more likely to be bad.")
    out(f"- Strongest pushes towards \"bad\": " + ", ".join(f"`{k}` ({v:+.3f})" for k, v in top_pos.items()) + ".")
    out(f"- Strongest pushes towards \"not bad\": " + ", ".join(f"`{k}` ({v:+.3f})" for k, v in top_neg.items()) + ".")
    out("- Month and day-of-week terms describe the season and weekly cycle; their sin and cos parts only make sense "
        "together (for example, the month pair together places each month on the yearly circle).")
    out()
    out("Figure: `phase5_logreg_coefficients.png`.")
    out()

    # =================================================================
    # 8. Save the fitted pipeline for the app
    # =================================================================
    joblib.dump(lr_best, MODEL_DIR / "logreg_pipeline.joblib")
    meta = dict(
        model="LogisticRegression pipeline (median imputer, StandardScaler, class_weight=balanced)",
        C=best_C, decision_threshold=DECISION_THRESHOLD, bad_day_aqi_threshold=THRESHOLD_AQI,
        features=FEATURES, trained_on=f"{train.index.min().date()} to {train.index.max().date()}",
        n_train=len(train), test_metrics_2019={k: float(v) for k, v in
                                               metrics.set_index("Model").loc["Logistic regression",
                                                                              ["Precision", "Recall", "F1", "AUC"]].items()},
        training_medians=dict(zip(FEATURES, lr_best.named_steps["impute"].statistics_.tolist())),
        sklearn_version=sklearn.__version__, seed=SEED,
    )
    (MODEL_DIR / "logreg_pipeline_meta.json").write_text(json.dumps(meta, indent=2))
    # Reload and check the saved file gives identical predictions.
    reloaded = joblib.load(MODEL_DIR / "logreg_pipeline.joblib")
    assert np.allclose(reloaded.predict_proba(X_test)[:, 1], probs["Logistic regression"])
    out("## 8. Saved model")
    out()
    out("- `models/logreg_pipeline.joblib`: the tuned logistic regression pipeline (imputer, scaler, model), fitted on "
        f"2015-2018. Reloading it reproduces the 2019 probabilities exactly.")
    out("- `models/logreg_pipeline_meta.json`: feature order, C, thresholds, training period, 2019 metrics and scikit-learn version.")
    out()

    text = "\n".join(lines) + "\n"
    assert "—" not in text and "–" not in text, "report contains an em or en dash"
    REPORT_PATH.write_text(text)

    section = ["<!-- phase5:start -->", f"## Phase 5: Tomorrow's bad air day (AQI > {THRESHOLD_AQI}), test = 2019 ({len(y_test)} days)", ""]
    section += table
    section += ["", f"Change days in 2019: {int(change.sum())} ({int(to_bad.sum())} turn bad, {int(to_good.sum())} turn better).", "",
                "| Model | Correct on change days | Turns bad | Turns better |", "|---|---|---|---|"]
    for r in change_rows:
        section.append(f"| {r['Model']} | {r['all']} of {int(change.sum())} | {r['bad']} of {int(to_bad.sum())} | "
                       f"{r['good']} of {int(to_good.sum())} |")
    section += ["", f"Logistic regression C = {best_C}; QDA reg_param = {qda_reg}.", "<!-- phase5:end -->"]
    block = "\n".join(section)
    existing = RESULTS_PATH.read_text()
    if "<!-- phase5:start -->" in existing:
        head, rest = existing.split("<!-- phase5:start -->", 1)
        existing = head + block + rest.split("<!-- phase5:end -->", 1)[1]
    else:
        existing = existing.rstrip() + "\n\n" + block + "\n"
    RESULTS_PATH.write_text(existing)
    print(f"\nSaved {REPORT_PATH.relative_to(ROOT)}, {RESULTS_PATH.relative_to(ROOT)} and models/")


if __name__ == "__main__":
    main()
