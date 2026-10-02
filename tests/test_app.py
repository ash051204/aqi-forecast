"""Smoke tests for app.py using Streamlit's AppTest (no browser needed).

Run from the project root:
    .venv/bin/python tests/test_app.py
Each check prints PASS or FAIL; the script exits non-zero if any fail.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
replay = pd.read_csv(ROOT / "reports" / "outputs" / "phase5_replay_2019.csv", parse_dates=["Date"])
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def all_text(at) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.title] + [e.value for e in at.subheader]
    parts += [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.warning]
    return "\n".join(str(p) for p in parts)


def new_app():
    return AppTest.from_file(APP, default_timeout=120).run()


# Every page loads without exceptions.
for page in ["Overview", "Forecast", "Tomorrow's risk", "Lockdown case study"]:
    at = new_app()
    at.sidebar.radio(key="page").set_value(page).run()
    check(f"page '{page}' loads", not at.exception, str([e.value for e in at.exception])[:300])
    text = all_text(at)
    check(f"page '{page}' has no em or en dashes", "\u2014" not in text and "\u2013" not in text)

# Forecast page: rolling horizon also renders.
at = new_app()
at.sidebar.radio(key="page").set_value("Forecast").run()
at.radio(key="horizon").set_value("Rolling one week ahead").run()
check("forecast page, rolling horizon loads", not at.exception)


def replay_day(date: pd.Timestamp):
    at = new_app()
    at.sidebar.radio(key="page").set_value("Tomorrow's risk").run()
    at.date_input(key="replay_date").set_value(date.date()).run()
    return at


def shown_probability(at) -> str:
    return next(m.value for m in at.metric if m.label.startswith("Probability"))


# Replay a normal day and a change day; the displayed probability must match
# the probability saved by src/classify.py for that date.
normal = replay[~replay["change_day"]].iloc[10]
change = replay[replay["change_day"]].iloc[0]
for kind, row in [("normal day", normal), ("change day", change)]:
    at = replay_day(row["Date"])
    ok = not at.exception
    check(f"replay {kind} {row['Date'].date()} loads", ok, str([e.value for e in at.exception])[:300])
    if ok:
        prob = shown_probability(at)
        check(f"replay {kind}: probability matches saved output", prob == f"{row['prob_logreg']:.1%}",
              f"shown {prob}, saved {row['prob_logreg']:.1%}")
        text = all_text(at)
        flagged = "This is a change day" in text
        check(f"replay {kind}: change-day flag shown correctly", flagged == bool(row["change_day"]))
        check(f"replay {kind}: next-day AQI shown", f"{row['next_day_aqi']:,.0f} ({row['next_day_bucket']})" in text)

# Jump-to-change-day selector moves the date.
at = new_app()
at.sidebar.radio(key="page").set_value("Tomorrow's risk").run()
target = at.selectbox(key="jump_to").options[3]
at.selectbox(key="jump_to").set_value(target).run()
check("jump to change day sets the date",
      str(at.date_input(key="replay_date").value) == target.split(" ")[0], target)
check("jump to change day shows change-day warning", "This is a change day" in all_text(at))

# Custom input: pre-filled with a real day gives the same probability as
# replaying that day; editing a value and pressing Predict changes it.
at = new_app()
at.sidebar.radio(key="page").set_value("Tomorrow's risk").run()
at.radio(key="risk_mode").set_value("Custom input").run()
check("custom input loads", not at.exception, str([e.value for e in at.exception])[:300])
first = replay.iloc[0]
check("custom input pre-filled result equals replay of the same day",
      shown_probability(at) == f"{first['prob_logreg']:.1%}", shown_probability(at))
check("custom input shows the 'impossible day' note", "cannot happen in reality" in all_text(at))
# This edit (AQI today alone) makes a physically impossible day. It only
# checks that the Predict button recomputes; the value itself means nothing.
before = shown_probability(at)
key = next(k for k in (w.key for w in at.number_input) if k.startswith("aqi_today"))
at.number_input(key=key).set_value(50.0)
at.button(key="predict").click().run()
check("custom input: editing AQI today and pressing Predict changes the probability",
      not at.exception and shown_probability(at) != before, f"{before} -> {shown_probability(at)}")

# The app source contains no em or en dashes.
src = Path(APP).read_text()
check("app.py has no em or en dashes", "\u2014" not in src and "\u2013" not in src)

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
