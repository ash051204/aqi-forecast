"""Run the whole project from the raw CSV, in order.

Steps:
  0. If data/raw/city_day.csv is missing, download it from Kaggle with
     kagglehub (dataset version 12, the one recorded in data/SOURCE.md) and
     check its SHA-256 hash, so we know it is the exact same file.
  1-5. Run each phase script: audit, clean, eda, forecast, classify.
  6. Run the app tests and the em/en dash check over all committed text files.

Stops at the first failure.

Run from the project root:
    .venv/bin/python run_all.py
"""

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "city_day.csv"
DATASET = "rohanrao/air-quality-data-in-india/versions/12"
# Hash of the file used in this project (also recorded in data/SOURCE.md).
EXPECTED_SHA256 = "0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847"

STEPS = [
    ("Phase 1: audit", ["src/audit.py"]),
    ("Phase 2: cleaning", ["src/clean.py"]),
    ("Phase 3: EDA", ["src/eda.py"]),
    ("Phase 4: forecasting", ["src/forecast.py"]),
    ("Phase 5: classification", ["src/classify.py"]),
    ("App tests", ["tests/test_app.py"]),
    ("Dash check", ["tests/test_no_dashes.py"]),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data() -> None:
    if not RAW.exists():
        print(f"{RAW.relative_to(ROOT)} not found; downloading {DATASET} with kagglehub ...")
        import kagglehub  # imported here so the rest works without it once data exists

        folder = Path(kagglehub.dataset_download(DATASET))
        RAW.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / "city_day.csv", RAW)
    digest = sha256(RAW)
    if digest != EXPECTED_SHA256:
        sys.exit(f"STOP: {RAW.relative_to(ROOT)} has SHA-256 {digest}, expected {EXPECTED_SHA256}. "
                 "This is not the file the project was built on.")
    print(f"Data OK: {RAW.relative_to(ROOT)} (SHA-256 matches data/SOURCE.md)")


def main() -> None:
    ensure_data()
    for name, args in STEPS:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        # Each step runs in its own process with this same Python, so every
        # script starts clean exactly as it would when run by hand.
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            sys.exit(f"STOP: {name} failed (exit code {result.returncode}).")
        print(f"{name}: OK in {time.perf_counter() - t0:.1f} s")
        if args[0].startswith("tests/"):
            print(result.stdout.strip().splitlines()[-1])
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
