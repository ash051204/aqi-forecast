"""Check that no committed text file contains an em dash or an en dash.

The design rules forbid them in any text. This scans every file tracked by
git (README.md, BRIEF.md, data/SOURCE.md, reports/*.md, code, CSV and JSON
outputs) and skips binary files such as PNG figures and the saved model.
Code that needs to mention the characters uses escape codes instead.

Run from the project root:
    .venv/bin/python tests/test_no_dashes.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHES = {"\u2014": "em dash", "\u2013": "en dash"}

files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
# These must be covered; fail loudly if one is ever missing from git.
required = ["README.md", "BRIEF.md", "data/SOURCE.md", "reports/viva_notes.md", "reports/results.md"]
missing = [f for f in required if f not in files]

problems, checked = [], 0
for name in files:
    try:
        text = (ROOT / name).read_text(encoding="utf-8")
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        continue  # binary files (figures, model)
    checked += 1
    for i, line in enumerate(text.splitlines(), 1):
        for ch, label in DASHES.items():
            if ch in line:
                problems.append(f"{name}:{i}: {label}")

print(f"Checked {checked} committed text files.")
for p in problems:
    print("FAIL ", p)
for f in missing:
    print("FAIL  required file not tracked by git:", f)
print("PASS  no em or en dashes" if not problems and not missing else f"{len(problems) + len(missing)} problem(s)")
sys.exit(1 if problems or missing else 0)
