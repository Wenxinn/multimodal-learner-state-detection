#!/usr/bin/env python3
"""
build_dataset.py  -  STEP 1: build the final 3-class dataset.

Usage:
    python3 build_dataset.py features.csv

Input : features.csv  (made by extract_features.py, one row per sub-question)
Output: dataset.csv   (same rows + 3-class label + keep flags)
        prints a summary you can copy into your slides

What it does
  1. Removes P48 (researcher's own test session).
  2. Makes the 3-class label:
        locating_information  -> locating
        understanding         -> understanding
        reasoning_integrating -> reasoning
        answer_construction   -> reasoning     (merged: rare, similar behaviour)
  3. Adds keep flags:
        use_gaze = 1 if the sub-question has >= 80% valid gaze (for gaze models)
        use_pen  = 1 always (pen data is usable for everyone)
     P1-P7 have no real pen pressure, so their pressure features are left empty
     (extract_features.py already does this; this script checks it).
"""
import sys
import pandas as pd

GAZE_MIN_VALID = 80.0
EXCLUDE = [48]  # researcher's own test run
MERGE = {
    "locating_information": "locating",
    "understanding": "understanding",
    "reasoning_integrating": "reasoning",
    "answer_construction": "reasoning",
}

path = sys.argv[1] if len(sys.argv) > 1 else "features.csv"
d = pd.read_csv(path)

# 1. exclude researcher session
d = d[~d.participant.isin(EXCLUDE)].copy()

# 2. 3-class label (keep the original 4-class label too, for the thesis)
d["label4"] = d["label"]
d["label3"] = d["label"].map(MERGE)
missing = d["label3"].isna().sum()
if missing:
    print(f"WARNING: {missing} rows have an unknown label and are dropped")
    d = d.dropna(subset=["label3"])

# 3. keep flags
d["use_gaze"] = (d["gaze_valid_pct"] >= GAZE_MIN_VALID).astype(int)
d["use_pen"] = 1

# check: pressure must be empty for P1-P7, filled for P8+
p17 = d[d.participant <= 7]["pressure_mean"].notna().sum()
p8 = d[d.participant >= 8]["pressure_mean"].notna().mean() * 100
d.to_csv("dataset.csv", index=False)

# ---- summary ----
print("=" * 55)
print("  STEP 1 SUMMARY  ->  dataset.csv")
print("=" * 55)
print(f"Participants          : {d.participant.nunique()}  (P48 excluded)")
print(f"Sub-questions (rows)  : {len(d)}")
print("\n3-class labels:")
for k, v in d["label3"].value_counts().items():
    print(f"  {k:14s} {v:4d}  ({100*v/len(d):.0f}%)")
print(f"\nGaze models  : {d.use_gaze.sum()} of {len(d)} rows have >= {GAZE_MIN_VALID:.0f}% valid gaze")
bad = d.groupby("participant").use_gaze.agg(["sum", "size"])
bad = bad[bad["sum"] < bad["size"] / 2]
if len(bad):
    print("  participants with < half usable gaze: " +
          ", ".join(f"P{p} ({r['sum']}/{r['size']})" for p, r in bad.iterrows()))
print(f"Pen models   : {d.use_pen.sum()} of {len(d)} rows")
print(f"Pressure     : P1-P7 rows with pressure = {p17} (should be 0); "
      f"P8+ rows with pressure = {p8:.0f}%")
print("=" * 55)
