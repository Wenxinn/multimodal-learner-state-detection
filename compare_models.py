#!/usr/bin/env python3
"""
compare_models.py  -  3 models x 3 feature sets, 3 learner states, leave-one-participant-out.

Usage:  python compare_models.py features.csv
Writes: results_3class.csv (one row per model x feature set, macro-F1 + F1 per state)

3 states: locating / understanding / reasoning (= reasoning_integrating + answer_construction)
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "features.csv")
MERGE = {"locating_information": "locating", "understanding": "understanding",
         "reasoning_integrating": "reasoning", "answer_construction": "reasoning"}
y_txt = d.label.map(MERGE)
STATES = ["locating", "understanding", "reasoning"]
enc = LabelEncoder().fit(STATES)
y = enc.transform(y_txt)
groups = d.participant

META = ["participant", "qid", "passage", "subq", "english_level", "label", "strategy",
        "confidence", "difficulty", "pen_pointer_type", "pressure_available"]
FEAT = [c for c in d.columns if c not in META]
TIMING = ["total_time_s", "time_to_first_stroke_s", "writing_phase_s"]
GAZE = [c for c in FEAT if c not in TIMING and any(k in c for k in
        ["gaze", "fix", "dwell", "trans", "aoi", "regression", "sacc", "pupil"])]
PEN = [c for c in FEAT if c not in TIMING and c not in GAZE]
SETS = {"Gaze": GAZE, "Pen": PEN + TIMING, "Gaze + Pen": GAZE + PEN + TIMING}

def models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight="balanced", random_state=0, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0,
                                 n_jobs=-1, eval_metric="mlogloss"),
        "SVM (RBF)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   SVC(kernel="rbf", C=1.0, class_weight="balanced")),
    }

logo = LeaveOneGroupOut()
rows, cms = [], {}

# Baseline: majority state per sub-question number, learned from training participants only
pred = np.empty(len(y), dtype=int)
for tr, te in logo.split(d, y, groups):
    rule = pd.Series(y[tr]).groupby(d.subq.values[tr]).agg(lambda s: s.value_counts().idxmax())
    pred[te] = d.subq.iloc[te].map(rule).fillna(pd.Series(y[tr]).mode()[0]).astype(int).values
f1s = f1_score(y, pred, average=None, labels=enc.transform(STATES))
rows.append({"model": "Baseline: sub-question number", "features": "-",
             "accuracy": accuracy_score(y, pred), "macro_F1": f1_score(y, pred, average="macro"),
             **{f"F1_{s}": v for s, v in zip(STATES, f1s)}})

for fs_name, cols in SETS.items():
    X = d[cols].astype(float).values
    for m_name in models():
        pred = np.empty(len(y), dtype=int)
        for tr, te in logo.split(X, y, groups):
            m = models()[m_name]
            if m_name == "XGBoost":
                m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            else:
                m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        f1s = f1_score(y, pred, average=None, labels=enc.transform(STATES))
        rows.append({"model": m_name, "features": fs_name,
                     "accuracy": accuracy_score(y, pred), "macro_F1": f1_score(y, pred, average="macro"),
                     **{f"F1_{s}": v for s, v in zip(STATES, f1s)}})
        cms[(m_name, fs_name)] = confusion_matrix(y, pred, labels=enc.transform(STATES))
        print(f"done: {m_name:14s} {fs_name}")

res = pd.DataFrame(rows).round(3)
res.to_csv("results_3class.csv", index=False)
print("\nClass counts:", dict(pd.Series(y_txt).value_counts()))
print(res.to_string(index=False))
best = res.iloc[1:].sort_values("macro_F1", ascending=False).iloc[0]
print(f"\nBest: {best.model} / {best.features}  macro-F1 {best.macro_F1}")
print("Confusion matrix (rows = true, cols = predicted):", STATES)
print(cms[(best.model, best.features)])
