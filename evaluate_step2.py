#!/usr/bin/env python3
"""
evaluate_step2.py  -  does each new feature group improve the 3-class result?

Usage:  python3 evaluate_step2.py dataset_v2.csv
Writes: results_step2.csv  (stage x model: accuracy, macro-F1, F1 per state)

Stages (each adds to the previous one):
  0  Original features (Step 1)
  1  + gaze before / during writing
  2  + look-backs while writing
  3  + passage reading pattern
  4  + pen rates
  5  + per-student normalisation (each feature relative to that student's own average)
All models: 3 classes, gaze + pen + timing, leave-one-participant-out, balanced classes.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

STATES = ["locating", "understanding", "reasoning"]
META = ["participant", "qid", "passage", "subq", "english_level", "label", "label4", "label3",
        "strategy", "confidence", "difficulty", "pen_pointer_type", "pressure_available",
        "use_gaze", "use_pen"]


def groups_of(cols):
    new = lambda p: [c for c in cols if c.startswith(p)]
    base = [c for c in cols if not any(c.startswith(p) for p in ("pre_", "wri_", "lb_", "rd_", "pr_"))]
    return base, new("pre_") + new("wri_"), new("lb_"), new("rd_"), new("pr_")


def models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight="balanced", random_state=0, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=0, n_jobs=-1, eval_metric="mlogloss"),
        "SVM (RBF)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   SVC(class_weight="balanced")),
    }


def per_student_z(df, cols, groups):
    """each feature expressed relative to the same student's mean and SD (no labels used)"""
    z = df[cols].groupby(groups).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    z.columns = [c + "_z" for c in cols]
    return z


def evaluate(X, y, groups):
    logo, out = LeaveOneGroupOut(), {}
    for name in models():
        pred = np.empty(len(y), dtype=int)
        for tr, te in logo.split(X, y, groups):
            m = models()[name]
            if name == "XGBoost":
                m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            else:
                m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out[name] = pred
    return out


def main():
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "dataset_v2.csv")
    y = d.label3.map({s: i for i, s in enumerate(STATES)}).values
    g = d.participant
    cols = [c for c in d.columns if c not in META]
    base, phase, lb, rd, pr = groups_of(cols)
    stages = [("0 Original features", base),
              ("1 + gaze before/during writing", base + phase),
              ("2 + look-backs while writing", base + phase + lb),
              ("3 + passage reading pattern", base + phase + lb + rd),
              ("4 + pen rates", base + phase + lb + rd + pr)]
    all_cols = stages[-1][1]
    Z = per_student_z(d, all_cols, g)
    rows = []
    for name, c in stages + [("5 + per-student normalisation", None)]:
        X = (pd.concat([d[all_cols], Z], axis=1) if c is None else d[c]).astype(float).values
        preds = evaluate(X, y, g)
        for m, p in preds.items():
            f1s = f1_score(y, p, average=None, labels=[0, 1, 2])
            rows.append({"stage": name, "model": m, "n_features": X.shape[1],
                         "accuracy": accuracy_score(y, p), "macro_F1": f1_score(y, p, average="macro"),
                         **{f"F1_{s}": v for s, v in zip(STATES, f1s)}})
        print(f"done: {name}", flush=True)
    r = pd.DataFrame(rows).round(3)
    r.to_csv("results_step2.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
