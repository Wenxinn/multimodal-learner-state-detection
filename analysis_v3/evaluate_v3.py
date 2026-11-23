#!/usr/bin/env python3
"""
evaluate_v3.py  -  Random Forest / XGBoost / LightGBM, flat 3-class vs hierarchy,
                   leave-one-participant-out, averaged over several random seeds.

Usage:  python3 evaluate_v3.py dataset_v3.csv [n_seeds]
Writes: results_v3_summary.csv   (mean and SD over seeds, every metric)
        results_v3_runs.csv      (one row per seed)
        results_v3_predictions.csv (predictions of seed 0, for confusion matrices)

Setups
  flat-v2      : one 3-class model, features up to step 2 (53 features)
  flat-v3      : one 3-class model, + Miyazaki-style features (98 features)
  hier-v3      : stage 1 locating vs other  ->  stage 2 understanding vs reasoning
  hier-v3-gaze / hier-v3-pen : hierarchy with gaze-only / pen-only (+timing) features
Metrics: accuracy, balanced accuracy, macro precision / recall / F1, weighted F1,
         Cohen's kappa, per-state precision / recall / F1, and for the hierarchy the
         stage-1 and stage-2 scores separately.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             f1_score, cohen_kappa_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

STATES = ["locating", "understanding", "reasoning"]
META = {"participant", "qid", "passage", "subq", "english_level", "label", "label4", "label3", "strategy",
        "confidence", "difficulty", "pen_pointer_type", "pressure_available", "use_gaze", "use_pen",
        "gaze_samples", "gaze_valid_pct", "gaze_valid_pct_before_writing", "gaze_valid_pct_while_writing"}
TIMING = ["total_time_s", "time_to_first_stroke_s", "writing_phase_s"]
GAZE_KEYS = ["fix", "dwell", "trans", "aoi", "regression", "sacc", "pupil", "blink", "lb_", "rd_", "m_gaze"]


def feature_sets(cols):
    v2 = [c for c in cols if not c.startswith("m_")]
    v3 = list(cols)
    gaze = [c for c in v3 if c not in TIMING and any(k in c for k in GAZE_KEYS)]
    pen = [c for c in v3 if c not in TIMING and c not in gaze]
    return {"v2": v2, "v3": v3, "gaze": gaze + TIMING, "pen": pen + TIMING}


def make(name, seed, binary):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                             eval_metric="logloss" if binary else "mlogloss")
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                          random_state=seed, n_jobs=-1, verbose=-1)


def fit(m, name, X, y):
    if name == "XGBoost":
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        m.fit(X, y)
    return m


def metrics(y, p, labels, prefix=""):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    out = {f"{prefix}accuracy": accuracy_score(y, p),
           f"{prefix}balanced_accuracy": balanced_accuracy_score(y, p),
           f"{prefix}macro_precision": pr.mean(), f"{prefix}macro_recall": rc.mean(),
           f"{prefix}macro_F1": f1.mean(),
           f"{prefix}weighted_F1": f1_score(y, p, labels=labels, average="weighted", zero_division=0),
           f"{prefix}kappa": cohen_kappa_score(y, p)}
    return out, pr, rc, f1


def run(d, cols, model, seed, hierarchical):
    X = d[cols].astype(float).values
    y = d.label3.map({s: i for i, s in enumerate(STATES)}).values
    g = d.participant.values
    pred = np.empty(len(y), int)
    s2_true, s2_pred = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if not hierarchical:
            pred[te] = fit(make(model, seed, False), model, X[tr], y[tr]).predict(X[te])
            continue
        # stage 1: locating (0) vs other (1)
        s1 = fit(make(model, seed, True), model, X[tr], (y[tr] != 0).astype(int))
        other = s1.predict(X[te]).astype(bool)
        # stage 2: understanding (0) vs reasoning (1), trained on non-locating rows only
        nl = y[tr] != 0
        s2 = fit(make(model, seed, True), model, X[tr][nl], (y[tr][nl] == 2).astype(int))
        p2 = s2.predict(X[te])
        pred[te] = np.where(~other, 0, np.where(p2 == 1, 2, 1))
        tnl = y[te] != 0                       # stage-2 scores on true non-locating rows
        s2_true.extend(y[te][tnl]); s2_pred.extend(np.where(p2[tnl] == 1, 2, 1))
    out, pr, rc, f1 = metrics(y, pred, [0, 1, 2])
    for i, s in enumerate(STATES):
        out[f"P_{s}"], out[f"R_{s}"], out[f"F1_{s}"] = pr[i], rc[i], f1[i]
    # stage 1 (locating vs other) is evaluated for flat models too, by merging their predictions
    o1, *_ = metrics((y != 0).astype(int), (pred != 0).astype(int), [0, 1], "stage1_")
    out.update(o1)
    if hierarchical:
        o2, *_ = metrics(np.array(s2_true), np.array(s2_pred), [1, 2], "stage2_")
        out.update(o2)
    return out, pred


def main():
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "dataset_v3.csv")
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    fs = feature_sets([c for c in d.columns if c not in META])
    print({k: len(v) for k, v in fs.items()}, flush=True)
    setups = [("flat-v2", "v2", False), ("flat-v3", "v3", False), ("hier-v3", "v3", True),
              ("hier-v3-gaze", "gaze", True), ("hier-v3-pen", "pen", True)]
    rows, preds = [], d[["participant", "qid", "label3"]].copy()
    for setup, fset, hier in setups:
        for model in ["Random Forest", "XGBoost", "LightGBM"]:
            for seed in range(n_seeds):
                out, p = run(d, fs[fset], model, seed, hier)
                rows.append({"setup": setup, "model": model, "seed": seed, "n_features": len(fs[fset]), **out})
                if seed == 0:
                    preds[f"{setup}|{model}"] = [STATES[i] for i in p]
            r = pd.DataFrame([x for x in rows if x["setup"] == setup and x["model"] == model])
            print(f"{setup:13s} {model:13s} macro-F1 {r.macro_F1.mean():.3f} ± {r.macro_F1.std():.3f}  "
                  f"acc {r.accuracy.mean():.3f}  F1 und {r.F1_understanding.mean():.3f}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv("results_v3_runs.csv", index=False)
    preds.to_csv("results_v3_predictions.csv", index=False)
    num = runs.drop(columns=["seed"]).groupby(["setup", "model"], sort=False)
    summ = num.mean().round(3).join(num.std().round(3).add_suffix("_sd"))
    summ.to_csv("results_v3_summary.csv")
    print("done")


if __name__ == "__main__":
    main()
