#!/usr/bin/env python3
"""Run the window evaluation of windows_v3.py one (setup, model, seed) at a time, so it
can be resumed. Same models and metrics as windows_v3.evaluate().
Usage: python3 windows_chunked.py windows_30s.csv [n_seeds] [time_budget_s]
Appends to windows_runs.csv; when all runs exist, writes results_windows_summary.csv."""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from evaluate_v3 import make, fit, metrics, STATES
warnings.filterwarnings("ignore")

w = pd.read_csv(sys.argv[1], dtype={"qid": str})
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
meta = {"participant", "qid", "window", "label3"}
cols = [c for c in w.columns if c not in meta]
X = w[cols].astype(float).values
y = w.label3.map({s: i for i, s in enumerate(STATES)}).values
g = w.participant.values
qkey = (w.participant.astype(str) + "|" + w.qid.astype(str)).values
qtrue = pd.Series(y, index=qkey).groupby(level=0, sort=False).first()

done = pd.read_csv("windows_runs.csv") if os.path.exists("windows_runs.csv") else pd.DataFrame()
todo = [(s, m, k) for s in ["flat", "hierarchy"] for m in ["Random Forest", "XGBoost", "LightGBM"]
        for k in range(n_seeds)]
if len(done):
    have = set(zip(done.setup, done.model, done.seed))
    todo = [t for t in todo if t not in have]
start = time.time()
for setup, model, seed in todo:
    if time.time() - start > budget:
        break
    pred = np.empty(len(y), int)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if setup == "flat":
            pred[te] = fit(make(model, seed, False), model, X[tr], y[tr]).predict(X[te])
        else:
            s1 = fit(make(model, seed, True), model, X[tr], (y[tr] != 0).astype(int))
            nl = y[tr] != 0
            s2 = fit(make(model, seed, True), model, X[tr][nl], (y[tr][nl] == 2).astype(int))
            other, p2 = s1.predict(X[te]).astype(bool), s2.predict(X[te])
            pred[te] = np.where(~other, 0, np.where(p2 == 1, 2, 1))
    wm, wpr, wrc, wf1 = metrics(y, pred, [0, 1, 2], "win_")
    vote = pd.Series(pred, index=qkey).groupby(level=0, sort=False).agg(
        lambda s: s.value_counts().sort_index().idxmax())
    qm, qpr, qrc, qf1 = metrics(qtrue.values, vote.loc[qtrue.index].values, [0, 1, 2], "q_")
    row = {"setup": setup, "model": model, "seed": seed, **wm, **qm}
    for i, s in enumerate(STATES):
        row[f"win_F1_{s}"], row[f"q_F1_{s}"] = wf1[i], qf1[i]
        row[f"win_R_{s}"], row[f"q_R_{s}"] = wrc[i], qrc[i]
    if seed == 0:
        pd.DataFrame({"participant": w.participant, "qid": w.qid, "window": w.window,
                      "true": w.label3, "pred": [STATES[i] for i in pred]}).to_csv(
            f"windows_pred_{setup}_{model.replace(' ', '')}.csv", index=False)
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv("windows_runs.csv", index=False)
    print(f"{setup:9s} {model:13s} seed {seed}: window macro-F1 {wm['win_macro_F1']:.3f}  "
          f"question macro-F1 {qm['q_macro_F1']:.3f}  ({time.time() - start:.0f}s)", flush=True)

if len(done) == 2 * 3 * n_seeds:
    grp = done.drop(columns="seed").groupby(["setup", "model"], sort=False)
    grp.mean().round(3).join(grp.std().round(3).add_suffix("_sd")).to_csv("results_windows_summary.csv")
    print("ALL DONE -> results_windows_summary.csv")
else:
    print(f"{len(done)}/{2 * 3 * n_seeds} runs finished; run again to continue")
