#!/usr/bin/env python3
"""Resumable version of evaluate_v3.main(): runs one (setup, model, seed) at a time and
appends to results_v3_runs.csv. Usage: python3 v3_chunked.py dataset_v3.csv [n_seeds] [budget_s]"""
import os, sys, time
import pandas as pd
from evaluate_v3 import run, feature_sets, META, STATES

d = pd.read_csv(sys.argv[1])
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
fs = feature_sets([c for c in d.columns if c not in META])
setups = [("flat-v2", "v2", False), ("flat-v3", "v3", False), ("hier-v3", "v3", True),
          ("hier-v3-gaze", "gaze", True), ("hier-v3-pen", "pen", True)]
models = ["Random Forest", "XGBoost", "LightGBM"]
done = pd.read_csv("results_v3_runs.csv") if os.path.exists("results_v3_runs.csv") else pd.DataFrame()
have = set(zip(done.setup, done.model, done.seed)) if len(done) else set()
preds = (pd.read_csv("results_v3_predictions.csv") if os.path.exists("results_v3_predictions.csv")
         else d[["participant", "qid", "label3"]].copy())
start = time.time()
for setup, fset, hier in setups:
    for model in models:
        for seed in range(n_seeds):
            if (setup, model, seed) in have:
                continue
            if time.time() - start > budget:
                print(f"{len(done)}/{len(setups) * 3 * n_seeds} runs finished; run again"); sys.exit()
            out, p = run(d, fs[fset], model, seed, hier)
            done = pd.concat([done, pd.DataFrame([{"setup": setup, "model": model, "seed": seed,
                                                   "n_features": len(fs[fset]), **out}])], ignore_index=True)
            done.to_csv("results_v3_runs.csv", index=False)
            if seed == 0:
                preds[f"{setup}|{model}"] = [STATES[i] for i in p]
                preds.to_csv("results_v3_predictions.csv", index=False)
            print(f"{setup:13s} {model:13s} seed {seed}: macro-F1 {out['macro_F1']:.3f}  "
                  f"F1 und {out['F1_understanding']:.3f}  ({time.time() - start:.0f}s)", flush=True)
num = done.drop(columns=["seed"]).groupby(["setup", "model"], sort=False)
num.mean().round(3).join(num.std().round(3).add_suffix("_sd")).to_csv("results_v3_summary.csv")
print("ALL DONE -> results_v3_summary.csv")
