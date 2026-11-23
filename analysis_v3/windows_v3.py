#!/usr/bin/env python3
"""
windows_v3.py  -  cut every sub-question into 30-second windows and evaluate.

Usage:  python3 windows_v3.py <data_root> dataset_v3.csv [window_s] [n_seeds]
Writes: windows_v3.csv            (one row per window, features + inherited label)
        results_windows_summary.csv (window level and question level, mean/SD over seeds)

* Each sub-question (question shown -> submit) is cut into windows of window_s seconds.
  A last piece shorter than 10 s is merged into the previous window.
* Every window inherits the self-reported label of its sub-question.
* Features per window: AOI dwell / fixations / transitions (step 1), Miyazaki-style
  mean/SD/max gaze and pen features (v3), and elapsed time since the question appeared.
  (Whole-question features such as total time are NOT used, since a window cannot know them.)
* Evaluation: leave-one-participant-out; window level, and question level by majority vote
  of the windows of each sub-question.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_features as ef
import extract_features_v3 as v3
from evaluate_v3 import make, fit, metrics, STATES, GAZE_KEYS
from sklearn.model_selection import LeaveOneGroupOut
warnings.filterwarnings("ignore")
MIN_LAST_S = 10


def window_bounds(t0, t1, w_ms):
    edges = list(range(int(t0), int(t1), w_ms)) + [int(t1)]
    if len(edges) > 2 and edges[-1] - edges[-2] < MIN_LAST_S * 1000:
        edges.pop(-2)
    return list(zip(edges[:-1], edges[1:]))


def build(root, ds, w_s):
    rows = []
    for pid in sorted(ds.participant.unique()):
        base = os.path.join(root, str(pid))
        lo, hi = v3.participant_pupil_range(base)
        events = pd.read_csv(os.path.join(base, "markers", "events.csv"), dtype={"qNumber": str})
        for _, r in ds[ds.participant == pid].iterrows():
            qid = r.qid
            q = json.load(open(os.path.join(base, "questionnaire", f"Q{qid}.json"), encoding="utf-8"))
            t0, t1 = q["subquestion_opened_ms"], q["answer_submitted_ms"]
            panes = ef.pane_rects(events, qid)
            g1 = ef.load_gaze(os.path.join(base, "gaze", f"Q{qid}.jsonl"))
            g3 = v3.load_gaze(os.path.join(base, "gaze", f"Q{qid}.jsonl"))
            p = pd.read_csv(os.path.join(base, "pen", f"Q{qid}.csv"))
            for k, (a, b) in enumerate(window_bounds(t0, t1, w_s * 1000)):
                qw = {"subquestion_opened_ms": a, "answer_submitted_ms": b, "writing_started_ms": None}
                f = {"participant": pid, "qid": qid, "window": k, "label3": r.label3,
                     "elapsed_s": (a - t0) / 1000, "window_s": (b - a) / 1000}
                base_g = ef.gaze_features(g1, qw, panes)
                f.update({c: base_g.get(c, np.nan) for c in
                          ["fix_count", "fix_rate_per_s", "fix_dur_mean_ms", "fix_dur_sd_ms",
                           "dwell_passage_pct", "dwell_question_pct", "dwell_answer_pct", "dwell_other_pct",
                           "aoi_transitions", "trans_passage_question", "trans_passage_answer",
                           "passage_regression_rate", "sacc_amp_mean_px"]})
                pw = p[(p.timestamp_ms >= a) & (p.timestamp_ms <= b)]
                base_p = ef.pen_features(pw, qw)
                f.update({c: base_p.get(c, np.nan) for c in
                          ["stroke_count", "ink_length_px", "stroke_dur_mean_ms", "pen_down_time_s",
                           "pause_count_gt1s", "pause_total_s"]})
                f.update(v3.gaze_features(g3, a, b, lo, hi))
                f.update(v3.pen_features(p, a, b, has_pressure=pid >= 8))
                rows.append(f)
        print(f"P{pid} done", flush=True)
    return pd.DataFrame(rows)


def evaluate(w, n_seeds):
    meta = {"participant", "qid", "window", "label3"}
    cols = [c for c in w.columns if c not in meta]
    X = w[cols].astype(float).values
    y = w.label3.map({s: i for i, s in enumerate(STATES)}).values
    g = w.participant.values
    qkey = w.participant.astype(str) + "|" + w.qid.astype(str)
    qtrue = w.groupby(qkey, sort=False).label3.first().map({s: i for i, s in enumerate(STATES)})
    rows = []
    for hier in [False, True]:
        for model in ["Random Forest", "XGBoost", "LightGBM"]:
            for seed in range(n_seeds):
                pred = np.empty(len(y), int)
                for tr, te in LeaveOneGroupOut().split(X, y, g):
                    if not hier:
                        pred[te] = fit(make(model, seed, False), model, X[tr], y[tr]).predict(X[te])
                    else:
                        s1 = fit(make(model, seed, True), model, X[tr], (y[tr] != 0).astype(int))
                        nl = y[tr] != 0
                        s2 = fit(make(model, seed, True), model, X[tr][nl], (y[tr][nl] == 2).astype(int))
                        other, p2 = s1.predict(X[te]).astype(bool), s2.predict(X[te])
                        pred[te] = np.where(~other, 0, np.where(p2 == 1, 2, 1))
                wm, *_ = metrics(y, pred, [0, 1, 2], "win_")
                # question level: majority vote; ties -> class with the most windows overall
                vote = pd.Series(pred, index=qkey.values).groupby(level=0, sort=False).agg(
                    lambda s: s.value_counts().sort_index().idxmax())
                qm, pr, rc, f1 = metrics(qtrue.values, vote.loc[qtrue.index].values, [0, 1, 2], "q_")
                rows.append({"setup": "hierarchy" if hier else "flat", "model": model, "seed": seed,
                             **wm, **qm, **{f"q_F1_{s}": f1[i] for i, s in enumerate(STATES)}})
            r = pd.DataFrame(rows[-n_seeds:])
            print(f"{'hier' if hier else 'flat':5s} {model:13s} window macro-F1 {r.win_macro_F1.mean():.3f}  "
                  f"question macro-F1 {r.q_macro_F1.mean():.3f}", flush=True)
    runs = pd.DataFrame(rows)
    grp = runs.drop(columns="seed").groupby(["setup", "model"], sort=False)
    grp.mean().round(3).join(grp.std().round(3).add_suffix("_sd")).to_csv("results_windows_summary.csv")
    return runs


def main():
    root, ds_path = sys.argv[1], sys.argv[2]
    w_s = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    ds = pd.read_csv(ds_path, dtype={"qid": str})
    if os.path.exists(f"windows_{w_s}s.csv"):
        w = pd.read_csv(f"windows_{w_s}s.csv", dtype={"qid": str})
    else:
        w = build(root, ds, w_s)
        w.to_csv(f"windows_{w_s}s.csv", index=False)
    print(f"{len(w)} windows:", dict(w.label3.value_counts()), flush=True)
    evaluate(w, n_seeds)
    print("done")


if __name__ == "__main__":
    main()
