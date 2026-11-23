#!/usr/bin/env python3
"""
extract_features_v2.py  -  STEP 2: extra features, added to dataset.csv

Usage:
    python3 extract_features_v2.py <data_root> dataset.csv
    e.g. python3 extract_features_v2.py public/userData dataset.csv
Writes dataset_v2.csv  (dataset.csv + new feature columns)

New feature groups (column prefix in brackets):
  A. Gaze before vs during writing      [pre_ / wri_]
       dwell % on passage / question / answer, mean fixation duration, fixation rate
  B. Look-backs while writing           [lb_]
       answer -> passage and answer -> question switches during writing (count, per minute)
  C. Passage reading pattern            [rd_]
       forward-reading saccades %, long jumps %, number of passage visits,
       mean visit length, how much of the passage was looked at (coverage)
  D. Pen rates                          [pr_]
       strokes per second of writing, ink per stroke, share of writing time spent pausing
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import load_gaze, idt_fixations, aoi_of, pane_rects, VIEW_W, VIEW_H, PAUSE_MS

AOIS = ["passage", "question", "answer"]


def fixations_with_aoi(g, t0, t1, panes):
    g = g[(g.timestamp_ms >= t0) & (g.timestamp_ms <= t1)]
    v = g[g.gaze_valid == 1].dropna(subset=["x", "y"])
    v = v[v.x.between(0, VIEW_W) & v.y.between(0, VIEW_H)]
    fx = idt_fixations(v.timestamp_ms.values, v.x.values, v.y.values)
    if len(fx):
        fx["dur"] = fx.end - fx.start
        fx["aoi"] = [aoi_of(a, b, panes) for a, b in zip(fx.x, fx.y)]
    return fx


def phase_feats(fx, prefix, dur_s):
    f = {}
    if fx is None or len(fx) == 0 or dur_s <= 0:
        for a in AOIS:
            f[f"{prefix}dwell_{a}_pct"] = np.nan
        f[f"{prefix}fix_dur_mean_ms"] = np.nan
        f[f"{prefix}fix_rate_per_s"] = np.nan
        return f
    tot = fx.dur.sum()
    for a in AOIS:
        f[f"{prefix}dwell_{a}_pct"] = 100 * fx.dur[fx.aoi == a].sum() / tot
    f[f"{prefix}fix_dur_mean_ms"] = fx.dur.mean()
    f[f"{prefix}fix_rate_per_s"] = len(fx) / dur_s
    return f


def visits(seq, aoi):
    """number of separate visits to an AOI and their lengths (in fixations)"""
    runs, cur = [], 0
    for a in seq:
        if a == aoi:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def extra_features(g, q, panes, pen):
    f = {}
    t0, ts = q["subquestion_opened_ms"], q["answer_submitted_ms"]
    ws = q.get("writing_started_ms") or ts
    fx = fixations_with_aoi(g, t0, ts, panes)

    # A. before vs during writing
    pre = fx[fx.start < ws] if len(fx) else fx
    wri = fx[fx.start >= ws] if len(fx) else fx
    f.update(phase_feats(pre, "pre_", (ws - t0) / 1000))
    f.update(phase_feats(wri, "wri_", (ts - ws) / 1000))

    # B. look-backs while writing
    wmin = (ts - ws) / 60000
    seq = [a for a in (wri.aoi.tolist() if len(wri) else []) if a != "other"]
    sw = list(zip(seq, seq[1:]))
    f["lb_answer_to_passage"] = sum(1 for a, b in sw if a == "answer" and b == "passage")
    f["lb_answer_to_question"] = sum(1 for a, b in sw if a == "answer" and b == "question")
    f["lb_passage_per_min"] = f["lb_answer_to_passage"] / wmin if wmin > 0 else np.nan

    # C. passage reading pattern (whole sub-question)
    allseq = [a for a in (fx.aoi.tolist() if len(fx) else []) if a != "other"]
    pv = visits(allseq, "passage")
    f["rd_passage_visits"] = len(pv)
    f["rd_passage_visit_len"] = float(np.mean(pv)) if pv else 0.0
    f["rd_question_visits"] = len(visits(allseq, "question"))
    p = fx[fx.aoi == "passage"].reset_index(drop=True) if len(fx) else fx
    if len(p) > 2:
        dx, dy = np.diff(p.x), np.diff(p.y)
        amp = np.hypot(dx, dy)
        f["rd_forward_pct"] = 100 * np.mean((dx > 0) & (dx < 200) & (np.abs(dy) < 20))
        f["rd_long_jump_pct"] = 100 * np.mean(amp > 300)
        r = panes["passage"]
        gx = np.clip(((p.x - r["left"]) / (r["right"] - r["left"]) * 6).astype(int), 0, 5)
        gy = np.clip(((p.y - r["top"]) / (r["bottom"] - r["top"]) * 10).astype(int), 0, 9)
        f["rd_coverage_pct"] = 100 * len(set(zip(gx, gy))) / 60
    else:
        f["rd_forward_pct"] = f["rd_long_jump_pct"] = f["rd_coverage_pct"] = np.nan

    # D. pen rates
    pen = pen[pen.pointerType != "touch"]
    downs = pen[pen.event == "pointerdown"].timestamp_ms.values
    ups = pen[pen.event == "pointerup"].timestamp_ms.values
    k = min(len(downs), len(ups))
    wsec = (ts - ws) / 1000
    m = pen[pen.event == "pointermove"]
    ink = np.hypot(np.diff(m.x), np.diff(m.y)).clip(max=200).sum() if len(m) > 1 else 0
    f["pr_strokes_per_s"] = len(downs) / wsec if wsec > 0 else 0
    f["pr_ink_per_stroke"] = ink / len(downs) if len(downs) else 0
    if k > 1 and wsec > 0:
        gaps = downs[1:k] - ups[:k - 1]
        f["pr_pause_share"] = gaps[gaps > PAUSE_MS].sum() / 1000 / wsec
    else:
        f["pr_pause_share"] = 0
    return f


def main():
    root, ds_path = sys.argv[1], sys.argv[2]
    ds = pd.read_csv(ds_path, dtype={"qid": str})
    out = []
    for pid in sorted(ds.participant.unique()):
        base = os.path.join(root, str(pid))
        events = pd.read_csv(os.path.join(base, "markers", "events.csv"), dtype={"qNumber": str})
        for qid in ds[ds.participant == pid].qid:
            q = json.load(open(os.path.join(base, "questionnaire", f"Q{qid}.json"), encoding="utf-8"))
            panes = pane_rects(events, qid)
            g = load_gaze(os.path.join(base, "gaze", f"Q{qid}.jsonl"))
            pen = pd.read_csv(os.path.join(base, "pen", f"Q{qid}.csv"))
            row = {"participant": pid, "qid": qid}
            row.update(extra_features(g, q, panes, pen))
            out.append(row)
        print(f"P{pid} done", flush=True)
    new = pd.DataFrame(out)
    res = ds.merge(new, on=["participant", "qid"], how="left")
    res.to_csv("dataset_v2.csv", index=False)
    print(f"wrote dataset_v2.csv: {res.shape[0]} rows, {new.shape[1] - 2} new features")


if __name__ == "__main__":
    main()
