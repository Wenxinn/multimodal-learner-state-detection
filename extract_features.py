#!/usr/bin/env python3
"""
extract_features.py  -  pilot feature extraction (one row per sub-question)

Usage:  python extract_features.py <data_root> <pid> [<pid> ...]
        e.g. python extract_features.py public/userData 1 47
Writes features.csv in the current folder.

Assumptions (check these!):
  * Experiment ran full-screen on a 1920x1080 display at 125% Windows scaling,
    so the browser viewport is 1536 x 864 CSS px. (Inferred from the logged
    pane rectangles: answer-pane right edge 1497 + 14 col padding + 1 border
    + 12 page padding + ... = 1536.)
  * gaze_x/gaze_y are normalised to the whole display (Tobii display area).
"""
import os, sys, csv, json
import numpy as np
import pandas as pd

VIEW_W, VIEW_H = 1536, 864          # CSS px of the full-screen viewport
IDT_DISP_PX = 40                    # I-DT dispersion threshold (~1 deg at ~60 cm)
IDT_MIN_MS = 100                    # minimum fixation duration
PAUSE_MS = 1000                     # pen pause threshold between strokes


def load_gaze(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line.replace("NaN", "null")))
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    g["x"] = g["gaze_x"].astype(float) * VIEW_W
    g["y"] = g["gaze_y"].astype(float) * VIEW_H
    return g


def idt_fixations(t, x, y):
    """Dispersion-threshold fixation detection on valid samples only."""
    fix = []
    n, i = len(t), 0
    while i < n:
        j = i
        while j < n and t[j] - t[i] < IDT_MIN_MS:
            j += 1
        if j >= n:
            break
        wx, wy = x[i:j + 1], y[i:j + 1]
        if (wx.max() - wx.min()) + (wy.max() - wy.min()) <= IDT_DISP_PX:
            while j + 1 < n:
                wx, wy = x[i:j + 2], y[i:j + 2]
                if (wx.max() - wx.min()) + (wy.max() - wy.min()) > IDT_DISP_PX:
                    break
                if t[j + 1] - t[j] > 100:   # gap in data -> end fixation
                    break
                j += 1
            fix.append((t[i], t[j], x[i:j + 1].mean(), y[i:j + 1].mean()))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(fix, columns=["start", "end", "x", "y"])


def aoi_of(px, py, panes):
    for name, r in panes.items():
        if r["left"] <= px <= r["right"] and r["top"] <= py <= r["bottom"]:
            return name
    return "other"


def pane_rects(events, qid):
    ev = events[(events.qNumber == qid) & (events.event_name == "question_shown")]
    if ev.empty:
        return None
    d = json.loads(ev.iloc[-1]["details"])
    # Extend panes down to the bottom of the grid column so the full
    # column counts (the logged rect is only the inner div's content box).
    # Use full-height columns: the logged rects are only the inner divs,
    # but each grid column runs to the bottom of the screen (minus 12px padding).
    out = {}
    for k, key in [("passage", "passage_pane"), ("question", "question_pane"), ("answer", "answer_pane")]:
        r = dict(d[key]); r["left"] -= 15; r["right"] += 15; r["top"] -= 15; r["bottom"] = VIEW_H - 12
        out[k] = r
    return out


def gaze_features(g, q, panes):
    f = {}
    t0, t_sub = q["subquestion_opened_ms"], q["answer_submitted_ms"]
    ws = q.get("writing_started_ms")
    g = g[(g.timestamp_ms >= t0) & (g.timestamp_ms <= t_sub)]
    f["gaze_samples"] = len(g)
    f["gaze_valid_pct"] = 100 * g.gaze_valid.mean() if len(g) else np.nan
    if ws:
        pre, wri = g[g.timestamp_ms < ws], g[g.timestamp_ms >= ws]
        f["gaze_valid_pct_before_writing"] = 100 * pre.gaze_valid.mean() if len(pre) else np.nan
        f["gaze_valid_pct_while_writing"] = 100 * wri.gaze_valid.mean() if len(wri) else np.nan
    v = g[g.gaze_valid == 1].dropna(subset=["x", "y"])
    v = v[(v.x.between(0, VIEW_W)) & (v.y.between(0, VIEW_H))]
    fx = idt_fixations(v.timestamp_ms.values, v.x.values, v.y.values)
    dur_s = (t_sub - t0) / 1000
    f["fix_count"] = len(fx)
    f["fix_rate_per_s"] = len(fx) / dur_s if dur_s else np.nan
    if len(fx):
        d = fx.end - fx.start
        f["fix_dur_mean_ms"] = d.mean()
        f["fix_dur_sd_ms"] = d.std()
        fx["aoi"] = [aoi_of(a, b, panes) for a, b in zip(fx.x, fx.y)] if panes else "other"
        tot = d.sum()
        for a in ["passage", "question", "answer", "other"]:
            f[f"dwell_{a}_pct"] = 100 * d[fx.aoi == a].sum() / tot
        seq = fx.aoi[fx.aoi != "other"].tolist()
        trans = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
        f["aoi_transitions"] = len(trans)
        f["trans_passage_question"] = sum(1 for a, b in trans if {a, b} == {"passage", "question"})
        f["trans_passage_answer"] = sum(1 for a, b in trans if {a, b} == {"passage", "answer"})
        # regressions inside the passage: saccade going back (left on same line, or up a line)
        p = fx[fx.aoi == "passage"].reset_index(drop=True)
        if len(p) > 1:
            dx, dy = np.diff(p.x), np.diff(p.y)
            reg = ((dx < -40) & (np.abs(dy) < 20)) | (dy < -20)
            f["passage_regression_rate"] = reg.mean()
        else:
            f["passage_regression_rate"] = np.nan
        sacc = np.hypot(np.diff(fx.x), np.diff(fx.y))
        f["sacc_amp_mean_px"] = sacc.mean() if len(sacc) else np.nan
    pup = pd.concat([g.left_pupil, g.right_pupil]).astype(float).dropna()
    f["pupil_mean_mm"] = pup.mean() if len(pup) else np.nan
    return f


def pen_features(p, q):
    f = {}
    p = p[p.pointerType != "touch"]
    f["pen_pointer_type"] = "/".join(sorted(p.pointerType.unique())) if len(p) else "none"
    downs = p[p.event == "pointerdown"].timestamp_ms.values
    ups = p[p.event == "pointerup"].timestamp_ms.values
    f["stroke_count"] = len(downs)
    m = p[p.event == "pointermove"]
    if len(m) > 1:
        f["ink_length_px"] = np.hypot(np.diff(m.x), np.diff(m.y)).clip(max=200).sum()
    else:
        f["ink_length_px"] = 0
    k = min(len(downs), len(ups))
    if k:
        sd = ups[:k] - downs[:k]
        f["stroke_dur_mean_ms"] = sd.mean()
        f["pen_down_time_s"] = sd.sum() / 1000
        gaps = downs[1:k] - ups[:k - 1]
        f["pause_count_gt1s"] = int((gaps > PAUSE_MS).sum())
        f["pause_total_s"] = gaps[gaps > PAUSE_MS].sum() / 1000 if len(gaps) else 0
        f["writing_speed_px_s"] = f["ink_length_px"] / f["pen_down_time_s"] if f["pen_down_time_s"] else np.nan
    pr = p[(p.pointerType == "pen")].pressure.astype(float)
    real_pressure = pr.nunique() >= 5
    f["pressure_available"] = int(real_pressure)
    f["pressure_mean"] = pr[pr > 0].mean() if real_pressure else np.nan
    f["pressure_sd"] = pr[pr > 0].std() if real_pressure else np.nan
    return f


def main():
    root, pids = sys.argv[1], sys.argv[2:]
    out = []
    for pid in pids:
        base = os.path.join(root, pid)
        info = json.load(open(os.path.join(base, "participant_info.json")))
        events = pd.read_csv(os.path.join(base, "markers", "events.csv"), dtype={"qNumber": str})
        for qf in sorted(os.listdir(os.path.join(base, "questionnaire"))):
            qname = qf[:-5]; qid = qname[1:]
            if qid.startswith("0-"):
                continue  # skip practice
            q = json.load(open(os.path.join(base, "questionnaire", qf)))
            row = {"participant": pid, "qid": qid,
                   "passage": int(qid.split("-")[0]), "subq": int(qid.split("-")[1]),
                   "english_level": info.get("englishTest"),
                   "label": q["dominant_state"], "strategy": q["main_strategy"],
                   "confidence": q["confidence_1to5"], "difficulty": q["difficulty_1to5"]}
            t0, ts = q["subquestion_opened_ms"], q["answer_submitted_ms"]
            ws = q.get("writing_started_ms")
            row["total_time_s"] = (ts - t0) / 1000
            row["time_to_first_stroke_s"] = (ws - t0) / 1000 if ws else np.nan
            row["writing_phase_s"] = (ts - ws) / 1000 if ws else 0
            row["clears"] = int(((events.qNumber == qid) & (events.event_name == "canvas_cleared")).sum())
            panes = pane_rects(events, qid)
            g = load_gaze(os.path.join(base, "gaze", f"{qname}.jsonl"))
            row.update(gaze_features(g, q, panes))
            p = pd.read_csv(os.path.join(base, "pen", f"{qname}.csv"))
            row.update(pen_features(p, q))
            out.append(row)
    df = pd.DataFrame(out)
    df.to_csv("features.csv", index=False)
    print(f"wrote features.csv: {df.shape[0]} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
