#!/usr/bin/env python3
"""
extract_features_v3.py  -  Miyazaki-style detailed features (mean / SD / max of raw signals)

Usage:
    python3 extract_features_v3.py <data_root> dataset_v2.csv
Writes dataset_v3.csv  (dataset_v2.csv + new columns starting with "m_")

Following Miyazaki (2025, Waseda Yamana lab), each raw signal is summarised by its
mean, standard deviation and maximum inside the analysis window:

  GAZE
    m_gaze_speed        gaze movement speed (px/s), consecutive valid samples <= 50 ms apart
    m_pupil             pupil diameter (mm, mean of both eyes)
    m_pupil_change      |change in pupil diameter| per second (mm/s)
    m_pupil_lr_diff     |left - right pupil diameter| (mm)
    m_saccade_dur       saccade duration (ms); I-VT filter, 30 deg/s
    m_fixation_dur      fixation duration (ms); non-saccade runs >= 200 ms
    m_blink_dur         blink duration (ms); invalid-data gaps of 75-425 ms
    + counts per second of saccades, fixations, blinks; gaze position SD (px)
  PEN
    m_pen_speed         writing speed inside strokes (px/s)
    m_pen_accel         |change in writing speed| per second (px/s^2)
    m_pressure          pen pressure (P8-P47 only; empty for P1-P7)
    m_pressure_change   |change in pressure| per second
    m_isi               inter-stroke interval: pen-up -> next pen-down (ms)
    m_isd               inter-stroke distance: pen-up position -> next pen-down position (px)
    + writing position SD (px), strokes per second

Cleaning (as in Miyazaki): samples within 50 ms of a blink are removed before pupil
features; pupil values outside each participant's Q1-1.5*IQR .. Q3+1.5*IQR are removed.

Assumptions (state these in the thesis): screen 1920 px wide = ~34.5 cm (15.6-inch laptop),
viewing distance 60 cm -> 30 deg/s = 2*60*tan(15 deg) cm/s = ~1790 px/s.
"""
import os, sys, json
import numpy as np
import pandas as pd

SCREEN_W, SCREEN_H = 1920, 1080          # physical px (gaze is normalised 0-1 on the display)
IVT_PX_PER_S = 2 * 60 * np.tan(np.radians(15)) / 34.5 * SCREEN_W   # ~1790 px/s
FIX_MIN_MS, BLINK_MIN, BLINK_MAX, BLINK_PAD = 200, 75, 425, 50
MAX_DT_MS = 50


def stats(prefix, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_sd": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_mean": x.mean(), f"{prefix}_sd": x.std(), f"{prefix}_max": x.max()}


def load_gaze(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line.replace("NaN", "null"))
                rows.append((r["timestamp_ms"], r.get("gaze_valid", 0), r.get("gaze_x"), r.get("gaze_y"),
                             r.get("left_pupil"), r.get("right_pupil")))
    g = pd.DataFrame(rows, columns=["t", "valid", "gx", "gy", "lp", "rp"]).astype(float)
    g["x"], g["y"] = g.gx * SCREEN_W, g.gy * SCREEN_H
    return g.sort_values("t").reset_index(drop=True)


def runs(mask):
    """start/end index pairs of consecutive True runs"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def gaze_features(g, t0, t1, pupil_lo, pupil_hi):
    f = {}
    w = g[(g.t >= t0) & (g.t <= t1)].reset_index(drop=True)
    dur_s = max((t1 - t0) / 1000, 1e-6)
    if len(w) < 3:
        return f
    t = w.t.values
    valid = (w.valid.values == 1) & np.isfinite(w.x.values) & np.isfinite(w.y.values)

    # blinks: invalid gaps of 75-425 ms
    blinks, near_blink = [], np.zeros(len(w), bool)
    for a, b in runs(~valid):
        start = t[a - 1] if a > 0 else t[a]
        end = t[b + 1] if b + 1 < len(w) else t[b]
        dur = end - start
        if BLINK_MIN <= dur < BLINK_MAX:
            blinks.append(dur)
            near_blink |= (t >= start - BLINK_PAD) & (t <= end + BLINK_PAD)
    f["m_blink_rate"] = len(blinks) / dur_s
    f.update(stats("m_blink_dur", blinks))

    v = w[valid].reset_index(drop=True)
    if len(v) < 3:
        return f
    dt = np.diff(v.t.values)
    dist = np.hypot(np.diff(v.x.values), np.diff(v.y.values))
    ok = (dt > 0) & (dt <= MAX_DT_MS)
    speed = np.where(ok, dist / np.where(dt > 0, dt, 1) * 1000, np.nan)
    f.update(stats("m_gaze_speed", speed[ok]))
    f["m_gaze_pos_sd"] = np.sqrt(v.x.var() + v.y.var())

    # I-VT: saccade samples = speed above threshold; fixation = other runs >= 200 ms
    is_sacc = np.concatenate([[False], np.nan_to_num(speed) > IVT_PX_PER_S])
    sacc = [v.t[b] - v.t[a] + (dt[a - 1] if a > 0 else 0) for a, b in runs(is_sacc)]
    fix = [v.t[b] - v.t[a] for a, b in runs(~is_sacc) if v.t[b] - v.t[a] >= FIX_MIN_MS]
    f["m_saccade_rate"] = len(sacc) / dur_s
    f["m_fixation_rate"] = len(fix) / dur_s
    f.update(stats("m_saccade_dur", sacc))
    f.update(stats("m_fixation_dur", fix))

    # pupils: drop samples near blinks and outside the participant's IQR range
    pw = w[valid & ~near_blink]
    lp, rp = pw.lp.values, pw.rp.values
    both = np.isfinite(lp) & np.isfinite(rp)
    pupil = np.where(both, (lp + rp) / 2, np.where(np.isfinite(lp), lp, rp))
    keep = np.isfinite(pupil) & (pupil >= pupil_lo) & (pupil <= pupil_hi)
    f.update(stats("m_pupil", pupil[keep]))
    f.update(stats("m_pupil_lr_diff", np.abs(lp - rp)[both & keep]))
    tp = pw.t.values[keep]
    pk = pupil[keep]
    if len(pk) > 2:
        dtp = np.diff(tp)
        okp = (dtp > 0) & (dtp <= MAX_DT_MS)
        f.update(stats("m_pupil_change", (np.abs(np.diff(pk)) / np.where(dtp > 0, dtp, 1) * 1000)[okp]))
    return f


def pen_features(p, t0, t1, has_pressure):
    f = {}
    w = p[(p.timestamp_ms >= t0) & (p.timestamp_ms <= t1)]
    w = w[w.pointerType != "touch"].reset_index(drop=True)
    dur_s = max((t1 - t0) / 1000, 1e-6)
    downs = w.index[w.event == "pointerdown"].tolist()
    ups = w.index[w.event == "pointerup"].tolist()
    f["m_strokes_per_s"] = len(downs) / dur_s
    if len(w) == 0:
        return f
    # speed / acceleration inside strokes (stroke id increases at each pointerdown)
    sid = (w.event == "pointerdown").cumsum()
    speeds, accels, pchg = [], [], []
    for _, s in w.groupby(sid):
        s = s[s.event != "pointerup"] if len(s) > 2 else s
        tt = s.timestamp_ms.values.astype(float)
        dtt = np.diff(tt)
        ok = dtt > 0
        if ok.sum() < 1:
            continue
        sp = np.hypot(np.diff(s.x.values), np.diff(s.y.values))[ok] / dtt[ok] * 1000
        speeds.extend(sp)
        if len(sp) > 1:
            accels.extend(np.abs(np.diff(sp)) / (dtt[ok][1:] / 1000))
        if has_pressure:
            pchg.extend(np.abs(np.diff(s.pressure.values.astype(float)))[ok] / dtt[ok] * 1000)
    f.update(stats("m_pen_speed", speeds))
    f.update(stats("m_pen_accel", accels))
    if has_pressure:
        pr = w.pressure.astype(float)
        f.update(stats("m_pressure", pr[pr > 0]))
        f.update(stats("m_pressure_change", pchg))
    else:
        f.update(stats("m_pressure", []))
        f.update(stats("m_pressure_change", []))
    # inter-stroke interval and distance
    isi, isd = [], []
    for u in ups:
        nxt = [d for d in downs if d > u]
        if nxt:
            d = nxt[0]
            isi.append(w.timestamp_ms[d] - w.timestamp_ms[u])
            isd.append(np.hypot(w.x[d] - w.x[u], w.y[d] - w.y[u]))
    f.update(stats("m_isi", isi))
    f.update(stats("m_isd", isd))
    f["m_pen_pos_sd"] = np.sqrt(w.x.var() + w.y.var()) if len(w) > 1 else np.nan
    return f


def participant_pupil_range(base):
    vals = []
    for fn in os.listdir(os.path.join(base, "gaze")):
        if fn.startswith("Q0-"):
            continue
        g = load_gaze(os.path.join(base, "gaze", fn))
        g = g[g.valid == 1]
        vals.append(np.nanmean(g[["lp", "rp"]].values, axis=1))
    v = np.concatenate(vals) if vals else np.array([])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return -np.inf, np.inf
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def main():
    root, ds_path = sys.argv[1], sys.argv[2]
    ds = pd.read_csv(ds_path, dtype={"qid": str})
    out = []
    for pid in sorted(ds.participant.unique()):
        base = os.path.join(root, str(pid))
        lo, hi = participant_pupil_range(base)
        for qid in ds[ds.participant == pid].qid:
            q = json.load(open(os.path.join(base, "questionnaire", f"Q{qid}.json"), encoding="utf-8"))
            t0, t1 = q["subquestion_opened_ms"], q["answer_submitted_ms"]
            g = load_gaze(os.path.join(base, "gaze", f"Q{qid}.jsonl"))
            p = pd.read_csv(os.path.join(base, "pen", f"Q{qid}.csv"))
            row = {"participant": pid, "qid": qid}
            row.update(gaze_features(g, t0, t1, lo, hi))
            row.update(pen_features(p, t0, t1, has_pressure=pid >= 8))
            out.append(row)
        print(f"P{pid} done", flush=True)
    new = pd.DataFrame(out)
    res = ds.merge(new, on=["participant", "qid"], how="left")
    res.to_csv("dataset_v3.csv", index=False)
    print(f"wrote dataset_v3.csv: {len(res)} rows, {new.shape[1] - 2} new features")


if __name__ == "__main__":
    main()
