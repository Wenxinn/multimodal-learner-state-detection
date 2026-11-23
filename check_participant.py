#!/usr/bin/env python3
# ============================================================
#  check_participant.py   (v3 - keep-list edition)
#  Quality-check ONE participant's data after their session.
#
#  HOW TO RUN:
#      python check_participant.py 11
#
#  where "11" is the participant number (folder under public/userData/).
#
#  Each sub-question gets a verdict:
#    PASS    = fully usable.
#    WARNING = usable, but check it (low-ish gaze, or a few stray
#              mouse rows that can simply be filtered out).
#    FATAL   = not usable (no pressure at all / all-mouse pen /
#              very low gaze / empty file).
#
#  At the end you get a KEEP-LIST: exactly which sub-questions are
#  usable for the multimodal (gaze+pen) model.
#
#  Practice round (Q0) is skipped by default.
#  Add  --include-practice  to include it.
# ============================================================

import os
import sys
import csv
import json
import glob

# ---- Settings you can change if needed -------------------------------
DATA_ROOT = os.path.join("public", "userData")   # where participant folders live

MIN_GAZE_VALID_PCT    = 80.0   # FATAL if valid gaze below this %
WARN_GAZE_VALID_PCT   = 90.0   # WARNING if below this (but above MIN)
MIN_PRESSURE_DISTINCT = 5      # FATAL if pressure has fewer distinct values
MIN_ONSCREEN_PCT      = 85.0   # WARNING if fewer valid gaze points on-screen
EXPECTED_RATE_HZ      = 60     # Tobii Pro Nano nominal rate

# A pen file with a FEW stray mouse rows is salvageable (just filter them).
# If more than this fraction is mouse, the pen tablet was really off -> FATAL.
MAX_MOUSE_FRACTION    = 0.15   # up to 15% stray mouse rows = WARNING, else FATAL
# ----------------------------------------------------------------------

FATAL, WARN = "FATAL", "WARN"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line.replace("NaN", "null")))
    return rows


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pen(path):
    """Returns (issues, summary). issues = list of (level, message)."""
    issues = []
    rows = load_csv(path)
    if not rows:
        return [(FATAL, "pen file is EMPTY")], "no rows"

    ptypes = [r.get("pointerType", "") for r in rows]
    ptype_set = set(ptypes)
    pressures = []
    for r in rows:
        try:
            pressures.append(float(r.get("pressure", "")))
        except (ValueError, TypeError):
            pass  # skip rows with missing/blank pressure
    if not pressures:
        return [(FATAL, "no readable pressure values in pen file")], "unreadable pressure"
    distinct_p = len(set(pressures))
    strokes = sum(1 for r in rows if r.get("event") == "pointerdown")

    n_mouse = sum(1 for p in ptypes if p != "pen")
    mouse_frac = n_mouse / len(rows)

    # --- Pen-type logic ---
    if ptype_set == {"pen"}:
        pass  # perfect
    elif "pen" not in ptype_set:
        # entirely mouse -> the tablet/Windows Ink was off -> unusable
        issues.append((FATAL,
            f"pointerType is all {sorted(ptype_set)} (no 'pen' at all) "
            "-> Windows Ink OFF / drawn with mouse"))
    elif mouse_frac <= MAX_MOUSE_FRACTION:
        # mostly pen with a few stray mouse rows -> SALVAGEABLE
        issues.append((WARN,
            f"{n_mouse}/{len(rows)} rows ({mouse_frac*100:.1f}%) are 'mouse' "
            "-> minor; filter out the mouse rows, keep the pen rows"))
    else:
        # too much mouse to trust
        issues.append((FATAL,
            f"{n_mouse}/{len(rows)} rows ({mouse_frac*100:.1f}%) are 'mouse' "
            "-> too much non-pen input to trust"))

    # --- Pressure logic (only fatal if it never varied at all) ---
    if distinct_p < MIN_PRESSURE_DISTINCT:
        issues.append((FATAL,
            f"pressure has only {distinct_p} distinct value(s) "
            "-> pressure not recorded (check Windows Ink)"))

    if strokes == 0:
        issues.append((WARN, "no pen strokes at all (participant wrote nothing?)"))

    summary = (f"{len(rows)} rows, {strokes} strokes, "
               f"pointerType={sorted(ptype_set)}, "
               f"pressure {min(pressures):.3f}-{max(pressures):.3f} "
               f"({distinct_p} distinct)")
    return issues, summary


def check_gaze(path):
    """Returns (issues, summary, (t0,tN))."""
    issues = []
    rows = load_jsonl(path)
    if not rows:
        return [(FATAL, "gaze file is EMPTY")], "no rows", None

    valid_rows = [r for r in rows if r.get("gaze_valid") == 1]
    valid = len(valid_rows)
    pct = 100.0 * valid / len(rows)
    t0 = rows[0]["timestamp_ms"]
    tN = rows[-1]["timestamp_ms"]
    dur = (tN - t0) / 1000.0
    rate = (len(rows) - 1) / dur if dur > 0 else 0

    if pct < MIN_GAZE_VALID_PCT:
        issues.append((FATAL, f"gaze valid only {pct:.1f}% (below {MIN_GAZE_VALID_PCT}%) "
                              "-> calibration/seating likely failed"))
    elif pct < WARN_GAZE_VALID_PCT:
        issues.append((WARN, f"gaze valid {pct:.1f}% (below {WARN_GAZE_VALID_PCT}%) "
                             "-> a bit low, check seating"))

    if rate and abs(rate - EXPECTED_RATE_HZ) > 15:
        issues.append((WARN, f"sampling rate ~{rate:.0f} Hz (expected ~{EXPECTED_RATE_HZ}) "
                             "-> possible dropped samples"))

    onscreen = sum(1 for r in valid_rows
                   if r.get("gaze_x") is not None
                   and 0 <= r["gaze_x"] <= 1 and 0 <= r["gaze_y"] <= 1)
    on_pct = 100.0 * onscreen / valid if valid else 0
    if valid and on_pct < MIN_ONSCREEN_PCT:
        issues.append((WARN, f"only {on_pct:.1f}% of valid gaze is on-screen "
                             "-> participant may be looking away"))

    summary = (f"{len(rows)} samples, {pct:.1f}% valid, "
               f"{on_pct:.0f}% on-screen, {dur:.1f}s, ~{rate:.0f} Hz")
    return issues, summary, (t0, tN)


def check_questionnaire(path):
    issues = []
    q = json.load(open(path, encoding="utf-8"))
    for k in ["dominant_state", "confidence_1to5", "difficulty_1to5"]:
        if q.get(k) in (None, ""):
            issues.append((WARN, f"questionnaire missing '{k}'"))

    op = q.get("subquestion_opened_ms")
    ws = q.get("writing_started_ms")
    sub = q.get("answer_submitted_ms")
    if op and sub and sub <= op:
        issues.append((WARN, "submit time is not after open time -> timing glitch"))
    if ws and sub and sub < ws:
        issues.append((WARN, "submit time is before writing started -> timing glitch"))

    summary = (f"state={q.get('dominant_state')}, "
               f"strategy={q.get('main_strategy')}, "
               f"conf={q.get('confidence_1to5')}, "
               f"diff={q.get('difficulty_1to5')}")
    return issues, summary


def check_image(path):
    if not os.path.isfile(path):
        return [(WARN, "answer image missing (canvas capture may have failed)")]
    size = os.path.getsize(path)
    if size < 1000:
        return [(WARN, f"answer image very small ({size} bytes) -> maybe blank")]
    return []


def check_events(events_path, qid):
    if not os.path.isfile(events_path):
        return [(WARN, "events.csv missing -> timeline markers unavailable")]
    wanted = {"question_shown", "writing_started", "submit_pressed",
              "survey_submitted"}
    seen = set()
    with open(events_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("qNumber")) == qid:
                seen.add(row.get("event_name"))
    missing = wanted - seen
    if missing:
        return [(WARN, f"missing event marker(s) for {qid}: {sorted(missing)}")]
    return []


def main():
    raw = sys.argv[1:]
    include_practice = "--include-practice" in raw
    args = [a for a in raw if not a.startswith("--")]

    if not args:
        print("Usage: python check_participant.py <participant_number> [--include-practice]")
        print("Example: python check_participant.py 11")
        sys.exit(1)

    pid = args[0].strip()
    base = os.path.join(DATA_ROOT, pid)

    if not os.path.isdir(base):
        print(f"ERROR: folder not found: {base}")
        print("Run this from your project folder (the one containing 'public').")
        sys.exit(1)

    print("=" * 62)
    print(f"  QUALITY CHECK  -  Participant {pid}")
    print("=" * 62)

    info_path = os.path.join(base, "participant_info.json")
    if os.path.isfile(info_path):
        info = json.load(open(info_path, encoding="utf-8"))
        print(f"Info: English={info.get('englishTest')}, "
              f"usage={info.get('englishUsage')}, "
              f"vision={info.get('wearingGlasses')}")
    else:
        print("  [WARN] participant_info.json missing")

    events_path = os.path.join(base, "markers", "events.csv")

    gaze_files = sorted(glob.glob(os.path.join(base, "gaze", "Q*.jsonl")))
    if not gaze_files:
        print("\nNo gaze files found. Nothing to check.")
        sys.exit(1)

    keep_list = []     # PASS or WARNING -> usable
    drop_list = []     # FATAL -> not usable
    n_pass = n_warn = n_fatal = 0

    # For label-variety check (collected from each real sub-question's survey)
    states_seen = []          # list of (qname, state) so we can tell which
                              # sub-question each label came from
    confs_seen = []
    diffs_seen = []

    for gpath in gaze_files:
        qname = os.path.basename(gpath).replace(".jsonl", "")  # Q1-4
        qid = qname[1:]                                        # 1-4

        if qid.startswith("0-") and not include_practice:
            continue

        print("\n" + "-" * 62)
        print(f"  {qname}")
        print("-" * 62)

        issues = []

        gi, gsumm, gwindow = check_gaze(gpath)
        print(f"Gaze  : {gsumm}")
        issues += gi

        ppath = os.path.join(base, "pen", f"{qname}.csv")
        if os.path.isfile(ppath):
            pi, psumm = check_pen(ppath)
            print(f"Pen   : {psumm}")
            issues += pi
            if gwindow:
                prows = load_csv(ppath)
                if prows:
                    pts = [int(r["timestamp_ms"]) for r in prows]
                    if not (min(pts) >= gwindow[0] and max(pts) <= gwindow[1]):
                        issues.append((WARN, "pen timestamps fall OUTSIDE the gaze "
                                             "window -> stream sync issue"))
        else:
            issues.append((FATAL, f"pen file missing ({qname}.csv)"))
            print("Pen   : MISSING")

        qpath = os.path.join(base, "questionnaire", f"{qname}.json")
        if os.path.isfile(qpath):
            qi, qsumm = check_questionnaire(qpath)
            print(f"Survey: {qsumm}")
            issues += qi
            # collect label values for the end-of-report variety check
            try:
                _q = json.load(open(qpath, encoding="utf-8"))
                if _q.get("dominant_state") not in (None, ""):
                    states_seen.append((qname, _q.get("dominant_state")))
                if _q.get("confidence_1to5") not in (None, ""):
                    confs_seen.append(_q.get("confidence_1to5"))
                if _q.get("difficulty_1to5") not in (None, ""):
                    diffs_seen.append(_q.get("difficulty_1to5"))
            except Exception:
                pass
        else:
            issues.append((WARN, f"questionnaire missing ({qname}.json)"))
            print("Survey: MISSING")

        issues += check_image(os.path.join(base, "img", f"{qname}.png"))
        issues += check_events(events_path, qid)

        fatals = [m for lvl, m in issues if lvl == FATAL]
        warns = [m for lvl, m in issues if lvl == WARN]
        if fatals:
            print("  >>> FATAL:")
            for m in fatals:
                print("        - " + m)
            for m in warns:
                print("        - (warn) " + m)
            n_fatal += 1
            drop_list.append(qname)
        elif warns:
            print("  >>> WARNING (usable):")
            for m in warns:
                print("        - " + m)
            n_warn += 1
            keep_list.append(qname)
        else:
            print("  >>> PASS")
            n_pass += 1
            keep_list.append(qname)

    # ---- Summary + KEEP-LIST ----
    print("\n" + "=" * 62)
    total = n_pass + n_warn + n_fatal
    print(f"  Sub-questions checked : {total}")
    print(f"  PASS {n_pass}  /  WARNING {n_warn}  /  FATAL {n_fatal}")
    print("-" * 62)
    usable = n_pass + n_warn
    print(f"  USABLE sub-questions  : {usable}  (PASS + WARNING)")
    if keep_list:
        print("    KEEP : " + ", ".join(keep_list))
    if drop_list:
        print(f"  DROP (do not use)     : {n_fatal}")
        print("    DROP : " + ", ".join(drop_list))
    print("-" * 62)
    if usable == total:
        print(f"  RESULT: ALL {total} sub-questions usable. ")
    elif usable == 0:
        print("  RESULT: NOTHING usable from this participant.")
    else:
        print(f"  RESULT: Keep the {usable} usable sub-questions above; "
              f"drop the {n_fatal} FATAL one(s).")
    print("=" * 62)

    # ---- Label-variety check (does NOT affect keep/drop) ----
    # Surfaces possible "straight-lining": same answer on every question,
    # which weakens the ground-truth labels even when sensors are perfect.
    #
    # NOTE: from the revised design, sub-questions 1-5 of each passage are
    # LOCATING items, and sub-questions 6-8 are REASONING (6,7) and
    # ANSWER-CONSTRUCTION (8) items. So a locating-heavy overall mix is now
    # EXPECTED and is NOT itself a problem. The meaningful red flag is a
    # participant who still reports 'locating_information' on the 6/7/8
    # items -- those questions cannot be answered by locating, so that
    # signals disengagement or a failed question.
    if states_seen:
        from collections import Counter

        def subq_index(qname):
            # qname like 'Q1-6' -> 6 ; returns None if unparseable
            try:
                return int(qname.split("-", 1)[1])
            except (IndexError, ValueError):
                return None

        all_states = [s for (_q, s) in states_seen]
        n = len(all_states)
        uniq_states = len(set(all_states))

        # split into locating-design vs reasoning/construction-design items
        reasoning_items = [(q, s) for (q, s) in states_seen
                           if (subq_index(q) or 0) >= 6]

        print("\n" + "-" * 62)
        print("  LABEL VARIETY (ground-truth quality, not sensor quality)")
        print("-" * 62)
        dist = ", ".join(f"{k}={v}" for k, v in Counter(all_states).items())
        print(f"  dominant_state : {uniq_states} unique / {n}  ({dist})")
        if confs_seen:
            print(f"  confidence     : {len(set(confs_seen))} unique "
                  f"(min {min(confs_seen)}, max {max(confs_seen)})")
        if diffs_seen:
            print(f"  difficulty     : {len(set(diffs_seen))} unique "
                  f"(min {min(diffs_seen)}, max {max(diffs_seen)})")

        flags = []
        # state-aware flag: locating reported on questions that CAN'T be located
        if reasoning_items:
            loc_on_reasoning = [q for (q, s) in reasoning_items
                                if s == "locating_information"]
            print(f"  reasoning/construction items (Q*-6/7/8) : {len(reasoning_items)} answered, "
                  f"{len(loc_on_reasoning)} marked 'locating_information'")
            if loc_on_reasoning:
                flags.append(
                    "reported 'locating_information' on inference/construction "
                    f"items {sorted(loc_on_reasoning)} -- these cannot be answered "
                    "by locating, so the participant may not have engaged with them")
        else:
            print("  reasoning/construction items : none found "
                  "(participant may predate the revised questions)")

        # keep the original whole-survey checks as secondary signals
        if uniq_states <= 1:
            flags.append("only ONE state used on every question")
        if diffs_seen and len(set(diffs_seen)) <= 1:
            flags.append("difficulty identical on every question")
        if confs_seen and len(set(confs_seen)) <= 1:
            flags.append("confidence identical on every question")

        if flags:
            print("  [!] Low label variety: " + "; ".join(flags) + ".")
            print("      Sensors may be fine, but the self-reported labels")
            print("      may be straight-lined. Worth noting for this participant.")
        print("-" * 62)


if __name__ == "__main__":
    main()