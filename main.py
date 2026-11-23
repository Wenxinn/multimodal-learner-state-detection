import os
import csv
import time
import json
import base64
import shutil
import threading
from typing import List, Optional, Literal

try:
    import tobii_research as tr
except ImportError:
    tr = None
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Admin Password
# =========================
# Set via environment, e.g.  export ADMIN_PASSWORD="..."
# If unset, admin endpoints reject every request.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

# IMPORTANT: Do NOT mount static files here. Mount at the VERY END.

# =========================
# Helpers
# =========================
def normalize_q(q: Optional[str]) -> Optional[str]:
    """Accept '1' or 'Q1' and return '1'."""
    if q is None:
        return None
    q = str(q).strip()
    if not q:
        return None
    if q.upper().startswith("Q"):
        q = q[1:]
    return q

def ensure_dirs(user: str):
    base = f"./public/userData/{user}"
    os.makedirs(base, exist_ok=True)
    os.makedirs(f"{base}/gaze", exist_ok=True)
    os.makedirs(f"{base}/pen", exist_ok=True)
    os.makedirs(f"{base}/img", exist_ok=True)
    os.makedirs(f"{base}/questionnaire", exist_ok=True)
    os.makedirs(f"{base}/markers", exist_ok=True)

# =========================
# Tobii connect (safe)
# =========================
eyetracker = None
try:
    trackers = tr.find_all_eyetrackers()
    if trackers:
        eyetracker = trackers[0]
        print("Connected eyetracker:", eyetracker)
    else:
        print("No eyetracker found.")
except Exception as e:
    print("Eyetracker init error:", e)

# ---- Startup self-check banner -------------------------------------
# Prints a clear summary so you can confirm the eye tracker before a
# session. The Wacom + Windows Ink status cannot be detected from the
# server (pen pressure only appears in the browser on a real stroke),
# so this just reminds you to verify the setting manually.
print("\n" + "=" * 50)
print("  STARTUP CHECK")
print("=" * 50)
if eyetracker is not None:
    try:
        print(f"  [OK]   Eye tracker connected: {eyetracker.model}")
    except Exception:
        print("  [OK]   Eye tracker connected.")
    print("         Did you calibrate it in Tobii Eye Tracker Manager?")
else:
    print("  [!!]   NO eye tracker found - gaze will NOT be recorded.")
    print("         Plug in the Tobii Pro Nano and restart the server.")
print("  [..]   Check Windows Ink is ON in the Wacom Tablet")
print("         Properties app, in the Mapping section.")
print("=" * 50 + "\n")
# --------------------------------------------------------------------

# =========================
# Global logging state
# =========================
current_user: Optional[str] = None
current_q: Optional[str] = None

gaze_subscribed = False

# file handles
gaze_fp = None         # jsonl
pen_fp = None          # csv
pen_writer = None

# paths (for returning to frontend)
gaze_path = None
pen_path = None

lock = threading.Lock()

PEN_HEADER = [
    "timestamp_ms", "x", "y", "pressure", "event", "pointerType"
]

def stop_logging():
    """Stop gaze subscription + close files safely."""
    global gaze_subscribed, gaze_fp, pen_fp, pen_writer
    global gaze_path, pen_path

    # unsubscribe gaze
    if eyetracker is not None and gaze_subscribed:
        try:
            eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        except Exception as e:
            print("unsubscribe error:", e)
        gaze_subscribed = False

    # close files
    with lock:
        try:
            if gaze_fp:
                gaze_fp.flush()
                gaze_fp.close()
        finally:
            gaze_fp = None
            gaze_path = None

        try:
            if pen_fp:
                pen_fp.flush()
                pen_fp.close()
        finally:
            pen_fp = None
            pen_writer = None
            pen_path = None

def start_logging(user: str, q: str):
    """Open gaze+pen files and subscribe gaze callback."""
    global current_user, current_q
    global gaze_fp, pen_fp, pen_writer
    global gaze_subscribed, gaze_path, pen_path

    # if already logging, stop first (prevents Tobii invalid_operation)
    if current_user is not None and current_q is not None:
        stop_logging()

    current_user, current_q = user, q
    ensure_dirs(user)

    # open gaze jsonl
    gaze_path = f"./public/userData/{user}/gaze/Q{q}.jsonl"
    gaze_fp = open(gaze_path, "w", encoding="utf-8")

    # open pen csv
    pen_path = f"./public/userData/{user}/pen/Q{q}.csv"
    pen_fp = open(pen_path, "w", newline="", encoding="utf-8")
    pen_writer = csv.DictWriter(pen_fp, fieldnames=PEN_HEADER)
    pen_writer.writeheader()
    pen_fp.flush()

    print("[INIT] gaze_path:", gaze_path)
    print("[INIT] pen_path :", pen_path)

    # subscribe gaze
    if eyetracker is not None and not gaze_subscribed:
        try:
            eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
            gaze_subscribed = True
        except Exception as e:
            # if subscribe fails, still keep pen logging
            print("subscribe error:", e)
            gaze_subscribed = False

def validity_is_valid(v) -> bool:
    # Tobii validity: 1 = Valid, 0 = Invalid
    return v == 1

def gaze_data_callback(gaze_data):
    """
    Write one gaze sample as one JSON line to gaze file.
    """
    global gaze_fp
    if gaze_fp is None:
        return

    try:
        ts_ms = int(time.time() * 1000)

        lg = gaze_data.get("left_gaze_point_on_display_area")
        rg = gaze_data.get("right_gaze_point_on_display_area")
        lv = gaze_data.get("left_gaze_point_validity")
        rv = gaze_data.get("right_gaze_point_validity")

        lp = gaze_data.get("left_pupil_diameter")
        rp = gaze_data.get("right_pupil_diameter")

        left_ok = (lg is not None) and validity_is_valid(lv)
        right_ok = (rg is not None) and validity_is_valid(rv)

        gx = gy = None
        gaze_valid = 0

        if left_ok and right_ok:
            gx = (lg[0] + rg[0]) / 2.0
            gy = (lg[1] + rg[1]) / 2.0
            gaze_valid = 1
        elif left_ok:
            gx, gy = lg[0], lg[1]
            gaze_valid = 1
        elif right_ok:
            gx, gy = rg[0], rg[1]
            gaze_valid = 1

        record = {
            "timestamp_ms": ts_ms,
            "gaze_x": gx,
            "gaze_y": gy,
            "gaze_valid": gaze_valid,
            "left_pupil": lp,
            "right_pupil": rp,
            "left_gaze_raw": lg,
            "right_gaze_raw": rg,
            "left_validity": lv,
            "right_validity": rv,
        }

        with lock:
            if gaze_fp is None:
                return
            gaze_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            gaze_fp.flush()

    except Exception as e:
        print("gaze_data_callback error:", e)

# =========================
# Request models
# =========================
class InitParam(BaseModel):
    userNumber: Optional[str] = None
    qNumber: Optional[str] = None
    user: Optional[str] = None
    question: Optional[int] = None
    englishUsage: Optional[str] = None
    englishTest: Optional[str] = None
    wearingGlasses: Optional[str] = None

class PenEvent(BaseModel):
    timestamp_ms: int
    x: float
    y: float
    pressure: float = 0.5
    event: Literal["pointerdown", "pointermove", "pointerup"]
    pointerType: str = "pen"

class PenBatchParam(BaseModel):
    userNumber: str
    qNumber: str
    events: List[PenEvent]

class CanvasParam(BaseModel):
    userNumber: Optional[str] = None
    qNumber: str
    data: List[str]

class QuestionnaireParam(BaseModel):
    participant_id: str = Field(..., min_length=1)
    question_id: str = Field(..., min_length=1) 

    submitted_at: Optional[str] = None
    prompt_text: Optional[str] = None
    answer_image_path: Optional[str] = None
    subquestion_opened_ms: Optional[int] = None
    writing_started_ms: Optional[int] = None
    last_penup_ms: Optional[int] = None
    answer_submitted_ms: Optional[int] = None
    survey_opened_ms: Optional[int] = None
    survey_submitted_ms: Optional[int] = None
    confidence_1to5: Optional[int] = Field(None, ge=1, le=5)
    difficulty_1to5: Optional[int] = Field(None, ge=1, le=5)
    main_strategy: Optional[str] = None
    dominant_state: Optional[str] = None
    perceived_correctness: Optional[Literal["sure", "maybe", "guess"]] = None
    notes: Optional[str] = None

class UIEvent(BaseModel):
    userNumber: str
    qNumber: str
    event_name: str
    timestamp_ms: int
    details: Optional[str] = None

class AdminParam(BaseModel):
    password: str
    userNumber: Optional[str] = None

# =========================
# APIs
# =========================
@app.get("/api/ping")
def ping():
    return {"ok": True}

@app.post("/api/init")
def api_init(param: InitParam):
    user = param.userNumber or param.user
    q = param.qNumber or (str(param.question) if param.question is not None else None)

    user = str(user).strip() if user else None
    q = normalize_q(q)

    if not user or not q:
        return JSONResponse(status_code=422, content={"detail": "Missing userNumber/qNumber (or user/question)."})

    # --- NEW: Save participant info to a JSON file ---
    ensure_dirs(user) # Make sure the participant's folder exists
    info_path = f"./public/userData/{user}/participant_info.json"
    
    # We only save this if the data was actually sent from the browser
    if param.englishUsage and param.englishTest:
        participant_data = {
            "userNumber": user,
            "englishUsage": param.englishUsage,
            "englishTest": param.englishTest,
            "wearingGlasses": param.wearingGlasses,
            "timestamp": time.time()
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(participant_data, f, indent=4)
    # --------------------------------------------------

    start_logging(user, q)
    
    return {
        "status": "started",
        "user": user,
        "q": q,
        "gaze_path": gaze_path,
        "pen_path": pen_path
    }

@app.post("/api/end")
def api_end():
    global current_user, current_q
    u, q = current_user, current_q
    stop_logging()
    current_user, current_q = None, None
    return {"status": "stopped", "user": u, "q": q}

# =========================
# Practice disaster-check (option a: only clear disasters)
# Reads the just-finished practice files (Q0-*) and reports whether
# the pen was all-mouse (Windows Ink off) or gaze was near-zero
# (tracker not working). Borderline values are NOT flagged here;
# those are caught later by check_participant.py.
# =========================
class CheckParam(BaseModel):
    userNumber: str

# Disaster thresholds (intentionally loose - practice gaze is noisy)
PRACTICE_MIN_GAZE_PCT = 50.0   # below this = tracker basically not working

def _check_practice_data(user: str):
    """Return (ok: bool, problems: list[str]) for this user's Q0 practice files.

    Disaster conditions (option a) - the experiment is stopped if ANY apply:
      - No usable eye-gaze data (file missing/empty, or validity under threshold)
      - No usable pen-stroke data (file missing/empty, all-mouse, or no real strokes)
    Both streams must be present and working to continue.
    """
    problems = []
    base = f"./public/userData/{user}"

    # ===== GAZE: is there usable eye-gaze data across the Q0 files? =====
    gaze_dir = f"{base}/gaze"
    gaze_total = 0          # total gaze samples seen
    gaze_valid = 0          # valid gaze samples seen
    if os.path.isdir(gaze_dir):
        for fname in os.listdir(gaze_dir):
            if fname.startswith("Q0-") and fname.endswith(".jsonl"):
                try:
                    with open(os.path.join(gaze_dir, fname), encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            gaze_total += 1
                            rec = json.loads(line.replace("NaN", "null"))
                            if rec.get("gaze_valid") == 1:
                                gaze_valid += 1
                except Exception:
                    continue

    gaze_ok = False
    if gaze_total == 0:
        # No gaze samples at all -> the eye tracker recorded nothing.
        problems.append(
            "No eye-gaze data was recorded. Check the eye tracker is connected "
            "and calibrated. / 視線データが記録されていません。"
            "アイトラッカーの接続とキャリブレーションを確認してください。")
    else:
        pct = 100.0 * gaze_valid / gaze_total
        if pct < PRACTICE_MIN_GAZE_PCT:
            problems.append(
                f"Eye tracking is not working ({pct:.0f}% valid). "
                "Re-seat the participant and re-calibrate. / "
                f"視線追跡(しせんついせき)が機能(きのう)していません（有効(ゆうこう){pct:.0f}%）。"
                "座(すわ)り直(なお)して再(さい)キャリブレーションしてください。")
        else:
            gaze_ok = True

    # ===== PEN: are there usable pen strokes across the Q0 files? =====
    pen_dir = f"{base}/pen"
    pen_rows = 0            # total pen rows seen
    pen_strokes = 0         # pen-down events (actual strokes)
    saw_pen_type = False    # at least one row with pointerType == 'pen'
    saw_any_type = False    # at least one row at all
    if os.path.isdir(pen_dir):
        for fname in os.listdir(pen_dir):
            if fname.startswith("Q0-") and fname.endswith(".csv"):
                try:
                    with open(os.path.join(pen_dir, fname), newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            pen_rows += 1
                            saw_any_type = True
                            if row.get("pointerType") == "pen":
                                saw_pen_type = True
                            if row.get("event") == "pointerdown":
                                pen_strokes += 1
                except Exception:
                    continue

    pen_ok = False
    if pen_rows == 0:
        # No pen rows at all -> nothing was written with the tablet.
        problems.append(
            "No pen-stroke data was recorded. Make sure the participant writes "
            "with the pen during practice. / ペンの筆跡(ひっせき)データが記録(きろく)"
            "されていません。練習中(れんしゅうちゅう)にペンで書(か)いてもらってください。")
    elif saw_any_type and not saw_pen_type:
        # All input was 'mouse' -> Windows Ink off / drawn with the mouse.
        problems.append(
            "Pen pressure is not being recorded (input is 'mouse'). "
            "Turn ON Windows Ink in Wacom Tablet Properties (Mapping). / "
            "ペン筆圧が記録されていません（入力が"
            "「マウス」）。WacomのプロパティでWindows Inkをオンにしてください。")
    elif pen_strokes == 0:
        # Rows exist but no real strokes -> nothing actually drawn.
        problems.append(
            "No pen strokes detected (nothing was written). / "
            "ペンの筆跡が検出されません（何も書かれて"
            "いません）。")
    else:
        pen_ok = True

    return (gaze_ok and pen_ok), problems

@app.post("/api/check_practice")
def api_check_practice(param: CheckParam):
    user = str(param.userNumber).strip()
    if not user:
        return JSONResponse(status_code=422, content={"detail": "Missing userNumber"})
    ok, problems = _check_practice_data(user)
    return {"ok": ok, "problems": problems}

@app.post("/api/pen")
def api_pen(batch: PenBatchParam):
    global pen_writer, pen_fp

    if pen_writer is None or pen_fp is None:
        return JSONResponse(status_code=409, content={"detail": "Not logging. Call /api/init first."})

    bq = normalize_q(batch.qNumber)
    if str(batch.userNumber) != str(current_user) or str(bq) != str(current_q):
        return JSONResponse(status_code=409, content={"detail": f"Session mismatch. Current={current_user},Q{current_q}"})

    with lock:
        for ev in batch.events:
            pen_writer.writerow({
                "timestamp_ms": ev.timestamp_ms,
                "x": ev.x,
                "y": ev.y,
                "pressure": ev.pressure,
                "event": ev.event,
                "pointerType": ev.pointerType,
            })
        pen_fp.flush()

    return {"status": "ok", "count": len(batch.events)}

@app.post("/api/canvasimg")
def api_canvas_img(param: CanvasParam):
    if current_user is None:
        return JSONResponse(status_code=409, content={"detail": "No active user session."})

    q = normalize_q(param.qNumber)
    if not q:
        return JSONResponse(status_code=422, content={"detail": "Missing qNumber."})

    out_path = f"./public/userData/{current_user}/img/Q{q}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = base64.b64decode(param.data[0].split(",")[1].encode("utf-8"))
    with open(out_path, "wb") as f:
        f.write(img)

    # Return relative path for web
    return {"status": "saved", "path": f"/userData/{current_user}/img/Q{q}.png"}

@app.post("/api/questionnaire")
def save_questionnaire(param: QuestionnaireParam):
    user = param.participant_id
    q = param.question_id

    q_dir = f"./public/userData/{user}/questionnaire"
    os.makedirs(q_dir, exist_ok=True)

    file_path = os.path.join(q_dir, f"Q{q}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(param.dict(), f, ensure_ascii=False, indent=2)

    return {"status": "saved", "user": user, "q": q, "path": file_path}

@app.post("/api/event")
def log_ui_event(ev: UIEvent):
    ensure_dirs(ev.userNumber)
    
    path = f"./public/userData/{ev.userNumber}/markers/events.csv"
    file_exists = os.path.isfile(path)
    
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_ms", "qNumber", "event_name", "details"])
        writer.writerow([ev.timestamp_ms, ev.qNumber, ev.event_name, ev.details or ""])
        
    return {"status": "saved"}

# =========================
# Admin APIs
# =========================
@app.post("/api/admin/verify")
def verify_admin(param: AdminParam):
    if param.password != ADMIN_PASSWORD:
        return JSONResponse(status_code=401, content={"detail": "Incorrect password"})
    return {"status": "ok"}

@app.post("/api/admin/download_questionnaires")
def download_questionnaires(param: AdminParam):
    if param.password != ADMIN_PASSWORD:
        return JSONResponse(status_code=401, content={"detail": "Incorrect password"})
    
    if not param.userNumber:
        return JSONResponse(status_code=422, content={"detail": "Missing userNumber"})

    q_dir = f"./public/userData/{param.userNumber}/questionnaire"
    if not os.path.exists(q_dir):
        return JSONResponse(status_code=404, content={"detail": "No data found for this participant."})

    all_data = {}
    for fname in os.listdir(q_dir):
        if fname.endswith(".json"):
            q_id = fname.replace(".json", "")
            with open(os.path.join(q_dir, fname), "r", encoding="utf-8") as f:
                all_data[q_id] = json.load(f)

    return all_data

@app.post("/api/admin/clear_user_data")
def clear_user_data(param: AdminParam):
    if param.password != ADMIN_PASSWORD:
        return JSONResponse(status_code=401, content={"detail": "Incorrect password"})
    
    if not param.userNumber:
        return JSONResponse(status_code=422, content={"detail": "Missing userNumber"})

    base = f"./public/userData/{param.userNumber}"
    if os.path.exists(base):
        shutil.rmtree(base)
        return {"status": "All data cleared successfully."}
    
    return JSONResponse(status_code=404, content={"detail": "No data found to clear."})

# =========================
# Static files LAST (prevents 405 on POST /api/*)
# =========================
os.makedirs("./public", exist_ok=True)
app.mount("/", StaticFiles(directory="public", html=True), name="static")