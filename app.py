from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import threading, time, os, sqlite3, cv2, numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)
app.secret_key = 'secretkey123'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['EMBEDDINGS_FOLDER'] = os.path.join('embeddings')
app.config['IMAGE_UPLOAD'] = os.path.join('static', 'images')
app.config['VIDEO_UPLOAD'] = os.path.join('static', 'videos')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EMBEDDINGS_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOAD'], exist_ok=True)
os.makedirs(app.config['VIDEO_UPLOAD'], exist_ok=True)

ADMIN_EMAIL = "medolin.502236@sxcce.edu.in"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = "sherin.502253@sxcce.edu.in"
EMAIL_PASSWORD = "hkoi kfct dpyj wvlg"
LOG_FILE = 'detection_logs.txt'
DB = 'security_system.db'

video_path = None
video_alert = ""
authorized_db = {}
uploaded_video_path = None
last_logged_offender = None
last_weapon_log_time = 0
last_offender_log_time = {}
VIDEO_LOG_COOLDOWN = 5
last_email_time = 0
EMAIL_COOLDOWN = 30

# ================================================================
# FIX 1: Shared state for the background face-recognition thread
# ================================================================
face_thread_lock = threading.Lock()
face_input_frame = None        # frame waiting to be processed
face_result = {                # last result from face thread
    "label": None,
    "box": None,
    "is_authorized": False,
    "is_offender": False,
}
face_thread_busy = False       # prevent queuing multiple jobs


def send_email_alert(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USERNAME
    msg['To'] = ADMIN_EMAIL
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed: {e}")


def log_detection(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {event}\n")


def trigger_email_alert(message):
    global last_email_time
    current_time = time.time()
    if current_time - last_email_time > EMAIL_COOLDOWN:
        send_email_alert("🚨 Smart Surveillance Alert", message)
        last_email_time = current_time


# ---------------- DB Setup ----------------
def init_db():
    with sqlite3.connect(DB) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                role TEXT CHECK(role IN ('user','admin')) NOT NULL
            )
        ''')
init_db()


# ---------------- Model Setup ----------------
weapon_model = YOLO('models/weapon_best.pt')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
offender_db = {}
camera_on = True
current_alert = None
lock = threading.Lock()


def load_offender_db():
    db = {}
    for fn in os.listdir(app.config['EMBEDDINGS_FOLDER']):
        if fn.endswith(".npy"):
            name = fn.split('_')[0]
            emb = np.load(os.path.join(app.config['EMBEDDINGS_FOLDER'], fn))
            if name not in db:
                db[name] = []
            db[name].append(emb)
    return db

offender_db = load_offender_db()


def load_authorized_db():
    db = {}
    for fn in os.listdir("authorized_embeddings"):
        if fn.endswith(".npy"):
            name = fn.split("_")[0]
            emb = np.load(os.path.join("authorized_embeddings", fn))
            if name not in db:
                db[name] = []
            db[name].append(emb)
    return db

authorized_db = load_authorized_db()


# ================================================================
# WEAPON VALIDATION FILTERS
# Fixes false positives: pens, scissors, remotes, syringes etc.
# Three filters run in sequence — all must pass to trigger alert.
# ================================================================

# Per-class minimum confidence (pens/scissors score 0.50–0.68 typically)
CLASS_CONF_THRESHOLD = {
    "gun":   0.72,
    "knife": 0.75,   # raised — knife class is more commonly confused
}

# Expected width:height ratio per class
# Gun held sideways = wide  (1.4 – 5.0)
# Knife held upright = tall (0.1 – 0.85)
CLASS_ASPECT_RATIO = {
    "gun":   (1.4, 5.0),
    "knife": (0.1, 0.85),
}

# Colors that real knives/guns NEVER are.
# Real knives: silver/gray/black blade.
# Real guns:   black/dark-gray/dark-brown.
FORBIDDEN_COLORS = [
    {"name": "blue",   "hue_lo": 100, "hue_hi": 130, "sat_min": 80},
    {"name": "green",  "hue_lo": 40,  "hue_hi": 85,  "sat_min": 100},
    {"name": "red_hi", "hue_lo": 160, "hue_hi": 180,  "sat_min": 120},
    {"name": "yellow", "hue_lo": 20,  "hue_hi": 35,  "sat_min": 130},
    {"name": "purple", "hue_lo": 130, "hue_hi": 155, "sat_min": 90},
]


def get_top_region(roi_hsv, top_fraction=0.4):
    """
    Return only the TOP portion of the bounding box.
    Why: the hand/skin grip at the bottom was averaging down saturation,
    hiding the object's true color (e.g. the blue pen case).
    We check the top 40% which shows the object body, not the hand.
    """
    h = roi_hsv.shape[0]
    top_rows = max(1, int(h * top_fraction))
    return roi_hsv[:top_rows, :, :]


def is_forbidden_color(roi_hsv):
    """
    Returns (True, color_name) if the object's top region
    matches any color that real weapons never are.
    """
    top = get_top_region(roi_hsv)
    for color in FORBIDDEN_COLORS:
        lo   = np.array([color["hue_lo"], color["sat_min"], 40])
        hi   = np.array([color["hue_hi"], 255, 255])
        mask = cv2.inRange(top, lo, hi)
        match_ratio = np.count_nonzero(mask) / max(top.size // 3, 1)
        if match_ratio > 0.25:
            return True, color["name"]
    return False, None


def is_valid_weapon_class(cls_name, conf, x1, y1, x2, y2):
    """
    Filter 1: class name + confidence + aspect ratio.
    Returns (is_valid, label, reason).
    """
    cls_name = cls_name.lower().strip()
    if cls_name not in CLASS_CONF_THRESHOLD:
        return False, cls_name, f"Unknown class '{cls_name}'"
    min_conf = CLASS_CONF_THRESHOLD[cls_name]
    if conf < min_conf:
        return False, cls_name, f"Conf {conf:.2f} < {min_conf} for {cls_name}"
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = w / h
    lo, hi = CLASS_ASPECT_RATIO[cls_name]
    if not (lo <= ratio <= hi):
        return False, cls_name, f"Aspect {ratio:.2f} invalid for {cls_name} (expected {lo}-{hi})"
    return True, cls_name, "OK"


def is_likely_real_weapon(frame, x1, y1, x2, y2):
    """
    Filter 2: color + size + edge sharpness.
    Analyses TOP of bounding box to avoid hand-skin color averaging.
    Returns (is_real, reason_string).
    """
    h_frame, w_frame = frame.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w_frame, x2); y2c = min(h_frame, y2)
    roi = frame[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return False, "empty ROI"

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    real_score = 0
    toy_score  = 0
    reasons    = []

    # Signal 1: Forbidden color — checks top 40% only
    forbidden, color_name = is_forbidden_color(roi_hsv)
    if forbidden:
        toy_score += 2   # Strong signal — double penalty
        reasons.append(f"forbidden color:{color_name}")
    else:
        top = get_top_region(roi_hsv)
        avg_sat = float(np.mean(top[:, :, 1]))
        if avg_sat > 100:
            toy_score += 1
            reasons.append(f"high sat top={avg_sat:.0f}")
        else:
            real_score += 1
            reasons.append(f"dark top sat={avg_sat:.0f}")

    # Signal 2: Size ratio
    size_ratio = ((x2c - x1c) * (y2c - y1c)) / (w_frame * h_frame)
    if size_ratio < 0.02:
        toy_score += 1
        reasons.append(f"tiny {size_ratio*100:.1f}%")
    else:
        real_score += 1
        reasons.append(f"size ok {size_ratio*100:.1f}%")

    # Signal 3: Edge sharpness
    gray_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    if sharpness < 30:
        toy_score += 1
        reasons.append(f"soft edges {sharpness:.1f}")
    else:
        real_score += 1
        reasons.append(f"sharp {sharpness:.1f}")

    is_real = real_score > toy_score
    verdict = "REAL" if is_real else "FILTERED"
    print(f"[WeaponFilter] {verdict} real={real_score} toy={toy_score} | {', '.join(reasons)}")
    return is_real, f"{', '.join(reasons)}"


def process_weapon_detections(frame, results):
    """
    Single entry point called in all 3 places (live/video/image).
    Returns (weapon_detected, annotated_frame).
    """
    weapon_detected = False
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf     = r.conf.item()
        cls_id   = int(r.cls.item())
        cls_name = weapon_model.names.get(cls_id, "unknown")

        # Filter 1
        valid, label, reason1 = is_valid_weapon_class(cls_name, conf, x1, y1, x2, y2)
        if not valid:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
            cv2.putText(frame, f"Filtered:{label}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (120, 120, 120), 1)
            print(f"[ClassFilter] Rejected — {reason1}")
            continue

        # Filter 2
        is_real, reason2 = is_likely_real_weapon(frame, x1, y1, x2, y2)
        if is_real:
            weapon_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label.capitalize()} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(frame, f"Not weapon",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (180, 180, 180), 1)

    return weapon_detected, frame


# ================================================================
# FIX 2: Background worker — face recognition runs here, NOT in
#         the streaming loop, so the camera never freezes.
# ================================================================
def face_recognition_worker():
    """
    Runs in a daemon thread forever.
    Picks up face_input_frame when available, processes it,
    writes result back to face_result.
    """
    global face_input_frame, face_result, face_thread_busy

    while True:
        # Wait until there is a frame to process
        with face_thread_lock:
            frame_to_process = face_input_frame
            face_input_frame = None

        if frame_to_process is None:
            time.sleep(0.01)   # idle sleep — very light on CPU
            continue

        # ---- Run DeepFace (slow, but now off the main thread) ----
        try:
            gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(60, 60)
            )

            best_result = {
                "label": None,
                "box": None,
                "is_authorized": False,
                "is_offender": False,
            }

            for (x, y, w, h) in faces:
                face_crop = frame_to_process[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (160, 160))

                rep = DeepFace.represent(
                    face_crop,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False
                )

                if not rep:
                    continue

                emb = np.array(rep[0]['embedding'])

                # Check authorized
                best_auth_score = 0
                best_auth = None
                for name, emb_list in authorized_db.items():
                    for db_emb in emb_list:
                        sim = np.dot(emb, db_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(db_emb)
                        )
                        if sim > best_auth_score:
                            best_auth_score = sim
                            best_auth = name

                if best_auth_score > 0.82:
                    best_result = {
                        "label": f"Authorized: {best_auth}",
                        "box": (x, y, w, h),
                        "is_authorized": True,
                        "is_offender": False,
                    }
                    break   # authorized found — no need to check offenders

                # Check offender
                best_score = 0
                best_match = None
                for name, emb_list in offender_db.items():
                    for db_emb in emb_list:
                        sim = np.dot(emb, db_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(db_emb)
                        )
                        if sim > best_score:
                            best_score = sim
                            best_match = name

                if best_score > 0.82:
                    best_result = {
                        "label": best_match,
                        "box": (x, y, w, h),
                        "is_authorized": False,
                        "is_offender": True,
                    }
                    log_detection(f"Offender Detected: {best_match}")
                    trigger_email_alert(f"🚨 Offender detected: {best_match}")

            # Write result (thread-safe)
            with face_thread_lock:
                face_result.update(best_result)

        except Exception as e:
            print(f"[FaceWorker] Error: {e}")

        finally:
            with face_thread_lock:
                face_thread_busy = False


# Start the background face thread once at startup
_face_worker_thread = threading.Thread(
    target=face_recognition_worker, daemon=True
)
_face_worker_thread.start()


# ================================================================
# FIX 3: Rewritten gen_frames() — streams smoothly, never blocks
# ================================================================
def gen_frames():
    global current_alert, face_input_frame, face_thread_busy

    # FIX 4: Remove CAP_DSHOW — it causes hangs on some Windows setups.
    #         If you're on Linux/Mac this is already correct.
    #         If you NEED CAP_DSHOW (e.g., certain USB cams on Windows),
    #         add it back: cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)

    # FIX 5: Set buffer size to 1 so we always get the LATEST frame,
    #         not a stale frame that was queued while we were processing.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    # Local copy of face result — updated from background thread
    last_face_result = {
        "label": None, "box": None,
        "is_authorized": False, "is_offender": False
    }

    while True:
        if not camera_on:
            off = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(off, "Camera Off", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buf = cv2.imencode('.jpg', off)
            time.sleep(0.1)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            # Camera read failed — try to reconnect once
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame = cv2.resize(frame, (640, 480))

        siren_triggered = False
        alert_msgs = []
        weapon_detected = False

        # -------- WEAPON DETECTION --------
        results = weapon_model.predict(frame, conf=0.5, verbose=False)
        weapon_detected, frame = process_weapon_detections(frame, results)

        # -------- FACE RECOGNITION (offloaded to background thread) --------
        # FIX 7: Every 10 frames, send a frame to the background thread.
        #         We never WAIT for the result — we use the LAST known result.
        if frame_count % 10 == 0:
            with face_thread_lock:
                if not face_thread_busy:
                    face_input_frame = frame.copy()
                    face_thread_busy = True

        # Read latest face result (non-blocking)
        with face_thread_lock:
            last_face_result = dict(face_result)

        # Draw the last known face result onto the current frame
        if last_face_result["box"] is not None:
            x, y, w, h = last_face_result["box"]
            label = last_face_result["label"]

            if last_face_result["is_authorized"]:
                color = (255, 255, 0)  # Yellow for authorized
            elif last_face_result["is_offender"]:
                color = (0, 255, 0)    # Green box for offender
                siren_triggered = True
                alert_msgs.append(f"{label} Detected")
            else:
                color = (200, 200, 200)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # -------- THREAT DECISION --------
        authorized_person = last_face_result["is_authorized"]
        if weapon_detected and not authorized_person:
            siren_triggered = True
            alert_msgs.append("Weapon Threat Detected")
            log_detection("Weapon Threat Detected")
            trigger_email_alert("🚨 Weapon detected by surveillance system.")

        # -------- SIREN OVERLAY --------
        if siren_triggered:
            cv2.putText(frame, "⚠ THREAT DETECTED!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        with lock:
            current_alert = ", ".join(alert_msgs) if alert_msgs else ""

        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

    cap.release()


# ================================================================
# gen_video_prediction() — same threading fix applied
# ================================================================
def gen_video_prediction():
    global uploaded_video_path, current_alert
    global last_weapon_log_time, last_offender_log_time

    if uploaded_video_path is None:
        return

    cap = cv2.VideoCapture(uploaded_video_path)
    frame_count = 0
    last_face_result = {
        "label": None, "box": None,
        "is_authorized": False, "is_offender": False
    }
    local_face_busy = False
    local_face_input = [None]
    local_face_result = [dict(last_face_result)]
    local_lock = threading.Lock()

    def local_face_worker():
        while True:
            with local_lock:
                f = local_face_input[0]
                local_face_input[0] = None
            if f is None:
                time.sleep(0.01)
                continue
            try:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(60,60))
                result = {"label": None, "box": None,
                          "is_authorized": False, "is_offender": False}
                for (x, y, w, h) in faces:
                    crop = cv2.resize(f[y:y+h, x:x+w], (160, 160))
                    rep = DeepFace.represent(crop, model_name='Facenet',
                                            detector_backend='opencv',
                                            enforce_detection=False)
                    if not rep:
                        continue
                    emb = np.array(rep[0]['embedding'])
                    # Authorized check
                    ba_score, ba_name = 0, None
                    for name, emb_list in authorized_db.items():
                        for db_emb in emb_list:
                            s = np.dot(emb, db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                            if s > ba_score:
                                ba_score, ba_name = s, name
                    if ba_score > 0.82:
                        result = {"label": f"Authorized: {ba_name}", "box": (x,y,w,h),
                                  "is_authorized": True, "is_offender": False}
                        break
                    # Offender check
                    bo_score, bo_name = 0, None
                    for name, emb_list in offender_db.items():
                        for db_emb in emb_list:
                            s = np.dot(emb, db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                            if s > bo_score:
                                bo_score, bo_name = s, name
                    if bo_score > 0.82:
                        result = {"label": bo_name, "box": (x,y,w,h),
                                  "is_authorized": False, "is_offender": True}
                        current_time = time.time()
                        if bo_name not in last_offender_log_time:
                            last_offender_log_time[bo_name] = 0
                        if current_time - last_offender_log_time[bo_name] > VIDEO_LOG_COOLDOWN:
                            log_detection(f"Offender Detected: {bo_name}")
                            trigger_email_alert(f"🚨 Offender detected: {bo_name}")
                            last_offender_log_time[bo_name] = current_time
                with local_lock:
                    local_face_result[0] = result
            except Exception as e:
                print(f"[VideoFaceWorker] {e}")

    t = threading.Thread(target=local_face_worker, daemon=True)
    t.start()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        siren_triggered = False
        alert_msgs = []
        weapon_detected = False

        results = weapon_model.predict(frame, conf=0.5, verbose=False)
        weapon_detected, frame = process_weapon_detections(frame, results)

        if frame_count % 10 == 0:
            with local_lock:
                if local_face_input[0] is None:
                    local_face_input[0] = frame.copy()

        with local_lock:
            fr = dict(local_face_result[0])

        if fr["box"] is not None:
            x, y, w, h = fr["box"]
            color = (255,255,0) if fr["is_authorized"] else (0,255,0)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            if fr["label"]:
                cv2.putText(frame, fr["label"], (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if fr["is_offender"]:
                siren_triggered = True
                alert_msgs.append(f"{fr['label']} Detected")

        if weapon_detected and not fr["is_authorized"]:
            siren_triggered = True
            alert_msgs.append("Weapon Threat Detected")
            current_time = time.time()
            if current_time - last_weapon_log_time > VIDEO_LOG_COOLDOWN:
                log_detection("Weapon Threat Detected")
                trigger_email_alert("🚨 Weapon detected.")
                last_weapon_log_time = current_time

        if siren_triggered:
            cv2.putText(frame, "⚠ THREAT DETECTED!", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        with lock:
            current_alert = ", ".join(alert_msgs) if alert_msgs else ""

        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

    cap.release()


# ---------------- Authentication ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form['username'].strip()
        pwd = request.form['password']
        role = request.form['role']
        if not user or not pwd:
            flash('Fields required', 'error')
        else:
            pw_hash = generate_password_hash(pwd)
            try:
                with sqlite3.connect(DB) as conn:
                    conn.execute(
                        'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                        (user, pw_hash, role)
                    )
                flash('Registration successful', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already taken', 'error')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['username'].strip()
        pwd = request.form['password']
        cur = sqlite3.connect(DB).execute(
            'SELECT password_hash, role FROM users WHERE username=?', (user,)
        )
        row = cur.fetchone()
        if row and check_password_hash(row[0], pwd):
            session['logged_in'] = True
            session['username'] = user
            session['role'] = row[1]
            if row[1] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------- Dashboard ----------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/user_dashboard')
def user_dashboard():
    if not session.get('logged_in') or session.get('role') != 'user':
        return redirect(url_for('login'))
    return render_template('user_dashboard.html')


@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html')


@app.route('/detect')
def detect():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('detect.html')


@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    result_img = None
    weapon_detected = False
    offender_detected = False
    authorized_detected = False

    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['IMAGE_UPLOAD'], filename)
            file.save(path)
            frame = cv2.imread(path)
            frame = cv2.resize(frame, (640, 480))

            results = weapon_model.predict(frame, conf=0.5, verbose=False)
            weapon_detected, frame = process_weapon_detections(frame, results)
            if weapon_detected:
                log_detection("Weapon Threat Detected")
                trigger_email_alert("🚨 Weapon detected.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 6, minSize=(60,60))
            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
                try:
                    rep = DeepFace.represent(face, model_name='Facenet',
                                            detector_backend='opencv',
                                            enforce_detection=False)
                    if not rep:
                        continue
                    emb = np.array(rep[0]['embedding'])

                    ba_score, ba_name = 0, None
                    for name, emb_list in authorized_db.items():
                        for db_emb in emb_list:
                            s = np.dot(emb,db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                            if s > ba_score:
                                ba_score, ba_name = s, name
                    if ba_score > 0.82:
                        authorized_detected = True
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                        cv2.putText(frame, f"Authorized: {ba_name}", (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        continue

                    bo_score, bo_name = 0, None
                    for name, emb_list in offender_db.items():
                        for db_emb in emb_list:
                            s = np.dot(emb,db_emb)/(np.linalg.norm(emb)*np.linalg.norm(db_emb))
                            if s > bo_score:
                                bo_score, bo_name = s, name
                    if bo_score > 0.82:
                        offender_detected = True
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(frame, bo_name, (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        log_detection(f"Offender Detected: {bo_name}")
                        trigger_email_alert(f"🚨 Offender detected: {bo_name}")
                except:
                    pass

            output_path = os.path.join(app.config['IMAGE_UPLOAD'], "result_" + filename)
            cv2.imwrite(output_path, frame)
            result_img = "images/result_" + filename

    return render_template("predict_image.html",
                           result=result_img,
                           weapon_detected=weapon_detected,
                           offender_detected=offender_detected,
                           authorized_detected=authorized_detected)


@app.route('/predict_video', methods=['GET', 'POST'])
def predict_video():
    global uploaded_video_path
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    video_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['VIDEO_UPLOAD'], filename)
            file.save(path)
            uploaded_video_path = path
            video_path = "videos/" + filename
    return render_template("predict_video.html", video=video_path)


@app.route('/video_prediction_feed')
def video_prediction_feed():
    return Response(gen_video_prediction(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_alert')
def get_alert():
    with lock:
        return jsonify({"alert": current_alert})


# ---------------- Admin Upload ----------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files or 'name' not in request.form:
            return render_template('upload.html', error="Missing name or file.")
        f = request.files['file']
        name = request.form['name'].strip().lower()
        if not name or f.filename == '' or not f.filename.lower().endswith(('.jpg','.jpeg','.png')):
            return render_template('upload.html', error="Invalid name or file.")
        try:
            existing_images = [x for x in os.listdir(app.config['UPLOAD_FOLDER'])
                               if x.startswith(name + "_")]
            img_index = len(existing_images) + 1
            filename = f"{name}_{img_index}.jpg"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(upload_path)
            emb = DeepFace.represent(img_path=upload_path, model_name='Facenet')[0]['embedding']
            emb_filename = f"{name}_{img_index}_embedding.npy"
            np.save(os.path.join(app.config['EMBEDDINGS_FOLDER'], emb_filename), emb)
            offender_db.clear()
            offender_db.update(load_offender_db())
            return render_template('upload.html',
                                   success=f"{name} image {img_index} added successfully.")
        except Exception as e:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return render_template('upload.html', error="Face not detected.")
    return render_template('upload.html')


@app.route('/upload_authorized', methods=['GET', 'POST'])
def upload_authorized():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        f = request.files['file']
        name = request.form['name'].strip().lower()
        if f.filename == "":
            return render_template("upload_authorized.html", error="Select image")
        filename = secure_filename(f.filename)
        path = os.path.join('static', 'authorized', filename)
        f.save(path)
        try:
            emb = DeepFace.represent(img_path=path, model_name="Facenet")[0]["embedding"]
            count = len(os.listdir("authorized_embeddings")) + 1
            save_name = f"{name}_{count}_embedding.npy"
            np.save(os.path.join("authorized_embeddings", save_name), emb)
            authorized_db.clear()
            authorized_db.update(load_authorized_db())
            return render_template("upload_authorized.html", success="Authorized person added")
        except:
            return render_template("upload_authorized.html", error="Face not detected")
    return render_template("upload_authorized.html")


@app.route('/view_images')
def view_images():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    image_folder = os.path.join(app.static_folder, 'uploads')
    images = os.listdir(image_folder)
    offenders_dict = {}
    for image in images:
        name = image.split('_')[0].capitalize()
        if name not in offenders_dict:
            offenders_dict[name] = image
    offenders = [{"name": n, "filename": f} for n, f in offenders_dict.items()]
    return render_template("view_images.html", offenders=offenders,
                           offender_count=len(offenders))


@app.route('/view_authorized')
def view_authorized():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    image_folder = os.path.join(app.static_folder, 'authorized')
    images = os.listdir(image_folder)
    personnel = [{'filename': img, 'name': img.split('_')[0].capitalize()} for img in images]
    return render_template("view_authorized.html", personnel=personnel)


@app.route('/delete_authorized_image', methods=['POST'])
def delete_authorized_image():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    filename = request.form.get("filename")
    if filename:
        filepath = os.path.join(app.static_folder, "authorized", filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            flash("Authorized image deleted", "success")
    return redirect(url_for('view_authorized'))


@app.route('/delete_offender_image', methods=['POST'])
def delete_offender_image():
    filename = request.form.get('filename')
    if filename:
        filepath = os.path.join(app.static_folder, 'uploads', filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            flash('Image deleted successfully!', 'success')
    return redirect(url_for('view_images'))


@app.route('/view_logs')
def view_logs():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = f.readlines()
    return render_template('view_logs.html', logs=logs)


@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return redirect(url_for('login'))
    if os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()
    flash("All detection logs cleared successfully!", "success")
    return redirect(url_for('view_logs'))


@app.route('/offender_profile/<name>')
def offender_profile(name):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    name_lower = name.lower()
    image_folder = os.path.join(app.static_folder, "uploads")
    images = []
    if os.path.exists(image_folder):
        for img in os.listdir(image_folder):
            if img.lower().startswith(name_lower):
                images.append(img)
    embedding_folder = app.config['EMBEDDINGS_FOLDER']
    embedding_count = sum(
        1 for emb in os.listdir(embedding_folder)
        if emb.lower().startswith(name_lower) and emb.endswith(".npy")
    ) if os.path.exists(embedding_folder) else 0
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                if name_lower in line.lower():
                    history.append(line.strip())
    return render_template("offender_profile.html",
                           name=name, images=images,
                           embedding_count=embedding_count,
                           history=history)


if __name__ == '__main__':
    # FIX 8: threaded=True is important — Flask must serve the
    #         video stream and other requests simultaneously
    app.run(debug=True, threaded=True)
