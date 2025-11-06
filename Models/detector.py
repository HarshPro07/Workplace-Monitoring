# Models/detector.py
import os
import json
import cv2
import numpy as np
from collections import deque, Counter
from typing import List, Tuple

# -----------------------
# Configuration / paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # Models/
TRAINER_FILE = os.path.join(BASE_DIR, "trainer", "trainer.yml")
NAMES_FILE = os.path.join(BASE_DIR, "trainer", "names.txt")

# Haar cascade (OpenCV ships this; no external download required)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# -----------------------
# Load recognizer & names
# -----------------------
recognizer = None
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(TRAINER_FILE):
        try:
            recognizer.read(TRAINER_FILE)
            print("✅ Loaded trainer:", TRAINER_FILE)
        except Exception as e:
            print("⚠️ Could not read trainer file:", e)
    else:
        print("⚠️ trainer.yml not found at", TRAINER_FILE)
except Exception as e:
    print("⚠️ Could not create LBPH recognizer (cv2.face might be missing):", e)
    recognizer = None

# load names map (names.txt: format `id,name` per line)
names = {}
if os.path.exists(NAMES_FILE):
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split(",", 1)
                try:
                    idx = int(parts[0])
                    names[idx] = parts[1].strip()
                except:
                    continue
        print("✅ Loaded names:", list(names.values()))
    except Exception as e:
        print("⚠️ Failed to read names.txt:", e)
else:
    print("⚠️ names.txt not found at", NAMES_FILE)

# load cascade
if os.path.exists(CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print("⚠️ Cascade loaded but classifier is empty:", CASCADE_PATH)
else:
    print("⚠️ Haar cascade not found at default OpenCV path:", CASCADE_PATH)
    # fallback: allow user to place cascade in Models/ directory
    alt = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    if os.path.exists(alt):
        face_cascade = cv2.CascadeClassifier(alt)
        print("✅ Loaded Haar cascade from Models/:", alt)
    else:
        face_cascade = None
        print("❗ No face cascade available. Put a cascade xml in Models/ or fix OpenCV install.")

# -----------------------
# Tunables for tracker & voting
# -----------------------
MIN_FACE_SIZE = 80            # px minimum face size to attempt recognition
VOTES_WINDOW = 12             # frames kept in vote window
VOTES_REQUIRED = 6            # votes required for commit
IOU_MATCH_THRESHOLD = 0.15
TRACK_MAX_MISSING = 18        # frames before a track is removed

# -----------------------
# FaceTracker implementation
# -----------------------
class FaceTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # tid -> {'box':(x1,y1,x2,y2), 'centroid':(...), 'missed':int, 'votes':deque}

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA + 1)
        interH = max(0, yB - yA + 1)
        inter = interW * interH
        areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
        areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
        denom = float(areaA + areaB - inter)
        return inter/denom if denom > 0 else 0.0

    @staticmethod
    def centroid(box):
        x1,y1,x2,y2 = box
        return (int((x1+x2)/2), int((y1+y2)/2))

    def register(self, box):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            'box': box,
            'centroid': self.centroid(box),
            'missed': 0,
            'votes': deque(maxlen=VOTES_WINDOW),
            'committed': None
        }
        return tid

    def deregister(self, tid):
        if tid in self.tracks:
            del self.tracks[tid]

    def update(self, boxes: List[Tuple[int,int,int,int]]):
        if face_cascade is None:
            return {}

        if len(boxes) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > TRACK_MAX_MISSING:
                    self.deregister(tid)
            return {tid: t['box'] for tid, t in self.tracks.items()}

        if not self.tracks:
            for b in boxes:
                self.register(b)
            return {tid: t['box'] for tid, t in self.tracks.items()}

        tids = list(self.tracks.keys())
        track_boxes = [self.tracks[t]['box'] for t in tids]
        iouM = np.zeros((len(track_boxes), len(boxes)), dtype=float)
        for i, tb in enumerate(track_boxes):
            for j, nb in enumerate(boxes):
                iouM[i, j] = self.iou(tb, nb)

        assigned_tracks = set()
        assigned_boxes = set()
        pairs = []
        rows, cols = np.where(iouM > 0)
        for r, c in zip(rows, cols):
            pairs.append((r, c, iouM[r, c]))
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        for r, c, score in pairs:
            if r in assigned_tracks or c in assigned_boxes:
                continue
            if score < IOU_MATCH_THRESHOLD:
                continue
            tid = tids[r]
            self.tracks[tid]['box'] = boxes[c]
            self.tracks[tid]['centroid'] = self.centroid(boxes[c])
            self.tracks[tid]['missed'] = 0
            assigned_tracks.add(r); assigned_boxes.add(c)

        # unmatched tracks -> missed
        for i, tid in enumerate(tids):
            if i not in assigned_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > TRACK_MAX_MISSING:
                    self.deregister(tid)

        # unmatched boxes -> register
        for j, b in enumerate(boxes):
            if j not in assigned_boxes:
                self.register(b)

        return {tid: t['box'] for tid, t in self.tracks.items()}

    def add_vote(self, tid, pred_id, conf):
        if tid not in self.tracks:
            return
        self.tracks[tid]['votes'].append((pred_id, conf))

    def get_committed(self, tid):
        t = self.tracks.get(tid)
        if not t:
            return None, {}
        votes = list(t['votes'])
        if len(votes) < VOTES_REQUIRED:
            return None, {'votes': len(votes)}
        ids = [v[0] for v in votes]
        cnt = Counter(ids)
        most_common_id, count = cnt.most_common(1)[0]
        confs = [v[1] for v in votes if v[1] is not None]
        avg_conf = float(sum(confs)/len(confs)) if confs else None

        if most_common_id != -1 and count >= VOTES_REQUIRED:
            return most_common_id, {'votes': count, 'avg_conf': avg_conf}
        if most_common_id == -1 and count >= VOTES_REQUIRED:
            return -1, {'votes': count, 'avg_conf': avg_conf}
        return None, {'votes': len(votes), 'most_common': most_common_id, 'count': count}

# instantiate tracker
_face_tracker = FaceTracker()

# -----------------------
# detect_employee(frame)
# -----------------------
def detect_employee(frame):
    """
    Input: BGR frame (numpy array)
    Output: annotated frame, list of (committed_id_or_-1, name) for visible faces
    """
    if face_cascade is None:
        # cascade missing; skip detection gracefully
        return frame, []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    boxes = []
    for (x, y, w, h) in faces:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        boxes.append((x1, y1, x2, y2))

    active_tracks = _face_tracker.update(boxes)
    detected_profiles = []

    for tid, box in active_tracks.items():
        x1, y1, x2, y2 = box
        H, W = gray.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W - 1, x2), min(H - 1, y2)
        w = max(1, x2c - x1c)
        h = max(1, y2c - y1c)

        # ignore very small faces -> treat as unknown vote
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE or recognizer is None:
            _face_tracker.add_vote(tid, -1, None)
            committed_id, info = _face_tracker.get_committed(tid)
            name = names.get(committed_id, "Unknown") if committed_id not in (None,) else "Unknown"
            label = "Unknown" if committed_id in (None, -1) else f"EID-{committed_id}: {name}"
            color = (0, 0, 255) if committed_id in (None, -1) else (0,255,0)
            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
            cv2.putText(frame, label, (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            detected_profiles.append((committed_id if committed_id is not None else -1, name))
            continue

        # crop, equalize, resize for recognizer
        face_crop = gray[y1c:y2c, x1c:x2c].copy()
        try:
            face_crop = cv2.equalizeHist(face_crop)
        except Exception:
            pass
        try:
            face_resized = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_LINEAR)
        except Exception:
            face_resized = face_crop

        try:
            pred_id, conf = recognizer.predict(face_resized)
        except Exception:
            pred_id, conf = -1, 999.0

        conf_val = float(conf) if isinstance(conf, (int, float)) and conf < 1000 else None
        _face_tracker.add_vote(tid, int(pred_id) if pred_id is not None else -1, conf_val)

        committed_id, info = _face_tracker.get_committed(tid)
        if committed_id is not None:
            name = names.get(committed_id, "Unknown") if committed_id != -1 else "Unknown"
            label = f"EID-{committed_id}: {name}" if committed_id != -1 else "Unknown"
            color = (0, 255, 0) if committed_id != -1 else (0, 0, 255)
        else:
            votes_list = _face_tracker.tracks[tid]['votes']
            if votes_list:
                last_pred = votes_list[-1][0]
                name = names.get(last_pred, "Unknown") if last_pred != -1 else "Unknown"
                label = f"(?) {name}"
                color = (0, 200, 200)
            else:
                name = "Unknown"
                label = "Unknown"
                color = (0,0,255)

        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
        cv2.putText(frame, label, (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        detected_profiles.append((committed_id if committed_id is not None else -1, name))

    return frame, detected_profiles
