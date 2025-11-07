# import requests
# import sys
# import cv2
# import atexit
# import time
# import threading
# import queue
# from collections import deque
# import numpy as np
# from flask import Flask, render_template, Response, jsonify
# from colorama import Fore, init
# from ultralytics import YOLO
# import os, importlib.util, torch

# # ---------- INITIAL SETUP ----------
# init(autoreset=True)

# MODELS_DIR = os.path.join(os.path.dirname(__file__), "Models")
# sys.path.append(MODELS_DIR if os.path.isdir(MODELS_DIR) else r"E:\Sem7\Project1\Project\Workplace-Monitoring\Models")

# # Load model modules safely
# try:
#     from detector import detect_employee
# except Exception:
#     spec = importlib.util.spec_from_file_location("detector", os.path.join(MODELS_DIR, "detector.py"))
#     detector = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(detector)
#     detect_employee = detector.detect_employee

# def _load_mod(name):
#     path = os.path.join(MODELS_DIR, f"{name}.py")
#     spec = importlib.util.spec_from_file_location(name, path)
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     return mod

# yawn = _load_mod("yawn")
# eye = _load_mod("eye")
# tilted = _load_mod("tilted")

# app = Flask(__name__)

# # ---------- Helper: bordered text (crisp at low res / MJPEG) ----------
# def draw_bordered_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
#                        font_scale=0.65, color=(0,255,0), thickness=1, outline_thickness=3):
#     x, y = org
#     # outline (black)
#     cv2.putText(img, text, (x, y), font, font_scale, (0,0,0), outline_thickness, cv2.LINE_AA)
#     # inner text
#     cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

# # =========================================================
# # CAMERA THREAD (non-blocking)
# # =========================================================
# class VideoCaptureThread:
#     def __init__(self, src=0, queue_size=6, width=640, height=480, fps=30):
#         self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         self.cap.set(cv2.CAP_PROP_FPS, fps)

#         if not self.cap.isOpened():
#             raise RuntimeError(Fore.RED + "⚠️ Could not open webcam.")

#         self.q = queue.Queue(maxsize=queue_size)
#         self.stopped = False
#         threading.Thread(target=self._reader, daemon=True).start()

#     def _reader(self):
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             if not ret:
#                 time.sleep(0.01)
#                 continue
#             if self.q.full():
#                 try: self.q.get_nowait()
#                 except queue.Empty: pass
#             self.q.put(frame)

#     def read(self, timeout=0.5):
#         try: return True, self.q.get(timeout=timeout)
#         except queue.Empty: return False, None

#     def release(self):
#         self.stopped = True
#         try: self.cap.release()
#         except: pass

# # =========================================================
# # PHONE DETECTOR (Optimized)
# # =========================================================
# # --- Replace existing CentroidTracker & PhoneDetector with this improved version ---
# from collections import deque
# import math

# class CentroidIoUTracker:
#     """
#     Simple tracker that registers objects and associates new detections using IoU + centroid fallback.
#     Keeps disappeared counter and deregisters if missing for too long.
#     """
#     def __init__(self, max_disappeared=15, max_distance=120):
#         self.next_object_id = 0
#         self.objects = {}          # id -> (bbox, centroid)
#         self.disappeared = {}      # id -> frames disappeared
#         self.max_disappeared = max_disappeared
#         self.max_distance = max_distance

#     @staticmethod
#     def _centroid_from_box(box):
#         x1,y1,x2,y2 = box
#         return (int((x1+x2)/2), int((y1+y2)/2))

#     @staticmethod
#     def _iou(boxA, boxB):
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[2], boxB[2])
#         yB = min(boxA[3], boxB[3])
#         interW = max(0, xB - xA + 1)
#         interH = max(0, yB - yA + 1)
#         inter = interW * interH
#         areaA = (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
#         areaB = (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
#         denom = float(areaA + areaB - inter)
#         return inter/denom if denom > 0 else 0.0

#     def register(self, box):
#         c = self._centroid_from_box(box)
#         self.objects[self.next_object_id] = (box, c)
#         self.disappeared[self.next_object_id] = 0
#         self.next_object_id += 1

#     def deregister(self, object_id):
#         self.objects.pop(object_id, None)
#         self.disappeared.pop(object_id, None)

#     def update(self, rects):
#         """
#         rects: list of boxes (x1,y1,x2,y2)
#         returns dict of id -> (box, centroid)
#         """
#         if len(rects) == 0:
#             # increment disappeared and deregister if needed
#             for oid in list(self.disappeared.keys()):
#                 self.disappeared[oid] += 1
#                 if self.disappeared[oid] > self.max_disappeared:
#                     self.deregister(oid)
#             return self.objects

#         input_centroids = [self._centroid_from_box(r) for r in rects]

#         if not self.objects:
#             for r in rects:
#                 self.register(r)
#             return self.objects

#         object_ids = list(self.objects.keys())
#         object_boxes = [self.objects[oid][0] for oid in object_ids]
#         object_centroids = [self.objects[oid][1] for oid in object_ids]

#         # compute IoU matrix between existing and new boxes
#         iouM = np.zeros((len(object_boxes), len(rects)), dtype=float)
#         for i, ob in enumerate(object_boxes):
#             for j, nb in enumerate(rects):
#                 iouM[i, j] = self._iou(ob, nb)

#         # Greedy match by highest IoU first
#         rows, cols = np.where(iouM > 0)
#         # fallback on centroid distances if IoU too small
#         assigned_rows, assigned_cols = set(), set()
#         pairs = []
#         # sort pairs by IoU desc
#         if rows.size > 0:
#             pairs = sorted([(i, j, iouM[i, j]) for i, j in zip(rows, cols)], key=lambda x: x[2], reverse=True)

#         for (row, col, score) in pairs:
#             if row in assigned_rows or col in assigned_cols:
#                 continue
#             if score < 0.15:  # low IoU threshold -> use centroid fallback
#                 continue
#             oid = object_ids[row]
#             self.objects[oid] = (rects[col], input_centroids[col])
#             self.disappeared[oid] = 0
#             assigned_rows.add(row)
#             assigned_cols.add(col)

#         # Centroid fallback for unmatched columns
#         if len(assigned_cols) < len(rects):
#             # compute centroid distance matrix for leftover
#             unused_rows = [r for r in range(len(object_centroids)) if r not in assigned_rows]
#             unused_cols = [c for c in range(len(rects)) if c not in assigned_cols]
#             if unused_rows and unused_cols:
#                 D = np.linalg.norm(np.array(object_centroids)[unused_rows][:, None] - np.array(input_centroids)[unused_cols][None, :], axis=2)
#                 for row_idx, col_idx in zip(*np.where(D == D.min(axis=1)[:,None])):
#                     row = unused_rows[row_idx]; col = unused_cols[col_idx]
#                     if row in assigned_rows or col in assigned_cols: continue
#                     if D[row_idx, col_idx] > self.max_distance: continue
#                     oid = object_ids[row]
#                     self.objects[oid] = (rects[col], input_centroids[col])
#                     self.disappeared[oid] = 0
#                     assigned_rows.add(row); assigned_cols.add(col)

#         # mark disappeared for unmatched rows
#         for r in range(len(object_boxes)):
#             if r not in assigned_rows:
#                 oid = object_ids[r]
#                 self.disappeared[oid] += 1
#                 if self.disappeared[oid] > self.max_disappeared:
#                     self.deregister(oid)

#         # register remaining unmatched cols
#         for c in range(len(rects)):
#             if c not in assigned_cols:
#                 self.register(rects[c])

#         return self.objects


# class PhoneDetector:
#     """
#     Improved phone detector using Ultralytics YOLO + temporal aggregator + tracker.
#     Tunable params at init for quick experimentation.
#     """
    
#     def __init__(self, weights="yolov8n.pt", imgsz=640, conf=0.25, iou=0.45,
#                  frame_skip=1, window_size=10, votes_needed=3, min_avg_conf=0.25):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = YOLO(weights)
#         try:
#             self.model.model.to(self.device)
#         except Exception:
#             pass

#         self.imgsz = imgsz
#         self.conf = conf
#         self.iou = iou
#         self.frame_skip = frame_skip
#         self.frame_counter = 0

#         # temporal aggregation
#         self.window_size = window_size
#         self.conf_deque = deque(maxlen=window_size)
#         self.presence_deque = deque(maxlen=window_size)
#         self.votes_needed = votes_needed
#         self.min_avg_conf = min_avg_conf

#         self.tracker = CentroidIoUTracker(max_disappeared=12, max_distance=140)
#         self.last_info = {"present": False, "detections": [], "votes": 0, "window": window_size, "max_conf": None}

#     def _predict(self, frame):
#         # ultralytics can accept numpy arrays (BGR). Use augment=True for TTA (slower but more robust).
#         results = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
#                                      device=self.device, augment=True, verbose=False)
#         return results

#     def detect(self, frame):
#         """
#         Returns info dict:
#          - present: bool
#          - detections: list of {box, conf}
#          - votes, window, max_conf
#         """
#         self.frame_counter += 1
#         if self.frame_counter % max(1, self.frame_skip) != 0:
#             return self.last_info

#         results = self._predict(frame)
#         boxes = []
#         detections = []
#         frame_hit = False

#         # results[0].boxes contains xyxy, cls, conf
#         for box in results[0].boxes:
#             cls_id = int(box.cls[0])
#             name = results[0].names.get(cls_id, "")
#             if name.lower() in ("cell phone", "cellphone", "phone"):   # be defensive about class names
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#                 conf = float(box.conf[0].cpu().numpy())
#                 # optionally ignore tiny boxes
#                 w = x2 - x1; h = y2 - y1
#                 if w < 24 or h < 24:   # ignore very small detections
#                     continue
#                 boxes.append((x1,y1,x2,y2))
#                 detections.append({"box": (x1,y1,x2,y2), "conf": conf})
#                 frame_hit = True

#         # update tracker and temporal queues
#         objects = self.tracker.update(boxes)
#         self.presence_deque.append(1 if frame_hit else 0)
#         max_conf = max((d["conf"] for d in detections), default=0.0)
#         self.conf_deque.append(max_conf)

#         votes = sum(self.presence_deque)
#         avg_conf = (sum(self.conf_deque)/len(self.conf_deque)) if len(self.conf_deque) else 0.0

#         # final present/hysteresis logic:
#         present = (votes >= self.votes_needed) and (avg_conf >= self.min_avg_conf)

#         # prepare id-linked detections (match tracker ids to detection boxes by IoU)
#         id_dets = []
#         for oid, (tbox, centroid) in objects.items():
#             # find detection whose IoU with tracker box is highest
#             best = None; best_iou = 0.0
#             for det in detections:
#                 i = CentroidIoUTracker._iou(tbox, det["box"])
#                 if i > best_iou:
#                     best_iou = i; best = det
#             if best and best_iou > 0.05:
#                 id_dets.append({"id": oid, "box": best["box"], "conf": best["conf"]})
#             else:
#                 # if no detection matches, still include track (low confidence)
#                 id_dets.append({"id": oid, "box": tbox, "conf": None})

#         info = {"present": bool(present),
#                 "votes": int(votes),
#                 "window": int(self.window_size),
#                 "max_conf": float(max_conf) if max_conf else None,
#                 "avg_conf": float(avg_conf),
#                 "detections": id_dets}
#         self.last_info = info
#         return info

#     def draw_overlay(self, frame, info):
#         out = frame.copy()
#         for det in info.get("detections", []):
#             x1,y1,x2,y2 = det.get("box", (0,0,0,0))
#             conf = det.get("conf")
#             oid = det.get("id", -1)
#             # translucent fill + border
#             overlay = out.copy()
#             cv2.rectangle(overlay, (x1,y1),(x2,y2),(0,200,0),-1)
#             cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
#             cv2.rectangle(out, (x1,y1),(x2,y2),(0,200,0),2)
#             txt = f"P{oid} {conf:.2f}" if conf else f"P{oid}"
#             draw_bordered_text(out, txt, (x1, y1-8), font_scale=0.55, color=(255,255,255), thickness=1, outline_thickness=2)
#         # small badge for presence
#         ptxt = f"Phone: {'Yes' if info.get('present') else 'No'} ({info.get('votes')}/{info.get('window')}, avg:{info.get('avg_conf',0):.2f})"
#         draw_bordered_text(out, ptxt, (12, 28), font_scale=0.6, color=(0,255,255), thickness=1, outline_thickness=3)
#         return out


# # ---------- CAMERA ----------
# cap_thread = VideoCaptureThread(src=0, queue_size=6, width=640, height=480, fps=30)

# # ---------- DETECTOR ----------
# phone_detector = PhoneDetector(weights="yolov8n.pt", imgsz=416, conf=0.45, frame_skip=3)

# # ---------- Live status (thread-safe) ----------
# latest_status = {}
# status_lock = threading.Lock()

# @atexit.register
# def release_resources():
#     cap_thread.release()
#     cv2.destroyAllWindows()
#     print(Fore.GREEN + "✅ Resources cleaned up.")

# def color_for_status(status):
#     s = (status or "").lower()
#     if any(k in s for k in ["alert","yawn","tilt","closed","using","detected"]): return (0,0,255)
#     elif any(k in s for k in ["partial","warning","slight"]): return (0,255,255)
#     return (0,255,0)

# # =========================================================
# # FRAME GENERATOR
# # =========================================================
# def gen_frames():
#     session_start = time.time()
#     phone_timer_start = None
#     phone_visible_duration = 0
#     fps_smooth, t_prev = 0, time.time()

#     while True:
#         ok, frame = cap_thread.read()
#         if not ok: continue
#         frame = cv2.flip(frame, 1)

#         # Employee
#         frame, profiles = detect_employee(frame)
#         employee_name = profiles[0][1] if profiles else "Unknown"
#         emp_color = (0,255,0) if profiles else (0,0,255)
#         emp_text = f"Employee: {employee_name}"

#         # Models
#         frame, yawn_status = yawn.detect(frame)
#         frame, drowsy_status = eye.detect(frame)
#         frame, tilt_status = tilted.detect(frame)

#         # Phone
#         # info = phone_detector.detect(frame)
#         # phone_status = "Detected" if info["present"] else "Not Detected"
#         # frame = phone_detector.draw_overlay(frame, info)

#                 # ---------------------------
#         # Run phone detection (but don't draw yet)
#         # ---------------------------
#                 # ---------------------------
#         # Run phone detection (but don't draw boxes yet)
#         # ---------------------------
#         info = phone_detector.detect(frame)
#         phone_status = "Detected" if info["present"] else "Not Detected"
#         votes = int(info.get('votes', 0))
#         window = int(info.get('window', 1))
#         avg_conf = float(info.get('avg_conf', 0.0)) if info.get('avg_conf') is not None else 0.0

#         # ---------------------------
#         # Draw header + statuses FIRST (background)
#         # ---------------------------
#         overlay = frame.copy()
#         header_x1, header_y1 = 10, 10
#         header_x2, header_y2 = 520, 160
#         cv2.rectangle(overlay, (header_x1, header_y1), (header_x2, header_y2), (0,0,0), -1)
#         frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

#         draw_bordered_text(frame, emp_text, (20,42), font_scale=0.9, color=emp_color, thickness=1, outline_thickness=4)

#         # build statuses with inline phone stats for the 4th line
#         statuses = [("1. Yawn", yawn_status), ("2. Drowsy", drowsy_status),
#                     ("3. Head Tilt", tilt_status), ("4. Phone", phone_status)]
#         y_pos = 72
#         for label, status in statuses:
#             color = color_for_status(status)
#             # draw the main status text on the left
#             draw_bordered_text(frame, f"{label}: {status}", (30, y_pos),
#                                font_scale=0.7, color=color, thickness=1, outline_thickness=3)

#             # if this is the Phone line, draw the votes/avg to the right side of the header
#             if label.startswith("4. Phone"):
#                 # compute right-side anchor (within header box) so it always looks part of header
#                 right_anchor_x = header_x2 - 12
#                 # text we want to show inline; smaller font
#                 phone_badge_txt = f"{'Using' if info.get('present') else 'No'} ({votes}/{window}, avg:{avg_conf:.2f})"
#                 # measure text width to align to the right_anchor_x
#                 (w_txt, h_txt), _ = cv2.getTextSize(phone_badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#                 # place so text right edge is near right_anchor_x
#                 txt_x = max( int(right_anchor_x - w_txt), 320 )
#                 txt_y = y_pos
#                 draw_bordered_text(frame, phone_badge_txt, (txt_x, txt_y),
#                                    font_scale=0.6, color=(0,200,255), thickness=1, outline_thickness=3)
#             y_pos += 26


#         # Summary (bottom-left)
#         h = frame.shape[0]
#         elapsed = int(time.time() - session_start)
#         overlay2 = frame.copy()
#         cv2.rectangle(overlay2, (20,h-85), (260,h-10), (0,0,0), -1)
#         frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
#         draw_bordered_text(frame, "📱 Phone", (30,h-58), font_scale=0.65, color=(0,255,255), outline_thickness=3)
#         draw_bordered_text(frame, f"Total: {elapsed}s", (30,h-36), font_scale=0.55, color=(255,255,255), outline_thickness=2)
#         draw_bordered_text(frame, f"Visible: {int(phone_visible_duration)}s", (30,h-16), font_scale=0.55, color=(0,255,0), outline_thickness=2)

#         # FPS
#         t_now = time.time()
#         fps = 1.0 / max(t_now - t_prev, 1e-6)
#         t_prev = t_now
#         fps_smooth = fps_smooth * 0.9 + fps * 0.1
#         draw_bordered_text(frame, f"FPS: {fps_smooth:.1f}", (frame.shape[1]-140,30), font_scale=0.6, color=(0,255,255), outline_thickness=2)

#         # Update latest_status (thread-safe)
#         with status_lock:
#             latest_status['employee'] = employee_name
#             latest_status['yawn'] = yawn_status
#             latest_status['drowsy'] = drowsy_status
#             latest_status['tilt'] = tilt_status
#             latest_status['phone_present'] = bool(info['present'])
#             latest_status['phone_votes'] = int(info.get('votes', 0))
#             latest_status['phone_window'] = int(info.get('window', 12))
#             latest_status['phone_conf'] = float(info['max_conf']) if info.get('max_conf') is not None else None
#             latest_status['phone_visible_s'] = int(phone_visible_duration)
#             latest_status['fps'] = round(fps_smooth, 1)
#             latest_status['elapsed_s'] = elapsed

#         # Encode (higher quality for crisper text)
#         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
#         if not ret: continue
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# #-----------PYTHON TO NODE-------------
# NODE_API_URL = "http://localhost:3000/api/logs/add"  # your Node endpoint

# def push_to_node():
#     while True:
#         time.sleep(5)  # send every 5 seconds (tweak as needed)
#         with status_lock:
#             data = dict(latest_status)
        
#         if not data:
#             continue
        
#         try:
#             # Build JSON payload to match your schema
#             payload = {
#                 "yawn": data.get("yawn", "").lower() in ["yes", "detected", "true"],
#                 "drowsy": data.get("drowsy", "").lower() in ["yes", "detected", "true"],
#                 "tilt": data.get("tilt", "").lower() in ["yes", "detected", "true"],
#                 "phone": data.get("phone_present", False),
#                 "fatigue_score": 100 - (20 if data.get("drowsy") else 0) - (20 if data.get("yawn") else 0),
#                 "productivity_score": 100 - (30 if data.get("phone_present") else 0)
#             }

#             response = requests.post(NODE_API_URL, json=payload, timeout=3)
#             if response.status_code == 201:
#                 print(Fore.GREEN + f"Log pushed to Node: {payload}")
#             else:
#                 print(Fore.YELLOW + f"Node response: {response.status_code} {response.text}")
#         except Exception as e:
#             print(Fore.RED + f"Failed to send log: {e}")

# # Start thread
# threading.Thread(target=push_to_node, daemon=True).start()

# # ---------- STATUS API ----------
# @app.route('/status')
# def get_status():
#     with status_lock:
#         # return a safe copy
#         return jsonify(dict(latest_status))

# # ---------- ROUTES ----------
# @app.route('/')
# def index(): return render_template('index.html')

# @app.route('/video_feed')
# def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     try:
#         print(Fore.CYAN + "\n🚀 Starting Flask app...\n")
#         app.run(debug=True, use_reloader=False, threaded=True)
#     finally:
#         release_resources()
