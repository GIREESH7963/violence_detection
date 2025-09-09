import os
import cv2
import torch
import numpy as np
import smtplib, ssl
from email.message import EmailMessage
from collections import deque
from datetime import datetime
from threading import Thread
import mediapipe as mp
from ultralytics import YOLO
import librosa
import time

# ======================
# CONFIGURATION
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 30
SEQUENCE_LENGTH = 16
ALERT_THRESHOLD = 0.85
EMA_ALPHA = 0.2
CONSECUTIVE_ALERTS_REQUIRED = 3

RTSP_LINK = "rtsp://username:password@ip_address:port/stream"

SENDER_EMAIL = ""
RECEIVER_EMAIL = ""
APP_PASSWORD = ""

EVIDENCE_DIR = "/content/evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

FRAME_SKIP = 1  # process every frame (adjust if laggy)

# ======================
# THREAD-BASED RTSP STREAM
# ======================
class RTSPStream:
    def __init__(self, link):
        self.link = link
        self.cap = cv2.VideoCapture(link)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print("[WARN] Stream disconnected, reconnecting...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.link)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ======================
# YOLO + POSE MODELS
# ======================
yolo_model = YOLO("")#download the weights of yolov8n-pose
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ======================
# LOAD TRAINED LSTM
# ======================
class LSTMAttention(torch.nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=256, num_layers=4, bidirectional=True, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.attn = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.attn(out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * out, dim=1)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits

lstm_model = LSTMAttention().to(DEVICE)
lm=""  #download the weights of lstm (best_lstm_attn.pt)  and place the path
lstm_model.load_state_dict(torch.load("lm", map_location=DEVICE))
lstm_model.eval()

# ======================
# EMAIL ALERT FUNCTION
# ======================
def send_email_alert(snapshot_path):
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "[ALERT] Violence Detected!"

        msg.set_content(f"Potential violent activity detected.\nSnapshot: {os.path.basename(snapshot_path)}")

        with open(snapshot_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=os.path.basename(snapshot_path))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print(f"[ALERT EMAIL SENT] -> {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"[EMAIL FAILED]: {e}")

# ======================
# FEATURE EXTRACTION
# ======================
def extract_audio_features_for_video(video_path, sr=22050):
    try:
        y, sr = librosa.load(video_path, sr=sr, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        flat = librosa.feature.spectral_flatness(y=y)[0]
        return dict(y=y, sr=sr, rms=rms, zcr=zcr, flat=flat)
    except:
        return None

def frame_audio_value(audio_feats, t_frame, total_frames, fps):
    if audio_feats is None: return (0.0, 0.0, 0.0)
    duration = len(audio_feats['y']) / audio_feats['sr']
    time_sec = t_frame / fps
    idx = int(np.clip((time_sec / duration) * len(audio_feats['rms']), 0, len(audio_feats['rms'])-1))
    return (float(audio_feats['rms'][idx]), float(audio_feats['zcr'][idx]), float(audio_feats['flat'][idx]))

def extract_features_from_frame(frame, yolo_model, mp_pose, prev_gray=None, prev_bbox=None, audio_feats=None, frame_idx=0, total_frames=1, fps=25):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # YOLO detection
    results = yolo_model.predict(frame, imgsz=640, verbose=False)
    boxes, confs = [], []
    for r in results:
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            for box in r.boxes:
                conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 0.0
                xywh = box.xywh.cpu().numpy().astype(float)
                boxes.append(xywh[0])
                confs.append(conf)
    num_persons = len(boxes)
    yolo_conf_mean = float(np.mean(confs)) if confs else 0.0
    yolo_conf_max = float(np.max(confs)) if confs else 0.0

    # Primary bbox
    bbox_feat = np.zeros(6, dtype=float)
    if boxes:
        areas = [b[2]*b[3] for b in boxes]
        idx = int(np.argmax(areas))
        x_c, y_c, bw, bh = boxes[idx]
        area = bw*bh
        aspect = bw/bh if bh!=0 else 0.0
        bbox_feat = np.array([x_c/w, y_c/h, bw/w, bh/h, area/(w*h), aspect], dtype=float)

    # Pose stats
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = mp_pose.process(rgb)
    pose_x, pose_y = [], []
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            pose_x.append(lm.x)
            pose_y.append(lm.y)
    pose_count = len(pose_x)
    if pose_count>0:
        pose_stats = np.array([np.mean(pose_x), np.mean(pose_y), np.std(pose_x), np.std(pose_y), pose_count/33.0])
    else:
        pose_stats = np.zeros(5)

    # Optical flow
    if prev_gray is None:
        flow_mean = flow_std = flow_frac = 0.0
    else:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_mean = float(np.mean(mag))
        flow_std = float(np.std(mag))
        flow_frac = float(np.count_nonzero(mag>1e-3)/(mag.size))

    # Color & intensity
    mean_bgr = cv2.mean(frame)[:3]
    mean_R, mean_G, mean_B = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
    mean_gray, std_gray = float(np.mean(gray)), float(np.std(gray))
    edge_density = float(np.count_nonzero(cv2.Canny(gray,100,200))/(gray.size))

    dx = (bbox_feat[0]-prev_bbox[0]) if prev_bbox is not None else 0.0
    dy = (bbox_feat[1]-prev_bbox[1]) if prev_bbox is not None else 0.0
    darea = (bbox_feat[4]-prev_bbox[4]) if prev_bbox is not None else 0.0

    rms_v, zcr_v, flat_v = frame_audio_value(audio_feats, frame_idx, total_frames, fps) if audio_feats is not None else (0.0,0.0,0.0)
    frame_var = float(np.var(frame))

    feat = np.array([
        bbox_feat[0], bbox_feat[1], bbox_feat[2], bbox_feat[3], bbox_feat[4], bbox_feat[5],
        float(num_persons),
        pose_stats[0], pose_stats[1], pose_stats[2], pose_stats[3], pose_stats[4],
        flow_mean, flow_std, flow_frac,
        mean_R, mean_G, mean_B,
        mean_gray, std_gray,
        edge_density,
        dx, dy, darea,
        yolo_conf_mean, yolo_conf_max,
        rms_v, zcr_v, flat_v,
        frame_var
    ], dtype=float)

    return feat, gray, bbox_feat

# ======================
# LIVE INFERENCE
# ======================
print("[INFO] Starting RTSP live inference...")
feature_queue = deque(maxlen=SEQUENCE_LENGTH)
ema_prob = 0.0
consecutive_alerts = 0
prev_gray = None
prev_bbox = None

stream = RTSPStream(RTSP_LINK)
frame_idx = 0

try:
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        features, prev_gray, prev_bbox = extract_features_from_frame(
            frame, yolo_model, mp_pose, prev_gray, prev_bbox
        )
        feature_queue.append(features)

        if len(feature_queue) == SEQUENCE_LENGTH:
            seq_input = torch.tensor(np.array(feature_queue, dtype=np.float32)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = lstm_model(seq_input)
                seq_prob = torch.softmax(logits, dim=1)[0,1].item()

            ema_prob = EMA_ALPHA*seq_prob + (1-EMA_ALPHA)*ema_prob

            if ema_prob > ALERT_THRESHOLD:
                consecutive_alerts +=1
                if consecutive_alerts >= CONSECUTIVE_ALERTS_REQUIRED:
                    snapshot_path = os.path.join(EVIDENCE_DIR, f"alert_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"[ALERT] Triggered! seq_prob={seq_prob:.3f}, ema={ema_prob:.3f}")
                    send_email_alert(snapshot_path)
                    consecutive_alerts = 0
            else:
                consecutive_alerts = 0

        frame_idx +=1
except KeyboardInterrupt:
    print("[INFO] Stopping live inference...")
    stream.stop()
