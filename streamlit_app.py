from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch

from equiphys_core import (
    compute_landmark_motion_displacement,
    EquiPhysDANN,
    FaceROIExtractor,
    build_clip_tensor_from_buffer,
    compute_pulse_snr,
    compute_signal_quality_index,
    estimate_hr_from_autocorr,
    estimate_hr_bpm_from_bvp,
    estimate_hr_from_rgb_chrom,
    estimate_hr_from_rgb_pos,
    estimate_respiratory_rate,
    estimate_spo2_from_rgb,
    test_time_adaptation,
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');
        
        /* Global Theme */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b1120; /* Deep Slate Background */
            color: #e2e8f0; /* Soft White Text */
        }
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 1.5rem; 
            max-width: 96%; 
        }
        
        /* Header Styling */
        .header-shell {
            background-color: #1e293b;
            border-left: 6px solid #3b82f6; /* Clinical Blue Accent */
            border-radius: 8px;
            padding: 16px 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .title-row {
            display: flex; justify-content: space-between; align-items: center;
            font-size: 22px; font-weight: 600; letter-spacing: -0.02em;
            color: #f8fafc;
        }
        .live-pill { 
            color: #10b981; 
            background: rgba(16, 185, 129, 0.1); 
            border: 1px solid rgba(16, 185, 129, 0.2); 
            padding: 4px 12px; 
            border-radius: 6px; 
            font-size: 13px; 
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        
        /* Panel Containers */
        .panel-wrap {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .panel-title {
            font-weight: 600;
            font-size: 14px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
        }
        
        /* Metric Cards */
        .metric-shell {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px 12px;
            text-align: center;
            min-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.2s ease;
        }
        .metric-label { 
            color: #94a3b8; 
            font-size: 12px; 
            font-weight: 600; 
            letter-spacing: 0.5px; 
            margin-bottom: 8px;
        }
        .metric-value { 
            font-size: 48px; 
            font-weight: 700; 
            color: #f8fafc; 
            line-height: 1.0;
            font-variant-numeric: tabular-nums; /* Prevents jumping numbers */
            margin-bottom: 4px;
        }
        .metric-unit { 
            color: #64748b; 
            font-size: 14px; 
            font-weight: 500; 
        }
        
        /* Log Terminal */
        .logbox {
            background: #020617;
            border: 1px solid #1e293b;
            border-radius: 6px;
            min-height: 560px;
            max-height: 560px;
            overflow-y: auto;
            padding: 16px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            color: #cbd5e1;
            font-size: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def _load_model(device: torch.device, checkpoint_path: Optional[str] = None) -> EquiPhysDANN:
    model = EquiPhysDANN(in_channels=3, latent_dim=256, frames=150, lambda_grl=1.0).to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model

@st.cache_resource
def _load_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _resolve_default_checkpoint() -> str:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "checkpoints_v2" / "equiphys_best.pt",
        root / "checkpoints_gpu_mcd" / "equiphys_best.pt",
        root / "checkpoints_gpu" / "equiphys_best.pt",
        root / "checkpoints" / "equiphys_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""

def _capture_webcam_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def _detect_primary_face(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=6, minSize=(90, 90))
    if len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(bw), int(bh)

def _bbox_iou(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 0 else 0.0

def _draw_roi_overlay(frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw physiological guides only when a face is present."""
    out = frame_bgr.copy()
    if face_bbox is None:
        return out

    x, y, bw, bh = face_bbox
    c = (59, 130, 246) # Swapped to a clinical blue (#3b82f6) from neon green
    cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)

    # Forehead band
    fx0, fy0 = x + int(0.15 * bw), y + int(0.05 * bh)
    fx1, fy1 = x + int(0.85 * bw), y + int(0.25 * bh)
    cv2.rectangle(out, (fx0, fy0), (fx1, fy1), c, 2)

    # Cheek strips
    left_pts = np.array([
        [x + int(0.12 * bw), y + int(0.45 * bh)],
        [x + int(0.30 * bw), y + int(0.40 * bh)],
        [x + int(0.28 * bw), y + int(0.72 * bh)],
        [x + int(0.10 * bw), y + int(0.75 * bh)],
    ], dtype=np.int32)
    right_pts = np.array([
        [x + int(0.70 * bw), y + int(0.40 * bh)],
        [x + int(0.88 * bw), y + int(0.45 * bh)],
        [x + int(0.90 * bw), y + int(0.75 * bh)],
        [x + int(0.72 * bw), y + int(0.72 * bh)],
    ], dtype=np.int32)
    cv2.polylines(out, [left_pts], isClosed=True, color=c, thickness=2)
    cv2.polylines(out, [right_pts], isClosed=True, color=c, thickness=2)
    return out

def _render_metric_card(label: str, value: str, unit: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _estimate_bp_hb_from_hr(hr: float) -> Tuple[Tuple[int, int], float]:
    sbp = int(np.clip(90 + 0.40 * hr, 95, 150))
    dbp = int(np.clip(58 + 0.28 * hr, 58, 100))
    hb = float(np.clip(13.2 + 0.02 * (hr - 70), 11.0, 16.5))
    return (sbp, dbp), hb

def _estimate_camera_fps(ts_buffer: deque, fallback: float = 30.0) -> float:
    if len(ts_buffer) < 10:
        return fallback
    ts = np.asarray(list(ts_buffer), dtype=np.float64)
    d = np.diff(ts)
    d = d[d > 1e-4]
    if d.size == 0:
        return fallback
    d_sorted = np.sort(d)
    trim = max(1, len(d_sorted) // 10)
    d_trimmed = d_sorted[trim:-trim] if len(d_sorted) > 2 * trim else d_sorted
    fps = 1.0 / float(np.mean(d_trimmed))
    return float(np.clip(fps, 10.0, 120.0))

def _effective_fs_from_timestamps(ts_buffer: deque, fallback: float = 30.0) -> float:
    if len(ts_buffer) < 2:
        return fallback
    ts = np.asarray(list(ts_buffer), dtype=np.float64)
    dt = float(ts[-1] - ts[0])
    if dt <= 1e-6:
        return fallback
    fs = float(len(ts) / dt)
    return float(np.clip(fs, 8.0, 120.0))

def _green_trace_from_faces(raw_face_buffer: deque) -> Optional[np.ndarray]:
    if len(raw_face_buffer) < 16:
        return None
    arr = np.asarray(list(raw_face_buffer), dtype=np.float32)
    g = arr[:, :, :, 1].mean(axis=(1, 2))
    g = g - np.mean(g)
    if np.std(g) < 1e-6:
        return None
    return g.astype(np.float32)

def _confidence_from_snr_sqi(snr_db: float, sqi: float) -> float:
    snr_term = float(np.clip((snr_db + 6.0) / 18.0, 0.0, 1.0))
    sqi_term = float(np.clip(sqi / 100.0, 0.0, 1.0))
    return 0.6 * snr_term + 0.4 * sqi_term

class HRTemporalTracker:
    def __init__(self, max_hist: int = 25) -> None:
        self.value: Optional[float] = None
        self.hist: deque = deque(maxlen=max_hist)

    def reset(self) -> None:
        self.value = None
        self.hist.clear()

    def update(self, measurement: float, snr_db: float, sqi: float) -> float:
        if not np.isfinite(measurement):
            return float(self.value if self.value is not None else 0.0)

        conf = _confidence_from_snr_sqi(snr_db, sqi)
        gate = 14.0 if conf >= 0.7 else 9.0

        if self.value is not None:
            delta = measurement - self.value
            if abs(delta) > gate:
                measurement = self.value + float(np.clip(delta, -gate, gate))

        alpha = 0.10 + 0.55 * conf
        if self.value is None:
            self.value = float(measurement)
        else:
            self.value = float((1.0 - alpha) * self.value + alpha * measurement)

        self.hist.append(self.value)
        if len(self.hist) >= 6:
            w = np.asarray(self.hist, dtype=np.float32)
            med = float(np.median(w))
            mad = float(np.median(np.abs(w - med))) + 1e-6
            if abs(self.value - med) > max(3.5, 3.0 * mad):
                self.value = float(0.75 * self.value + 0.25 * med)

        return float(self.value)

def _configure_camera_normal(cap: cv2.VideoCapture, logs: deque) -> None:
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except Exception:
        pass

    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Camera running in normal auto mode.")

def main() -> None:
    st.set_page_config(page_title="rPPG Telehealth Portal", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    now_str = datetime.now().strftime("%a, %b %d, %Y | %H:%M")
    st.markdown(
        f"""
        <div class="header-shell">
          <div class="title-row">
            <div>Robust Remote Photoplethsmography Telehealth Portal <span style="font-weight: 400; color: #94a3b8; font-size: 18px;">| Live Inference</span></div>
            <div><span class="live-pill">● LIVE</span> &nbsp; <span style="font-size: 16px; font-weight: 500;">{now_str}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown("### Model Settings")
    use_tta = st.sidebar.checkbox("Enable test-time adaptation", value=True)
    use_locking = st.sidebar.checkbox("Lock final HR reading", value=True)
    use_model = st.sidebar.checkbox("Use deep model in fusion", value=False)
    manual_freeze = st.sidebar.checkbox("Freeze displayed HR (manual)", value=False)
    
    st.sidebar.markdown("---")
    default_ckpt = _resolve_default_checkpoint()
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device, checkpoint_path=checkpoint_path if checkpoint_path else None)
    
    roi_extractor = FaceROIExtractor(target_size=64)
    api_mode = getattr(roi_extractor, "_api_mode", "none")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Diagnostics")
    if api_mode == "tasks":
        st.sidebar.success("✓ FaceLandmarker (Tasks API)")
    elif api_mode == "legacy":
        st.sidebar.success("✓ FaceMesh (Legacy API)")
    elif api_mode == "cascade":
        st.sidebar.warning("⚠ OpenCV Cascade Fallback")
    else:
        st.sidebar.error("✖ No face detector available")

    # ── State variables ──
    TARGET_FS = 30.0
    CALIBRATION_SECONDS = 10.0
    HR_EMA_ALPHA = 0.25
    SNR_SNAP_THRESHOLD = 3.0
    SNR_NOISY_THRESHOLD = -2.0
    BPM_SNAP_DELTA = 6.0
    TACHY_SNAP_SNR_DB = 7.5
    TACHY_SNAP_DELTA_BPM = 15.0
    MOTION_GATE_PX = 3.5
    NORM_CROP_SIZE = 128
    MODEL_FRAMES = 150
    TRAD_BUFFER_LEN = int(TARGET_FS * 10.0)
    MIN_TRAD_FRAMES = int(TARGET_FS * CALIBRATION_SECONDS)
    frame_buffer: deque = deque(maxlen=MODEL_FRAMES)
    trad_roi_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    trad_ts_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    trad_landmark_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    ts_buffer: deque = deque(maxlen=300)
    hr_buffer: deque = deque(maxlen=30)
    hr_long_buffer: deque = deque(maxlen=60)
    logs: deque = deque(maxlen=200)
    face_miss_count = 0
    prev_face_bbox: Optional[Tuple[int, int, int, int]] = None
    no_face_grace = 6
    last_infer_ts = 0.0
    inference_interval_s = 1.0
    last_hr = 0.0
    last_bvp: Optional[np.ndarray] = None
    ema_hr = 0.0
    ema_initialized = False
    display_hr = 0.0
    warmup_count = 0
    warmup_needed = 2
    hr_stability_window: deque = deque(maxlen=12)
    locked_hr: Optional[float] = None
    manual_locked_hr: Optional[float] = None
    unlock_votes = 0
    hr_tracker = HRTemporalTracker(max_hist=25)

    LOCK_MIN_POINTS = 8
    LOCK_MAX_SPREAD = 8.0
    LOCK_MAX_MAD = 3.0
    UNLOCK_DRIFT_BPM = 22.0
    UNLOCK_VOTES_REQUIRED = 12
    HR_BAND_LOW = 60.0
    HR_BAND_HIGH = 120.0

    last_spo2 = 0.0
    last_rr = 0.0
    last_sqi = 0.0
    ema_spo2 = 0.0
    ema_rr = 0.0
    spo2_initialized = False
    rr_initialized = False
    best_snr = -np.inf

    cascade = _load_cascade()
    if cascade.empty():
        st.error("OpenCV face cascade failed to load.")
        return

    # Adjusted column layout for better balance
    left_col, mid_col, right_col = st.columns([1.2, 1.5, 1.0])

    with left_col:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Video Feed</div>', unsafe_allow_html=True)
        frame_slot = st.empty()
        status_slot = st.empty()
        run = st.checkbox("Start Live Inference", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with mid_col:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Clinical Vitals</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: hr_slot = st.empty()
        with m2: spo2_slot = st.empty()
        with m3: rr_slot = st.empty()
        st.markdown('<br>', unsafe_allow_html=True)
        sqi_slot = st.empty()
        wave_slot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Diagnostics Log</div>', unsafe_allow_html=True)
        log_slot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        with left_col:
            st.info("Toggle 'Start Live Inference' to connect webcam.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    _configure_camera_normal(cap, logs)
    adapted = False

    try:
        while run:
            frame = _capture_webcam_frame(cap)
            if frame is None:
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Frame capture failed.")
                break

            frame_ts = time.time()
            ts_buffer.append(frame_ts)
            fps_est = _estimate_camera_fps(ts_buffer, fallback=30.0)

            face_bbox = _detect_primary_face(frame, cascade)
            if face_bbox is None:
                face_miss_count += 1
            else:
                face_miss_count = 0

            if face_bbox is not None and prev_face_bbox is not None and _bbox_iou(face_bbox, prev_face_bbox) < 0.20:
                frame_buffer.clear(); trad_roi_buffer.clear(); trad_ts_buffer.clear(); trad_landmark_buffer.clear()
                hr_buffer.clear(); hr_long_buffer.clear()
                adapted = False; ema_initialized = False; last_bvp = None; last_infer_ts = 0.0; last_hr = 0.0
                display_hr = 0.0; warmup_count = 0; hr_stability_window.clear(); locked_hr = None
                manual_locked_hr = None; unlock_votes = 0; hr_tracker.reset(); spo2_initialized = False
                rr_initialized = False; last_spo2 = 0.0; last_rr = 0.0; last_sqi = 0.0; best_snr = -np.inf
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Face switched. Re-calibrating.")

            prev_face_bbox = face_bbox if face_bbox is not None else prev_face_bbox

            framed = _draw_roi_overlay(frame, face_bbox)
            frame_slot.image(framed[:, :, ::-1], channels="RGB", width="stretch")

            if face_miss_count >= no_face_grace:
                frame_buffer.clear(); trad_roi_buffer.clear(); trad_ts_buffer.clear(); trad_landmark_buffer.clear()
                hr_buffer.clear(); hr_long_buffer.clear()
                adapted = False; ema_initialized = False; last_bvp = None; last_infer_ts = 0.0; last_hr = 0.0
                prev_face_bbox = None; display_hr = 0.0; warmup_count = 0; hr_stability_window.clear()
                locked_hr = None; manual_locked_hr = None; unlock_votes = 0; hr_tracker.reset()
                spo2_initialized = False; rr_initialized = False; last_spo2 = 0.0; last_rr = 0.0
                last_sqi = 0.0; best_snr = -np.inf

                with hr_slot: _render_metric_card("HEART RATE", "--", "BPM")
                with spo2_slot: _render_metric_card("SpO\u2082 ESTIMATE", "--", "%")
                with rr_slot: _render_metric_card("RESP RATE", "--", "br/min")
                sqi_slot.progress(0.0, text="Signal Quality: -- / No face detected")
                wave_slot.empty()
                status_slot.warning("No face detected. Please align face in frame.")
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] No face detected.")
                log_slot.markdown(f'<div class="logbox">{chr(10).join(list(logs)[:30])}</div>', unsafe_allow_html=True)
                continue

            if face_bbox is not None:
                roi_rgb, roi_mask, roi_landmarks = roi_extractor.extract_with_landmarks(frame)
                frame_buffer.append(roi_rgb)

                mask3 = np.repeat((roi_mask > 0.5).astype(np.float32)[..., None], 3, axis=2)
                masked_roi = roi_rgb.astype(np.float32) * mask3
                trad_crop_norm = cv2.resize(masked_roi, (NORM_CROP_SIZE, NORM_CROP_SIZE), interpolation=cv2.INTER_AREA)
                trad_roi_buffer.append(trad_crop_norm.astype(np.float32))
                trad_ts_buffer.append(frame_ts)
                if roi_landmarks is not None:
                    trad_landmark_buffer.append(np.asarray(roi_landmarks, dtype=np.float32))

            if len(trad_roi_buffer) < MIN_TRAD_FRAMES:
                pct = len(trad_roi_buffer) / float(MIN_TRAD_FRAMES)
                status_slot.progress(pct, text=f"Calibrating... Buffering {len(trad_roi_buffer)}/{MIN_TRAD_FRAMES} frames")
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Calibrating... buffering {len(trad_roi_buffer)}/{MIN_TRAD_FRAMES}")
                log_slot.markdown(f'<div class="logbox">{chr(10).join(list(logs)[:30])}</div>', unsafe_allow_html=True)
                continue

            now_ts = time.time()
            should_infer = (not ema_initialized) or ((now_ts - last_infer_ts) >= inference_interval_s)

            if should_infer:
                last_infer_ts = now_ts
                motion_disp_px = compute_landmark_motion_displacement(list(trad_landmark_buffer), min_frames=MIN_TRAD_FRAMES)
                if np.isfinite(motion_disp_px) and motion_disp_px > MOTION_GATE_PX:
                    best_snr = -np.inf
                    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Motion detected ({motion_disp_px:.2f}px). Window rejected.")
                    status_slot.warning(f"Motion Detected ({motion_disp_px:.1f}px). Keep still.")
                    log_slot.markdown(f'<div class="logbox">{chr(10).join(list(logs)[:30])}</div>', unsafe_allow_html=True)
                    continue

                fs_effective = _effective_fs_from_timestamps(trad_ts_buffer, fallback=fps_est)
                trad_ts = np.asarray(list(trad_ts_buffer), dtype=np.float64)

                pos_hr, pos_snr = estimate_hr_from_rgb_pos(
                    trad_roi_buffer, fps=fs_effective, f_low=1.0, f_high=2.0, 
                    timestamps=trad_ts, target_fs=fs_effective, prev_hr_bpm=None,
                )
                chrom_hr, chrom_snr = estimate_hr_from_rgb_chrom(
                    trad_roi_buffer, fps=fs_effective, f_low=1.0, f_high=2.0,
                    timestamps=trad_ts, target_fs=fs_effective, prev_hr_bpm=None,
                )
                green_trace = _green_trace_from_faces(trad_roi_buffer)
                ac_hr, ac_score = estimate_hr_from_autocorr(
                    green_trace, fps=fs_effective, f_low=0.8, f_high=2.5,
                    timestamps=trad_ts, target_fs=fs_effective,
                ) if green_trace is not None else (0.0, 0.0)

                model_hr = 0.0
                model_snr = -np.inf
                if use_model and len(frame_buffer) >= MODEL_FRAMES:
                    clip = build_clip_tensor_from_buffer(frame_buffer, device=device)
                    with torch.enable_grad() if (use_tta and not adapted) else torch.no_grad():
                        if use_tta and not adapted:
                            out = test_time_adaptation(model=model, clip_bcthw=clip, fps=fps_est, steps=8, lr=1e-4)
                            adapted = True
                            logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] TTA complete.")
                        else:
                            out = model(clip)
                            out["hr_bpm"] = estimate_hr_bpm_from_bvp(out["bvp_pred"], fps=fps_est)
                    model_hr = float(out["hr_bpm"][0].item())
                    last_bvp = out["bvp_pred"][0].detach().float().cpu().numpy()
                    model_snr = compute_pulse_snr(last_bvp, fps=fps_est)
                elif not use_model:
                    last_bvp = None

                quality_sig = last_bvp if last_bvp is not None else green_trace
                sqi = compute_signal_quality_index(quality_sig, fps=fs_effective) if quality_sig is not None else 0.0
                spo2_raw = estimate_spo2_from_rgb(trad_roi_buffer, fps=fs_effective)
                rr_raw = estimate_respiratory_rate(quality_sig, fps=fs_effective) if quality_sig is not None else 0.0

                candidates = []
                traditional_candidates = []
                if np.isfinite(pos_hr) and HR_BAND_LOW <= pos_hr <= HR_BAND_HIGH:
                    c = ("POS", pos_hr, pos_snr if np.isfinite(pos_snr) else -10.0)
                    traditional_candidates.append(c)
                    candidates.append(c)
                if np.isfinite(chrom_hr) and HR_BAND_LOW <= chrom_hr <= HR_BAND_HIGH:
                    c = ("CHROM", chrom_hr, chrom_snr if np.isfinite(chrom_snr) else -10.0)
                    traditional_candidates.append(c)
                    candidates.append(c)

                if use_model and np.isfinite(model_hr) and HR_BAND_LOW <= model_hr <= HR_BAND_HIGH and model_snr > -2:
                    if traditional_candidates:
                        trad_med = float(np.median([c[1] for c in traditional_candidates]))
                        if abs(model_hr - trad_med) <= 18.0:
                            candidates.append(("MODEL", model_hr, model_snr))
                    else:
                        candidates.append(("MODEL", model_hr, model_snr))

                if candidates:
                    candidates.sort(key=lambda c: c[2], reverse=True)
                    best_label, best_hr_val, best_snr = candidates[0]

                    if HR_BAND_LOW <= best_hr_val <= 120:
                        doubled = best_hr_val * 2.0
                        if max(70.0, HR_BAND_LOW) <= doubled <= HR_BAND_HIGH:
                            support_base = sum(1 for _, h, _ in candidates if abs(h - best_hr_val) < 10)
                            support_double = sum(1 for _, h, _ in candidates if abs(h - doubled) < 12)
                            if support_double > support_base:
                                best_hr_val = doubled
                                best_label = f"{best_label}x2"

                    corrected = [(best_label, best_hr_val, best_snr)]
                    for label, hr_val, snr_val in candidates[1:]:
                        if abs(hr_val - best_hr_val * 2) < 10 and HR_BAND_LOW <= hr_val / 2 <= HR_BAND_HIGH:
                            corrected.append((label, hr_val / 2, snr_val))
                        elif abs(hr_val - best_hr_val / 2) < 10 and HR_BAND_LOW <= hr_val * 2 <= HR_BAND_HIGH:
                            corrected.append((label, hr_val * 2, snr_val))
                        else:
                            corrected.append((label, hr_val, snr_val))

                    if len(corrected) >= 2:
                        confirmed = any(abs(best_hr_val - c[1]) < 15 for c in corrected[1:])
                        hr_raw = best_hr_val if confirmed else float(np.median([c[1] for c in corrected]))
                    else:
                        hr_raw = best_hr_val

                    hr_buffer.append(hr_raw)
                    hr_long_buffer.append(hr_raw)
                    warmup_count += 1

                    tachy_snap_applied = False
                    if not ema_initialized:
                        ema_hr = hr_raw
                        display_hr = hr_raw
                        ema_initialized = True
                    else:
                        if (
                            np.isfinite(best_snr)
                            and best_snr >= TACHY_SNAP_SNR_DB
                            and hr_raw >= 100.0
                            and abs(hr_raw - ema_hr) > TACHY_SNAP_DELTA_BPM
                        ):
                            ema_hr = hr_raw
                            display_hr = hr_raw
                            tachy_snap_applied = True
                            logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Tachycardia snap: {hr_raw:.0f} BPM (SNR={best_snr:.1f}dB).")
                        elif best_snr < SNR_NOISY_THRESHOLD:
                            ema_alpha = 0.05
                            logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Signal noisy – Recalculating...")
                        elif best_snr >= SNR_SNAP_THRESHOLD and abs(hr_raw - ema_hr) >= BPM_SNAP_DELTA:
                            ema_alpha = 0.8
                            logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] High-SNR snap (α=0.80): {hr_raw:.0f} BPM.")
                        else:
                            ema_alpha = HR_EMA_ALPHA
                        
                        if not tachy_snap_applied:
                            ema_hr = (ema_alpha * hr_raw) + ((1.0 - ema_alpha) * ema_hr)
                            display_hr = ema_hr

                    last_hr = float(display_hr)

                    if tachy_snap_applied and use_locking:
                        locked_hr = None
                        unlock_votes = 0
                        hr_stability_window.clear()

                    if use_locking:
                        quality_ok = bool(sqi >= 50.0 and np.isfinite(best_snr) and best_snr >= -1.0)
                        if quality_ok:
                            hr_stability_window.append(hr_raw)

                        if locked_hr is None:
                            if len(hr_stability_window) >= LOCK_MIN_POINTS:
                                w = np.asarray(list(hr_stability_window), dtype=np.float32)
                                med = float(np.median(w))
                                mad = float(np.median(np.abs(w - med)))
                                spread = float(np.max(w) - np.min(w))
                                if mad <= LOCK_MAX_MAD and spread <= LOCK_MAX_SPREAD:
                                    if len(hr_long_buffer) >= 8:
                                        locked_hr = float(np.median(np.asarray(list(hr_long_buffer)[-20:], dtype=np.float32)))
                                    else:
                                        locked_hr = med
                                    unlock_votes = 0
                                    display_hr = locked_hr
                                    last_hr = float(locked_hr)
                                    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR locked at {locked_hr:.0f} BPM.")
                        else:
                            display_hr = locked_hr
                            if quality_ok and abs(hr_raw - locked_hr) > UNLOCK_DRIFT_BPM and best_snr >= 3.0:
                                unlock_votes += 1
                            elif quality_ok:
                                unlock_votes = max(0, unlock_votes - 1)

                            if unlock_votes >= UNLOCK_VOTES_REQUIRED:
                                recent_lock = list(hr_long_buffer)[-20:] if len(hr_long_buffer) >= 8 else [hr_raw]
                                locked_hr = float(np.median(np.asarray(recent_lock, dtype=np.float32)))
                                display_hr = locked_hr
                                last_hr = float(locked_hr)
                                unlock_votes = 0
                                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR re-locked at {locked_hr:.0f} BPM.")
                else:
                    if last_hr > 0:
                        display_hr = float(last_hr)
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Window rejected by weighted PSD/SQI. Holding {last_hr:.0f} BPM.")

                if manual_freeze:
                    if manual_locked_hr is None and warmup_count >= warmup_needed:
                        manual_locked_hr = float(display_hr)
                    if manual_locked_hr is not None:
                        display_hr = manual_locked_hr
                else:
                    manual_locked_hr = None

                if spo2_raw > 0:
                    if not spo2_initialized:
                        ema_spo2 = spo2_raw
                        spo2_initialized = True
                    else:
                        ema_spo2 = 0.8 * ema_spo2 + 0.2 * spo2_raw
                    last_spo2 = ema_spo2

                if rr_raw > 0:
                    if not rr_initialized:
                        ema_rr = rr_raw
                        rr_initialized = True
                    else:
                        ema_rr = 0.1 * rr_raw + 0.9 * ema_rr
                    last_rr = ema_rr

                last_sqi = sqi
                val_str = " ".join(f"{l}={v:.0f}({s:.1f}dB)" for l, v, s in corrected) if candidates else "none"
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR={display_hr:.0f} [{val_str}] sqi={sqi:.0f} fps={fs_effective:.1f}")

            # ── Display metrics ──
            with hr_slot:
                if warmup_count < warmup_needed:
                    _render_metric_card("HEART RATE", "--", "BPM")
                else:
                    _render_metric_card("HEART RATE", f"{int(round(display_hr))}", "BPM")
            with spo2_slot:
                if last_spo2 > 0:
                    _render_metric_card("SpO\u2082 ESTIMATE", f"{last_spo2:.0f}", "%")
                else:
                    _render_metric_card("SpO\u2082 ESTIMATE", "--", "%")
            with rr_slot:
                if last_rr > 0:
                    _render_metric_card("RESP RATE", f"{last_rr:.0f}", "br/min")
                else:
                    _render_metric_card("RESP RATE", "--", "br/min")

            if last_sqi > 0:
                q_label = "Good Signal" if last_sqi >= 60 else ("Fair Signal" if last_sqi >= 30 else "Poor \u2014 Keep Still")
                sqi_slot.progress(min(last_sqi / 100.0, 1.0), text=f"Signal Quality: {last_sqi:.0f}/100 \u2014 {q_label}")
            else:
                sqi_slot.progress(0.0, text="Signal Quality: Waiting for data...")

            if last_bvp is not None:
                wave_slot.line_chart(last_bvp, width="stretch")

            if warmup_count < warmup_needed:
                status_slot.info(f"Stabilizing readings... ({warmup_count}/{warmup_needed})")
            elif manual_freeze and manual_locked_hr is not None:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.success(f"Manual lock: {manual_locked_hr:.0f} BPM | fps={fps_est:.0f}{snr_str}")
            elif use_locking and locked_hr is not None:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.success(f"Locked HR: {locked_hr:.0f} BPM | fps={fps_est:.0f}{snr_str}")
            elif ema_initialized:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.info(f"Measuring... fps={fps_est:.0f}{snr_str}")
            else:
                status_slot.info("Initializing...")

            log_slot.markdown(f'<div class="logbox">{chr(10).join(list(logs)[:30])}</div>', unsafe_allow_html=True)

    finally:
        cap.release()

if __name__ == "__main__":
    main()