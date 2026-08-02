"""video_analysis.py — offline rPPG analysis of uploaded video clips.

Designed for researchers who want to validate EquiPhys against existing
dataset videos (UBFC, MMPD, PURE, etc.) or any face video where ground-truth
HR/RR is available from a reference instrument.

Pipeline
--------
1. Decode the uploaded video frame-by-frame with OpenCV.
2. Extract the face ROI from every frame using the same FaceROIExtractor
   used in live inference (MediaPipe Tasks → FaceMesh → cascade fallback).
3. Apply optional camera calibration (colour gain, lighting normalisation).
4. Run the full classical rPPG pipeline (POS + CHROM patch-weighted fusion)
   on the complete buffer — no rolling window, one estimate per clip.
5. Compute HR, RR, SpO2 (provisional), SQI, and a per-second HR trace for
   plotting against ground truth.
6. If the user provides a ground-truth HR value, compute MAE and relative error.

All results are returned as a plain dict so the Streamlit page can render
them however it likes.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from equiphys_core import (
    FaceROIExtractor,
    _extract_patch_weighted_pulse,
    _resample_with_timestamps,
    _welch_peak_hz,
    compute_signal_quality_index,
    estimate_respiratory_rate,
    estimate_spo2_from_rgb,
)
from facial_vitals import (
    clean_frame_mask,
    frame_diff_motion,
    reconstruct_facial_bvp,
    select_clean_frames,
)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoAnalysisResult:
    # Core estimates
    hr_bpm: Optional[float] = None          # final fused HR
    hr_pos_bpm: Optional[float] = None      # POS channel estimate
    hr_chrom_bpm: Optional[float] = None    # CHROM channel estimate
    rr_bpm: Optional[float] = None          # respiratory rate
    spo2_pct: Optional[float] = None        # provisional SpO2

    # Quality
    snr_db: float = -np.inf
    sqi: float = 0.0
    clean_frame_frac: float = 0.0           # fraction of frames that passed motion gate
    fps_detected: float = 0.0
    total_frames: int = 0
    face_detected_frames: int = 0
    duration_s: float = 0.0

    # Waveforms for plotting
    bvp_waveform: Optional[np.ndarray] = None   # full-clip POS pulse waveform
    hr_trace: Optional[np.ndarray] = None        # per-second HR estimate (sliding windows)
    hr_trace_times: Optional[np.ndarray] = None  # timestamps for hr_trace

    # Validation (filled if ground truth provided)
    gt_hr_bpm: Optional[float] = None
    mae_bpm: Optional[float] = None
    rel_err_pct: Optional[float] = None

    # Metadata
    video_name: str = ""
    analysis_duration_s: float = 0.0
    method_used: str = "pos+chrom"
    warnings: List[str] = field(default_factory=list)
    per_window: List[Dict] = field(default_factory=list)  # per-second window details


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis function
# ──────────────────────────────────────────────────────────────────────────────

def analyse_video_file(
    video_path: str,
    gt_hr_bpm: Optional[float] = None,
    target_size: int = 64,
    norm_crop_size: int = 128,
    hr_band_low: float = 0.8,
    hr_band_high: float = 2.5,
    motion_mad_k: float = 3.5,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cam_cal=None,
    max_frames: Optional[int] = None,
    window_stride_s: float = 1.0,
    window_len_s: float = 10.0,
    snr_ratio_threshold: float = 0.12,
) -> VideoAnalysisResult:
    """Run the full rPPG pipeline on a video file.

    Args:
        video_path:      path to any OpenCV-readable video (mp4, avi, mov …).
        gt_hr_bpm:       optional ground-truth HR for validation metrics.
        target_size:     ROI resize target fed to FaceROIExtractor.
        norm_crop_size:  spatial resolution used for RGB trace extraction.
        hr_band_low/high: physiological HR search band in Hz.
        motion_mad_k:    MAD multiplier for per-frame motion rejection.
        progress_cb:     optional callback(fraction 0–1, message) for UI.
        cam_cal:         optional CameraCalibration from camera_calibration.py.
        max_frames:      cap total frames decoded (useful for quick previews).
        window_stride_s: stride of the sliding HR-trace window in seconds.
        window_len_s:    length of each sliding window in seconds.

    Returns:
        VideoAnalysisResult with all estimates and waveforms filled.
    """
    t0 = time.time()
    res = VideoAnalysisResult(video_name=Path(video_path).name, gt_hr_bpm=gt_hr_bpm)

    def _prog(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(float(np.clip(frac, 0.0, 1.0)), msg)

    # ── 1. Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        res.warnings.append(f"Cannot open video: {video_path}")
        return res

    fps_meta = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_meta <= 0:
        total_frames_meta = 99999  # unknown length

    limit = max_frames if max_frames else total_frames_meta
    res.fps_detected = fps_meta
    res.total_frames = min(total_frames_meta, limit)
    res.duration_s = res.total_frames / max(fps_meta, 1.0)

    _prog(0.0, f"Opened '{res.video_name}' — {res.total_frames} frames @ {fps_meta:.1f} fps")

    # ── 2. ROI extraction ────────────────────────────────────────────────────
    roi_extractor = FaceROIExtractor(target_size=target_size)

    roi_buffer: List[np.ndarray] = []
    ts_buffer: List[float] = []
    face_detected = 0
    frame_idx = 0

    while frame_idx < limit:
        ok, bgr = cap.read()
        if not ok:
            break

        frame_ts = frame_idx / max(fps_meta, 1.0)
        roi_rgb, roi_mask, _ = roi_extractor.extract_with_landmarks(bgr)

        # Check face was actually found (non-trivial mask coverage)
        if roi_mask.mean() > 0.1:
            face_detected += 1

            # Optional camera calibration
            if cam_cal is not None:
                try:
                    from camera_calibration import apply_camera_calibration
                    roi_rgb = apply_camera_calibration(roi_rgb, cam_cal)
                except Exception:
                    pass

            mask3 = np.repeat((roi_mask > 0.5)[..., None], 3, axis=2).astype(np.float32)
            masked = roi_rgb.astype(np.float32) * mask3
            crop = cv2.resize(masked, (norm_crop_size, norm_crop_size),
                              interpolation=cv2.INTER_AREA)
            roi_buffer.append(crop.astype(np.float32))
            ts_buffer.append(frame_ts)

        frame_idx += 1
        if frame_idx % 30 == 0:
            _prog(0.1 + 0.5 * frame_idx / limit,
                  f"Extracting ROIs… {frame_idx}/{limit} frames")

    cap.release()

    res.face_detected_frames = face_detected
    res.total_frames = frame_idx

    if len(roi_buffer) < 60:
        res.warnings.append(
            f"Only {len(roi_buffer)} usable face frames extracted — "
            "need at least 60. Check that the video contains a visible frontal face."
        )
        res.analysis_duration_s = time.time() - t0
        return res

    _prog(0.6, f"Extracted {len(roi_buffer)} ROI frames. Running motion filter…")

    # ── 3. Motion filtering ──────────────────────────────────────────────────
    motion = frame_diff_motion(roi_buffer)
    keep_mask = clean_frame_mask(motion, k=motion_mad_k)
    clean_frames, clean_ts = select_clean_frames(
        roi_buffer, np.asarray(ts_buffer, dtype=np.float64), keep_mask
    )
    res.clean_frame_frac = float(keep_mask.mean()) if keep_mask.size else 0.0

    if len(clean_frames) < 45:
        res.warnings.append(
            f"Too many high-motion frames — only {len(clean_frames)} clean frames remain. "
            "Results may be unreliable."
        )

    _prog(0.65, f"Motion filter: {len(clean_frames)}/{len(roi_buffer)} frames kept "
                f"({res.clean_frame_frac*100:.0f}%). Running spectral analysis…")

    ts_arr = np.asarray(clean_ts, dtype=np.float64) if clean_ts is not None else None
    fs_eff = _effective_fs(ts_arr, fallback=fps_meta)

    # ── 4. Full-clip POS and CHROM estimates ────────────────────────────────
    pos_hr, pos_snr, pos_pulse = _run_method(clean_frames, ts_arr, fs_eff,
                                              "pos", hr_band_low, hr_band_high,
                                              snr_ratio_threshold=snr_ratio_threshold)
    chrom_hr, chrom_snr, chrom_pulse = _run_method(clean_frames, ts_arr, fs_eff,
                                                    "chrom", hr_band_low, hr_band_high,
                                                    snr_ratio_threshold=snr_ratio_threshold)

    res.hr_pos_bpm = pos_hr if pos_hr > 0 else None
    res.hr_chrom_bpm = chrom_hr if chrom_hr > 0 else None

    # Pick the higher-SNR estimate as the primary
    if pos_snr >= chrom_snr and pos_hr > 0:
        best_hr, best_snr, best_pulse = pos_hr, pos_snr, pos_pulse
        res.method_used = "pos"
    elif chrom_hr > 0:
        best_hr, best_snr, best_pulse = chrom_hr, chrom_snr, chrom_pulse
        res.method_used = "chrom"
    else:
        best_hr, best_snr, best_pulse = 0.0, -np.inf, None
        res.warnings.append("Neither POS nor CHROM found a valid HR peak.")

    # If both valid and agree, average them (reduces single-method bias)
    if pos_hr > 0 and chrom_hr > 0 and abs(pos_hr - chrom_hr) < 10.0:
        w_pos = max(0.0, pos_snr + 10.0)
        w_chr = max(0.0, chrom_snr + 10.0)
        denom = w_pos + w_chr
        if denom > 0:
            best_hr = (w_pos * pos_hr + w_chr * chrom_hr) / denom
            res.method_used = "pos+chrom (fused)"

    res.hr_bpm = best_hr if best_hr > 0 else None
    res.snr_db = best_snr
    res.bvp_waveform = best_pulse

    _prog(0.75, f"HR estimate: {best_hr:.1f} BPM (SNR={best_snr:.1f} dB). "
                "Computing RR and SpO₂…")

    # ── 5. RR and SpO2 ───────────────────────────────────────────────────────
    if best_pulse is not None and len(best_pulse) >= 90:
        rr = estimate_respiratory_rate(best_pulse, fps=fs_eff)
        res.rr_bpm = rr if rr > 0 else None
        sqi = compute_signal_quality_index(best_pulse, fps=fs_eff)
        res.sqi = sqi
    else:
        res.warnings.append("BVP waveform too short for RR / SQI.")

    if len(clean_frames) >= 60:
        spo2 = estimate_spo2_from_rgb(clean_frames, fps=fs_eff)
        res.spo2_pct = spo2 if spo2 > 0 else None

    # ── 6. Sliding-window HR trace ───────────────────────────────────────────
    _prog(0.85, "Building per-second HR trace…")
    res.hr_trace, res.hr_trace_times = _sliding_hr_trace(
        clean_frames, ts_arr, fs_eff,
        window_len_s=window_len_s,
        stride_s=window_stride_s,
        f_low=hr_band_low,
        f_high=hr_band_high,
        snr_ratio_threshold=snr_ratio_threshold,
    )

    # ── 7. Per-window detail table ───────────────────────────────────────────
    res.per_window = _build_window_table(
        clean_frames, ts_arr, fs_eff, hr_band_low, hr_band_high,
        window_len_s=window_len_s, stride_s=window_stride_s,
        snr_ratio_threshold=snr_ratio_threshold,
    )

    # ── 8. Validation metrics ─────────────────────────────────────────────────
    if gt_hr_bpm is not None and res.hr_bpm is not None:
        res.mae_bpm = abs(res.hr_bpm - gt_hr_bpm)
        res.rel_err_pct = 100.0 * res.mae_bpm / max(gt_hr_bpm, 1.0)

    res.analysis_duration_s = time.time() - t0
    _prog(1.0, f"Done in {res.analysis_duration_s:.1f} s.")
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _effective_fs(ts: Optional[np.ndarray], fallback: float = 30.0) -> float:
    if ts is None or len(ts) < 2:
        return fallback
    dt = float(ts[-1] - ts[0])
    if dt <= 1e-6:
        return fallback
    return float(np.clip(len(ts) / dt, 8.0, 120.0))


def _run_method(
    frames: List[np.ndarray],
    ts: Optional[np.ndarray],
    fs: float,
    method: str,
    f_low: float,
    f_high: float,
    snr_ratio_threshold: float = 0.12,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """Run one projection method on the full frame buffer. Returns (hr_bpm, snr, pulse).

    snr_ratio_threshold is deliberately lower than the live-inference default (0.35)
    because dataset videos often have heavier compression and different noise profiles.
    0.12 accepts peaks that hold at least 12% of total in-band power.
    """
    try:
        pulse, fs_used = _extract_patch_weighted_pulse(
            frames, timestamps=ts, target_fs=fs, method=method
        )
        peak_hz, snr = _welch_peak_hz(
            pulse, fs_used, f_low=f_low, f_high=f_high,
            snr_ratio_threshold=snr_ratio_threshold,
        )
        hr = peak_hz * 60.0 if peak_hz > 0 else 0.0
        return hr, snr, pulse.astype(np.float32)
    except Exception as exc:
        return 0.0, -np.inf, None


def _sliding_hr_trace(
    frames: List[np.ndarray],
    ts: Optional[np.ndarray],
    fs: float,
    window_len_s: float,
    stride_s: float,
    f_low: float,
    f_high: float,
    snr_ratio_threshold: float = 0.12,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute a per-second HR trace using overlapping windows."""
    win_n = max(30, int(window_len_s * fs))
    stride_n = max(1, int(stride_s * fs))
    n = len(frames)
    if n < win_n:
        return None, None

    hrs, times = [], []
    for start in range(0, n - win_n + 1, stride_n):
        end = start + win_n
        seg = frames[start:end]
        seg_ts = ts[start:end] if ts is not None else None
        hr, _, _ = _run_method(seg, seg_ts, fs, "pos", f_low, f_high,
                               snr_ratio_threshold=snr_ratio_threshold)
        if hr > 0:
            t_mid = float(ts[(start + end) // 2]) if ts is not None else (start + win_n / 2) / fs
            hrs.append(hr)
            times.append(t_mid)

    if not hrs:
        return None, None
    return np.array(hrs, dtype=np.float32), np.array(times, dtype=np.float32)


def _build_window_table(
    frames: List[np.ndarray],
    ts: Optional[np.ndarray],
    fs: float,
    f_low: float,
    f_high: float,
    window_len_s: float,
    stride_s: float,
    snr_ratio_threshold: float = 0.12,
) -> List[Dict]:
    """Build a per-window detail table (time, POS HR, CHROM HR, SNR)."""
    win_n = max(30, int(window_len_s * fs))
    stride_n = max(1, int(stride_s * fs))
    n = len(frames)
    rows = []
    for start in range(0, n - win_n + 1, stride_n):
        end = start + win_n
        seg = frames[start:end]
        seg_ts = ts[start:end] if ts is not None else None
        t_start = float(ts[start]) if ts is not None else start / fs
        t_end = float(ts[min(end - 1, len(ts) - 1)]) if ts is not None else end / fs
        pos_hr, pos_snr, _ = _run_method(seg, seg_ts, fs, "pos", f_low, f_high,
                                          snr_ratio_threshold=snr_ratio_threshold)
        chrom_hr, chrom_snr, _ = _run_method(seg, seg_ts, fs, "chrom", f_low, f_high,
                                             snr_ratio_threshold=snr_ratio_threshold)
        rows.append({
            "t_start_s": round(t_start, 2),
            "t_end_s": round(t_end, 2),
            "pos_hr": round(pos_hr, 1) if pos_hr > 0 else None,
            "chrom_hr": round(chrom_hr, 1) if chrom_hr > 0 else None,
            "pos_snr_db": round(pos_snr, 1) if np.isfinite(pos_snr) else None,
            "chrom_snr_db": round(chrom_snr, 1) if np.isfinite(chrom_snr) else None,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_video_analysis_tab(st, cam_cal=None) -> None:
    """Render the full Video Analysis tab inside Streamlit.

    Call this from inside a `with tab:` block in streamlit_app.py.
    """
    st.markdown(
        "<div style='color:#94a3b8;font-size:13px;margin-bottom:16px;'>"
        "Upload any face video (mp4, avi, mov, mkv) to get HR, RR, and SpO₂ estimates "
        "using the same POS+CHROM pipeline as live inference. Useful for validating "
        "against dataset ground truth (UBFC, MMPD, PURE, iBVP, etc.)."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

    with ctrl1:
        uploaded = st.file_uploader(
            "Upload video clip",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            key="_vid_uploader",
            help="Any face video. Longer clips (≥15 s) give more stable estimates.",
        )

    with ctrl2:
        gt_hr = st.number_input(
            "Ground-truth HR (BPM)",
            min_value=30.0, max_value=220.0, value=None, step=0.5,
            placeholder="leave empty to skip",
            key="_vid_gt_hr",
            help="Optional. Enter the reference HR from your dataset metadata to compute MAE.",
        )
        gt_hr_val = float(gt_hr) if gt_hr is not None else None

    with ctrl3:
        max_dur = st.number_input(
            "Max clip duration (s, 0=full)",
            min_value=0, max_value=600, value=0, step=10,
            key="_vid_max_dur",
            help="Set > 0 to analyse only the first N seconds (faster for long videos).",
        )

    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    with adv_col1:
        win_len = st.slider("HR window length (s)", 5, 30, 10, key="_vid_win_len")
    with adv_col2:
        hr_low = st.slider("HR band low (BPM)", 40, 80, 48, key="_vid_hr_low")
    with adv_col3:
        hr_high = st.slider("HR band high (BPM)", 100, 200, 150, key="_vid_hr_high")
    with adv_col4:
        snr_thresh = st.slider(
            "SNR threshold", 0.05, 0.40, 0.12, step=0.01,
            key="_vid_snr_thresh",
            help="Min fraction of in-band power a peak must hold to be accepted. "
                 "Lower = more permissive. Live inference uses 0.35; "
                 "compressed dataset videos typically need 0.08–0.15.",
        )

    run_btn = st.button("Analyse video", key="_vid_run", type="primary",
                        disabled=uploaded is None)

    if uploaded is None:
        st.info("Upload a video file above to begin.")
        return

    if not run_btn and "vid_result" not in st.session_state:
        return

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn:
        # Save upload to temp file (OpenCV needs a real path)
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        prog_bar = st.progress(0.0, text="Starting analysis…")
        status_txt = st.empty()

        def _cb(frac: float, msg: str) -> None:
            prog_bar.progress(frac, text=msg)
            status_txt.markdown(
                f"<div style='font-size:12px;color:#64748b;'>{msg}</div>",
                unsafe_allow_html=True,
            )

        # Compute max_frames from duration slider
        cap_probe = cv2.VideoCapture(tmp_path)
        _fps = float(cap_probe.get(cv2.CAP_PROP_FPS)) or 30.0
        cap_probe.release()
        max_f = int(max_dur * _fps) if max_dur > 0 else None

        result = analyse_video_file(
            video_path=tmp_path,
            gt_hr_bpm=gt_hr_val,
            hr_band_low=hr_low / 60.0,
            hr_band_high=hr_high / 60.0,
            window_len_s=float(win_len),
            max_frames=max_f,
            progress_cb=_cb,
            cam_cal=cam_cal,
            snr_ratio_threshold=float(snr_thresh),
        )
        st.session_state["vid_result"] = result
        prog_bar.progress(1.0, text="Analysis complete.")
        status_txt.empty()

    # ── Render results ────────────────────────────────────────────────────────
    result: VideoAnalysisResult = st.session_state.get("vid_result")
    if result is None:
        return

    _render_results(st, result)


def _render_results(st, r: VideoAnalysisResult) -> None:
    """Render the result cards, charts, and table."""

    # Warnings banner
    if r.warnings:
        for w in r.warnings:
            st.warning(w)

    # ── Top metric cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    def _card(col, label, value, unit, color_theme="blue", sub=""):
        val_str = f"{value:.1f}" if isinstance(value, float) else (str(value) if value is not None else "—")
        theme_colors = {
            "red": {"border": "#f43f5e", "glow": "rgba(244, 63, 94, 0.15)"},
            "cyan": {"border": "#06b6d4", "glow": "rgba(6, 182, 212, 0.15)"},
            "emerald": {"border": "#10b981", "glow": "rgba(16, 185, 129, 0.15)"},
            "purple": {"border": "#8b5cf6", "glow": "rgba(139, 92, 246, 0.15)"},
            "blue": {"border": "#3b82f6", "glow": "rgba(59, 130, 246, 0.15)"},
            "pink": {"border": "#ec4899", "glow": "rgba(236, 72, 153, 0.15)"},
        }
        c = theme_colors.get(color_theme, theme_colors["blue"])
        sub_html = f'<div class="metric-status">{sub}</div>' if sub else ""
        col.markdown(
            f"""
            <div class="metric-shell-new" style="border-left: 4px solid {c['border']}; --glow-color: {c['glow']};">
              <div class="metric-label-new">{label}</div>
              <div class="metric-value-new">{val_str}</div>
              <div class="metric-unit-new">{unit}</div>
              {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    _card(c1, "HEART RATE", r.hr_bpm, "BPM", "red", r.method_used)
    _card(c2, "RESP RATE", r.rr_bpm, "br/min", "emerald", "Spectral")
    _card(c3, "SpO₂ (prov.)", r.spo2_pct, "%", "cyan", "Provisional")
    _card(c4, "SIGNAL SNR", round(r.snr_db, 1) if np.isfinite(r.snr_db) else None, "dB", "blue", "POS peak")
    _card(c5, "SQI", round(r.sqi, 0), "/ 100", "pink", "Quality Index")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Validation row (only shown when GT provided) ──────────────────────────
    if r.gt_hr_bpm is not None:
        v1, v2, v3, v4 = st.columns(4)
        _card(v1, "GT HEART RATE", r.gt_hr_bpm, "BPM", "blue", "Reference")
        _card(v2, "MAE", r.mae_bpm, "BPM",
              "emerald" if r.mae_bpm is not None and r.mae_bpm < 5 else "red",
              "Mean Abs Error")
        _card(v3, "REL ERROR", r.rel_err_pct, "%",
              "emerald" if r.rel_err_pct is not None and r.rel_err_pct < 5 else "red",
              "Percentage")
        _card(v4, "POS vs GT", 
              round(abs(r.hr_pos_bpm - r.gt_hr_bpm), 1) if r.hr_pos_bpm and r.gt_hr_bpm else None,
              "BPM Δ", "blue", "Difference")
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column detail section ─────────────────────────────────────────────
    left, right = st.columns([1.6, 1.0])

    with left:
        # BVP waveform
        if r.bvp_waveform is not None and len(r.bvp_waveform) > 0:
            with st.container(border=True):
                st.markdown('<div class="panel-title">BVP Waveform</div>', unsafe_allow_html=True)
                import pandas as pd
                bvp_df = pd.DataFrame({"BVP": r.bvp_waveform})
                st.line_chart(bvp_df, height=160)

        # Per-second HR trace
        if r.hr_trace is not None and len(r.hr_trace) > 1:
            with st.container(border=True):
                st.markdown('<div class="panel-title">HR TRACE (BPM over time)</div>', unsafe_allow_html=True)
                import pandas as pd
                trace_df = pd.DataFrame({
                    "HR (BPM)": r.hr_trace,
                }, index=np.round(r.hr_trace_times, 1))
                if r.gt_hr_bpm is not None:
                    trace_df["Ground Truth"] = r.gt_hr_bpm
                st.line_chart(trace_df, height=180)

    with right:
        # Video + clip metadata
        with st.container(border=True):
            st.markdown('<div class="panel-title">Clip Metadata</div>', unsafe_allow_html=True)

            def _row(label, value):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:12px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                    f"<span style='color:#64748b;font-weight:500;'>{label}</span>"
                    f"<span style='color:#e2e8f0;font-family:monospace;font-weight:600;'>{value}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            _row("File", r.video_name)
            _row("Duration", f"{r.duration_s:.1f} s")
            _row("FPS (detected)", f"{r.fps_detected:.1f}")
            _row("Total frames", str(r.total_frames))
            _row("Face detected", f"{r.face_detected_frames} frames")
            _row("Clean frames", f"{r.clean_frame_frac*100:.0f}%")
            _row("POS HR", f"{r.hr_pos_bpm:.1f} BPM" if r.hr_pos_bpm else "—")
            _row("CHROM HR", f"{r.hr_chrom_bpm:.1f} BPM" if r.hr_chrom_bpm else "—")
            _row("Analysis time", f"{r.analysis_duration_s:.1f} s")

    # ── Per-window table ──────────────────────────────────────────────────────
    if r.per_window:
        with st.expander("Per-window detail table", expanded=False):
            import pandas as pd
            df = pd.DataFrame(r.per_window)
            df.columns = ["Start (s)", "End (s)", "POS HR", "CHROM HR",
                          "POS SNR (dB)", "CHROM SNR (dB)"]
            st.dataframe(df, use_container_width=True, height=260)

            csv = df.to_csv(index=False).encode()
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{Path(r.video_name).stem}_rppg_windows.csv",
                mime="text/csv",
                key="_vid_dl_csv",
            )