"""camera_calibration.py — runtime camera and signal calibration for EquiPhys.

This module handles four calibration axes that directly improve rPPG accuracy:

1. Camera colour calibration
   Captures RGB from known matte surfaces (grey card, white card, coloured
   patches) and computes per-channel gain corrections + white-balance offsets.
   Applied before any RGB trace extraction so POS/CHROM see corrected colour.

2. Per-session lighting baseline
   Records the illumination fingerprint of the current environment from a
   short still window. Used to normalise each new frame against the session
   baseline rather than relying solely on per-segment normalisation.

3. Motion gate calibration
   Estimates the noise floor of "zero motion" from the first few seconds so
   the MAD threshold adapts to the current camera's jitter level, not a
   hard-coded constant.

4. SpO2 device calibration (extends facial_vitals.Spo2Calibration)
   Wraps the existing Spo2Calibration.fit() and exposes it in the sidebar
   with a grey-card-based ratio-R capture helper so users can compute R on
   their own camera without owning a separate oximeter first.

Usage in streamlit_app.py
--------------------------
    from camera_calibration import (
        CameraCalibration,
        sidebar_camera_calibration_panel,
        apply_camera_calibration,
    )

    # In the sidebar block:
    cam_cal = sidebar_camera_calibration_panel(st, st.session_state)

    # Per-frame, before appending to trad_roi_buffer:
    corrected_roi = apply_camera_calibration(roi_rgb, cam_cal)
    trad_roi_buffer.append(corrected_roi)

    # Pass motion floor to clean_frame_mask:
    keep_mask = clean_frame_mask(
        motion, k=cam_cal.motion_mad_k, min_keep_frac=MOTION_CLEAN_MIN_FRAC
    )
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ColourCalibrationState:
    """Per-channel gain and white-balance correction derived from reference surfaces."""

    # Multiplicative gains applied to [R, G, B] of every ROI frame.
    gain: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))

    # Additive offsets (after gain) — used to remove residual DC bias per channel.
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # True once at least a grey-card capture has been completed.
    calibrated: bool = False

    # Human-readable summary of the last calibration.
    note: str = "Not yet calibrated — using unity gain."

    # Raw readings keyed by surface name: {name: [R, G, B] mean in [0,1]}
    surface_readings: Dict[str, np.ndarray] = field(default_factory=dict)

    def apply(self, roi_float: np.ndarray) -> np.ndarray:
        """Apply gain + offset correction to a float ROI [H,W,3] in [0,1]."""
        if not self.calibrated:
            return roi_float
        corrected = roi_float * self.gain[None, None, :] + self.offset[None, None, :]
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)


@dataclass
class LightingBaseline:
    """Per-session illumination fingerprint captured during a still window."""

    # Mean RGB of the ROI during the calibration window [3] in [0,1].
    mean_rgb: Optional[np.ndarray] = None

    # Standard deviation of each channel across the calibration window [3].
    std_rgb: Optional[np.ndarray] = None

    # Timestamp when the baseline was captured.
    captured_at: float = 0.0

    # True once at least BASELINE_FRAMES frames have been accumulated.
    ready: bool = False

    note: str = "No session baseline captured yet."

    def normalise(self, roi_float: np.ndarray) -> np.ndarray:
        """Normalise ROI against the captured illumination baseline.

        Divides each channel by the session mean so that slow lighting drift
        is suppressed without distorting the relative channel balance that
        POS/CHROM rely on.
        """
        if not self.ready or self.mean_rgb is None:
            return roi_float
        safe_mean = np.maximum(self.mean_rgb[None, None, :], 1e-4)
        normalised = roi_float / safe_mean
        return np.clip(normalised, 0.0, 2.0).astype(np.float32)


@dataclass
class MotionFloorState:
    """Noise floor of "zero-motion" pixel variance, adapted to this camera."""

    # Median frame-difference during the still calibration window.
    noise_floor: float = 0.002

    # Adaptive MAD multiplier: tighter for stable cameras, looser for noisy ones.
    motion_mad_k: float = 3.5

    ready: bool = False
    note: str = "Using default motion threshold."


@dataclass
class CameraCalibration:
    """Bundle of all calibration state passed through the inference loop."""
    colour: ColourCalibrationState = field(default_factory=ColourCalibrationState)
    lighting: LightingBaseline = field(default_factory=LightingBaseline)
    motion: MotionFloorState = field(default_factory=MotionFloorState)

    # Whether to apply colour correction in the inference loop.
    apply_colour: bool = True

    # Whether to apply per-session lighting normalisation.
    apply_lighting: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Colour calibration  (grey card + optional multi-patch)
# ──────────────────────────────────────────────────────────────────────────────

# Reference reflectance of a standard 18% grey card in [0,1] linear space.
_GREY_CARD_REFLECTANCE = 0.18

# Reference RGB for a pure-white diffuse surface.
_WHITE_REFLECTANCE = np.array([1.0, 1.0, 1.0], dtype=np.float64)


def _mean_rgb_from_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Spatially and temporally average a list of [H,W,3] float32 frames."""
    if not frames:
        return np.array([0.5, 0.5, 0.5], dtype=np.float64)
    arr = np.stack([f.mean(axis=(0, 1)) for f in frames], axis=0)  # [T, 3]
    return arr.mean(axis=0).astype(np.float64)


def calibrate_grey_card(
    grey_frames: List[np.ndarray],
    target_reflectance: float = _GREY_CARD_REFLECTANCE,
) -> ColourCalibrationState:
    """Compute per-channel gain from a grey-card capture.

    The grey card should be neutral (equal R=G=B when correctly illuminated).
    We scale each channel so its mean equals `target_reflectance`, then set
    offset to zero (pure gain-only correction).

    Args:
        grey_frames: frames captured while a grey card fills the ROI.
        target_reflectance: desired output level after correction (default 0.18).

    Returns:
        ColourCalibrationState with gains set and calibrated=True.
    """
    measured = _mean_rgb_from_frames(grey_frames)
    measured = np.maximum(measured, 1e-4)

    gain = np.full(3, target_reflectance, dtype=np.float64) / measured
    # Cap gains: a channel ratio > 3× suggests misuse (e.g., coloured light).
    gain = np.clip(gain, 0.3, 3.0)

    note = (
        f"Grey-card cal: measured R={measured[0]:.3f} G={measured[1]:.3f} "
        f"B={measured[2]:.3f} → gains {gain[0]:.3f}/{gain[1]:.3f}/{gain[2]:.3f}"
    )

    state = ColourCalibrationState(
        gain=gain,
        offset=np.zeros(3, dtype=np.float64),
        calibrated=True,
        note=note,
        surface_readings={"grey_card": measured.copy()},
    )
    return state


def add_white_balance_patch(
    cal: ColourCalibrationState,
    white_frames: List[np.ndarray],
) -> ColourCalibrationState:
    """Refine colour calibration with a white patch (optional, runs after grey).

    White patch constrains the *relative* channel balance more tightly.
    """
    measured_white = _mean_rgb_from_frames(white_frames)
    measured_white = np.maximum(measured_white, 1e-4)

    # Apply existing gain first to see what the corrected white looks like.
    corrected_white = measured_white * cal.gain
    max_ch = float(corrected_white.max())
    if max_ch < 1e-4:
        return cal

    # Additional balance factor to make all channels equal at the white point.
    balance = max_ch / np.maximum(corrected_white, 1e-4)
    balance = np.clip(balance, 0.5, 2.0)

    new_gain = cal.gain * balance
    new_gain = np.clip(new_gain, 0.3, 3.0)
    cal.surface_readings["white_patch"] = measured_white.copy()
    cal.gain = new_gain
    cal.note += f" | white-balance refined (balance {balance[0]:.3f}/{balance[1]:.3f}/{balance[2]:.3f})"
    return cal


def add_skin_patch(
    cal: ColourCalibrationState,
    skin_frames: List[np.ndarray],
) -> ColourCalibrationState:
    """Record a skin-patch reading (informational — used for SpO2 ratio logging)."""
    measured_skin = _mean_rgb_from_frames(skin_frames)
    cal.surface_readings["skin_patch"] = measured_skin.copy()
    r_rg = measured_skin[0] / max(measured_skin[1], 1e-4)
    cal.note += f" | skin R/G={r_rg:.3f}"
    return cal


# ──────────────────────────────────────────────────────────────────────────────
# Per-session lighting baseline
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_FRAMES_NEEDED = 60  # ~2 s at 30 fps


def capture_lighting_baseline(frames: List[np.ndarray]) -> LightingBaseline:
    """Compute the illumination baseline from a still-subject window.

    Args:
        frames: list of [H,W,3] float32 ROI frames (already colour-corrected).

    Returns:
        LightingBaseline ready for use in the inference loop.
    """
    if len(frames) < 10:
        lb = LightingBaseline()
        lb.note = f"Too few frames ({len(frames)}) for baseline — need {BASELINE_FRAMES_NEEDED}."
        return lb

    arr = np.stack([f.mean(axis=(0, 1)) for f in frames], axis=0)  # [T,3]
    mean_rgb = arr.mean(axis=0).astype(np.float64)
    std_rgb = arr.std(axis=0).astype(np.float64)

    note = (
        f"Session baseline: R={mean_rgb[0]:.3f}±{std_rgb[0]:.4f}  "
        f"G={mean_rgb[1]:.3f}±{std_rgb[1]:.4f}  "
        f"B={mean_rgb[2]:.3f}±{std_rgb[2]:.4f}  "
        f"({len(frames)} frames)"
    )
    return LightingBaseline(
        mean_rgb=mean_rgb,
        std_rgb=std_rgb,
        captured_at=time.time(),
        ready=True,
        note=note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Motion noise floor
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_motion_floor(frames: List[np.ndarray]) -> MotionFloorState:
    """Estimate per-camera motion noise floor from a still-subject capture.

    Computes frame differences during the still window and uses the median as
    the noise floor. The adaptive MAD multiplier is set so that a frame must
    be roughly 5× the noise floor to count as motion, which is less aggressive
    than the default for cameras with higher JPEG noise.

    Args:
        frames: list of [H,W,3] float32 frames captured while the subject is still.

    Returns:
        MotionFloorState with adapted threshold.
    """
    if len(frames) < 8:
        s = MotionFloorState()
        s.note = f"Too few frames ({len(frames)}) — keeping default motion threshold."
        return s

    small = [f[::4, ::4].mean(axis=2) for f in frames]  # downsample for speed
    h = min(a.shape[0] for a in small)
    w = min(a.shape[1] for a in small)
    arr = np.stack([a[:h, :w] for a in small], axis=0)  # [T,h,w]
    diffs = np.abs(np.diff(arr, axis=0)).reshape(len(frames) - 1, -1).mean(axis=1)

    noise_floor = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - noise_floor)))

    # If the camera is unusually noisy (noise_floor > 0.01 in [0,1] space),
    # tighten the multiplier slightly so we don't reject too many frames.
    if noise_floor > 0.01:
        k = 4.5
    elif noise_floor < 0.002:
        k = 3.0
    else:
        k = 3.5

    note = (
        f"Motion floor: noise={noise_floor:.4f}, MAD={mad:.4f} → k={k:.1f}  "
        f"({len(frames)} still frames)"
    )
    return MotionFloorState(noise_floor=noise_floor, motion_mad_k=k, ready=True, note=note)


# ──────────────────────────────────────────────────────────────────────────────
# Apply calibration to a single ROI frame (called in the inference loop)
# ──────────────────────────────────────────────────────────────────────────────

def apply_camera_calibration(
    roi_float: np.ndarray,
    cal: CameraCalibration,
) -> np.ndarray:
    """Apply colour correction and lighting normalisation to a single ROI frame.

    This is the single function you drop into the per-frame path between
    roi_extractor.extract_with_landmarks() and trad_roi_buffer.append().

    Args:
        roi_float: [H,W,3] float32 ROI in [0,1].
        cal: CameraCalibration from the sidebar panel.

    Returns:
        Corrected [H,W,3] float32 ROI.
    """
    out = roi_float.astype(np.float32)

    if cal.apply_colour and cal.colour.calibrated:
        out = cal.colour.apply(out)

    if cal.apply_lighting and cal.lighting.ready:
        out = cal.lighting.normalise(out)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# SpO2 ratio helper
# ──────────────────────────────────────────────────────────────────────────────

def estimate_spo2_ratio_from_frames(
    frames: List[np.ndarray],
    fps: float,
) -> Optional[float]:
    """Estimate the raw red/green pulsatility ratio R from a set of ROI frames.

    This is the same ratio that facial_vitals.Spo2Calibration maps to SpO2 %.
    Expose it here so the sidebar can show the user the R value for each
    oximeter reference reading they enter.

    Returns R or None if the signal is too flat.
    """
    if len(frames) < 60:
        return None
    try:
        from facial_vitals import spo2_ratio_from_faces
        R, conf, _ = spo2_ratio_from_faces(frames, fps)
        if R is not None and conf > 0.1:
            return float(R)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar UI panel
# ──────────────────────────────────────────────────────────────────────────────

# How long (seconds) each ROI capture window runs.
_CAPTURE_DURATION_S = 3.0
# Minimum still frames for the motion floor calibration.
_MOTION_CAL_FRAMES = 90  # ~3 s


def sidebar_camera_calibration_panel(st, session_state) -> CameraCalibration:
    """Render the full camera calibration panel in the Streamlit sidebar.

    Manages capture state via session_state so captures survive Streamlit reruns.
    Returns the current CameraCalibration that the inference loop should use.

    Initialise persistent state on first run.
    """
    # ── Persistent calibration objects ────────────────────────────────────────
    if "cam_cal" not in session_state:
        session_state.cam_cal = CameraCalibration()
    if "cal_capture_mode" not in session_state:
        session_state.cal_capture_mode = None     # None | "grey" | "white" | "skin" | "lighting" | "motion"
    if "cal_capture_frames" not in session_state:
        session_state.cal_capture_frames = []
    if "cal_capture_start" not in session_state:
        session_state.cal_capture_start = 0.0
    if "cal_log" not in session_state:
        session_state.cal_log = deque(maxlen=12)

    cal: CameraCalibration = session_state.cam_cal

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:13px;font-weight:700;color:#94a3b8;"
        "text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
        "📐 Camera Calibration</div>",
        unsafe_allow_html=True,
    )

    # ── Toggle switches ────────────────────────────────────────────────────────
    cal.apply_colour = st.sidebar.checkbox(
        "Apply colour correction", value=cal.apply_colour,
        help="Applies per-channel gain from the grey-card capture to every ROI frame."
    )
    cal.apply_lighting = st.sidebar.checkbox(
        "Apply session lighting normalisation", value=cal.apply_lighting,
        help="Divides each channel by the session mean captured during the baseline window."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Section 1: Colour calibration
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar.expander("🎨 Colour / White-balance", expanded=not cal.colour.calibrated):

        st.caption(
            "**What to do:** hold a neutral grey card (or piece of white paper) "
            "so it fills most of the camera view, then press the button. "
            "Do this under your normal room lighting before starting inference."
        )

        _render_surface_status(st, cal)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📷 Capture grey card", key="btn_grey",
                         disabled=session_state.cal_capture_mode is not None):
                session_state.cal_capture_mode = "grey"
                session_state.cal_capture_frames = []
                session_state.cal_capture_start = time.time()
                session_state.cal_log.appendleft("Grey-card capture started (3 s).")

        with col_b:
            if st.button("⬜ Add white patch", key="btn_white",
                         disabled=session_state.cal_capture_mode is not None
                         or not cal.colour.calibrated):
                session_state.cal_capture_mode = "white"
                session_state.cal_capture_frames = []
                session_state.cal_capture_start = time.time()
                session_state.cal_log.appendleft("White-patch capture started (3 s).")

        if st.button("🟤 Add skin patch", key="btn_skin",
                     disabled=session_state.cal_capture_mode is not None):
            session_state.cal_capture_mode = "skin"
            session_state.cal_capture_frames = []
            session_state.cal_capture_start = time.time()
            session_state.cal_log.appendleft("Skin-patch capture started (3 s).")

        if session_state.cal_capture_mode in ("grey", "white", "skin"):
            elapsed = time.time() - session_state.cal_capture_start
            remaining = max(0.0, _CAPTURE_DURATION_S - elapsed)
            st.progress(
                min(elapsed / _CAPTURE_DURATION_S, 1.0),
                text=f"Capturing {session_state.cal_capture_mode}… {remaining:.1f}s"
            )

        if cal.colour.calibrated:
            g = cal.colour.gain
            st.markdown(
                f"<div style='font-size:11px;color:#64748b;margin-top:6px;'>"
                f"Gains — R: <b style='color:#f87171'>{g[0]:.3f}</b> "
                f"G: <b style='color:#4ade80'>{g[1]:.3f}</b> "
                f"B: <b style='color:#60a5fa'>{g[2]:.3f}</b></div>",
                unsafe_allow_html=True,
            )
            if st.button("↺ Reset colour cal", key="btn_reset_colour"):
                session_state.cam_cal.colour = ColourCalibrationState()
                session_state.cal_log.appendleft("Colour calibration reset.")
        else:
            st.info("No colour calibration yet — unity gain active.", icon="ℹ️")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 2: Lighting baseline
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar.expander("💡 Session lighting baseline", expanded=not cal.lighting.ready):

        st.caption(
            "**What to do:** sit naturally, face the camera, keep still for 3 s, "
            "then press Capture. Do this once per session or whenever your room "
            "lighting changes significantly."
        )

        if cal.lighting.ready and cal.lighting.mean_rgb is not None:
            m = cal.lighting.mean_rgb
            st.markdown(
                f"<div style='font-size:11px;color:#64748b;'>"
                f"Baseline — R: <b style='color:#f87171'>{m[0]:.3f}</b> "
                f"G: <b style='color:#4ade80'>{m[1]:.3f}</b> "
                f"B: <b style='color:#60a5fa'>{m[2]:.3f}</b></div>",
                unsafe_allow_html=True,
            )

        if st.button("🌅 Capture lighting baseline", key="btn_lighting",
                     disabled=session_state.cal_capture_mode is not None):
            session_state.cal_capture_mode = "lighting"
            session_state.cal_capture_frames = []
            session_state.cal_capture_start = time.time()
            session_state.cal_log.appendleft("Lighting baseline capture started (3 s).")

        if session_state.cal_capture_mode == "lighting":
            elapsed = time.time() - session_state.cal_capture_start
            remaining = max(0.0, _CAPTURE_DURATION_S - elapsed)
            st.progress(
                min(elapsed / _CAPTURE_DURATION_S, 1.0),
                text=f"Capturing baseline… {remaining:.1f}s"
            )

        if cal.lighting.ready:
            if st.button("↺ Reset lighting baseline", key="btn_reset_lighting"):
                session_state.cam_cal.lighting = LightingBaseline()
                session_state.cal_log.appendleft("Lighting baseline reset.")
        else:
            st.info("No session baseline — raw frames used.", icon="ℹ️")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 3: Motion floor calibration
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar.expander("🏃 Motion gate calibration", expanded=not cal.motion.ready):

        st.caption(
            "**What to do:** keep completely still for 3 s, then press Calibrate. "
            "This adapts the motion rejection threshold to your camera's own noise "
            "level so shaky frames are dropped without over-rejecting good ones."
        )

        if cal.motion.ready:
            st.markdown(
                f"<div style='font-size:11px;color:#64748b;'>"
                f"Noise floor: <b>{cal.motion.noise_floor:.4f}</b> &nbsp;|&nbsp; "
                f"MAD k: <b>{cal.motion.motion_mad_k:.1f}</b></div>",
                unsafe_allow_html=True,
            )

        if st.button("🎯 Calibrate motion floor", key="btn_motion",
                     disabled=session_state.cal_capture_mode is not None):
            session_state.cal_capture_mode = "motion"
            session_state.cal_capture_frames = []
            session_state.cal_capture_start = time.time()
            session_state.cal_log.appendleft("Motion floor capture started (3 s).")

        if session_state.cal_capture_mode == "motion":
            elapsed = time.time() - session_state.cal_capture_start
            remaining = max(0.0, _CAPTURE_DURATION_S - elapsed)
            st.progress(
                min(elapsed / _CAPTURE_DURATION_S, 1.0),
                text=f"Calibrating motion… {remaining:.1f}s"
            )

        if cal.motion.ready:
            if st.button("↺ Reset motion cal", key="btn_reset_motion"):
                session_state.cam_cal.motion = MotionFloorState()
                session_state.cal_log.appendleft("Motion calibration reset.")
        else:
            st.info(f"Using default k={cal.motion.motion_mad_k:.1f}", icon="ℹ️")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 4: SpO2 device calibration (wraps facial_vitals.Spo2Calibration)
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar.expander("🩸 SpO₂ device calibration", expanded=False):

        st.caption(
            "**Optional but recommended.** Enter one or more reference pulse-oximeter "
            "readings paired with the ratio-R value measured simultaneously on this "
            "camera. The ratio-R for the current live clip is shown in the diagnostics "
            "log. Without this, SpO₂ is labelled **provisional** (relative trend only)."
        )

        # Widget keys (prefixed _) differ from the storage keys so Streamlit
        # never sees a write to a key already owned by a widget.
        ref_text = st.text_area(
            "SpO₂ ref %  ,  ratio-R  (one pair per line)",
            key="_spo2_cal_text_widget",
            height=80,
            help="Example:\n98, 0.55\n95, 0.82\n\nObtain ratio-R from the diagnostics log."
        )

        spo2_cal_result = None
        if ref_text.strip():
            pcts, ratios = [], []
            for line in ref_text.strip().splitlines():
                try:
                    parts = line.replace(";", ",").split(",")
                    pcts.append(float(parts[0].strip()))
                    ratios.append(float(parts[1].strip()))
                except Exception:
                    continue
            if pcts:
                try:
                    from facial_vitals import Spo2Calibration
                    spo2_cal_result = Spo2Calibration.fit(ratios=ratios, spo2_ref=pcts)
                    st.success(f"✓ SpO₂ calibrated ({spo2_cal_result.note}).", icon="✅")
                except Exception as exc:
                    st.error(f"Calibration failed: {exc}")

        # Checkbox: widget owns "_spo2_show_prov_widget"; we read it back and
        # store under "spo2_show_prov" (no widget with that key, so safe to write).
        st.checkbox(
            "Show provisional SpO₂ (uncalibrated)",
            key="_spo2_show_prov_widget",
            help="Uncalibrated SpO₂ is a relative trend, not a validated percentage.",
        )

        session_state["spo2_cal_result"] = spo2_cal_result
        session_state["spo2_show_prov"] = bool(session_state.get("_spo2_show_prov_widget", True))

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration status summary + log
    # ──────────────────────────────────────────────────────────────────────────
    st.sidebar.markdown("**Calibration status**")
    _render_calibration_badges(st, cal)

    if session_state.cal_log:
        with st.sidebar.expander("📋 Calibration log", expanded=False):
            st.markdown(
                "<div style='font-size:10.5px;color:#64748b;font-family:monospace;"
                "line-height:1.6;'>"
                + "<br>".join(list(session_state.cal_log))
                + "</div>",
                unsafe_allow_html=True,
            )

    return session_state.cam_cal


def _render_surface_status(st, cal: CameraCalibration) -> None:
    """Show a compact table of captured surface readings."""
    readings = cal.colour.surface_readings
    if not readings:
        return
    rows = []
    for name, rgb in readings.items():
        label = {"grey_card": "Grey card", "white_patch": "White patch", "skin_patch": "Skin patch"}.get(name, name)
        rows.append(
            f"<tr>"
            f"<td style='padding:2px 6px;color:#94a3b8;'>{label}</td>"
            f"<td style='padding:2px 6px;color:#f87171;font-family:monospace;'>{rgb[0]:.3f}</td>"
            f"<td style='padding:2px 6px;color:#4ade80;font-family:monospace;'>{rgb[1]:.3f}</td>"
            f"<td style='padding:2px 6px;color:#60a5fa;font-family:monospace;'>{rgb[2]:.3f}</td>"
            f"</tr>"
        )
    st.markdown(
        "<table style='font-size:11px;border-collapse:collapse;width:100%;margin-bottom:6px;'>"
        "<tr><th style='color:#475569;text-align:left;padding:2px 6px;'>Surface</th>"
        "<th style='color:#f87171;padding:2px 6px;'>R</th>"
        "<th style='color:#4ade80;padding:2px 6px;'>G</th>"
        "<th style='color:#60a5fa;padding:2px 6px;'>B</th></tr>"
        + "".join(rows)
        + "</table>",
        unsafe_allow_html=True,
    )


def _render_calibration_badges(st, cal: CameraCalibration) -> None:
    """Render compact green/amber/red badges for each calibration axis."""
    def badge(label: str, ok: bool, warn: bool = False) -> str:
        if ok:
            colour = "#10b981"; icon = "✓"
        elif warn:
            colour = "#f59e0b"; icon = "~"
        else:
            colour = "#ef4444"; icon = "✗"
        return (
            f"<span style='background:{colour}22;border:1px solid {colour}44;"
            f"color:{colour};padding:2px 8px;border-radius:4px;font-size:10.5px;"
            f"font-weight:600;margin:2px;display:inline-block;'>{icon} {label}</span>"
        )

    badges = (
        badge("Colour", cal.colour.calibrated)
        + badge("Lighting", cal.lighting.ready)
        + badge("Motion", cal.motion.ready, warn=not cal.motion.ready)
        + badge("SpO₂ cal", False, warn=True)  # replaced per-call by SpO2 section
    )
    st.sidebar.markdown(
        f"<div style='line-height:2;margin-bottom:4px;'>{badges}</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feed frames into active captures (called from the inference loop each frame)
# ──────────────────────────────────────────────────────────────────────────────

def feed_calibration_frame(
    roi_float: np.ndarray,
    session_state,
    logs: "deque[str]",
) -> None:
    """Feed a single ROI frame into the active calibration capture window.

    Call this once per inference frame, after ROI extraction but before
    apply_camera_calibration(). When a capture window completes it automatically
    runs the appropriate calibration function and updates session_state.cam_cal.

    Args:
        roi_float: [H,W,3] float32 ROI in [0,1] — raw, uncorrected.
        session_state: Streamlit session_state (must contain cam_cal, cal_capture_*).
        logs: the existing diagnostics log deque from streamlit_app.py.
    """
    import datetime as _dt

    mode: Optional[str] = session_state.get("cal_capture_mode")
    if mode is None:
        return

    session_state.cal_capture_frames.append(roi_float.copy())
    elapsed = time.time() - session_state.get("cal_capture_start", time.time())

    if elapsed < _CAPTURE_DURATION_S:
        return  # still collecting

    # ── Capture window complete ───────────────────────────────────────────────
    frames: List[np.ndarray] = session_state.cal_capture_frames
    ts = _dt.datetime.now().strftime("%H:%M:%S")

    if mode == "grey":
        session_state.cam_cal.colour = calibrate_grey_card(frames)
        msg = f"[{ts}] Grey-card calibration complete. {session_state.cam_cal.colour.note}"
        session_state.cal_log.appendleft(msg)
        logs.appendleft(msg)

    elif mode == "white":
        session_state.cam_cal.colour = add_white_balance_patch(
            session_state.cam_cal.colour, frames
        )
        msg = f"[{ts}] White-patch WB applied. {session_state.cam_cal.colour.note}"
        session_state.cal_log.appendleft(msg)
        logs.appendleft(msg)

    elif mode == "skin":
        session_state.cam_cal.colour = add_skin_patch(
            session_state.cam_cal.colour, frames
        )
        msg = f"[{ts}] Skin patch recorded. {session_state.cam_cal.colour.note}"
        session_state.cal_log.appendleft(msg)
        logs.appendleft(msg)

    elif mode == "lighting":
        session_state.cam_cal.lighting = capture_lighting_baseline(frames)
        msg = f"[{ts}] Lighting baseline set. {session_state.cam_cal.lighting.note}"
        session_state.cal_log.appendleft(msg)
        logs.appendleft(msg)

    elif mode == "motion":
        session_state.cam_cal.motion = calibrate_motion_floor(frames)
        msg = f"[{ts}] Motion floor calibrated. {session_state.cam_cal.motion.note}"
        session_state.cal_log.appendleft(msg)
        logs.appendleft(msg)

    session_state.cal_capture_mode = None
    session_state.cal_capture_frames = []