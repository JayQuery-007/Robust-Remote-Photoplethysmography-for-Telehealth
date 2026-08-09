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
    """Camera calibration — renders with plain st.* so caller wraps in one expander."""
    if "cam_cal" not in session_state:
        session_state.cam_cal = CameraCalibration()
    if "cal_capture_mode" not in session_state:
        session_state.cal_capture_mode = None
    if "cal_capture_frames" not in session_state:
        session_state.cal_capture_frames = []
    if "cal_capture_start" not in session_state:
        session_state.cal_capture_start = 0.0
    if "cal_log" not in session_state:
        session_state.cal_log = deque(maxlen=12)

    cal = session_state.cam_cal

    def _badge(label, ok, warn=False):
        col = "#10b981" if ok else ("#f59e0b" if warn else "#64748b")
        txt = "OK" if ok else ("~" if warn else "-")
        return (
            "<span style='background:" + col + "22;border:1px solid " + col + "55;"
            "color:" + col + ";padding:2px 7px;border-radius:4px;font-size:10px;"
            "font-weight:600;margin:2px;display:inline-block;'>" + txt + " " + label + "</span>"
        )

    bp_anchored = bool(
        session_state.get("bp_estimator") and
        getattr(session_state.get("bp_estimator"), "baseline", None)
    )
    spo2_done = bool(session_state.get("spo2_cal_result"))

    st.markdown(
        _badge("Colour",  cal.colour.calibrated) +
        _badge("Lighting", cal.lighting.ready, warn=True) +
        _badge("Motion",   cal.motion.ready,   warn=True) +
        _badge("SpO2",     spo2_done,           warn=not spo2_done) +
        _badge("BP",       bp_anchored,         warn=not bp_anchored),
        unsafe_allow_html=True,
    )

    tc1, tc2 = st.columns(2)
    cal.apply_colour   = tc1.checkbox("Colour correction",  value=cal.apply_colour,   key="_cal_col_on")
    cal.apply_lighting = tc2.checkbox("Lighting normalise", value=cal.apply_lighting, key="_cal_lit_on")

    st.markdown("---")
    st.markdown("**Colour / White-balance**")
    ca, cb, cc = st.columns(3)
    busy = session_state.cal_capture_mode is not None
    if ca.button("Grey card",   key="btn_grey",      disabled=busy):
        session_state.cal_capture_mode  = "grey"
        session_state.cal_capture_frames = []
        session_state.cal_capture_start = time.time()
        session_state.cal_log.appendleft("Grey-card capture started (3 s).")
    if cb.button("White patch", key="btn_white",     disabled=busy or not cal.colour.calibrated):
        session_state.cal_capture_mode  = "white"
        session_state.cal_capture_frames = []
        session_state.cal_capture_start = time.time()
        session_state.cal_log.appendleft("White-patch capture started (3 s).")
    if cc.button("Reset",       key="btn_reset_col", disabled=not cal.colour.calibrated):
        session_state.cam_cal.colour = ColourCalibrationState()
        session_state.cal_log.appendleft("Colour calibration reset.")
    if session_state.cal_capture_mode in ("grey", "white", "skin"):
        el = time.time() - session_state.cal_capture_start
        st.progress(min(el / _CAPTURE_DURATION_S, 1.0),
                    text="Capturing... %.1fs" % max(0, _CAPTURE_DURATION_S - el))
    elif cal.colour.calibrated:
        g = cal.colour.gain
        st.caption("Gains  R:%.3f  G:%.3f  B:%.3f" % (g[0], g[1], g[2]))
    else:
        st.caption("Not calibrated — unity gain active.")

    st.markdown("---")
    st.markdown("**Session lighting**")
    la, lb = st.columns(2)
    if la.button("Capture baseline", key="btn_lighting",    disabled=busy):
        session_state.cal_capture_mode  = "lighting"
        session_state.cal_capture_frames = []
        session_state.cal_capture_start = time.time()
        session_state.cal_log.appendleft("Lighting baseline started (3 s).")
    if lb.button("Reset##light",     key="btn_reset_light", disabled=not cal.lighting.ready):
        session_state.cam_cal.lighting = LightingBaseline()
        session_state.cal_log.appendleft("Lighting baseline reset.")
    if session_state.cal_capture_mode == "lighting":
        el = time.time() - session_state.cal_capture_start
        st.progress(min(el / _CAPTURE_DURATION_S, 1.0),
                    text="Capturing... %.1fs" % max(0, _CAPTURE_DURATION_S - el))
    elif cal.lighting.ready and cal.lighting.mean_rgb is not None:
        m = cal.lighting.mean_rgb
        st.caption("Baseline  R:%.3f  G:%.3f  B:%.3f" % (m[0], m[1], m[2]))
    else:
        st.caption("Not captured — raw frames used.")

    st.markdown("---")
    st.markdown("**Motion gate**")
    ma, mb = st.columns(2)
    if ma.button("Calibrate",     key="btn_motion",       disabled=busy):
        session_state.cal_capture_mode  = "motion"
        session_state.cal_capture_frames = []
        session_state.cal_capture_start = time.time()
        session_state.cal_log.appendleft("Motion floor capture started — keep still (3 s).")
    if mb.button("Reset##motion", key="btn_reset_motion", disabled=not cal.motion.ready):
        session_state.cam_cal.motion = MotionFloorState()
        session_state.cal_log.appendleft("Motion calibration reset.")
    if session_state.cal_capture_mode == "motion":
        el = time.time() - session_state.cal_capture_start
        st.progress(min(el / _CAPTURE_DURATION_S, 1.0),
                    text="Keep still... %.1fs" % max(0, _CAPTURE_DURATION_S - el))
    elif cal.motion.ready:
        st.caption("Noise floor: %.4f  |  k: %.1f" % (cal.motion.noise_floor, cal.motion.motion_mad_k))
    else:
        st.caption("Default k=%.1f" % cal.motion.motion_mad_k)

    st.markdown("---")
    st.markdown("**SpO2 calibration**")
    ref_text = st.text_area(
        "Oximeter readings (spo2%,  ratioR — one per line)",
        key="_spo2_cal_widget", height=68,
    )
    spo2_cal_result = None
    if (ref_text or "").strip():
        pcts, ratios = [], []
        for line in ref_text.strip().splitlines():
            try:
                pts = line.replace(";", ",").split(",")
                pcts.append(float(pts[0].strip()))
                ratios.append(float(pts[1].strip()))
            except Exception:
                continue
        if pcts:
            try:
                from facial_vitals import Spo2Calibration
                spo2_cal_result = Spo2Calibration.fit(ratios=ratios, spo2_ref=pcts)
                st.success("SpO2 calibrated (%s)." % spo2_cal_result.note)
            except Exception as exc:
                st.error("Calibration failed: %s" % exc)
    st.checkbox("Show uncalibrated SpO2", key="_spo2_prov_widget", value=True)
    session_state["spo2_cal_result"] = spo2_cal_result
    session_state["spo2_show_prov"]  = bool(session_state.get("_spo2_prov_widget", True))

    st.markdown("---")
    st.markdown("**Blood pressure anchor**")
    if "bp_estimator" not in session_state:
        from facial_vitals import PersonalizedBPEstimator
        session_state.bp_estimator = PersonalizedBPEstimator()
    bp = session_state.bp_estimator
    ba, bb = st.columns(2)
    sbp0 = ba.number_input("SBP (mmHg)", 70, 220, 120, key="bp_sbp0")
    dbp0 = bb.number_input("DBP (mmHg)", 40, 140, 80,  key="bp_dbp0")
    anchor_on = st.checkbox("Use as BP anchor", value=False, key="bp_anchor_on")
    if bp.baseline is not None:
        st.success("Anchored at %.0f/%.0f mmHg" % (bp.baseline.sbp, bp.baseline.dbp))
    elif anchor_on:
        st.info("Waiting for stable pulse...")

    if session_state.cal_log:
        st.markdown("---")
        with st.expander("Calibration log"):
            st.markdown(
                "<div style='font-size:10px;color:#64748b;font-family:monospace;line-height:1.7;'>"
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