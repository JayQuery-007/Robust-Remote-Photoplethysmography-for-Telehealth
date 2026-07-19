"""facial_vitals.py — honest facial-feature extraction for vital signs.

Design principle
----------------
A value shown to a user must be traceable to something physically measured on
the face. This module makes provenance explicit and refuses to fabricate.

    HR   -> dominant frequency of the reconstructed facial pulse        (MEASURED)
    RR   -> respiratory modulation of the pulse envelope               (MEASURED)
    SpO2 -> red/green pulsatility ratio-of-ratios                      (MEASURED feature)
            mapped to % ONLY through an explicit device calibration.
    BP   -> pulse-wave MORPHOLOGY features (rise time, reflection/     (MEASURED features)
            augmentation index, APG wave ratios) mapped to mmHg ONLY
            through a per-subject cuff baseline or a trained model.
    Hb   -> experimental; feature exposed, number withheld unless a
            trained model is supplied.

Anything that is not calibrated returns value=None with a reason, rather than a
plausible-looking guess. This is NOT a medical device and must not be used for
diagnosis; absolute BP/SpO2/Hb require validation against reference instruments.

The frame->pulse step reuses the validated POS/CHROM patch weighting in
equiphys_core; that import is lazy so the pure-signal maths here can run without
OpenCV/PyTorch installed.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch


# --------------------------------------------------------------------------- #
# Provenance / result container
# --------------------------------------------------------------------------- #

SRC_SPECTRAL = "rppg_spectral"                 # HR from facial pulse spectrum
SRC_ENVELOPE = "rppg_envelope"                 # RR from facial pulse envelope
SRC_RATIO_CAL = "rppg_ratio_calibrated"        # SpO2 from ratio + device calibration
SRC_RATIO_PROVISIONAL = "rppg_ratio_uncalibrated"
SRC_MORPH_PERSONAL = "rppg_morphology_personalized"  # BP from morphology + cuff baseline
SRC_MORPH_MODEL = "rppg_morphology_model"      # BP/Hb from a trained model
SRC_UNAVAILABLE = "unavailable"


@dataclass
class VitalSign:
    name: str
    value: Optional[float]
    unit: str
    confidence: float          # 0..1
    source: str
    note: str = ""
    extra: Optional[dict] = None

    @property
    def available(self) -> bool:
        return self.value is not None


# --------------------------------------------------------------------------- #
# Small self-contained DSP helpers (numpy/scipy only)
# --------------------------------------------------------------------------- #

def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    low = max(lo / nyq, 1e-3)
    high = min(hi / nyq, 0.999)
    if low >= high or x.size < max(15, 3 * order):
        return x - np.mean(x)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def _welch_peak(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> Tuple[float, float]:
    """Return (peak_hz, snr_db) of the dominant in-band spectral peak.

    snr_db compares power within +/-0.2 Hz of the peak to the rest of the band.
    Returns (0.0, -inf) when nothing usable is found.
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    if T < 45 or fs <= 1.0:
        return 0.0, -np.inf

    sig = _bandpass(x - np.mean(x), fs, f_lo, f_hi, order=4)
    nperseg = int(min(T, max(4 * fs, 64)))
    nfft = int(max(4096, T * 4))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4,
                       nfft=nfft, window="hann", detrend=False)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band) or psd[band].max() < 1e-15:
        return 0.0, -np.inf

    bf, bp = freqs[band], psd[band]
    peak_idx = int(np.argmax(bp))
    peak_hz = float(bf[peak_idx])

    sig_mask = (bf >= peak_hz - 0.2) & (bf <= peak_hz + 0.2)
    sig_pow = float(bp[sig_mask].sum()) if np.any(sig_mask) else float(bp[peak_idx])
    noise_pow = max(float(bp.sum()) - sig_pow, 1e-15)
    snr = float(np.clip(10.0 * np.log10(sig_pow / noise_pow), -10.0, 30.0))
    return peak_hz, snr


def _snr_to_conf(snr_db: float) -> float:
    return float(np.clip((snr_db + 3.0) / 15.0, 0.0, 1.0))


def _ac_dc(channel: np.ndarray, fs: float, f_lo: float, f_hi: float) -> Tuple[float, float]:
    """Robust AC amplitude (pulsatile) and DC level for one colour channel."""
    c = np.asarray(channel, dtype=np.float64)
    dc = float(np.median(c))
    if dc <= 1e-6:
        return 0.0, 0.0
    ac = _bandpass(c / (dc + 1e-6) - 1.0, fs, f_lo, f_hi, order=3)
    amp = float(np.percentile(ac, 95) - np.percentile(ac, 5))  # outlier-robust p-p
    return amp, dc


# --------------------------------------------------------------------------- #
# Motion robustness  (drop shaky frames instead of rejecting whole windows)
# --------------------------------------------------------------------------- #

def frame_diff_motion(frame_buffer: Sequence[np.ndarray], stride: int = 4) -> np.ndarray:
    """Per-frame motion proxy: mean abs difference between consecutive ROI frames.

    Pure numpy (no OpenCV). The pulsatile colour change is ~0.1-1% of intensity,
    far below motion/lighting changes, so a simple frame difference cleanly flags
    the frames that are unusable. frames: list of [H,W,3] in [0,1].
    Returns [T] float; index 0 is 0 (no predecessor).
    """
    T = len(frame_buffer)
    if T == 0:
        return np.zeros(0, dtype=np.float64)
    small = []
    for f in frame_buffer:
        a = np.asarray(f, dtype=np.float64)
        a = a[::stride, ::stride].mean(axis=2) if a.ndim == 3 else a[::stride, ::stride]
        small.append(a)
    h = min(s.shape[0] for s in small)
    w = min(s.shape[1] for s in small)
    arr = np.stack([s[:h, :w] for s in small], axis=0)  # [T,h,w]
    motion = np.zeros(T, dtype=np.float64)
    if T >= 2:
        motion[1:] = np.abs(np.diff(arr, axis=0)).reshape(T - 1, -1).mean(axis=1)
    return motion


def clean_frame_mask(motion: np.ndarray, k: float = 3.5, min_keep_frac: float = 0.6) -> np.ndarray:
    """Boolean mask of low-motion frames via a robust median+k*MAD threshold.

    If motion is elevated almost everywhere, keeps the calmest `min_keep_frac`
    rather than discarding the whole window.
    """
    m = np.asarray(motion, dtype=np.float64)
    n = m.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-12
    keep = m <= (med + k * 1.4826 * mad)
    keep[0] = True
    if keep.mean() < 0.3:
        order = np.argsort(m)
        keep = np.zeros(n, dtype=bool)
        keep[order[: max(1, int(min_keep_frac * n))]] = True
    return keep


def select_clean_frames(frame_buffer: Sequence[np.ndarray], timestamps: Optional[np.ndarray],
                        mask: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Keep only masked-in frames and their timestamps.

    Because the downstream estimators resample on true timestamps, dropping the
    shaky frames leaves gaps the interpolator simply bridges.
    """
    mask = np.asarray(mask, dtype=bool)
    frames = [f for f, keep in zip(frame_buffer, mask) if keep]
    if timestamps is None:
        return frames, None
    ts = np.asarray(timestamps, dtype=np.float64)
    ts = ts[mask] if ts.shape[0] == mask.shape[0] else ts
    return frames, ts


# --------------------------------------------------------------------------- #
# 1) Facial pulse reconstruction  (frame buffer -> clean BVP waveform)
# --------------------------------------------------------------------------- #

def reconstruct_facial_bvp(
    frame_buffer: Sequence[np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    target_fs: float = 30.0,
) -> Tuple[Optional[np.ndarray], float, float, str]:
    """Reconstruct the facial blood-volume-pulse waveform.

    Runs POS and CHROM (via equiphys_core's validated patch weighting) and keeps
    whichever gives the stronger in-band SNR.

    Returns (bvp, fs_used, snr_db, method). bvp is None if reconstruction fails.
    """
    try:  # lazy: only this path needs OpenCV/PyTorch-backed helpers
        from equiphys_core import _extract_patch_weighted_pulse
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"equiphys_core unavailable for pulse extraction: {exc}")
        return None, float(target_fs), -np.inf, SRC_UNAVAILABLE

    best: Optional[Tuple[np.ndarray, float, float, str]] = None
    for method in ("pos", "chrom"):
        try:
            pulse, fs_used = _extract_patch_weighted_pulse(
                frame_buffer, timestamps=timestamps, target_fs=target_fs, method=method
            )
        except Exception:
            continue
        _, snr = _welch_peak(pulse, fs_used, 0.7, 3.5)
        if best is None or snr > best[2]:
            best = (np.asarray(pulse, dtype=np.float64), float(fs_used), float(snr), method)

    if best is None:
        return None, float(target_fs), -np.inf, SRC_UNAVAILABLE
    return best


# --------------------------------------------------------------------------- #
# 2) Heart rate  (spectral peak of the pulse)
# --------------------------------------------------------------------------- #

def heart_rate_from_bvp(bvp: np.ndarray, fs: float,
                        f_lo: float = 0.7, f_hi: float = 3.5) -> VitalSign:
    peak_hz, snr = _welch_peak(bvp, fs, f_lo, f_hi)
    if peak_hz <= 0:
        return VitalSign("heart_rate", None, "bpm", 0.0, SRC_UNAVAILABLE,
                         "No dominant pulse peak in band.")
    return VitalSign("heart_rate", float(peak_hz * 60.0), "bpm",
                     _snr_to_conf(snr), SRC_SPECTRAL,
                     extra={"snr_db": snr})


# --------------------------------------------------------------------------- #
# 3) Respiratory rate  (envelope modulation of the pulse)
# --------------------------------------------------------------------------- #

def respiratory_rate_from_bvp(bvp: np.ndarray, fs: float) -> VitalSign:
    x = np.asarray(bvp, dtype=np.float64)
    if x.size < 90 or np.std(x) < 1e-8:
        return VitalSign("respiratory_rate", None, "br/min", 0.0, SRC_UNAVAILABLE,
                         "Clip too short / flat for respiration.")
    env = np.abs(hilbert(x - np.mean(x)))
    env = env - np.mean(env)
    resp = _bandpass(env, fs, 0.15, 0.40, order=2)
    peak_hz, snr = _welch_peak(resp, fs, 0.15, 0.40)
    if peak_hz <= 0:
        return VitalSign("respiratory_rate", None, "br/min", 0.0, SRC_UNAVAILABLE,
                         "No respiratory peak found.")
    rr = float(np.clip(peak_hz * 60.0, 6.0, 30.0))
    return VitalSign("respiratory_rate", rr, "br/min",
                     _snr_to_conf(snr) * 0.8, SRC_ENVELOPE, extra={"snr_db": snr})


# --------------------------------------------------------------------------- #
# 4) SpO2  (red/green ratio-of-ratios + explicit device calibration)
# --------------------------------------------------------------------------- #

@dataclass
class Spo2Calibration:
    """Device-specific mapping SpO2 = A - B * R.

    Defaults are a generic starting point and are NOT valid until calibrated
    against a reference oximeter on the same camera/lighting.
    """
    A: float = 109.0
    B: float = 15.0
    calibrated: bool = False
    note: str = "uncalibrated defaults"

    @classmethod
    def fit(cls, ratios: Sequence[float], spo2_ref: Sequence[float]) -> "Spo2Calibration":
        R = np.asarray(ratios, dtype=np.float64)
        S = np.asarray(spo2_ref, dtype=np.float64)
        ok = np.isfinite(R) & np.isfinite(S)
        R, S = R[ok], S[ok]
        if R.size >= 2 and (R.max() - R.min()) > 0.05:
            B, A = np.polyfit(R, S, 1)          # S = A + B*R  ->  store A, -B
            return cls(A=float(A), B=float(-B), calibrated=True,
                       note=f"linear fit on {R.size} points")
        if R.size >= 1:                          # single point: offset-only
            A = float(np.mean(S) + cls().B * np.mean(R))
            return cls(A=A, B=cls().B, calibrated=True, note="single-point offset")
        return cls()

    def to_spo2(self, R: float) -> float:
        return float(np.clip(self.A - self.B * float(R), 70.0, 100.0))


def spo2_ratio_from_faces(frame_buffer: Sequence[np.ndarray],
                          fs: float) -> Tuple[Optional[float], float, int]:
    """Compute the pulsatility ratio-of-ratios R across skin patches.

    Returns (R, confidence, n_valid_patches). R is device-agnostic; converting it
    to a % needs Spo2Calibration.
    """
    try:
        from equiphys_core import _extract_patch_rgb_traces, _extract_rgb_trace
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"equiphys_core unavailable for SpO2 patches: {exc}")
        return None, 0.0, 0

    patch_rgb = _extract_patch_rgb_traces(frame_buffer, normalize_size=128,
                                          grid_size=4, min_patch_coverage=0.35)
    if patch_rgb.shape[1] == 0:
        rgb = _extract_rgb_trace(frame_buffer, normalize_size=128)
        patch_rgb = rgb[:, None, :]

    r_vals: List[float] = []
    weights: List[float] = []
    for p in range(patch_rgb.shape[1]):
        red, green = patch_rgb[:, p, 0], patch_rgb[:, p, 1]
        ac_r, dc_r = _ac_dc(red, fs, 0.7, 3.0)
        ac_g, dc_g = _ac_dc(green, fs, 0.7, 3.0)
        if ac_r < 6e-5 or ac_g < 6e-5:
            continue
        # ratio-of-ratios: (AC/DC)_red / (AC/DC)_green  (AC already DC-normalised)
        r = ac_r / (ac_g + 1e-9)
        if not np.isfinite(r) or not (0.1 < r < 3.0):
            continue
        _, snr = _welch_peak(_bandpass(green / (dc_g + 1e-6) - 1.0, fs, 0.7, 3.0), fs, 0.7, 3.0)
        w = max(0.0, snr + 8.0) + 60.0 * min(ac_r, ac_g)
        if w > 0:
            r_vals.append(r)
            weights.append(w)

    if not r_vals:
        return None, 0.0, 0

    r_arr = np.asarray(r_vals)
    w_arr = np.asarray(weights)
    order = np.argsort(r_arr)
    cw = np.cumsum(w_arr[order]) / w_arr.sum()
    R = float(r_arr[order][int(np.clip(np.searchsorted(cw, 0.5), 0, r_arr.size - 1))])
    conf = float(np.clip((np.mean(weights) - 4.0) / 12.0, 0.0, 1.0)) * min(1.0, len(r_vals) / 4.0)
    return R, conf, len(r_vals)


def spo2_from_faces(frame_buffer, fs, calib: Optional[Spo2Calibration] = None,
                    min_conf: float = 0.35) -> VitalSign:
    R, conf, n = spo2_ratio_from_faces(frame_buffer, fs)
    if R is None or conf < min_conf:
        return VitalSign("spo2", None, "%", conf, SRC_UNAVAILABLE,
                         "Pulsatility too weak/noisy for a reliable ratio.",
                         extra={"ratio_R": R, "n_patches": n})
    if calib is None:
        calib = Spo2Calibration()
    value = calib.to_spo2(R)
    src = SRC_RATIO_CAL if calib.calibrated else SRC_RATIO_PROVISIONAL
    note = calib.note if calib.calibrated else ("PROVISIONAL — uncalibrated device "
                                                "constants; treat as relative only")
    return VitalSign("spo2", value, "%", conf if calib.calibrated else conf * 0.5,
                     src, note, extra={"ratio_R": R, "n_patches": n})


# --------------------------------------------------------------------------- #
# 5) Pulse-wave morphology  (the real basis for cuffless BP)
# --------------------------------------------------------------------------- #

def _width_at_fraction(beat: np.ndarray, frac: float) -> float:
    """Width (in samples) of a single normalised beat at `frac` of its height."""
    peak = int(np.argmax(beat))
    thr = beat[peak] * frac
    left = peak
    while left > 0 and beat[left] > thr:
        left -= 1
    right = peak
    while right < beat.size - 1 and beat[right] > thr:
        right += 1
    return float(right - left)


def _apg_waves(template: np.ndarray) -> Dict[str, float]:
    """Second-derivative (APG) a,b,c,d,e wave ratios — established BP correlates."""
    vpg = np.gradient(template)
    apg = np.gradient(vpg)
    # a = first prominent positive peak; then alternating extrema b,c,d,e
    maxima, _ = find_peaks(apg)
    minima, _ = find_peaks(-apg)
    out = {k: np.nan for k in ("apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "apg_aging")}
    if maxima.size == 0:
        return out
    a_idx = int(maxima[np.argmax(apg[maxima])])
    a = apg[a_idx]
    if abs(a) < 1e-9:
        return out

    def _first_after(idxs, ref):
        after = idxs[idxs > ref]
        return int(after[0]) if after.size else None

    b_idx = _first_after(minima, a_idx)
    c_idx = _first_after(maxima, b_idx) if b_idx is not None else None
    d_idx = _first_after(minima, c_idx) if c_idx is not None else None
    e_idx = _first_after(maxima, d_idx) if d_idx is not None else None
    b = apg[b_idx] if b_idx is not None else np.nan
    c = apg[c_idx] if c_idx is not None else np.nan
    d = apg[d_idx] if d_idx is not None else np.nan
    e = apg[e_idx] if e_idx is not None else np.nan
    out["apg_b_a"] = float(b / a)
    out["apg_c_a"] = float(c / a)
    out["apg_d_a"] = float(d / a)
    out["apg_e_a"] = float(e / a)
    if np.all(np.isfinite([b, c, d, e])):
        out["apg_aging"] = float((b - c - d - e) / a)
    return out


def extract_pulse_morphology(bvp: np.ndarray, fs: float,
                             template_len: int = 128) -> Dict[str, float]:
    """Extract genuine per-beat morphology features from the facial pulse.

    Returns a feature dict plus 'morph_quality' in [0,1]. Fine features (dicrotic
    notch, APG waves) are only trustworthy when quality is high — webcam pulses at
    ~30 fps frequently cannot resolve them, hence the quality gate.
    """
    feats: Dict[str, float] = {k: np.nan for k in (
        "rise_time_s", "width50", "width25", "width75",
        "reflection_index", "augmentation_index",
        "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "apg_aging",
    )}
    feats["morph_quality"] = 0.0

    sig = _bandpass(np.asarray(bvp, dtype=np.float64), fs, 0.7, 4.0, order=4)
    if sig.size < int(2 * fs) or np.std(sig) < 1e-9:
        return feats

    min_dist = max(1, int(0.4 * fs))
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=np.std(sig) * 0.3)
    if peaks.size < 4:
        return feats

    # Feet = minima between consecutive systolic peaks.
    feet = [int(peaks[i] + np.argmin(sig[peaks[i]:peaks[i + 1]]))
            for i in range(peaks.size - 1)]

    rise_times, widths50, widths25, widths75, amps, beats = [], [], [], [], [], []
    for i in range(len(feet) - 1):
        f0, f1 = feet[i], feet[i + 1]
        seg = sig[f0:f1]
        if seg.size < 5:
            continue
        seg = seg - seg.min()
        amp = float(seg.max())
        if amp < 1e-9:
            continue
        pk = int(np.argmax(seg))
        rise_times.append(pk / fs)
        norm = seg / amp
        widths50.append(_width_at_fraction(norm, 0.50) / fs)
        widths25.append(_width_at_fraction(norm, 0.25) / fs)
        widths75.append(_width_at_fraction(norm, 0.75) / fs)
        amps.append(amp)
        beats.append(np.interp(np.linspace(0, seg.size - 1, template_len),
                               np.arange(seg.size), norm))

    if len(beats) < 3:
        return feats

    feats["rise_time_s"] = float(np.median(rise_times))
    feats["width50"] = float(np.median(widths50))
    feats["width25"] = float(np.median(widths25))
    feats["width75"] = float(np.median(widths75))

    beats_arr = np.asarray(beats)
    template = beats_arr.mean(axis=0)
    # Beat consistency: 1 - mean pairwise deviation from the template.
    consistency = float(np.clip(1.0 - np.mean(np.std(beats_arr, axis=0)), 0.0, 1.0))
    amp_cv = float(np.std(amps) / (np.mean(amps) + 1e-9))

    # Reflection / augmentation from the template (needs a resolvable notch).
    notch_found = False
    sys_idx = int(np.argmax(template))
    tail = template[sys_idx:]
    tmin, _ = find_peaks(-tail)
    if tmin.size:
        notch_i = sys_idx + int(tmin[0])
        dia = template[notch_i:]
        dmax, _ = find_peaks(dia)
        if dmax.size:
            dia_idx = notch_i + int(dmax[0])
            sys_amp = template[sys_idx] - template[0]
            dia_amp = template[dia_idx] - template[notch_i]
            if sys_amp > 1e-6:
                feats["reflection_index"] = float(np.clip(template[dia_idx] / template[sys_idx], 0, 1.5))
                feats["augmentation_index"] = float(np.clip(dia_amp / sys_amp, -1, 1.5))
                notch_found = True

    feats.update(_apg_waves(template))

    q = 0.5 * consistency + 0.3 * min(1.0, len(beats) / 6.0) + 0.2 * (1.0 if notch_found else 0.0)
    q *= float(np.clip(1.0 - amp_cv, 0.3, 1.0))
    feats["morph_quality"] = float(np.clip(q, 0.0, 1.0))
    return feats


# --------------------------------------------------------------------------- #
# 6) Blood pressure  (personalized baseline OR trained model — never fabricated)
# --------------------------------------------------------------------------- #

@dataclass
class BPBaseline:
    features: Dict[str, float]
    sbp: float
    dbp: float


class PersonalizedBPEstimator:
    """Cuffless BP from facial morphology, with honest provenance.

    Two supported, defensible paths:

      * baseline  — record ONE reference cuff reading with concurrent features,
        then report BP as that baseline plus a capped response to *changes* in a
        few well-established BP-linked features. Output is a personalized TREND
        anchored to a real measurement, not an absolute clinical claim.

      * model     — load / fit a regression trained on paired (features, cuff BP)
        data. Only as good as the training set; use for research.

    With neither configured, estimate() returns value=None with a reason.
    """

    # Approximate *directional* sensitivities (mmHg per unit feature change).
    # Intentionally conservative and capped; they encode sign, not calibrated gain.
    _SBP_SENS = {"heart_rate": 0.35, "rise_time_s": -60.0, "augmentation_index": 20.0}
    _MAX_DEV = 25.0  # cap deviation from baseline (mmHg)

    def __init__(self) -> None:
        self.baseline: Optional[BPBaseline] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.feat_order: List[str] = []

    # ---- baseline path ---------------------------------------------------- #
    def set_baseline(self, features: Dict[str, float], sbp: float, dbp: float) -> None:
        self.baseline = BPBaseline(dict(features), float(sbp), float(dbp))

    # ---- model path ------------------------------------------------------- #
    def fit(self, feat_dicts: Sequence[Dict[str, float]],
            sbp: Sequence[float], dbp: Sequence[float],
            feat_order: Optional[Sequence[str]] = None, ridge: float = 10.0) -> None:
        self.feat_order = list(feat_order or sorted(feat_dicts[0].keys()))
        X = np.array([[fd.get(k, np.nan) for k in self.feat_order] for fd in feat_dicts], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_med = np.nanmedian(X, axis=0)            # median ignoring NaNs
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)  # all-NaN column -> 0 (inert)
        X = np.where(np.isfinite(X), X, col_med)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        Y = np.column_stack([np.asarray(sbp, float), np.asarray(dbp, float)])
        reg = ridge * np.eye(Xb.shape[1]); reg[-1, -1] = 0.0
        W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)
        self.coef_, self.intercept_ = W[:-1], W[-1]

    def save(self, path: str) -> None:
        json.dump({"feat_order": self.feat_order,
                   "coef": None if self.coef_ is None else self.coef_.tolist(),
                   "intercept": None if self.intercept_ is None else self.intercept_.tolist()},
                  open(path, "w"))

    def load(self, path: str) -> None:
        d = json.load(open(path))
        self.feat_order = d["feat_order"]
        self.coef_ = None if d["coef"] is None else np.asarray(d["coef"], float)
        self.intercept_ = None if d["intercept"] is None else np.asarray(d["intercept"], float)

    # ---- inference -------------------------------------------------------- #
    def estimate(self, features: Dict[str, float]) -> Tuple[VitalSign, VitalSign]:
        if self.coef_ is not None:
            x = np.array([features.get(k, np.nan) for k in self.feat_order], float)
            x = np.where(np.isfinite(x), x, 0.0)
            sbp, dbp = (x @ self.coef_ + self.intercept_).tolist()
            note = "trained model (research use; validate before any real use)"
            return (VitalSign("systolic_bp", float(sbp), "mmHg", 0.4, SRC_MORPH_MODEL, note),
                    VitalSign("diastolic_bp", float(dbp), "mmHg", 0.4, SRC_MORPH_MODEL, note))

        if self.baseline is not None:
            dsbp = 0.0
            for k, s in self._SBP_SENS.items():
                b = self.baseline.features.get(k, np.nan)
                v = features.get(k, np.nan)
                if np.isfinite(b) and np.isfinite(v):
                    dsbp += s * (v - b)
            dsbp = float(np.clip(dsbp, -self._MAX_DEV, self._MAX_DEV))
            sbp = self.baseline.sbp + dsbp
            dbp = self.baseline.dbp + 0.6 * dsbp
            q = float(features.get("morph_quality", 0.0))
            note = "relative to your cuff baseline — TREND only, not an absolute reading"
            return (VitalSign("systolic_bp", round(sbp), "mmHg", 0.3 + 0.3 * q, SRC_MORPH_PERSONAL, note),
                    VitalSign("diastolic_bp", round(dbp), "mmHg", 0.3 + 0.3 * q, SRC_MORPH_PERSONAL, note))

        reason = "No cuff baseline or trained model — absolute BP cannot be inferred from a face."
        return (VitalSign("systolic_bp", None, "mmHg", 0.0, SRC_UNAVAILABLE, reason),
                VitalSign("diastolic_bp", None, "mmHg", 0.0, SRC_UNAVAILABLE, reason))


# --------------------------------------------------------------------------- #
# 7) Calibration bundle + orchestrator
# --------------------------------------------------------------------------- #

@dataclass
class VitalsCalibration:
    spo2: Optional[Spo2Calibration] = None
    bp: PersonalizedBPEstimator = field(default_factory=PersonalizedBPEstimator)
    allow_provisional_spo2: bool = True   # show uncalibrated SpO2 (clearly flagged)
    bp_anchor: Optional[Tuple[float, float]] = None  # (cuff SBP, cuff DBP) to auto-anchor live


def estimate_all_vitals(
    frame_buffer: Sequence[np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    fps: float = 30.0,
    calib: Optional[VitalsCalibration] = None,
) -> Dict[str, VitalSign]:
    """Single entry point: face frames -> vitals with explicit provenance.

    Returns dict keyed by: heart_rate, respiratory_rate, spo2, systolic_bp,
    diastolic_bp, signal_quality. Each is a VitalSign with .value possibly None.
    """
    calib = calib or VitalsCalibration()
    out: Dict[str, VitalSign] = {}

    bvp, fs_used, snr, _ = reconstruct_facial_bvp(frame_buffer, timestamps, fps)
    if bvp is None:
        for k, u in (("heart_rate", "bpm"), ("respiratory_rate", "br/min"), ("spo2", "%"),
                     ("systolic_bp", "mmHg"), ("diastolic_bp", "mmHg")):
            out[k] = VitalSign(k, None, u, 0.0, SRC_UNAVAILABLE, "No facial pulse reconstructed.")
        out["signal_quality"] = VitalSign("signal_quality", 0.0, "0-100", 0.0, SRC_UNAVAILABLE)
        return out

    hr = heart_rate_from_bvp(bvp, fs_used)
    out["heart_rate"] = hr
    out["respiratory_rate"] = respiratory_rate_from_bvp(bvp, fs_used)

    min_conf = 0.35
    spo2 = spo2_from_faces(frame_buffer, fs_used, calib.spo2, min_conf=min_conf)
    if (spo2.source == SRC_RATIO_PROVISIONAL) and not calib.allow_provisional_spo2:
        spo2 = VitalSign("spo2", None, "%", spo2.confidence, SRC_UNAVAILABLE,
                         "SpO2 hidden until device calibration is provided.")
    out["spo2"] = spo2

    morph = extract_pulse_morphology(bvp, fs_used)
    if hr.value is not None:
        morph["heart_rate"] = hr.value
    sbp, dbp = calib.bp.estimate(morph)
    sbp.extra = {"morph_quality": morph.get("morph_quality", 0.0)}
    out["systolic_bp"], out["diastolic_bp"] = sbp, dbp

    # Composite signal quality (0..100) from pulse SNR + spectral sharpness.
    sqi = float(np.clip((snr + 5.0) * 6.0, 0.0, 100.0))
    out["signal_quality"] = VitalSign("signal_quality", sqi, "0-100",
                                      _snr_to_conf(snr), SRC_SPECTRAL,
                                      extra={"snr_db": snr, "morphology": morph})
    return out


__all__ = [
    "VitalSign", "Spo2Calibration", "PersonalizedBPEstimator", "VitalsCalibration",
    "frame_diff_motion", "clean_frame_mask", "select_clean_frames",
    "reconstruct_facial_bvp", "heart_rate_from_bvp", "respiratory_rate_from_bvp",
    "spo2_ratio_from_faces", "spo2_from_faces", "extract_pulse_morphology",
    "estimate_all_vitals",
]