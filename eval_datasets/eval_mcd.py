"""eval_mcd.py — Evaluate EquiPhysDANN on MCD (Multi-task Clinical Dataset).

MCD provides rPPG video alongside clinical reference measurements:
  - Systolic blood pressure (SBP)
  - Diastolic blood pressure (DBP)
  - Haemoglobin concentration (Hb)

Because MCD is used for clinical regression transfer (not direct BVP supervision),
this script evaluates two things separately:

  1. HR accuracy  — EquiPhysDANN BVP head + POS + CHROM vs any available HR GT
  2. Clinical regression — if a trained ClinicalRegressor checkpoint is provided,
     evaluates SBP, DBP, Hb MAE against the clinical ground truth

Outputs:
  mcd_results.csv             — per-subject HR + clinical predictions vs GT
  mcd_results_summary.csv     — aggregate HR metrics per method
  mcd_clinical_summary.csv    — aggregate clinical regression MAE / RMSE / Pearson
                                 (only written if --clinical-checkpoint is passed)

MCD folder structure:
  <root>/
    subject_001/   (or p01 / S1 — any name)
      vid.mp4
      clinical.csv    (columns: sbp, dbp, hb  OR  SBP, DBP, Hb)
      bvp.csv         (optional: if absent, HR GT is derived via Welch)
    subject_002/
      ...

Usage:
  python eval_mcd.py --root E:/datasets/MCD --checkpoint ../checkpoints_v2/equiphys_best.pt
  python eval_mcd.py --root E:/datasets/MCD --checkpoint best.pt --clinical-checkpoint equiphys_clinical_best.pt
  python eval_mcd.py --root E:/datasets/MCD  # auto-finds checkpoints
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── path resolution ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent]:
    if (_candidate / "equiphys_core.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from equiphys_core import (
    ClinicalRegressor,
    EquiPhysDANN,
    FaceROIExtractor,
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    compute_signal_quality_index,
    estimate_hr_bpm_from_bvp,
    estimate_respiratory_rate,
)
from facial_vitals import clean_frame_mask, frame_diff_motion, select_clean_frames


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_FRAMES   = 150
NORM_CROP      = 64
VIDEO_EXTS     = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
CLINICAL_COLS  = ["sbp", "dbp", "hb"]  # canonical names (case-insensitive)


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────────────────

def load_mcd_clinical(clinical_csv: str) -> Dict[str, Optional[float]]:
    """Load a single-row (or averaged) clinical ground truth from MCD CSV.

    Returns dict with keys 'sbp', 'dbp', 'hb' (all float or None).
    The CSV may hold a single reference reading or multiple readings to average.
    """
    import csv as _csv
    rows = []
    with open(clinical_csv, newline="", encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        for r in reader:
            rows.append({k.strip().lower(): v.strip() for k, v in r.items()})

    if not rows:
        return {"sbp": None, "dbp": None, "hb": None}

    def _extract(col_candidates):
        for c in col_candidates:
            if c in headers or c in rows[0]:
                vals = []
                for r in rows:
                    v = r.get(c, "").replace(",", ".")
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
                return float(np.mean(vals)) if vals else None
        return None

    return {
        "sbp": _extract(["sbp", "systolic", "sys_bp", "systolic_bp"]),
        "dbp": _extract(["dbp", "diastolic", "dia_bp", "diastolic_bp"]),
        "hb":  _extract(["hb", "haemoglobin", "hemoglobin", "hgb", "haemo"]),
    }


def load_mcd_bvp_gt(bvp_csv: str) -> Optional[Dict]:
    """Load optional BVP / HR ground truth from MCD. Returns None if absent."""
    if not Path(bvp_csv).exists():
        return None
    try:
        import csv as _csv
        lines = Path(bvp_csv).read_text(encoding="utf-8-sig").strip().splitlines()
        if not lines:
            return None

        def _is_num(s):
            try: float(s.replace(",", ".").strip()); return True
            except ValueError: return False

        first = [f.strip() for f in lines[0].split(",")]
        has_header = not all(_is_num(f) for f in first if f)

        bvp_col = None
        ts_col  = None

        if has_header:
            hdrs = [h.lower() for h in first]
            for c in ["bvp", "ppg", "signal", "value"]:
                if c in hdrs: bvp_col = hdrs.index(c); break
            for c in ["timestamp", "time", "t", "time_s"]:
                if c in hdrs: ts_col = hdrs.index(c); break
            data = lines[1:]
        else:
            bvp_col = 1 if len(first) == 2 else 0
            ts_col  = 0 if len(first) == 2 else None
            data    = lines

        bvp_vals, ts_vals = [], []
        for line in data:
            fs = [f.strip() for f in line.split(",")]
            if not fs: continue
            try:
                bvp_vals.append(float(fs[bvp_col]))
                if ts_col is not None and ts_col < len(fs):
                    ts_vals.append(float(fs[ts_col]))
            except (ValueError, IndexError):
                continue

        if not bvp_vals:
            return None

        bvp = np.array(bvp_vals, dtype=np.float64)
        fps = float(len(bvp) / (ts_vals[-1] - ts_vals[0])) if ts_vals and ts_vals[-1] > ts_vals[0] else 30.0
        peak_hz, _ = _welch_peak_hz(bvp, fps, f_low=0.8, f_high=2.5,
                                     snr_ratio_threshold=0.0)
        hr_mean = peak_hz * 60.0 if peak_hz > 0 else float("nan")
        return {"bvp": bvp, "mean_hr": hr_mean, "fps": fps}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> EquiPhysDANN:
    model = EquiPhysDANN(in_channels=3, latent_dim=256,
                         frames=MODEL_FRAMES, lambda_grl=1.0).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    model.eval()
    print(f"  Loaded backbone: {checkpoint_path}")
    return model


def load_clinical_model(
    ckpt_path: str,
    backbone: EquiPhysDANN,
    device: torch.device,
) -> Optional[ClinicalRegressor]:
    """Load the ClinicalRegressor that was trained on top of the frozen extractor."""
    if not ckpt_path:
        return None
    try:
        ckpt   = torch.load(ckpt_path, map_location=device)
        # ClinicalRegressor wraps backbone.extractor
        clin   = ClinicalRegressor(backbone.extractor, latent_dim=256, out_targets=3).to(device)
        state  = ckpt.get("clinical_model_state", ckpt)
        clin.load_state_dict(state, strict=False)
        clin.eval()
        print(f"  Loaded clinical model: {ckpt_path}")
        return clin
    except Exception as e:
        print(f"  [WARN] Could not load clinical model ({ckpt_path}): {e}")
        return None


def _resolve_checkpoint(explicit: Optional[str], name: str = "equiphys_best.pt") -> str:
    if explicit:
        return explicit
    root = _HERE.parent
    for folder in ["checkpoints_v4", "checkpoints_v3", "checkpoints_v2",
                   "checkpoints_gpu_mcd", "checkpoints_gpu", "checkpoints"]:
        p = root / folder / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No checkpoint found for {name}. Pass --checkpoint.")


# ──────────────────────────────────────────────────────────────────────────────
# ROI extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_roi_frames(
    video_path: str,
    target_size: int = NORM_CROP,
    max_frames: Optional[int] = None,
    progress_every: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], float]:
    roi_extractor = FaceROIExtractor(target_size=target_size)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total

    roi_unmasked, roi_masked, timestamps = [], [], []
    idx = 0
    face_missed = 0

    while idx < limit:
        ok, bgr = cap.read()
        if not ok:
            break
        roi_rgb, roi_mask, _ = roi_extractor.extract_with_landmarks(bgr)
        if roi_mask.mean() > 0.1:
            unmasked = roi_rgb.astype(np.float32)
            if unmasked.shape[0] != target_size or unmasked.shape[1] != target_size:
                unmasked = cv2.resize(unmasked, (target_size, target_size),
                                      interpolation=cv2.INTER_AREA)
            roi_unmasked.append(unmasked)
            mask3  = np.repeat((roi_mask > 0.5)[..., None], 3, axis=2).astype(np.float32)
            masked = cv2.resize(roi_rgb.astype(np.float32) * mask3,
                                (128, 128), interpolation=cv2.INTER_AREA)
            roi_masked.append(masked)
            timestamps.append(idx / fps)
        else:
            face_missed += 1
        idx += 1
        if idx % progress_every == 0:
            print(f"      frame {idx}/{limit}  ROIs={len(roi_unmasked)}", flush=True)

    cap.release()
    if face_missed > 0:
        print(f"      face missed on {face_missed}/{idx} frames")
    return roi_unmasked, roi_masked, timestamps, fps


# ──────────────────────────────────────────────────────────────────────────────
# Inference — HR
# ──────────────────────────────────────────────────────────────────────────────

def run_deep_model(
    roi_frames: List[np.ndarray],
    fps: float,
    model: EquiPhysDANN,
    device: torch.device,
    hr_band_low: float,
    hr_band_high: float,
    snr_thresh: float,
    n_clips: int = 3,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    n = len(roi_frames)
    if n < MODEL_FRAMES:
        print(f"      [WARN] only {n} frames, need {MODEL_FRAMES}")
        return None, None, None

    from scipy.signal import butter, filtfilt, detrend

    def _clean(sig, fs, lo, hi):
        sig  = detrend(sig.astype(np.float64))
        nyq  = 0.5 * fs
        lo_n = max(0.01, min(0.99, lo / nyq))
        hi_n = max(lo_n + 0.001, min(0.99, hi / nyq))
        try:
            b, a = butter(3, [lo_n, hi_n], btype="band")
            return filtfilt(b, a, sig)
        except Exception:
            return sig

    step = max(1, (n - MODEL_FRAMES) // max(n_clips - 1, 1))
    starts = list(range(0, n - MODEL_FRAMES + 1, step))[:n_clips]
    per_clip_hrs, per_clip_snrs, bvp_list = [], [], []

    with torch.no_grad():
        for s in starts:
            seg  = roi_frames[s : s + MODEL_FRAMES]
            buf  = deque(seg, maxlen=MODEL_FRAMES)
            clip = build_clip_tensor_from_buffer(buf, device=device)
            out  = model(clip)
            bvp_pred  = out["bvp_pred"][0].cpu().numpy().astype(np.float32)
            bvp_clean = _clean(bvp_pred, fps, hr_band_low, hr_band_high)
            bvp_list.append(bvp_clean)
            peak_hz_i, snr_i = _welch_peak_hz(
                bvp_clean, fps,
                f_low=hr_band_low, f_high=hr_band_high,
                snr_ratio_threshold=snr_thresh,
            )
            if peak_hz_i > 0:
                per_clip_hrs.append(peak_hz_i * 60.0)
                per_clip_snrs.append(float(snr_i) if np.isfinite(snr_i) else 0.0)

    if not per_clip_hrs:
        return None, None, np.mean(bvp_list, axis=0) if bvp_list else None
    return (float(np.median(per_clip_hrs)),
            float(np.mean(per_clip_snrs)),
            np.mean(bvp_list, axis=0))


# ──────────────────────────────────────────────────────────────────────────────
# Inference — clinical regression
# ──────────────────────────────────────────────────────────────────────────────

def run_clinical_model(
    roi_frames: List[np.ndarray],
    model: ClinicalRegressor,
    device: torch.device,
    n_clips: int = 3,
) -> Optional[np.ndarray]:
    """Run ClinicalRegressor on overlapping clips and average predictions.

    Returns [sbp_pred, dbp_pred, hb_pred] or None.
    """
    n = len(roi_frames)
    if n < MODEL_FRAMES:
        return None

    step   = max(1, (n - MODEL_FRAMES) // max(n_clips - 1, 1))
    starts = list(range(0, n - MODEL_FRAMES + 1, step))[:n_clips]
    preds  = []

    with torch.no_grad():
        for s in starts:
            seg  = roi_frames[s : s + MODEL_FRAMES]
            buf  = deque(seg, maxlen=MODEL_FRAMES)
            clip = build_clip_tensor_from_buffer(buf, device=device)
            pred = model(clip)          # [1, 3]  → [sbp, dbp, hb]
            preds.append(pred[0].cpu().numpy().astype(np.float32))

    return np.mean(preds, axis=0) if preds else None


def run_classical(
    roi_frames: List[np.ndarray],
    timestamps: Optional[np.ndarray],
    fps: float,
    method: str,
    hr_band_low: float,
    hr_band_high: float,
    snr_thresh: float,
) -> Tuple[Optional[float], Optional[float]]:
    try:
        pulse, fs_used = _extract_patch_weighted_pulse(
            roi_frames, timestamps=timestamps, target_fs=fps, method=method
        )
        peak_hz, snr = _welch_peak_hz(
            pulse, fs_used,
            f_low=hr_band_low, f_high=hr_band_high,
            snr_ratio_threshold=snr_thresh,
        )
        hr = peak_hz * 60.0 if peak_hz > 0 else None
        return hr, float(snr) if np.isfinite(snr) else None
    except Exception as e:
        print(f"      [{method}] error: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_subjects(root: str) -> List[Dict]:
    """Discover MCD subjects.

    Each subject folder must have a video and clinical.csv.
    bvp.csv is optional.
    """
    subjects = []
    root_path = Path(root)
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Clinical GT
        clinical_csv = None
        for name in ["clinical.csv", "labels.csv", "gt_clinical.csv", "reference.csv"]:
            p = subdir / name
            if p.exists():
                clinical_csv = str(p)
                break
        if clinical_csv is None:
            continue

        # Video
        video_path = None
        for ext in VIDEO_EXTS:
            hits = list(subdir.glob(f"*{ext}"))
            if hits:
                video_path = str(sorted(hits)[0])
                break
        if video_path is None:
            continue

        # Optional BVP GT
        bvp_csv = None
        for name in ["bvp.csv", "ppg.csv", "gt.csv", "bvp_gt.csv"]:
            p = subdir / name
            if p.exists():
                bvp_csv = str(p)
                break

        subjects.append({
            "subject":      subdir.name,
            "video":        video_path,
            "clinical_csv": clinical_csv,
            "bvp_csv":      bvp_csv,
        })
    return subjects


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subject(
    subject: str,
    video_path: str,
    clinical_csv: str,
    bvp_csv: Optional[str],
    model: EquiPhysDANN,
    clinical_model: Optional[ClinicalRegressor],
    device: torch.device,
    hr_band_low: float,
    hr_band_high: float,
    snr_thresh: float,
    max_frames: Optional[int],
    n_clips: int,
) -> Dict:
    row = {"subject": subject, "status": "ok"}

    # Clinical GT
    try:
        clin_gt = load_mcd_clinical(clinical_csv)
    except Exception as e:
        return {**row, "status": f"clinical_gt_error:{e}"}

    row["gt_sbp"] = round(clin_gt["sbp"], 2) if clin_gt["sbp"] is not None else None
    row["gt_dbp"] = round(clin_gt["dbp"], 2) if clin_gt["dbp"] is not None else None
    row["gt_hb"]  = round(clin_gt["hb"],  2) if clin_gt["hb"]  is not None else None

    # Optional BVP / HR GT
    bvp_gt = load_mcd_bvp_gt(bvp_csv) if bvp_csv else None
    row["gt_mean_hr"] = round(bvp_gt["mean_hr"], 2) if bvp_gt and np.isfinite(bvp_gt["mean_hr"]) else None

    # ROI extraction
    print(f"    Extracting ROIs…")
    try:
        roi_unmasked, roi_masked, timestamps, fps = extract_roi_frames(
            video_path, target_size=NORM_CROP, max_frames=max_frames
        )
    except Exception as e:
        return {**row, "status": f"roi_error:{e}"}

    row["n_roi_frames"] = len(roi_unmasked)
    row["detected_fps"] = round(fps, 2)

    if len(roi_unmasked) < 60:
        return {**row, "status": f"too_few_frames:{len(roi_unmasked)}"}

    motion = frame_diff_motion(roi_masked)
    keep   = clean_frame_mask(motion, k=3.5)
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    clean_masked,   clean_ts   = select_clean_frames(roi_masked,   ts_arr, keep)
    clean_unmasked, _          = select_clean_frames(roi_unmasked, ts_arr, keep)
    row["clean_frame_frac"] = round(float(keep.mean()), 3)
    fs_eff = (float(len(clean_masked) / (clean_ts[-1] - clean_ts[0]))
              if clean_ts is not None and len(clean_ts) > 1 and clean_ts[-1] > clean_ts[0]
              else fps)

    f_low_hz  = hr_band_low  / 60.0
    f_high_hz = hr_band_high / 60.0

    # Deep model — HR
    print(f"    Running EquiPhysDANN (HR)…")
    t0 = time.time()
    deep_hr, deep_snr, deep_bvp = run_deep_model(
        roi_frames=roi_unmasked, fps=fps,
        model=model, device=device,
        hr_band_low=f_low_hz, hr_band_high=f_high_hz,
        snr_thresh=snr_thresh, n_clips=n_clips,
    )
    row["deep_hr"]     = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"] = round(deep_snr, 2) if deep_snr is not None else None
    row["deep_time_s"] = round(time.time() - t0, 1)

    if deep_bvp is not None and len(deep_bvp) >= 60:
        row["deep_sqi"] = round(compute_signal_quality_index(deep_bvp, fps=fps), 2)
        rr = estimate_respiratory_rate(deep_bvp, fps=fps)
        row["deep_rr"]  = round(rr, 2) if rr > 0 else None
    else:
        row["deep_sqi"] = None
        row["deep_rr"]  = None

    # Clinical regression
    if clinical_model is not None:
        print(f"    Running clinical regression…")
        clin_pred = run_clinical_model(roi_unmasked, clinical_model, device, n_clips=n_clips)
        if clin_pred is not None:
            row["pred_sbp"] = round(float(clin_pred[0]), 2)
            row["pred_dbp"] = round(float(clin_pred[1]), 2)
            row["pred_hb"]  = round(float(clin_pred[2]), 2)
            for col, pred_val, gt_val in [
                ("sbp", clin_pred[0], clin_gt["sbp"]),
                ("dbp", clin_pred[1], clin_gt["dbp"]),
                ("hb",  clin_pred[2], clin_gt["hb"]),
            ]:
                if gt_val is not None:
                    row[f"{col}_mae"]  = round(abs(float(pred_val) - gt_val), 2)
                    row[f"{col}_bias"] = round(float(pred_val) - gt_val, 2)
                else:
                    row[f"{col}_mae"]  = None
                    row[f"{col}_bias"] = None
        else:
            for col in ["pred_sbp", "pred_dbp", "pred_hb",
                        "sbp_mae", "dbp_mae", "hb_mae",
                        "sbp_bias", "dbp_bias", "hb_bias"]:
                row[col] = None

    # POS
    print(f"    Running POS…")
    pos_hr, pos_snr = run_classical(
        clean_masked, clean_ts, fs_eff, "pos", f_low_hz, f_high_hz, snr_thresh
    )
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None

    # CHROM
    print(f"    Running CHROM…")
    chrom_hr, chrom_snr = run_classical(
        clean_masked, clean_ts, fs_eff, "chrom", f_low_hz, f_high_hz, snr_thresh
    )
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None

    # HR error metrics (only when BVP GT available)
    if row.get("gt_mean_hr") is not None:
        gt_hr_val = row["gt_mean_hr"]
        for tag, pred_hr in [("deep", deep_hr), ("pos", pos_hr), ("chrom", chrom_hr)]:
            if pred_hr is not None:
                row[f"{tag}_mae"]     = round(abs(pred_hr - gt_hr_val), 2)
                row[f"{tag}_rmse_sq"] = round((pred_hr - gt_hr_val) ** 2, 4)
                row[f"{tag}_bias"]    = round(pred_hr - gt_hr_val, 2)
            else:
                for s in ("mae", "rmse_sq", "bias"):
                    row[f"{tag}_{s}"] = None

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_hr(rows: List[Dict], tag: str) -> Dict:
    valid = [r for r in rows if r.get(f"{tag}_mae") is not None]
    n = len(valid)
    if n == 0:
        return {"method": tag, "n_valid": 0}
    maes    = np.array([r[f"{tag}_mae"]     for r in valid])
    rmse_sq = np.array([r[f"{tag}_rmse_sq"] for r in valid])
    bias    = np.array([r[f"{tag}_bias"]    for r in valid])
    snrs    = np.array([r.get(f"{tag}_snr_db") for r in valid
                        if r.get(f"{tag}_snr_db") is not None])
    return {
        "method":           tag,
        "n_with_hr_gt":     len(rows),
        "n_valid":          n,
        "n_failed":         len(rows) - n,
        "mae_mean":         round(float(maes.mean()), 3),
        "mae_median":       round(float(np.median(maes)), 3),
        "rmse":             round(float(np.sqrt(rmse_sq.mean())), 3),
        "mean_bias":        round(float(bias.mean()), 3),
        "pct_within_5bpm":  round(float(np.mean(maes <= 5.0) * 100), 1),
        "pct_within_10bpm": round(float(np.mean(maes <= 10.0) * 100), 1),
        "mean_snr_db":      round(float(snrs.mean()), 2) if len(snrs) else None,
    }


def aggregate_clinical(rows: List[Dict]) -> Dict:
    """Aggregate SBP, DBP, Hb regression metrics across subjects."""
    out = {"n_subjects": len(rows)}
    for col in CLINICAL_COLS:
        mae_key  = f"{col}_mae"
        bias_key = f"{col}_bias"
        gt_key   = f"gt_{col}"
        pred_key = f"pred_{col}"
        valid = [r for r in rows
                 if r.get(mae_key) is not None and r.get(gt_key) is not None]
        n = len(valid)
        if n == 0:
            out.update({f"{col}_n": 0})
            continue
        maes   = np.array([r[mae_key]  for r in valid])
        biases = np.array([r[bias_key] for r in valid])
        preds  = np.array([r[pred_key] for r in valid])
        gts    = np.array([r[gt_key]   for r in valid])
        pearson = 0.0
        if np.std(preds) > 1e-8 and np.std(gts) > 1e-8:
            pearson = float(np.corrcoef(preds, gts)[0, 1])
            if not np.isfinite(pearson):
                pearson = 0.0
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        out.update({
            f"{col}_n":          n,
            f"{col}_mae_mean":   round(float(maes.mean()), 3),
            f"{col}_mae_std":    round(float(maes.std()),  3),
            f"{col}_mae_median": round(float(np.median(maes)), 3),
            f"{col}_rmse":       round(rmse, 3),
            f"{col}_bias":       round(float(biases.mean()), 3),
            f"{col}_pearson":    round(pearson, 4),
        })
    return out


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate EquiPhysDANN + POS + CHROM + clinical regression on MCD"
    )
    p.add_argument("--root",                required=True,  help="MCD dataset root")
    p.add_argument("--checkpoint",          default=None,   help="Path to equiphys_best.pt")
    p.add_argument("--clinical-checkpoint", default=None,
                   help="Path to equiphys_clinical_best.pt (optional)")
    p.add_argument("--out",        default="results/mcd_results.csv")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hr-low",     type=float, default=48.0)
    p.add_argument("--hr-high",    type=float, default=150.0)
    p.add_argument("--snr-thresh", type=float, default=0.12)
    p.add_argument("--max-frames", type=int,   default=None)
    p.add_argument("--n-clips",    type=int,   default=3)
    p.add_argument("--subjects",   nargs="*",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    try:
        ckpt_path = _resolve_checkpoint(args.checkpoint)
        model     = load_model(ckpt_path, device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    clinical_model = None
    if args.clinical_checkpoint:
        clinical_model = load_clinical_model(args.clinical_checkpoint, model, device)
    else:
        # Try auto-discovery
        try:
            clin_path = _resolve_checkpoint(None, name="equiphys_clinical_best.pt")
            clinical_model = load_clinical_model(clin_path, model, device)
        except FileNotFoundError:
            print("  No clinical checkpoint found — HR-only evaluation mode.")

    subjects = discover_subjects(args.root)
    if not subjects:
        print(f"No subjects found under {args.root}")
        sys.exit(1)
    if args.subjects:
        subjects = [s for s in subjects if s["subject"] in args.subjects]

    print(f"\nMCD: evaluating {len(subjects)} subjects  device={args.device}")
    print(f"Backbone   : {ckpt_path}")
    print(f"Clinical   : {'loaded' if clinical_model else 'not available'}")
    print(f"HR band    : {args.hr_low}–{args.hr_high} BPM\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, s in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {s['subject']}")
        try:
            r = evaluate_subject(
                subject=s["subject"],
                video_path=s["video"],
                clinical_csv=s["clinical_csv"],
                bvp_csv=s["bvp_csv"],
                model=model,
                clinical_model=clinical_model,
                device=device,
                hr_band_low=args.hr_low,
                hr_band_high=args.hr_high,
                snr_thresh=args.snr_thresh,
                max_frames=args.max_frames,
                n_clips=args.n_clips,
            )
        except Exception as e:
            r = {"subject": s["subject"], "status": f"crash:{e}"}
            traceback.print_exc()
        results.append(r)
        print(f"  deep={r.get('deep_hr','—')} gt_hr={r.get('gt_mean_hr','—')} "
              f"sbp={r.get('pred_sbp','—')}/gt={r.get('gt_sbp','—')} "
              f"dbp={r.get('pred_dbp','—')}/gt={r.get('gt_dbp','—')} "
              f"hb={r.get('pred_hb','—')}/gt={r.get('gt_hb','—')} "
              f"[{r.get('status','ok')}]")

    results.sort(key=lambda x: x["subject"])
    write_csv(results, str(out_path))

    # HR summary (only subjects with BVP GT)
    has_hr_gt = [r for r in results if r.get("gt_mean_hr") is not None]
    if has_hr_gt:
        summary_rows = [aggregate_hr(has_hr_gt, tag) for tag in ("deep", "pos", "chrom")]
        write_csv(summary_rows, str(out_path.with_stem(out_path.stem + "_summary")))
    else:
        print("  No BVP GT found — skipping HR summary CSV.")
        summary_rows = []

    # Clinical summary
    if clinical_model is not None:
        clin_summary = aggregate_clinical(results)
        write_csv([clin_summary],
                  str(out_path.with_stem(out_path.stem + "_clinical_summary")))

    # Console table
    print("\n" + "=" * 70)
    print("MCD  HR COMPARISON" + (" (subjects with BVP GT)" if has_hr_gt else " — no HR GT"))
    print("=" * 70)
    if summary_rows:
        header = f"{'Metric':<28} {'EquiPhysDANN':>14} {'POS':>10} {'CHROM':>10}"
        print(header); print("-" * len(header))
        metrics = [
            ("n_valid",          "Valid subjects"),
            ("mae_mean",         "MAE mean (BPM)"),
            ("mae_median",       "MAE median"),
            ("rmse",             "RMSE (BPM)"),
            ("mean_bias",        "Bias (BPM)"),
            ("pct_within_5bpm",  "Within ±5 BPM %"),
            ("pct_within_10bpm", "Within ±10 BPM %"),
        ]
        by_m = {r["method"]: r for r in summary_rows}
        for key, lbl in metrics:
            print(f"{lbl:<28} "
                  f"{str(by_m.get('deep',{}).get(key,'—')):>14} "
                  f"{str(by_m.get('pos', {}).get(key,'—')):>10} "
                  f"{str(by_m.get('chrom',{}).get(key,'—')):>10}")

    if clinical_model is not None:
        print("\nClinical regression (ClinicalRegressor):")
        print(f"  {'Metric':<20} {'SBP (mmHg)':>12} {'DBP (mmHg)':>12} {'Hb (g/dL)':>12}")
        print("  " + "-" * 58)
        for metric, row_key_tpl in [
            ("MAE mean",  "{}_mae_mean"),
            ("MAE median","{}_mae_median"),
            ("RMSE",      "{}_rmse"),
            ("Bias",      "{}_bias"),
            ("Pearson r", "{}_pearson"),
        ]:
            vals = [str(clin_summary.get(row_key_tpl.format(c), "—")) for c in CLINICAL_COLS]
            print(f"  {metric:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print(f"\nResults CSVs in: {out_path.parent}/")


if __name__ == "__main__":
    main()
