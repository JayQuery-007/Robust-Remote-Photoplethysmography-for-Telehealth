"""eval_ibvp.py — Evaluate EquiPhysDANN + POS + CHROM on iBVP.

iBVP (individualized Blood Volume Pulse) is recorded in controlled conditions
with both a contact reference sensor and camera, making it well-suited for
waveform-level Pearson evaluation in addition to HR MAE.

Runs three methods side-by-side on every subject:
  1. EquiPhysDANN  — deep model
  2. POS           — classical baseline
  3. CHROM         — classical baseline

Outputs:
  ibvp_results.csv         — per-subject HR, MAE, BVP Pearson for the deep model
  ibvp_results_summary.csv — aggregate metrics per method

iBVP folder structure (two common variants — both handled):
  Variant A (per-subject folder):
    <root>/
      S1/
        face.mp4   (or S1.avi)
        bvp.csv    (columns: timestamp, bvp  OR  just a single BVP column)
      S2/
        ...

  Variant B (task sub-folders):
    <root>/
      S1/
        T1/
          vid.avi
          bvp.csv
        T2/
          ...

Usage:
  python eval_ibvp.py --root E:/datasets/iBVP --checkpoint ../checkpoints_v2/equiphys_best.pt
  python eval_ibvp.py --root E:/datasets/iBVP  # auto-finds checkpoint
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
    EquiPhysDANN,
    FaceROIExtractor,
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    estimate_hr_bpm_from_bvp,
    compute_signal_quality_index,
    estimate_respiratory_rate,
)
from facial_vitals import clean_frame_mask, frame_diff_motion, select_clean_frames


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_FRAMES = 150
NORM_CROP    = 64
VIDEO_EXTS   = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────────────────

def load_ibvp_gt(bvp_csv: str) -> Dict:
    """Load iBVP ground-truth BVP CSV.

    Handles:
      - Single-column file: raw BVP signal, one sample per row (no header)
      - Two-column CSV: timestamp, bvp  (with or without header)
      - Multi-column CSV: looks for columns named 'bvp', 'ppg', or 'signal'
    """
    import csv as _csv

    lines = Path(bvp_csv).read_text(encoding="utf-8-sig").strip().splitlines()
    if not lines:
        raise ValueError(f"Empty BVP file: {bvp_csv}")

    # Detect if first line is a header (contains non-numeric text)
    def _is_numeric(s: str) -> bool:
        try:
            float(s.replace(",", ".").strip())
            return True
        except ValueError:
            return False

    first_fields = [f.strip() for f in lines[0].split(",")]
    has_header   = not all(_is_numeric(f) for f in first_fields if f)

    bvp_col = None
    ts_col  = None
    bvp_vals: List[float] = []
    ts_vals:  List[float] = []

    if has_header:
        headers = [h.lower() for h in first_fields]
        for c in ["bvp", "ppg", "signal", "value"]:
            if c in headers:
                bvp_col = headers.index(c)
                break
        for c in ["timestamp", "time", "t", "time_s"]:
            if c in headers:
                ts_col = headers.index(c)
                break
        data_lines = lines[1:]
    else:
        # Single column or two columns without header
        if len(first_fields) == 1:
            bvp_col = 0
        elif len(first_fields) == 2:
            ts_col, bvp_col = 0, 1
        else:
            bvp_col = 0  # fallback: take first column
        data_lines = lines

    for line in data_lines:
        fields = [f.strip() for f in line.split(",")]
        if not fields or not any(fields):
            continue
        try:
            bvp_vals.append(float(fields[bvp_col]))
            if ts_col is not None and ts_col < len(fields):
                ts_vals.append(float(fields[ts_col]))
        except (ValueError, IndexError):
            continue

    if not bvp_vals:
        raise ValueError(f"Could not parse BVP values from {bvp_csv}")

    bvp = np.array(bvp_vals, dtype=np.float64)

    if ts_vals:
        ts  = np.array(ts_vals, dtype=np.float64)
        fps = float(len(bvp) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else 30.0
    else:
        ts  = None
        fps = 30.0  # iBVP default

    # Derive instantaneous HR via a sliding window Welch estimate
    # (iBVP does not always ship a separate HR column)
    hr_mean = _bvp_to_mean_hr(bvp, fps)

    return {
        "bvp":        bvp,
        "timestamps": ts,
        "mean_hr":    hr_mean,
        "fps":        fps,
        "duration_s": float(len(bvp) / fps),
        "n_samples":  len(bvp),
    }


def _bvp_to_mean_hr(bvp: np.ndarray, fps: float,
                    f_low: float = 0.8, f_high: float = 2.5) -> float:
    """Derive a mean HR from a raw BVP signal via Welch PSD."""
    peak_hz, _ = _welch_peak_hz(bvp, fps, f_low=f_low, f_high=f_high,
                                 snr_ratio_threshold=0.0)
    return peak_hz * 60.0 if peak_hz > 0 else float(np.nan)


def _bvp_pearson(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Pearson correlation between two BVP waveforms, aligned by length."""
    n = min(len(pred), len(target))
    if n < 30:
        return None
    p = pred[:n] - pred[:n].mean()
    t = target[:n] - target[:n].mean()
    sp = np.std(p)
    st = np.std(t)
    if sp < 1e-8 or st < 1e-8:
        return None
    r = float(np.mean(p * t) / (sp * st))
    return r if np.isfinite(r) else None


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
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
    print(f"  Loaded: {checkpoint_path}")
    return model


def _resolve_checkpoint(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    root = _HERE.parent
    for folder in ["checkpoints_v4", "checkpoints_v3", "checkpoints_v2",
                   "checkpoints_gpu_mcd", "checkpoints_gpu", "checkpoints"]:
        p = root / folder / "equiphys_best.pt"
        if p.exists():
            return str(p)
    raise FileNotFoundError("No checkpoint found. Pass --checkpoint path/to/equiphys_best.pt")


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
# Inference
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
        sig = detrend(sig.astype(np.float64))
        nyq = 0.5 * fs
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

    # Return the full concatenated BVP (first clip) for Pearson comparison
    return (float(np.median(per_clip_hrs)),
            float(np.mean(per_clip_snrs)),
            bvp_list[0])  # use first clip BVP for waveform Pearson


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
    """Discover iBVP subjects/tasks.

    Handles both flat (S1/face.mp4 + S1/bvp.csv) and nested
    (S1/T1/vid.avi + S1/T1/bvp.csv) layouts. Each discovered
    pairing is treated as one evaluation unit (a "subject-task").
    """
    subjects = []

    def _find_bvp_csv(folder: Path) -> Optional[Path]:
        for name in ["bvp.csv", "ppg.csv", "gt.csv", "signal.csv"]:
            p = folder / name
            if p.exists():
                return p
        hits = list(folder.glob("*.csv"))
        return sorted(hits)[0] if hits else None

    def _find_video(folder: Path) -> Optional[Path]:
        for ext in VIDEO_EXTS:
            hits = list(folder.glob(f"*{ext}"))
            if hits:
                return sorted(hits)[0]
        return None

    root_path = Path(root)
    for subdir in sorted(root_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Try flat layout first
        vid = _find_video(subdir)
        bvp = _find_bvp_csv(subdir)
        if vid and bvp:
            subjects.append({
                "subject": subdir.name,
                "task":    "",
                "video":   str(vid),
                "gt":      str(bvp),
            })
            continue

        # Nested layout: look one level deeper
        for task_dir in sorted(subdir.iterdir()):
            if not task_dir.is_dir():
                continue
            vid = _find_video(task_dir)
            bvp = _find_bvp_csv(task_dir)
            if vid and bvp:
                subjects.append({
                    "subject": subdir.name,
                    "task":    task_dir.name,
                    "video":   str(vid),
                    "gt":      str(bvp),
                })

    return subjects


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subject(
    subject: str,
    task: str,
    video_path: str,
    gt_path: str,
    model: EquiPhysDANN,
    device: torch.device,
    hr_band_low: float,
    hr_band_high: float,
    snr_thresh: float,
    max_frames: Optional[int],
    n_clips: int,
) -> Dict:
    label = f"{subject}/{task}" if task else subject
    row   = {"subject": subject, "task": task, "label": label, "status": "ok"}

    # Ground truth
    try:
        gt = load_ibvp_gt(gt_path)
    except Exception as e:
        return {**row, "status": f"gt_error:{e}"}

    if not np.isfinite(gt["mean_hr"]):
        return {**row, "status": "gt_hr_invalid"}

    row["gt_mean_hr"]    = round(gt["mean_hr"],   2)
    row["gt_duration_s"] = round(gt["duration_s"],2)
    row["gt_fps"]        = round(gt["fps"],        2)
    row["gt_n_samples"]  = gt["n_samples"]

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

    # Deep model
    print(f"    Running EquiPhysDANN…")
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

    # BVP Pearson — iBVP specific: compare predicted waveform vs GT BVP
    if deep_bvp is not None and gt["bvp"] is not None:
        r_val = _bvp_pearson(deep_bvp, gt["bvp"])
        row["deep_bvp_pearson"] = round(r_val, 4) if r_val is not None else None
    else:
        row["deep_bvp_pearson"] = None

    if deep_bvp is not None and len(deep_bvp) >= 60:
        row["deep_sqi"] = round(compute_signal_quality_index(deep_bvp, fps=fps), 2)
        rr = estimate_respiratory_rate(deep_bvp, fps=fps)
        row["deep_rr"]  = round(rr, 2) if rr > 0 else None
    else:
        row["deep_sqi"] = None
        row["deep_rr"]  = None

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

    # Error metrics
    gt_hr = gt["mean_hr"]
    for tag, pred_hr in [("deep", deep_hr), ("pos", pos_hr), ("chrom", chrom_hr)]:
        if pred_hr is not None:
            row[f"{tag}_mae"]     = round(abs(pred_hr - gt_hr), 2)
            row[f"{tag}_rmse_sq"] = round((pred_hr - gt_hr) ** 2, 4)
            row[f"{tag}_bias"]    = round(pred_hr - gt_hr, 2)
            row[f"{tag}_rel_err"] = round(100 * abs(pred_hr - gt_hr) / gt_hr, 2)
        else:
            for s in ("mae", "rmse_sq", "bias", "rel_err"):
                row[f"{tag}_{s}"] = None

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(rows: List[Dict], tag: str) -> Dict:
    valid = [r for r in rows if r.get(f"{tag}_mae") is not None]
    n = len(valid)
    if n == 0:
        return {"method": tag, "n_valid": 0}
    maes    = np.array([r[f"{tag}_mae"]     for r in valid])
    rmse_sq = np.array([r[f"{tag}_rmse_sq"] for r in valid])
    bias    = np.array([r[f"{tag}_bias"]    for r in valid])
    rels    = np.array([r[f"{tag}_rel_err"] for r in valid])
    snrs    = np.array([r.get(f"{tag}_snr_db") for r in valid
                        if r.get(f"{tag}_snr_db") is not None])
    # BVP Pearson — only available for deep model
    pears   = np.array([r.get("deep_bvp_pearson") for r in valid
                        if r.get("deep_bvp_pearson") is not None]) if tag == "deep" else np.array([])
    out = {
        "method":           tag,
        "n_subjects":       len(rows),
        "n_valid":          n,
        "n_failed":         len(rows) - n,
        "mae_mean":         round(float(maes.mean()), 3),
        "mae_std":          round(float(maes.std()),  3),
        "mae_median":       round(float(np.median(maes)), 3),
        "rmse":             round(float(np.sqrt(rmse_sq.mean())), 3),
        "mean_bias":        round(float(bias.mean()), 3),
        "mean_rel_err_pct": round(float(rels.mean()), 3),
        "pct_within_2bpm":  round(float(np.mean(maes <= 2.0) * 100), 1),
        "pct_within_5bpm":  round(float(np.mean(maes <= 5.0) * 100), 1),
        "pct_within_10bpm": round(float(np.mean(maes <= 10.0) * 100), 1),
        "mean_snr_db":      round(float(snrs.mean()), 2) if len(snrs) else None,
    }
    if len(pears) > 0:
        out["mean_bvp_pearson"] = round(float(pears.mean()), 4)
        out["pct_pearson_pos"]  = round(float(np.mean(pears > 0) * 100), 1)
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
        description="Evaluate EquiPhysDANN + POS + CHROM on iBVP"
    )
    p.add_argument("--root",       required=True,  help="iBVP dataset root")
    p.add_argument("--checkpoint", default=None,   help="Path to equiphys_best.pt")
    p.add_argument("--out",        default="results/ibvp_results.csv")
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

    subjects = discover_subjects(args.root)
    if not subjects:
        print(f"No subjects found under {args.root}")
        sys.exit(1)
    if args.subjects:
        subjects = [s for s in subjects if s["subject"] in args.subjects]

    print(f"\niBVP: evaluating {len(subjects)} subject-tasks  device={args.device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"HR band    : {args.hr_low}–{args.hr_high} BPM\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, s in enumerate(subjects, 1):
        label = f"{s['subject']}/{s['task']}" if s["task"] else s["subject"]
        print(f"[{i}/{len(subjects)}] {label}")
        try:
            r = evaluate_subject(
                subject=s["subject"], task=s["task"],
                video_path=s["video"], gt_path=s["gt"],
                model=model, device=device,
                hr_band_low=args.hr_low, hr_band_high=args.hr_high,
                snr_thresh=args.snr_thresh, max_frames=args.max_frames,
                n_clips=args.n_clips,
            )
        except Exception as e:
            r = {"subject": s["subject"], "task": s["task"],
                 "label": label, "status": f"crash:{e}"}
            traceback.print_exc()
        results.append(r)
        print(f"  deep={r.get('deep_hr','—')} pos={r.get('pos_hr','—')} "
              f"chrom={r.get('chrom_hr','—')} gt={r.get('gt_mean_hr','—')} "
              f"pearson={r.get('deep_bvp_pearson','—')} "
              f"MAE(deep)={r.get('deep_mae','—')} [{r.get('status','ok')}]")

    results.sort(key=lambda x: (x["subject"], x.get("task", "")))
    write_csv(results, str(out_path))

    summary_rows = [aggregate(results, tag) for tag in ("deep", "pos", "chrom")]
    write_csv(summary_rows, str(out_path.with_stem(out_path.stem + "_summary")))

    # Console table
    print("\n" + "=" * 70)
    print("iBVP  METHOD COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<32} {'EquiPhysDANN':>14} {'POS':>10} {'CHROM':>10}"
    print(header)
    print("-" * len(header))
    metrics = [
        ("n_valid",           "Valid subjects/tasks"),
        ("mae_mean",          "MAE mean (BPM)"),
        ("mae_std",           "MAE std"),
        ("mae_median",        "MAE median"),
        ("rmse",              "RMSE (BPM)"),
        ("mean_bias",         "Bias (BPM)"),
        ("pct_within_5bpm",   "Within ±5 BPM %"),
        ("pct_within_10bpm",  "Within ±10 BPM %"),
        ("mean_snr_db",       "Mean SNR (dB)"),
        ("mean_bvp_pearson",  "BVP Pearson (deep)"),
        ("pct_pearson_pos",   "Pearson > 0 % (deep)"),
    ]
    by_method = {r["method"]: r for r in summary_rows}
    for key, label in metrics:
        deep_val  = by_method.get("deep",  {}).get(key, "—")
        pos_val   = by_method.get("pos",   {}).get(key, "—") if key not in ("mean_bvp_pearson", "pct_pearson_pos") else "N/A"
        chrom_val = by_method.get("chrom", {}).get(key, "—") if key not in ("mean_bvp_pearson", "pct_pearson_pos") else "N/A"
        print(f"{label:<32} {str(deep_val):>14} {str(pos_val):>10} {str(chrom_val):>10}")

    print(f"\nResults CSVs in: {out_path.parent}/")


if __name__ == "__main__":
    main()
