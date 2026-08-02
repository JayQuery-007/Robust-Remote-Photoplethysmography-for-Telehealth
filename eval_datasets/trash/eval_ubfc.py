"""eval_ubfc.py — Evaluate EquiPhysDANN deep model against UBFC-rPPG ground truth.

Runs three methods side-by-side on every subject so you can compare directly:
  1. EquiPhysDANN  — your deep learning model (the thing you actually want to test)
  2. POS           — classical baseline
  3. CHROM         — classical baseline

Outputs:
  ubfc_results.csv         — per-subject: all three HR predictions vs GT, MAE, RMSE, etc.
  ubfc_results_summary.csv — aggregate metrics for each method

UBFC folder structure:
  <root>/
    subject1/  (or subj1 / p1 / 01 — any name)
      vid.avi
      ground_truth.txt   (row0=BVP, row1=HR_bpm, row2=timestamps)
    subject2/
      ...

Usage:
  python eval_ubfc.py --root E:/datasets/UBFC --checkpoint ../checkpoints_v2/equiphys_best.pt
  python eval_ubfc.py --root E:/datasets/UBFC --checkpoint ../checkpoints_v2/equiphys_best.pt --device cuda
  python eval_ubfc.py --root E:/datasets/UBFC  # auto-finds checkpoint in parent dirs
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

# ── path: scripts may live in a subfolder (eval_datasets/) ───────────────────
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

MODEL_FRAMES  = 150          # clip length the model was trained on
NORM_CROP     = 64           # spatial crop size fed to the model (matches training)
VIDEO_EXTS    = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────────────────

def load_ubfc_gt(gt_path: str) -> Dict:
    with open(gt_path, "r") as f:
        rows = f.read().strip().split("\n")
    bvp = np.array([float(x) for x in rows[0].split()])
    hr  = np.array([float(x) for x in rows[1].split()])
    ts  = np.array([float(x) for x in rows[2].split()]) if len(rows) >= 3 else None
    fps = float(len(hr) / ts[-1]) if ts is not None and ts[-1] > 0 else 30.0
    return {
        "bvp": bvp, "hr": hr, "timestamps": ts,
        "mean_hr": float(np.mean(hr)), "median_hr": float(np.median(hr)),
        "fps": fps, "duration_s": float(ts[-1]) if ts is not None else len(hr)/30.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> EquiPhysDANN:
    model = EquiPhysDANN(in_channels=3, latent_dim=256,
                         frames=MODEL_FRAMES, lambda_grl=1.0).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys in checkpoint: {missing[:5]}{'...' if len(missing)>5 else ''}")
    model.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")
    return model


def _resolve_checkpoint(explicit: Optional[str]) -> str:
    """Find checkpoint automatically if not specified."""
    if explicit:
        return explicit
    root = _HERE.parent  # one level up from eval_datasets/
    for folder in ["checkpoints_v2", "checkpoints_gpu_mcd", "checkpoints_gpu", "checkpoints"]:
        p = root / folder / "equiphys_best.pt"
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint path/to/equiphys_best.pt"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Video → ROI frame buffer
# ──────────────────────────────────────────────────────────────────────────────

def extract_roi_frames(
    video_path: str,
    target_size: int = NORM_CROP,
    max_frames: Optional[int] = None,
    progress_every: int = 100,
) -> Tuple[List[np.ndarray], List[float], float]:
    """Decode video and extract face ROI frames.

    Returns (roi_frames, timestamps, fps).
    Each roi_frame is [H,W,3] float32 in [0,1].
    """
    roi_extractor = FaceROIExtractor(target_size=target_size)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total

    # Extract both unmasked ROIs (for the deep model — matches training) and
    # masked ROIs (for POS/CHROM — matches streamlit classical path).
    roi_frames_unmasked = []  # for the deep model
    roi_frames_masked   = []  # for classical POS / CHROM
    timestamps = []
    idx = 0
    face_missed = 0

    while idx < limit:
        ok, bgr = cap.read()
        if not ok:
            break
        roi_rgb, roi_mask, _ = roi_extractor.extract_with_landmarks(bgr)
        if roi_mask.mean() > 0.1:
            # Unmasked: raw ROI, resized. This is what the model was trained on.
            unmasked = roi_rgb.astype(np.float32)
            if unmasked.shape[0] != target_size or unmasked.shape[1] != target_size:
                unmasked = cv2.resize(unmasked, (target_size, target_size),
                                       interpolation=cv2.INTER_AREA)
            roi_frames_unmasked.append(unmasked)

            # Masked + resized to 128: matches streamlit classical path.
            mask3 = np.repeat((roi_mask > 0.5)[..., None], 3, axis=2).astype(np.float32)
            masked = roi_rgb.astype(np.float32) * mask3
            masked = cv2.resize(masked, (128, 128), interpolation=cv2.INTER_AREA)
            roi_frames_masked.append(masked)

            timestamps.append(idx / fps)
        else:
            face_missed += 1
        idx += 1
        if idx % progress_every == 0:
            print(f"      frame {idx}/{limit}  ROIs={len(roi_frames_unmasked)}", flush=True)

    cap.release()
    if face_missed > 0:
        print(f"      face missed on {face_missed}/{idx} frames")
    return roi_frames_unmasked, roi_frames_masked, timestamps, fps


# ──────────────────────────────────────────────────────────────────────────────
# Method 1 — EquiPhysDANN inference
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
    """Run EquiPhysDANN on overlapping clips and average the HR predictions.

    Strategy: divide the full buffer into `n_clips` overlapping windows of
    MODEL_FRAMES each, run the model on each, then average the predicted BVP
    and derive a final HR via Welch PSD.

    Returns (hr_bpm, snr_db, bvp_waveform).
    """
    n = len(roi_frames)
    if n < MODEL_FRAMES:
        print(f"      [WARN] only {n} frames, need {MODEL_FRAMES} for deep model")
        return None, None, None

    # Build overlapping clip start indices
    step = max(1, (n - MODEL_FRAMES) // max(n_clips - 1, 1))
    starts = list(range(0, n - MODEL_FRAMES + 1, step))[:n_clips]

    from scipy.signal import butter, filtfilt, detrend

    def _clean(sig, fs, lo, hi):
        """Detrend + bandpass filter the BVP before HR estimation."""
        sig = detrend(sig.astype(np.float64))
        nyq = 0.5 * fs
        lo_n = max(0.01, min(0.99, lo / nyq))
        hi_n = max(lo_n + 0.001, min(0.99, hi / nyq))
        try:
            b, a = butter(3, [lo_n, hi_n], btype="band")
            return filtfilt(b, a, sig)
        except Exception:
            return sig

    # Run each clip and estimate HR PER CLIP, then use per-clip HRs directly
    # rather than averaging BVPs. Averaging misaligned-phase BVPs cancels the
    # signal and pushes the Welch peak toward DC — this is likely the -36 BPM bug.
    per_clip_hrs  = []
    per_clip_snrs = []
    bvp_list = []

    with torch.no_grad():
        for s in starts:
            seg = roi_frames[s : s + MODEL_FRAMES]
            buf = deque(seg, maxlen=MODEL_FRAMES)
            clip = build_clip_tensor_from_buffer(buf, device=device)  # [1,3,T,H,W]
            out  = model(clip)
            bvp_pred = out["bvp_pred"][0].cpu().numpy().astype(np.float32)  # [T]

            # Detrend + bandpass to remove DC/slow drift the model tends to output
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

    # SNR-weighted median: robust against outlier clips
    hrs  = np.array(per_clip_hrs)
    snrs = np.array(per_clip_snrs)
    # Median is more robust than weighted mean when one clip fails hard
    hr   = float(np.median(hrs))
    snr  = float(np.mean(snrs))
    return hr, snr, np.mean(bvp_list, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Methods 2 & 3 — POS and CHROM (classical baselines)
# ──────────────────────────────────────────────────────────────────────────────

def run_classical(
    roi_frames: List[np.ndarray],
    timestamps: Optional[np.ndarray],
    fps: float,
    method: str,
    hr_band_low: float,
    hr_band_high: float,
    snr_thresh: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Run one classical method. Returns (hr_bpm, snr_db)."""
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
    subjects = []
    for subdir in sorted(Path(root).iterdir()):
        if not subdir.is_dir():
            continue
        gt_path = subdir / "ground_truth.txt"
        if not gt_path.exists():
            deep = list(subdir.glob("*/ground_truth.txt"))
            if deep:
                gt_path = deep[0]
                subdir  = gt_path.parent
            else:
                continue
        video_path = None
        for ext in VIDEO_EXTS:
            hits = list(subdir.glob(f"*{ext}"))
            if hits:
                video_path = hits[0]
                break
        if video_path:
            subjects.append({"subject": subdir.name,
                              "video":   str(video_path),
                              "gt":      str(gt_path)})
    return subjects


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_subject(
    subject: str,
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
    row = {"subject": subject, "status": "ok"}

    # Ground truth
    try:
        gt = load_ubfc_gt(gt_path)
    except Exception as e:
        return {**row, "status": f"gt_error:{e}"}

    row["gt_mean_hr"]   = round(gt["mean_hr"],   2)
    row["gt_median_hr"] = round(gt["median_hr"], 2)
    row["gt_duration_s"]= round(gt["duration_s"],2)
    row["gt_fps"]       = round(gt["fps"],        2)

    # ROI extraction
    print(f"    Extracting ROIs…")
    try:
        roi_unmasked, roi_masked, timestamps, fps = extract_roi_frames(
            video_path, target_size=NORM_CROP, max_frames=max_frames
        )
    except Exception as e:
        return {**row, "status": f"roi_error:{e}"}

    row["n_roi_frames"]  = len(roi_unmasked)
    row["detected_fps"]  = round(fps, 2)

    if len(roi_unmasked) < 60:
        return {**row, "status": f"too_few_frames:{len(roi_unmasked)}"}

    # Motion filter — computed on the masked ROIs (better motion signal)
    motion   = frame_diff_motion(roi_masked)
    keep     = clean_frame_mask(motion, k=3.5)
    ts_arr   = np.asarray(timestamps, dtype=np.float64)
    # Apply the same keep mask to both variants so they stay aligned
    clean_masked,   clean_ts       = select_clean_frames(roi_masked,   ts_arr, keep)
    clean_unmasked, _              = select_clean_frames(roi_unmasked, ts_arr, keep)
    row["clean_frame_frac"] = round(float(keep.mean()), 3)
    fs_eff = float(len(clean_masked) / (clean_ts[-1] - clean_ts[0])) \
             if clean_ts is not None and len(clean_ts) > 1 and clean_ts[-1] > clean_ts[0] \
             else fps

    f_low_hz  = hr_band_low  / 60.0
    f_high_hz = hr_band_high / 60.0

    # ── Deep model ────────────────────────────────────────────────────────────
    print(f"    Running EquiPhysDANN…")
    t0 = time.time()
    deep_hr, deep_snr, deep_bvp = run_deep_model(
        roi_frames=roi_unmasked,   # UNMASKED — matches training preprocessing
        fps=fps,
        model=model, device=device,
        hr_band_low=f_low_hz, hr_band_high=f_high_hz,
        snr_thresh=snr_thresh, n_clips=n_clips,
    )
    deep_time = time.time() - t0
    row["deep_hr"]      = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"]  = round(deep_snr, 2) if deep_snr is not None else None
    row["deep_time_s"]  = round(deep_time, 1)

    # SQI and RR from deep BVP
    if deep_bvp is not None and len(deep_bvp) >= 60:
        row["deep_sqi"] = round(compute_signal_quality_index(deep_bvp, fps=fps), 2)
        rr = estimate_respiratory_rate(deep_bvp, fps=fps)
        row["deep_rr"]  = round(rr, 2) if rr > 0 else None
    else:
        row["deep_sqi"] = None
        row["deep_rr"]  = None

    # ── POS baseline ──────────────────────────────────────────────────────────
    print(f"    Running POS…")
    pos_hr, pos_snr = run_classical(
        clean_masked, clean_ts, fs_eff, "pos", f_low_hz, f_high_hz, snr_thresh
    )
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None

    # ── CHROM baseline ────────────────────────────────────────────────────────
    print(f"    Running CHROM…")
    chrom_hr, chrom_snr = run_classical(
        clean_masked, clean_ts, fs_eff, "chrom", f_low_hz, f_high_hz, snr_thresh
    )
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None

    # ── Error metrics for all three ───────────────────────────────────────────
    gt_hr = gt["mean_hr"]
    for tag, pred_hr in [("deep", deep_hr), ("pos", pos_hr), ("chrom", chrom_hr)]:
        if pred_hr is not None:
            row[f"{tag}_mae"]      = round(abs(pred_hr - gt_hr), 2)
            row[f"{tag}_rmse_sq"]  = round((pred_hr - gt_hr) ** 2, 4)
            row[f"{tag}_bias"]     = round(pred_hr - gt_hr, 2)
            row[f"{tag}_rel_err"]  = round(100 * abs(pred_hr - gt_hr) / gt_hr, 2)
        else:
            for suffix in ("mae", "rmse_sq", "bias", "rel_err"):
                row[f"{tag}_{suffix}"] = None

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
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
    snr_key = f"{tag}_snr_db"
    snrs    = np.array([r[snr_key] for r in valid if r.get(snr_key) is not None])
    return {
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
        description="Evaluate EquiPhysDANN + classical baselines on UBFC-rPPG"
    )
    p.add_argument("--root",        required=True,  help="UBFC dataset root")
    p.add_argument("--checkpoint",  default=None,   help="Path to equiphys_best.pt")
    p.add_argument("--out",         default="results/ubfc_results.csv")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hr-low",      type=float, default=48.0)
    p.add_argument("--hr-high",     type=float, default=150.0)
    p.add_argument("--snr-thresh",  type=float, default=0.12)
    p.add_argument("--max-frames",  type=int,   default=None)
    p.add_argument("--n-clips",     type=int,   default=3,
                   help="Number of overlapping clips to run the model on per video")
    p.add_argument("--subjects",    nargs="*",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Load model
    try:
        ckpt_path = _resolve_checkpoint(args.checkpoint)
        model     = load_model(ckpt_path, device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(args.root)
    if not subjects:
        print(f"No subjects found under {args.root}")
        sys.exit(1)
    if args.subjects:
        subjects = [s for s in subjects if s["subject"] in args.subjects]

    print(f"\nEvaluating {len(subjects)} subjects on device={args.device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"HR band    : {args.hr_low}–{args.hr_high} BPM")
    print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, s in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {s['subject']}")
        try:
            r = evaluate_subject(
                subject=s["subject"], video_path=s["video"], gt_path=s["gt"],
                model=model, device=device,
                hr_band_low=args.hr_low, hr_band_high=args.hr_high,
                snr_thresh=args.snr_thresh, max_frames=args.max_frames,
                n_clips=args.n_clips,
            )
        except Exception as e:
            r = {"subject": s["subject"], "status": f"crash:{e}"}
            traceback.print_exc()

        results.append(r)
        # Quick print
        print(f"  deep={r.get('deep_hr','—')} pos={r.get('pos_hr','—')} "
              f"chrom={r.get('chrom_hr','—')} gt={r.get('gt_mean_hr','—')} "
              f"MAE(deep)={r.get('deep_mae','—')} [{r.get('status','ok')}]")

    results.sort(key=lambda x: x["subject"])
    write_csv(results, str(out_path))

    # Summary per method
    summary_rows = [aggregate(results, tag) for tag in ("deep", "pos", "chrom")]
    summary_path = out_path.with_stem(out_path.stem + "_summary")
    write_csv(summary_rows, str(summary_path))

    # Print comparison table
    print("\n" + "="*70)
    print("UBFC  METHOD COMPARISON")
    print("="*70)
    header = f"{'Metric':<28} {'EquiPhysDANN':>14} {'POS':>10} {'CHROM':>10}"
    print(header)
    print("-"*len(header))
    metrics = [
        ("n_valid",          "Valid subjects"),
        ("mae_mean",         "MAE mean (BPM)"),
        ("mae_std",          "MAE std"),
        ("mae_median",       "MAE median"),
        ("rmse",             "RMSE (BPM)"),
        ("mean_bias",        "Bias (BPM)"),
        ("pct_within_5bpm",  "Within ±5 BPM %"),
        ("pct_within_10bpm", "Within ±10 BPM %"),
        ("mean_snr_db",      "Mean SNR (dB)"),
    ]
    by_method = {r["method"]: r for r in summary_rows}
    for key, label in metrics:
        deep_val  = by_method.get("deep",  {}).get(key, "—")
        pos_val   = by_method.get("pos",   {}).get(key, "—")
        chrom_val = by_method.get("chrom", {}).get(key, "—")
        print(f"{label:<28} {str(deep_val):>14} {str(pos_val):>10} {str(chrom_val):>10}")

    print(f"\nPer-subject CSV : {out_path}")
    print(f"Summary CSV     : {summary_path}")


if __name__ == "__main__":
    main()