"""eval_mcd.py — Evaluate EquiPhysDANNV2 + POS + CHROM on MCD (mcd_rppg).

MCD dataset structure
---------------------
<root>/
  video/        {ID}_{Camera}_{timing}.mp4   e.g. 1020_FullHDwebcam_after.mp4
  ppg_sync/     {ID}_{Camera}_{timing}       matching PPG text file (no ext)
  meta/         {ID}_{Camera}_{timing}       metadata text file (HR, fps, etc.)
  ppg/          {ID}_{timing}.PW             finger PPG
  ecg/          {ID}_{timing}.json           ECG JSON
  db            Excel file with SBP/DBP/Hb/HR per subject-session

Each (ID, Camera, timing) triple is one evaluation unit.
Camera types seen: FullHDwebcam, IriunWebcam, USBVideo
Timing: before / after

Usage:
  python eval_mcd.py --root "D:/Datasets/Datasets/mcd_rppg" ^
                     --checkpoint checkpoints_v2/equiphys_best.pt
  
  # Specific camera only:
  python eval_mcd.py --root "D:/Datasets/Datasets/mcd_rppg" ^
                     --checkpoint checkpoints_v2/equiphys_best.pt ^
                     --camera FullHDwebcam

  # Quick sanity run (10 clips):
  python eval_mcd.py --root "D:/Datasets/Datasets/mcd_rppg" ^
                     --checkpoint checkpoints_v2/equiphys_best.pt ^
                     --max-clips 10
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── path resolution ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _c in [_HERE, _HERE.parent]:
    if (_c / "equiphys_core.py").exists():
        sys.path.insert(0, str(_c))
        break

try:
    from equiphys_v2 import EquiPhysDANNV2 as _ModelCls
    _MODEL_TAG = "v2"
except ImportError:
    from equiphys_core import EquiPhysDANN as _ModelCls
    _MODEL_TAG = "v1"

from equiphys_core import (
    FaceROIExtractor,
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    compute_signal_quality_index,
    estimate_hr_bpm_from_bvp,
)

MODEL_FRAMES = 150
NORM_CROP    = 64
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_clips(root: str,
                   camera_filter: Optional[str] = None) -> List[Dict]:
    """Match video files to their ppg_sync ground-truth files.

    Naming convention:  {ID}_{Camera}_{timing}.mp4
    PPG sync file:      {ID}_{Camera}_{timing}   (no extension, in ppg_sync/)
    Meta file:          {ID}_{Camera}_{timing}   (no extension, in meta/)
    """
    root_path  = Path(root)
    video_dir  = root_path / "video"
    ppg_dir    = root_path / "ppg_sync"
    meta_dir   = root_path / "meta"

    if not video_dir.exists():
        raise FileNotFoundError(f"video/ folder not found at {video_dir}")

    clips = []
    for vf in sorted(video_dir.iterdir()):
        if vf.suffix.lower() not in VIDEO_EXTS:
            continue

        stem = vf.stem   # e.g. 1020_FullHDwebcam_after

        # Parse:  {ID}_{Camera}_{timing}
        # Camera names contain no underscores in practice, but timing is last part
        # Split on the last _ to get timing, then the rest
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        id_cam, timing = parts          # "1020_FullHDwebcam", "after"
        parts2 = id_cam.split("_", 1)
        if len(parts2) != 2:
            continue
        subject_id, camera = parts2     # "1020", "FullHDwebcam"

        if camera_filter and camera_filter.lower() not in camera.lower():
            continue

        ppg_path  = ppg_dir / stem if ppg_dir.exists() else None
        meta_path = meta_dir / stem if meta_dir.exists() else None

        # ppg_sync files have no extension
        if ppg_path and not ppg_path.exists():
            ppg_path = None
        if meta_path and not meta_path.exists():
            meta_path = None

        clips.append({
            "subject":    subject_id,
            "camera":     camera,
            "timing":     timing,
            "stem":       stem,
            "video_path": str(vf),
            "ppg_path":   str(ppg_path) if ppg_path else None,
            "meta_path":  str(meta_path) if meta_path else None,
        })

    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ppg_sync(ppg_path: str) -> Optional[Dict]:
    """Load PPG sync text file.

    MCD ppg_sync files are typically space/newline separated float values
    representing the PPG waveform sampled at a fixed rate.
    """
    try:
        text = Path(ppg_path).read_text(encoding="utf-8", errors="ignore").strip()
        # Try newline-separated first, then space-separated
        values = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line.split()[0]))
            except (ValueError, IndexError):
                continue
        if not values:
            # Try space-separated single line
            values = [float(v) for v in text.split() if v.strip()]
        if not values:
            return None
        bvp = np.array(values, dtype=np.float64)
        return {"bvp": bvp, "n_samples": len(bvp)}
    except Exception as e:
        return None


def load_meta(meta_path: str) -> Dict:
    """Load MCD meta text file.

    Meta files contain HR, fps, SBP, DBP, SpO2 etc. as key=value pairs
    or as tab/space separated columns.
    """
    result = {}
    try:
        text = Path(meta_path).read_text(encoding="utf-8", errors="ignore")
        # Try key=value or key: value format
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sep in ["=", ":", "\t"]:
                if sep in line:
                    parts = line.split(sep, 1)
                    key = parts[0].strip().lower().replace(" ", "_")
                    val = parts[1].strip()
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
                    break
    except Exception:
        pass
    return result


def load_db_excel(root: str) -> Optional[Dict[str, Dict]]:
    """Load the MCD db Excel file for clinical GT (SBP, DBP, Hb, HR).

    Returns dict keyed by '{ID}_{timing}' -> {sbp, dbp, hr, hb, ...}
    """
    try:
        import openpyxl
    except ImportError:
        try:
            import pandas as pd
        except ImportError:
            return None

    db_path = Path(root) / "db"
    if not db_path.exists():
        # Try with .xlsx extension
        for ext in [".xlsx", ".xls", ".csv"]:
            p = Path(root) / f"db{ext}"
            if p.exists():
                db_path = p
                break
        else:
            return None

    try:
        import pandas as pd
        if db_path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(db_path)
        else:
            df = pd.read_csv(db_path)

        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        result = {}
        for _, row in df.iterrows():
            # Try to find subject ID column
            subj_id = None
            for col in ["id", "subject", "subject_id", "participant"]:
                if col in df.columns:
                    subj_id = str(row[col]).strip()
                    break
            if subj_id is None:
                continue
            timing = str(row.get("timing", row.get("session", ""))).strip().lower()
            key = f"{subj_id}_{timing}" if timing else subj_id

            entry = {}
            for src, dst in [
                (["sbp", "systolic", "sys_bp"], "sbp"),
                (["dbp", "diastolic", "dia_bp"], "dbp"),
                (["hr", "heart_rate", "bpm", "gt_hr"], "hr"),
                (["hb", "haemoglobin", "hemoglobin", "hgb"], "hb"),
                (["spo2", "spo2_%"], "spo2"),
            ]:
                for s in src:
                    if s in df.columns:
                        try:
                            entry[dst] = float(row[s])
                        except (ValueError, TypeError):
                            pass
                        break
            result[key] = entry
        return result if result else None
    except Exception as e:
        print(f"  [db] could not load Excel: {e}")
        return None


def derive_hr_from_bvp(bvp: np.ndarray, fps: float,
                        f_low: float = 0.7, f_high: float = 3.0) -> Optional[float]:
    """Derive mean HR from a BVP waveform via Welch PSD."""
    peak_hz, _ = _welch_peak_hz(bvp, fps, f_low=f_low, f_high=f_high,
                                 snr_ratio_threshold=0.0)
    return peak_hz * 60.0 if peak_hz > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    model = _ModelCls(in_channels=3, latent_dim=256,
                      frames=MODEL_FRAMES, lambda_grl=1.0).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded ({_MODEL_TAG}): {ckpt_path}")
    return model


def _resolve_checkpoint(explicit: Optional[str]) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    root = _HERE.parent
    for folder in ["checkpoints_v2", "checkpoints_v4", "checkpoints_v3",
                   "checkpoints_gpu_mcd", "checkpoints_gpu", "checkpoints"]:
        p = root / folder / "equiphys_best.pt"
        if p.exists():
            return str(p)
    raise FileNotFoundError("No checkpoint found. Pass --checkpoint.")


# ─────────────────────────────────────────────────────────────────────────────
# ROI extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi_frames(
    video_path: str,
    target_size: int = NORM_CROP,
    max_frames: Optional[int] = None,
    progress_every: int = 200,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], float]:
    roi_extractor = FaceROIExtractor(target_size=target_size)
    cap   = cv2.VideoCapture(video_path)
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames) if max_frames else total

    roi_unmasked, roi_masked, timestamps = [], [], []
    idx = 0; face_missed = 0

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
        print(f"      face missed {face_missed}/{idx} frames")
    return roi_unmasked, roi_masked, timestamps, fps


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_deep_model(roi_frames, fps, model, device,
                   f_low, f_high, snr_thresh, n_clips=3):
    n = len(roi_frames)
    if n < MODEL_FRAMES:
        return None, None, None

    from scipy.signal import butter, filtfilt, detrend as sp_detrend

    def _clean(sig, fs, lo, hi):
        sig = sp_detrend(sig.astype(np.float64))
        nyq = 0.5 * fs
        lo_n = max(0.01, min(0.99, lo / nyq))
        hi_n = max(lo_n + 0.001, min(0.99, hi / nyq))
        try:
            b, a = butter(3, [lo_n, hi_n], btype="band")
            return filtfilt(b, a, sig)
        except Exception:
            return sig

    step   = max(1, (n - MODEL_FRAMES) // max(n_clips - 1, 1))
    starts = list(range(0, n - MODEL_FRAMES + 1, step))[:n_clips]
    hrs, snrs, bvps = [], [], []

    with torch.no_grad():
        for s in starts:
            seg  = roi_frames[s : s + MODEL_FRAMES]
            buf  = deque(seg, maxlen=MODEL_FRAMES)
            clip = build_clip_tensor_from_buffer(buf, device=device)
            out  = model(clip)
            bvp_raw = out["bvp_pred"][0].cpu().numpy().astype(np.float32)
            bvp_c   = _clean(bvp_raw, fps, f_low, f_high)
            bvps.append(bvp_c)
            peak_hz, snr = _welch_peak_hz(bvp_c, fps, f_low=f_low, f_high=f_high,
                                          snr_ratio_threshold=snr_thresh)
            if peak_hz > 0:
                hrs.append(peak_hz * 60.0)
                snrs.append(float(snr) if np.isfinite(snr) else 0.0)

    if not hrs:
        return None, None, np.mean(bvps, axis=0) if bvps else None
    return float(np.median(hrs)), float(np.mean(snrs)), np.mean(bvps, axis=0)


def run_classical(roi_frames, fps, method, f_low, f_high, snr_thresh):
    try:
        pulse, fs_used = _extract_patch_weighted_pulse(
            roi_frames, timestamps=None, target_fs=fps, method=method)
        peak_hz, snr = _welch_peak_hz(
            pulse, fs_used, f_low=f_low, f_high=f_high,
            snr_ratio_threshold=snr_thresh)
        hr = peak_hz * 60.0 if peak_hz > 0 else None
        return hr, float(snr) if np.isfinite(snr) else None
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clip(clip: Dict, model, device,
                  f_low, f_high, snr_thresh,
                  n_clips, db_lookup: Optional[Dict],
                  max_video_frames: Optional[int]) -> Dict:
    row = {
        "subject":  clip["subject"],
        "camera":   clip["camera"],
        "timing":   clip["timing"],
        "stem":     clip["stem"],
        "status":   "ok",
    }

    # ── Ground truth HR from ppg_sync ────────────────────────────────────────
    gt_hr   = None
    gt_bvp  = None
    gt_fps  = 30.0

    if clip["ppg_path"]:
        ppg_data = load_ppg_sync(clip["ppg_path"])
        if ppg_data:
            gt_bvp = ppg_data["bvp"]
            # fps from meta if available
            if clip["meta_path"]:
                meta = load_meta(clip["meta_path"])
                gt_fps = float(meta.get("fps", meta.get("frame_rate", 30.0)))
            gt_hr = derive_hr_from_bvp(gt_bvp, gt_fps)

    # Fallback: DB Excel lookup
    if gt_hr is None and db_lookup:
        key1 = f"{clip['subject']}_{clip['timing']}"
        key2 = clip["subject"]
        entry = db_lookup.get(key1) or db_lookup.get(key2, {})
        if "hr" in entry:
            gt_hr = float(entry["hr"])
        row["gt_sbp"] = entry.get("sbp")
        row["gt_dbp"] = entry.get("dbp")
        row["gt_hb"]  = entry.get("hb")

    row["gt_hr"]  = round(gt_hr, 2) if gt_hr is not None else None
    row["gt_fps"] = gt_fps

    if row["gt_hr"] is None:
        return {**row, "status": "no_gt_hr"}

    # ── ROI extraction ────────────────────────────────────────────────────────
    print(f"    Extracting ROIs from video…")
    try:
        roi_unmasked, roi_masked, timestamps, fps = extract_roi_frames(
            clip["video_path"], target_size=NORM_CROP,
            max_frames=max_video_frames)
    except Exception as e:
        return {**row, "status": f"roi_error:{e}"}

    row["n_roi_frames"] = len(roi_unmasked)
    row["detected_fps"] = round(fps, 2)

    if len(roi_unmasked) < 60:
        return {**row, "status": f"too_short:{len(roi_unmasked)}"}

    # ── Deep model ────────────────────────────────────────────────────────────
    print(f"    Running deep model…")
    deep_hr, deep_snr, _ = run_deep_model(
        roi_unmasked, fps, model, device, f_low, f_high, snr_thresh, n_clips)
    row["deep_hr"]     = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"] = round(deep_snr, 2) if deep_snr is not None else None

    # ── POS ───────────────────────────────────────────────────────────────────
    print(f"    Running POS…")
    pos_hr, pos_snr = run_classical(
        roi_masked, fps, "pos", f_low, f_high, snr_thresh)
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None

    # ── CHROM ─────────────────────────────────────────────────────────────────
    print(f"    Running CHROM…")
    chrom_hr, chrom_snr = run_classical(
        roi_masked, fps, "chrom", f_low, f_high, snr_thresh)
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None

    # ── Error metrics ─────────────────────────────────────────────────────────
    gt = row["gt_hr"]
    for tag, pred in [("deep", deep_hr), ("pos", pos_hr), ("chrom", chrom_hr)]:
        if pred is not None:
            row[f"{tag}_mae"]     = round(abs(pred - gt), 2)
            row[f"{tag}_rmse_sq"] = round((pred - gt) ** 2, 4)
            row[f"{tag}_bias"]    = round(pred - gt, 2)
        else:
            row[f"{tag}_mae"] = row[f"{tag}_rmse_sq"] = row[f"{tag}_bias"] = None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows, tag, label=""):
    valid = [r for r in rows if r.get(f"{tag}_mae") is not None]
    n = len(valid)
    if n == 0:
        return {"method": tag, "group": label, "n_valid": 0}
    maes    = np.array([r[f"{tag}_mae"]     for r in valid])
    rmse_sq = np.array([r[f"{tag}_rmse_sq"] for r in valid])
    bias    = np.array([r[f"{tag}_bias"]    for r in valid])
    snrs    = np.array([r.get(f"{tag}_snr_db") for r in valid
                        if r.get(f"{tag}_snr_db") is not None])
    return {
        "method":           tag,
        "group":            label,
        "n_clips":          len(rows),
        "n_valid":          n,
        "n_failed":         len(rows) - n,
        "mae_mean":         round(float(maes.mean()), 3),
        "mae_std":          round(float(maes.std()),  3),
        "mae_median":       round(float(np.median(maes)), 3),
        "rmse":             round(float(np.sqrt(rmse_sq.mean())), 3),
        "mean_bias":        round(float(bias.mean()), 3),
        "pct_within_5bpm":  round(float(np.mean(maes <= 5.0)  * 100), 1),
        "pct_within_10bpm": round(float(np.mean(maes <= 10.0) * 100), 1),
        "mean_snr_db":      round(float(snrs.mean()), 2) if len(snrs) else None,
    }


def aggregate_by_group(rows, tag, key):
    out = []
    for g in sorted({r.get(key) for r in rows if r.get(key)}):
        subset = [r for r in rows if r.get(key) == g]
        out.append(aggregate(subset, tag, label=f"{key}={g}"))
    return out


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on MCD (mcd_rppg)")
    p.add_argument("--root",       required=True,
                   help="MCD root folder (contains video/, ppg_sync/, meta/, db)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out",        default="results/mcd_results.csv")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hr-low",     type=float, default=48.0)
    p.add_argument("--hr-high",    type=float, default=150.0)
    p.add_argument("--snr-thresh", type=float, default=0.12)
    p.add_argument("--n-clips",    type=int,   default=3)
    p.add_argument("--camera",     default=None,
                   help="Filter by camera name, e.g. FullHDwebcam, IriunWebcam, USBVideo")
    p.add_argument("--max-clips",  type=int,   default=None)
    p.add_argument("--max-frames", type=int,   default=None,
                   help="Max video frames per clip (useful for speed testing)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    try:
        ckpt_path = _resolve_checkpoint(args.checkpoint)
        model     = load_model(ckpt_path, device)
    except Exception as e:
        print(f"ERROR loading model: {e}"); sys.exit(1)

    try:
        clips = discover_clips(args.root, camera_filter=args.camera)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    if not clips:
        print(f"No clips found. Check that {args.root}/video/ exists and "
              f"contains .mp4 files named {{ID}}_{{Camera}}_{{timing}}.mp4")
        sys.exit(1)

    if args.max_clips:
        clips = clips[:args.max_clips]

    # Load Excel DB for clinical GT
    db_lookup = load_db_excel(args.root)
    if db_lookup:
        print(f"  DB loaded: {len(db_lookup)} entries")
    else:
        print("  DB not loaded (no Excel file found or openpyxl/pandas missing)")
        print("  Install with:  pip install openpyxl pandas")

    f_low  = args.hr_low  / 60.0
    f_high = args.hr_high / 60.0

    print(f"\nMCD: {len(clips)} clips  device={args.device}  model={_MODEL_TAG}")
    if args.camera:
        print(f"Camera filter: {args.camera}")
    print(f"HR band: {args.hr_low}–{args.hr_high} BPM\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, c in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {c['stem']}", flush=True)
        try:
            r = evaluate_clip(
                clip=c, model=model, device=device,
                f_low=f_low, f_high=f_high,
                snr_thresh=args.snr_thresh,
                n_clips=args.n_clips,
                db_lookup=db_lookup,
                max_video_frames=args.max_frames,
            )
        except Exception as e:
            r = {"subject": c["subject"], "camera": c["camera"],
                 "timing": c["timing"], "stem": c["stem"],
                 "status": f"crash:{e}"}
            traceback.print_exc()
        results.append(r)
        print(f"  deep={r.get('deep_hr','—')} pos={r.get('pos_hr','—')} "
              f"chrom={r.get('chrom_hr','—')} gt={r.get('gt_hr','—')} "
              f"MAE(pos)={r.get('pos_mae','—')} [{r.get('status','ok')}]",
              flush=True)

    results.sort(key=lambda x: (x.get("subject",""), x.get("camera",""), x.get("timing","")))
    write_csv(results, str(out_path))

    # Overall summary
    summary = [aggregate(results, t) for t in ("deep", "pos", "chrom")]
    write_csv(summary, str(out_path.with_stem(out_path.stem + "_summary")))

    # Breakdown by camera and timing
    for grp in ("camera", "timing"):
        grp_rows = sum([aggregate_by_group(results, t, grp)
                        for t in ("deep","pos","chrom")], [])
        write_csv(grp_rows,
                  str(out_path.with_stem(f"{out_path.stem}_by_{grp}")))

    # Console table
    by_m = {r["method"]: r for r in summary}
    print("\n" + "=" * 65)
    print(f"MCD RESULTS  ({len(results)} clips"
          + (f", camera={args.camera}" if args.camera else "") + ")")
    print("=" * 65)
    hdr = f"{'Metric':<28} {'Deep':>10} {'POS':>10} {'CHROM':>10}"
    print(hdr); print("-" * len(hdr))
    for key, lbl in [
        ("n_valid",          "Valid clips"),
        ("mae_mean",         "MAE mean (BPM)"),
        ("mae_median",       "MAE median"),
        ("rmse",             "RMSE (BPM)"),
        ("mean_bias",        "Bias (BPM)"),
        ("pct_within_5bpm",  "Within ±5 BPM %"),
        ("pct_within_10bpm", "Within ±10 BPM %"),
    ]:
        print(f"{lbl:<28} "
              f"{str(by_m.get('deep', {}).get(key,'—')):>10} "
              f"{str(by_m.get('pos',  {}).get(key,'—')):>10} "
              f"{str(by_m.get('chrom',{}).get(key,'—')):>10}")

    print(f"\nCSVs in: {out_path.parent}/")


if __name__ == "__main__":
    main()