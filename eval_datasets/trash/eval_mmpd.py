"""eval_mmpd.py — Evaluate EquiPhysDANN v2 + POS + CHROM on MMPD.

MMPD stores data as .mat files (MATLAB format). Each file contains:
  video       : (T, H, W, 3) uint8 face video
  gt_HR       : scalar ground-truth HR in BPM
  gt_bvp      : (T,) ground-truth BVP waveform
  skin_color  : Fitzpatrick scale 1–6
  light       : lighting condition 0–3
  fps         : frame rate (30.0)

Folder structure:
  <root>/
    subject1/
      subject1/
        p1_0.mat  ... p1_19.mat   (20 clips per subject)
    subject2/
      subject2/
        p2_0.mat  ... p2_19.mat
    ...

Usage:
  python eval_mmpd.py --root "D:/Datasets/Datasets/MMPD" --checkpoint checkpoints_v2/equiphys_best.pt
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

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _c in [_HERE, _HERE.parent]:
    if (_c / "equiphys_core.py").exists():
        sys.path.insert(0, str(_c))
        break

try:
    from scipy.io import loadmat as _loadmat
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

try:
    from equiphys_v2 import EquiPhysDANNV2 as _ModelCls
    _MODEL_TAG = "v2"
except ImportError:
    from equiphys_core import EquiPhysDANN as _ModelCls
    _MODEL_TAG = "v1"

from equiphys_core import (
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    compute_signal_quality_index,
    estimate_hr_bpm_from_bvp,
    estimate_respiratory_rate,
)

# ─────────────────────────────────────────────────────────────────────────────
MODEL_FRAMES = 150
NORM_CROP    = 64
LIGHT_NAMES  = {0: "LED-low", 1: "LED-high", 2: "incandescent", 3: "natural"}
SKIN_NAMES   = {1:"TypeI", 2:"TypeII", 3:"TypeIII", 4:"TypeIV", 5:"TypeV", 6:"TypeVI"}


# ─────────────────────────────────────────────────────────────────────────────
# .mat loading
# ─────────────────────────────────────────────────────────────────────────────

# String→int maps for MMPD fields stored as text
_LIGHT_STR2INT = {
    "led-low": 0, "led_low": 0, "low": 0,
    "led-high": 1, "led_high": 1, "high": 1,
    "incandescent": 2, "incan": 2,
    "natural": 3, "sunlight": 3, "daylight": 3,
}
_SKIN_STR2INT = {
    "type i": 1, "type_i": 1, "i": 1, "1": 1,
    "type ii": 2, "type_ii": 2, "ii": 2, "2": 2,
    "type iii": 3, "type_iii": 3, "iii": 3, "3": 3,
    "type iv": 4, "type_iv": 4, "iv": 4, "4": 4,
    "type v": 5, "type_v": 5, "v": 5, "5": 5,
    "type vi": 6, "type_vi": 6, "vi": 6, "6": 6,
}

def _scalar(v, str_map: dict = None):
    """Flatten any numpy scalar / 1-element array to a Python float.
    If the value is a string and str_map is provided, maps it to an int.
    """
    if v is None:
        return None
    arr = np.asarray(v).flatten()
    if arr.size == 0:
        return None
    val = arr[0]
    # Handle string values
    if isinstance(val, (str, np.str_)):
        s = str(val).strip().lower()
        if str_map is not None:
            return float(str_map.get(s, -1))
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_mat(path: str) -> Optional[Dict]:
    """Load one MMPD .mat clip. Returns None on failure."""
    try:
        mat = _loadmat(path, simplify_cells=True)
    except Exception:
        # older scipy: simplify_cells not supported
        try:
            mat = _loadmat(path)
        except Exception as e:
            print(f"      [mat] cannot load {Path(path).name}: {e}")
            return None

    # video
    # Can't use `or` with numpy arrays — check keys explicitly
    video = None
    for _vkey in ("video", "Video", "img", "frames", "face_video"):
        _v = mat.get(_vkey)
        if _v is not None and hasattr(_v, "shape"):
            video = _v
            break
    if video is None:
        return None
    video = np.asarray(video)
    if video.ndim == 3:                          # (T, H, W) grayscale → expand
        video = np.stack([video]*3, axis=-1)
    if video.ndim != 4 or video.shape[-1] != 3:
        return None
    video = video.astype(np.uint8)

    # ground truth
    _hr_raw = mat.get("gt_HR") or mat.get("HR") or mat.get("bpm") or mat.get("hr")
    gt_hr  = _scalar(_hr_raw)
    gt_bvp = None
    for _bkey in ("gt_bvp", "bvp", "BVP", "ppg", "PPG"):
        _b = mat.get(_bkey)
        if _b is not None and hasattr(np.asarray(_b).flatten(), "__len__"):
            gt_bvp = _b
            break
    gt_bvp = np.asarray(gt_bvp).flatten().astype(np.float64) if gt_bvp is not None else None
    fps    = _scalar(mat.get("fps") or mat.get("Fs")) or 30.0

    _skin_raw = mat.get("skin_color") or mat.get("skin") or mat.get("Skin_color")
    skin   = _scalar(_skin_raw, str_map=_SKIN_STR2INT)
    _light_raw = mat.get("light") or mat.get("lighting") or mat.get("Light")
    light  = _scalar(_light_raw, str_map=_LIGHT_STR2INT)
    motion = _scalar(mat.get("motion"))
    exercise = _scalar(mat.get("exercise"))

    return {
        "video":    video,        # (T, H, W, 3)
        "gt_hr":    gt_hr,
        "gt_bvp":   gt_bvp,
        "fps":      fps,
        "skin":     int(skin)   if skin   is not None else None,
        "light":    int(light)  if light  is not None else None,
        "motion":   int(motion) if motion is not None else None,
        "exercise": int(exercise) if exercise is not None else None,
        "n_frames": video.shape[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery — handles the double-nested subject1/subject1/ layout
# ─────────────────────────────────────────────────────────────────────────────

def discover_clips(root: str) -> List[Dict]:
    """Return a list of clip dicts with keys: subject, clip_id, mat_path."""
    clips = []
    root_path = Path(root)

    for outer in sorted(root_path.iterdir()):
        if not outer.is_dir():
            continue

        # Find the actual folder that contains .mat files
        # Could be outer/ directly or outer/outer/ (double-nested)
        mat_dirs = []
        if any(outer.glob("*.mat")):
            mat_dirs.append(outer)
        for inner in outer.iterdir():
            if inner.is_dir() and any(inner.glob("*.mat")):
                mat_dirs.append(inner)

        for mat_dir in mat_dirs:
            for mat_file in sorted(mat_dir.glob("*.mat")):
                clips.append({
                    "subject":  outer.name,
                    "clip_id":  mat_file.stem,
                    "mat_path": str(mat_file),
                })

    return clips


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
    raise FileNotFoundError("No checkpoint. Pass --checkpoint path/to/equiphys_best.pt")


# ─────────────────────────────────────────────────────────────────────────────
# ROI extraction from mat video
# ─────────────────────────────────────────────────────────────────────────────

def extract_roi_from_video(video_uint8: np.ndarray,
                           target_size: int = NORM_CROP) -> List[np.ndarray]:
    """Resize every frame to target_size × target_size float32 [0,1].

    MMPD videos are already face-cropped, so no face detection needed.
    """
    import cv2
    rois = []
    for frame in video_uint8:
        resized = cv2.resize(frame, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)
        rois.append(resized.astype(np.float32) / 255.0)
    return rois


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_deep_model(
    roi_frames: List[np.ndarray],
    fps: float,
    model,
    device: torch.device,
    f_low: float,
    f_high: float,
    snr_thresh: float,
    n_clips: int = 3,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
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
            peak_hz, snr = _welch_peak_hz(
                bvp_c, fps, f_low=f_low, f_high=f_high,
                snr_ratio_threshold=snr_thresh)
            if peak_hz > 0:
                hrs.append(peak_hz * 60.0)
                snrs.append(float(snr) if np.isfinite(snr) else 0.0)

    if not hrs:
        return None, None, np.mean(bvps, axis=0) if bvps else None
    return float(np.median(hrs)), float(np.mean(snrs)), np.mean(bvps, axis=0)


def run_classical(
    roi_frames: List[np.ndarray],
    fps: float,
    method: str,
    f_low: float,
    f_high: float,
    snr_thresh: float,
) -> Tuple[Optional[float], Optional[float]]:
    try:
        # classical methods expect uint8-scaled [0,255] frames at 128×128
        import cv2
        scaled = [cv2.resize((f * 255).astype(np.float32),
                             (128, 128), interpolation=cv2.INTER_AREA)
                  for f in roi_frames]
        pulse, fs_used = _extract_patch_weighted_pulse(
            scaled, timestamps=None, target_fs=fps, method=method)
        peak_hz, snr = _welch_peak_hz(
            pulse, fs_used, f_low=f_low, f_high=f_high,
            snr_ratio_threshold=snr_thresh)
        hr = peak_hz * 60.0 if peak_hz > 0 else None
        return hr, float(snr) if np.isfinite(snr) else None
    except Exception as e:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clip(
    subject: str,
    clip_id: str,
    mat_path: str,
    model,
    device: torch.device,
    f_low: float,
    f_high: float,
    snr_thresh: float,
    n_clips: int,
) -> Dict:
    row = {"subject": subject, "clip_id": clip_id, "status": "ok"}

    data = load_mat(mat_path)
    if data is None:
        return {**row, "status": "mat_load_failed"}

    row["gt_hr"]   = round(data["gt_hr"], 2) if data["gt_hr"] is not None else None
    row["fps"]     = data["fps"]
    row["n_frames"]= data["n_frames"]
    row["skin"]    = data["skin"]
    row["light"]   = data["light"]
    row["motion"]  = data["motion"]
    row["exercise"]= data["exercise"]
    row["skin_name"]  = SKIN_NAMES.get(data["skin"],  "?") if data["skin"]  else None
    row["light_name"] = LIGHT_NAMES.get(data["light"], "?") if data["light"] else None

    if row["gt_hr"] is None:
        return {**row, "status": "no_gt_hr"}

    # ROI frames (MMPD is already face-cropped)
    roi_frames = extract_roi_from_video(data["video"], target_size=NORM_CROP)
    row["n_roi_frames"] = len(roi_frames)

    if len(roi_frames) < 60:
        return {**row, "status": f"too_short:{len(roi_frames)}"}

    fps = data["fps"]

    # Deep model
    deep_hr, deep_snr, _ = run_deep_model(
        roi_frames, fps, model, device, f_low, f_high, snr_thresh, n_clips)
    row["deep_hr"]     = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"] = round(deep_snr, 2) if deep_snr is not None else None

    # POS
    pos_hr, pos_snr = run_classical(roi_frames, fps, "pos", f_low, f_high, snr_thresh)
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None

    # CHROM
    chrom_hr, chrom_snr = run_classical(roi_frames, fps, "chrom", f_low, f_high, snr_thresh)
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None

    # Error metrics
    gt = row["gt_hr"]
    for tag, pred in [("deep", deep_hr), ("pos", pos_hr), ("chrom", chrom_hr)]:
        if pred is not None and gt is not None:
            row[f"{tag}_mae"]     = round(abs(pred - gt), 2)
            row[f"{tag}_rmse_sq"] = round((pred - gt) ** 2, 4)
            row[f"{tag}_bias"]    = round(pred - gt, 2)
        else:
            row[f"{tag}_mae"] = row[f"{tag}_rmse_sq"] = row[f"{tag}_bias"] = None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows: List[Dict], tag: str, label: str = "") -> Dict:
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


def aggregate_by_group(rows, tag, key, names) -> List[Dict]:
    out = []
    for g in sorted({r.get(key) for r in rows if r.get(key) is not None}):
        subset = [r for r in rows if r.get(key) == g]
        label  = f"{names.get(g, str(g))} ({key}={g})"
        out.append(aggregate(subset, tag, label=label))
    return out


def write_csv(rows: List[Dict], path: str):
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
    p = argparse.ArgumentParser(description="Evaluate on MMPD (.mat format)")
    p.add_argument("--root",       required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out",        default="results/mmpd_results.csv")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hr-low",     type=float, default=48.0)
    p.add_argument("--hr-high",    type=float, default=150.0)
    p.add_argument("--snr-thresh", type=float, default=0.12)
    p.add_argument("--n-clips",    type=int,   default=3)
    p.add_argument("--subjects",   nargs="*",  default=None)
    p.add_argument("--max-clips",  type=int,   default=None,
                   help="Limit total clips (useful for a quick sanity run)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    try:
        ckpt_path = _resolve_checkpoint(args.checkpoint)
        model     = load_model(ckpt_path, device)
    except Exception as e:
        print(f"ERROR loading model: {e}"); sys.exit(1)

    clips = discover_clips(args.root)
    if not clips:
        print(f"No .mat clips found under {args.root}")
        print("Expected structure: <root>/subjectN/subjectN/*.mat")
        sys.exit(1)

    if args.subjects:
        clips = [c for c in clips if c["subject"] in args.subjects]
    if args.max_clips:
        clips = clips[:args.max_clips]

    f_low  = args.hr_low  / 60.0
    f_high = args.hr_high / 60.0

    print(f"\nMMPD: {len(clips)} clips  device={args.device}  model={_MODEL_TAG}")
    print(f"HR band: {args.hr_low}–{args.hr_high} BPM\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, c in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {c['subject']}/{c['clip_id']}", flush=True)
        try:
            r = evaluate_clip(
                subject=c["subject"], clip_id=c["clip_id"],
                mat_path=c["mat_path"], model=model, device=device,
                f_low=f_low, f_high=f_high,
                snr_thresh=args.snr_thresh, n_clips=args.n_clips,
            )
        except Exception as e:
            r = {"subject": c["subject"], "clip_id": c["clip_id"],
                 "status": f"crash:{e}"}
            traceback.print_exc()
        results.append(r)
        print(f"  deep={r.get('deep_hr','—')} pos={r.get('pos_hr','—')} "
              f"chrom={r.get('chrom_hr','—')} gt={r.get('gt_hr','—')} "
              f"skin={r.get('skin_name','?')} "
              f"MAE(pos)={r.get('pos_mae','—')} [{r.get('status','ok')}]",
              flush=True)

    write_csv(results, str(out_path))

    # Summaries
    summary = [aggregate(results, t) for t in ("deep", "pos", "chrom")]
    write_csv(summary, str(out_path.with_stem(out_path.stem + "_summary")))

    skin_rows  = sum([aggregate_by_group(results, t, "skin",  SKIN_NAMES)
                      for t in ("deep","pos","chrom")], [])
    light_rows = sum([aggregate_by_group(results, t, "light", LIGHT_NAMES)
                      for t in ("deep","pos","chrom")], [])
    write_csv(skin_rows,  str(out_path.with_stem(out_path.stem + "_by_skin")))
    write_csv(light_rows, str(out_path.with_stem(out_path.stem + "_by_lighting")))

    # Console table
    print("\n" + "=" * 65)
    print(f"MMPD RESULTS ({len(results)} clips)")
    print("=" * 65)
    by_m = {r["method"]: r for r in summary}
    hdr  = f"{'Metric':<28} {'Deep':>10} {'POS':>10} {'CHROM':>10}"
    print(hdr); print("-" * len(hdr))
    for key, lbl in [
        ("n_valid",         "Valid clips"),
        ("mae_mean",        "MAE mean (BPM)"),
        ("mae_median",      "MAE median"),
        ("rmse",            "RMSE (BPM)"),
        ("mean_bias",       "Bias (BPM)"),
        ("pct_within_5bpm", "Within ±5 BPM %"),
        ("pct_within_10bpm","Within ±10 BPM %"),
        ("mean_snr_db",     "Mean SNR (dB)"),
    ]:
        print(f"{lbl:<28} "
              f"{str(by_m.get('deep', {}).get(key,'—')):>10} "
              f"{str(by_m.get('pos',  {}).get(key,'—')):>10} "
              f"{str(by_m.get('chrom',{}).get(key,'—')):>10}")

    # Skin breakdown (POS, best classical)
    pos_skin = [r for r in skin_rows if r["method"] == "pos" and r.get("n_valid",0) > 0]
    if pos_skin:
        print("\nPOS MAE by Fitzpatrick skin type:")
        for r in pos_skin:
            print(f"  {r['group']:<40} n={r['n_valid']:>3}  MAE={r.get('mae_mean','—')}")

    print(f"\nCSVs in: {out_path.parent}/")


if __name__ == "__main__":
    main()