"""eval_npz.py — Unified evaluation on preprocessed NPZ clips.

Evaluates EquiPhysDANNV2 (deep), POS, CHROM, and the SNR-weighted fusion
directly on the preprocessed `.npz` clips produced by
`preprocess_raw_datasets.py`.

Why NPZ instead of raw video
----------------------------
The NPZ clips are already ROI-extracted, resized to 64x64, and have their
ground-truth BVP synchronised. This removes every source of failure that
plagued the raw-video evaluators (dataset folder layouts, .mat parsing,
face-detection misses, video codec issues) and makes evaluation ~50x faster.

Critically, ground-truth HR is derived from the clip's `bvp` field using the
SAME `estimate_hr_bpm_from_bvp()` call that `train_equiphys.py` used to compute
its validation metrics — so numbers here are directly comparable to the
training logs.

Expected NPZ schema (per clip)
------------------------------
    video       [T, H, W, 3] float32 in [0, 1]
    roi_mask    [H, W]       float32 in [0, 1]
    bvp         [T]          float32   (pulse / HR reference)
    skin_label  scalar int   0-5  (Fitzpatrick, MMPD only)
    light_label scalar int   0-3  (lighting condition, MMPD only)
    clinical    [3] float32  [SBP, DBP, Hb]  (MCD only, optional)

Usage
-----
    # Single dataset
    python eval_npz.py --npz-root processed_npz/ubfc --dataset ubfc \
        --checkpoint checkpoints_v2/equiphys_best.pt

    # All four in one run (writes one CSV set per dataset)
    python eval_npz.py --npz-root processed_npz --all \
        --checkpoint checkpoints_v2/equiphys_best.pt

    # Quick sanity check
    python eval_npz.py --npz-root processed_npz/ubfc --dataset ubfc \
        --checkpoint checkpoints_v2/equiphys_best.pt --max-clips 20
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

# ── locate equiphys_core / equiphys_v2 ───────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _c in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_c / "equiphys_core.py").exists():
        sys.path.insert(0, str(_c))
        break

try:
    from equiphys_v2 import EquiPhysDANNV2 as _ModelCls
    _MODEL_TAG = "v2"
except ImportError:
    from equiphys_core import EquiPhysDANN as _ModelCls  # type: ignore
    _MODEL_TAG = "v1"

from equiphys_core import (
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    compute_signal_quality_index,
    estimate_respiratory_rate,
)

MODEL_FRAMES = 150

SKIN_NAMES = {0: "Type I", 1: "Type II", 2: "Type III",
              3: "Type IV", 4: "Type V", 5: "Type VI"}
LIGHT_NAMES = {0: "LED-low", 1: "LED-high", 2: "Incandescent", 3: "Natural"}

DATASETS = ["ubfc", "ibvp", "mmpd", "mcd"]


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device, frames: int = MODEL_FRAMES):
    model = _ModelCls(in_channels=3, latent_dim=256,
                      frames=frames, lambda_grl=1.0).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Model loaded ({_MODEL_TAG}): {ckpt_path}")
    if missing:
        print(f"  [warn] {len(missing)} missing keys "
              f"(expected if checkpoint arch differs)")
    ep = ckpt.get("epoch")
    if ep is not None:
        print(f"  Checkpoint epoch: {ep}")
    return model


def resolve_checkpoint(explicit: Optional[str]) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    for base in [_HERE, _HERE.parent]:
        for folder in ["checkpoints_v5", "checkpoints_v4", "checkpoints_v3",
                       "checkpoints_v2", "checkpoints_gpu_mcd",
                       "checkpoints_gpu", "checkpoints"]:
            p = base / folder / "equiphys_best.pt"
            if p.exists():
                return str(p)
    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint path/to/equiphys_best.pt")


# ═════════════════════════════════════════════════════════════════════════════
# Signal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _bandpass_clean(sig: np.ndarray, fs: float,
                    f_low: float, f_high: float) -> np.ndarray:
    """Detrend + Butterworth bandpass. Falls back to detrend-only on failure."""
    from scipy.signal import butter, filtfilt, detrend as sp_detrend
    sig = sp_detrend(np.asarray(sig, dtype=np.float64))
    nyq  = 0.5 * fs
    lo_n = max(0.01, min(0.99, f_low  / nyq))
    hi_n = max(lo_n + 1e-3, min(0.99, f_high / nyq))
    try:
        b, a = butter(3, [lo_n, hi_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def gt_hr_from_bvp(bvp: np.ndarray, fps: float,
                   f_low: float, f_high: float) -> Optional[float]:
    """Ground-truth HR from the clip's reference BVP via Welch PSD.

    Uses snr_ratio_threshold=0.0 so the reference is never rejected — the
    reference signal is trusted by definition.
    """
    cleaned = _bandpass_clean(bvp, fps, f_low, f_high)
    peak_hz, _ = _welch_peak_hz(cleaned, fps, f_low=f_low, f_high=f_high,
                                snr_ratio_threshold=0.0)
    return float(peak_hz * 60.0) if peak_hz > 0 else None


def gt_quality(bvp: np.ndarray, fps: float,
               f_low: float, f_high: float) -> Tuple[float, float, bool]:
    """Assess whether the reference BVP is trustworthy.

    Returns (inband_pct, above_pct, usable).

    A reference whose energy sits mostly above the physiological band is
    aliased or noise-dominated; any HR derived from it is meaningless, so
    metrics computed against it would be measuring noise. We surface this
    per clip rather than silently averaging bad references into the results.
    """
    from scipy.signal import welch, detrend as sp_detrend
    n = bvp.size
    if n < 60 or np.std(bvp) < 1e-9:
        return 0.0, 0.0, False
    f, P = welch(sp_detrend(bvp), fs=fps, nperseg=min(128, n), nfft=2048)
    tot = float(P.sum())
    if tot <= 0:
        return 0.0, 0.0, False
    inband = 100.0 * float(P[(f >= f_low) & (f <= f_high)].sum()) / tot
    above  = 100.0 * float(P[f > f_high].sum()) / tot
    return inband, above, (inband >= 50.0 and above <= 30.0)


def pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 30:
        return None
    x = np.asarray(a[:n], dtype=np.float64); x -= x.mean()
    y = np.asarray(b[:n], dtype=np.float64); y -= y.mean()
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-9 or sy < 1e-9:
        return None
    r = float(np.mean(x * y) / (sx * sy))
    return r if np.isfinite(r) else None


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def run_deep(video_thwc: np.ndarray, fps: float, model, device: torch.device,
             f_low: float, f_high: float, snr_thresh: float
             ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """Deep-model HR from a single NPZ clip.

    video_thwc: [T, H, W, 3] float32 in [0,1]
    """
    T = video_thwc.shape[0]
    if T < MODEL_FRAMES:
        # pad by repeating the final frame
        pad = np.repeat(video_thwc[-1:], MODEL_FRAMES - T, axis=0)
        video_thwc = np.concatenate([video_thwc, pad], axis=0)
    elif T > MODEL_FRAMES:
        video_thwc = video_thwc[:MODEL_FRAMES]

    frames = [video_thwc[i] for i in range(video_thwc.shape[0])]
    buf    = deque(frames, maxlen=MODEL_FRAMES)

    with torch.no_grad():
        clip = build_clip_tensor_from_buffer(buf, device=device)
        out  = model(clip)
        bvp  = out["bvp_pred"][0].cpu().numpy().astype(np.float64)

    bvp_clean = _bandpass_clean(bvp, fps, f_low, f_high)
    peak_hz, snr = _welch_peak_hz(bvp_clean, fps, f_low=f_low, f_high=f_high,
                                  snr_ratio_threshold=snr_thresh)
    hr = float(peak_hz * 60.0) if peak_hz > 0 else None
    return hr, (float(snr) if np.isfinite(snr) else None), bvp_clean


def run_classical(video_thwc: np.ndarray, roi_mask: np.ndarray, fps: float,
                  method: str, f_low: float, f_high: float, snr_thresh: float
                  ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """POS / CHROM HR from a single NPZ clip.

    The classical patch extractor expects masked ROI frames where non-face
    pixels are zero, so we apply the roi_mask before extraction.
    """
    try:
        mask3  = np.repeat((roi_mask > 0.5)[..., None], 3, axis=2).astype(np.float32)
        masked = [video_thwc[i] * mask3 for i in range(video_thwc.shape[0])]

        pulse, fs_used = _extract_patch_weighted_pulse(
            masked, timestamps=None, target_fs=fps, method=method)
        peak_hz, snr = _welch_peak_hz(pulse, fs_used, f_low=f_low, f_high=f_high,
                                      snr_ratio_threshold=snr_thresh)
        hr = float(peak_hz * 60.0) if peak_hz > 0 else None
        return hr, (float(snr) if np.isfinite(snr) else None), pulse
    except Exception:
        return None, None, None


def snr_weighted_fusion(estimates: List[Tuple[str, Optional[float], Optional[float]]]
                        ) -> Tuple[Optional[float], Optional[float], str]:
    """Fuse (label, hr, snr) triples by SNR-derived weights.

    Drops candidates > 8 BPM from the median before weighting, so a single
    wild estimate cannot drag the fused value away from consensus.
    """
    valid = [(l, h, s) for (l, h, s) in estimates
             if h is not None and h > 0 and s is not None]
    if not valid:
        return None, None, "none"
    if len(valid) == 1:
        return valid[0][1], valid[0][2], valid[0][0]

    hrs  = np.array([h for _, h, _ in valid], dtype=np.float64)
    snrs = np.array([s for _, _, s in valid], dtype=np.float64)
    labs = [l for l, _, _ in valid]

    med  = float(np.median(hrs))
    keep = np.abs(hrs - med) <= 8.0
    if keep.sum() >= 2:
        hrs, snrs = hrs[keep], snrs[keep]
        labs = [l for l, k in zip(labs, keep) if k]

    w = np.exp(np.clip(snrs / 6.0, -3.0, 3.0))
    w = np.maximum(w, 0.05)
    w /= w.sum()
    return float((w * hrs).sum()), float(np.mean(snrs)), "+".join(labs)


# ═════════════════════════════════════════════════════════════════════════════
# Per-clip evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_clip(npz_path: Path, npz_root: Path, model, device: torch.device,
                  fps: float, f_low: float, f_high: float,
                  snr_thresh: float, require_usable_gt: bool = False) -> Dict:
    rel     = npz_path.relative_to(npz_root)
    subject = rel.parts[0] if len(rel.parts) > 1 else npz_path.parent.name

    row: Dict = {
        "subject": subject,
        "clip":    npz_path.stem,
        "status":  "ok",
    }

    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return {**row, "status": f"load_error:{e}"}

    if "video" not in data:
        return {**row, "status": "no_video"}

    video    = data["video"].astype(np.float32)          # [T,H,W,3]
    roi_mask = (data["roi_mask"].astype(np.float32)
                if "roi_mask" in data
                else np.ones(video.shape[1:3], dtype=np.float32))

    row["n_frames"] = int(video.shape[0])

    # ── domain labels ────────────────────────────────────────────────────────
    skin = int(data["skin_label"]) if "skin_label" in data else None
    if skin is not None and 1 <= skin <= 6:      # MMPD stores 1..6
        skin -= 1
    light = int(data["light_label"]) if "light_label" in data else None
    row["skin_label"] = skin
    row["light_label"] = light
    row["skin_name"]  = SKIN_NAMES.get(skin)   if skin  is not None else None
    row["light_name"] = LIGHT_NAMES.get(light) if light is not None else None

    # ── clinical targets (MCD) ───────────────────────────────────────────────
    if "clinical" in data:
        clin = np.asarray(data["clinical"], dtype=np.float64).flatten()
        if clin.size >= 3:
            row["gt_sbp"] = round(float(clin[0]), 2)
            row["gt_dbp"] = round(float(clin[1]), 2)
            row["gt_hb"]  = round(float(clin[2]), 2)

    # ── ground-truth HR ──────────────────────────────────────────────────────
    if "bvp" not in data:
        return {**row, "status": "no_bvp"}

    gt_bvp = data["bvp"].astype(np.float64)

    inband, above, usable = gt_quality(gt_bvp, fps, f_low, f_high)
    row["gt_inband_pct"] = round(inband, 2)
    row["gt_above_pct"]  = round(above, 2)
    row["gt_usable"]     = int(usable)

    gt_hr  = gt_hr_from_bvp(gt_bvp, fps, f_low, f_high)
    if gt_hr is None:
        return {**row, "status": "gt_hr_undetectable"}
    row["gt_hr"] = round(gt_hr, 2)

    if require_usable_gt and not usable:
        return {**row, "status": "gt_unusable"}

    # ── deep model ───────────────────────────────────────────────────────────
    deep_hr, deep_snr, deep_bvp = run_deep(
        video, fps, model, device, f_low, f_high, snr_thresh)
    row["deep_hr"]     = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"] = round(deep_snr, 2) if deep_snr is not None else None

    if deep_bvp is not None:
        r = pearson(deep_bvp, gt_bvp)
        row["deep_bvp_pearson"] = round(r, 4) if r is not None else None
        row["deep_sqi"] = round(compute_signal_quality_index(deep_bvp, fps=fps), 2)
        rr = estimate_respiratory_rate(deep_bvp, fps=fps)
        row["deep_rr"] = round(rr, 2) if rr > 0 else None

    # ── classical ────────────────────────────────────────────────────────────
    pos_hr, pos_snr, pos_bvp = run_classical(
        video, roi_mask, fps, "pos", f_low, f_high, snr_thresh)
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None

    chrom_hr, chrom_snr, chrom_bvp = run_classical(
        video, roi_mask, fps, "chrom", f_low, f_high, snr_thresh)
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None

    if pos_bvp is not None:
        r = pearson(pos_bvp, gt_bvp)
        row["pos_bvp_pearson"] = round(r, 4) if r is not None else None
    if chrom_bvp is not None:
        r = pearson(chrom_bvp, gt_bvp)
        row["chrom_bvp_pearson"] = round(r, 4) if r is not None else None

    # ── fusion ───────────────────────────────────────────────────────────────
    fused_hr, fused_snr, fused_label = snr_weighted_fusion([
        ("POS",   pos_hr,   pos_snr),
        ("CHROM", chrom_hr, chrom_snr),
        ("DEEP",  deep_hr,  deep_snr),
    ])
    row["fusion_hr"]     = round(fused_hr, 2)  if fused_hr  is not None else None
    row["fusion_snr_db"] = round(fused_snr, 2) if fused_snr is not None else None
    row["fusion_label"]  = fused_label

    # ── errors ───────────────────────────────────────────────────────────────
    for tag, pred in [("deep", deep_hr), ("pos", pos_hr),
                      ("chrom", chrom_hr), ("fusion", fused_hr)]:
        if pred is not None:
            row[f"{tag}_mae"]     = round(abs(pred - gt_hr), 3)
            row[f"{tag}_sqerr"]   = round((pred - gt_hr) ** 2, 4)
            row[f"{tag}_bias"]    = round(pred - gt_hr, 3)
            row[f"{tag}_ape"]     = round(100.0 * abs(pred - gt_hr) / max(gt_hr, 1e-6), 3)
        else:
            for s in ("mae", "sqerr", "bias", "ape"):
                row[f"{tag}_{s}"] = None

    return row


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═════════════════════════════════════════════════════════════════════════════

def aggregate(rows: List[Dict], tag: str, group: str = "overall",
              gate_snr_db: Optional[float] = None) -> Dict:
    valid = [r for r in rows if r.get(f"{tag}_mae") is not None]
    n = len(valid)
    base = {"method": tag.upper(), "group": group,
            "n_clips": len(rows), "n_valid": n, "n_failed": len(rows) - n}
    if n == 0:
        return base

    maes  = np.array([r[f"{tag}_mae"]   for r in valid], dtype=np.float64)
    sq    = np.array([r[f"{tag}_sqerr"] for r in valid], dtype=np.float64)
    bias  = np.array([r[f"{tag}_bias"]  for r in valid], dtype=np.float64)
    apes  = np.array([r[f"{tag}_ape"]   for r in valid], dtype=np.float64)
    snrs  = np.array([r[f"{tag}_snr_db"] for r in valid
                      if r.get(f"{tag}_snr_db") is not None], dtype=np.float64)

    preds = np.array([r[f"{tag}_hr"] for r in valid], dtype=np.float64)
    gts   = np.array([r["gt_hr"]     for r in valid], dtype=np.float64)

    # Pearson + CCC across clips
    pear = 0.0
    if n > 1 and np.std(preds) > 1e-9 and np.std(gts) > 1e-9:
        pear = float(np.corrcoef(preds, gts)[0, 1])
        if not np.isfinite(pear):
            pear = 0.0
    mx, my = float(np.mean(preds)), float(np.mean(gts))
    vx, vy = float(np.var(preds)),  float(np.var(gts))
    cov    = float(np.mean((preds - mx) * (gts - my)))
    den    = vx + vy + (mx - my) ** 2
    ccc    = float(2.0 * cov / den) if den > 1e-12 else 0.0

    # Bland-Altman limits of agreement
    sd     = float(np.std(bias, ddof=1)) if n > 1 else 0.0
    loa_lo = float(np.mean(bias) - 1.96 * sd)
    loa_hi = float(np.mean(bias) + 1.96 * sd)

    # BVP waveform Pearson (per-clip, if computed)
    wave = np.array([r[f"{tag}_bvp_pearson"] for r in valid
                     if r.get(f"{tag}_bvp_pearson") is not None], dtype=np.float64)

    # Deployment coverage: fraction of clips clearing the reported SNR gate
    coverage = None
    if snrs.size and gate_snr_db is not None:
        coverage = round(float(np.mean(snrs >= gate_snr_db) * 100), 1)

    base.update({
        "mae_mean":        round(float(maes.mean()),   3),
        "mae_std":         round(float(maes.std()),    3),
        "mae_median":      round(float(np.median(maes)), 3),
        "rmse":            round(float(np.sqrt(sq.mean())), 3),
        "mape_pct":        round(float(apes.mean()),   3),
        "bias":            round(float(bias.mean()),   3),
        "loa_low":         round(loa_lo, 3),
        "loa_high":        round(loa_hi, 3),
        "pearson_r":       round(pear, 4),
        "ccc":             round(ccc, 4),
        "within_2bpm_pct": round(float(np.mean(maes <= 2.0)  * 100), 1),
        "within_5bpm_pct": round(float(np.mean(maes <= 5.0)  * 100), 1),
        "within_10bpm_pct":round(float(np.mean(maes <= 10.0) * 100), 1),
        "mean_snr_db":     round(float(snrs.mean()), 2) if snrs.size else None,
        "bvp_pearson":     round(float(wave.mean()), 4) if wave.size else None,
        "coverage_pct":    coverage,
    })
    return base


def aggregate_by(rows: List[Dict], tag: str, key: str, names: Dict,
                 gate_snr_db: Optional[float] = None) -> List[Dict]:
    out = []
    for g in sorted({r.get(key) for r in rows if r.get(key) is not None}):
        subset = [r for r in rows if r.get(key) == g]
        out.append(aggregate(subset, tag, group=f"{names.get(g, g)} ({key}={g})",
                             gate_snr_db=gate_snr_db))
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:                      # union of keys, order-preserving
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    wrote {len(rows):>5} rows -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset driver
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(npz_root: Path, name: str, model, device, args) -> List[Dict]:
    clips = sorted(npz_root.rglob("*.npz"))
    if not clips:
        print(f"  [{name}] no .npz files under {npz_root} — skipping")
        return []

    if args.stride > 1:
        clips = clips[::args.stride]
    if args.max_clips:
        clips = clips[:args.max_clips]

    f_low  = args.hr_low  / 60.0
    f_high = args.hr_high / 60.0

    print(f"\n[{name.upper()}] {len(clips)} clips")
    t0 = time.time()
    results: List[Dict] = []

    for i, c in enumerate(clips, 1):
        try:
            r = evaluate_clip(c, npz_root, model, device,
                              args.fps, f_low, f_high, args.snr_thresh,
                              require_usable_gt=args.require_usable_gt)
        except Exception as e:
            r = {"subject": c.parent.name, "clip": c.stem,
                 "status": f"crash:{e}"}
            if args.verbose:
                traceback.print_exc()
        results.append(r)

        if i % args.log_every == 0 or i == len(clips):
            done = sum(1 for x in results if x.get("status") == "ok")
            el   = time.time() - t0
            rate = i / max(el, 1e-6)
            eta  = (len(clips) - i) / max(rate, 1e-6)
            print(f"  {i:>5}/{len(clips)}  ok={done:<5} "
                  f"{rate:5.1f} clip/s  eta {eta/60:5.1f} min", flush=True)

    scored = [r for r in results if r.get("gt_usable") is not None]
    if scored:
        good = sum(r["gt_usable"] for r in scored)
        pct  = 100.0 * good / len(scored)
        print(f"  [{name}] reference-BVP usable on {good}/{len(scored)} "
              f"clips ({pct:.1f}%)")
        if pct < 50 and not args.require_usable_gt:
            print(f"  [{name}] WARNING: most references are aliased or noisy. "
                  f"MAE below is measured against unreliable ground truth.")
            print(f"  [{name}]          Re-run with --require-usable-gt, or fix "
                  f"preprocessing first.")

    print(f"  [{name}] done in {(time.time()-t0)/60:.1f} min")
    return results


def print_table(summary: List[Dict], title: str) -> None:
    by = {r["method"]: r for r in summary}
    order = ["POS", "CHROM", "DEEP", "FUSION"]
    order = [m for m in order if m in by]

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    hdr = f"{'Metric':<24}" + "".join(f"{m:>13}" for m in order)
    print(hdr)
    print("-" * len(hdr))

    metrics = [
        ("n_valid",          "Valid clips"),
        ("mae_mean",         "MAE (BPM)"),
        ("mae_median",       "MAE median"),
        ("rmse",             "RMSE (BPM)"),
        ("mape_pct",         "MAPE (%)"),
        ("bias",             "Bias (BPM)"),
        ("pearson_r",        "Pearson r"),
        ("ccc",              "CCC"),
        ("within_5bpm_pct",  "Within +/-5 BPM %"),
        ("within_10bpm_pct", "Within +/-10 BPM %"),
        ("mean_snr_db",      "Mean SNR (dB)"),
        ("bvp_pearson",      "BVP waveform r"),
        ("coverage_pct",     "SNR-gate coverage %"),
    ]
    for key, label in metrics:
        cells = "".join(f"{str(by[m].get(key, '-')):>13}" for m in order)
        print(f"{label:<24}{cells}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate deep + POS + CHROM + fusion on preprocessed NPZ clips")
    p.add_argument("--npz-root",   required=True,
                   help="Path to processed_npz/<dataset> or processed_npz (with --all)")
    p.add_argument("--dataset",    default=None, choices=DATASETS,
                   help="Dataset name (used for output filenames)")
    p.add_argument("--all",        action="store_true",
                   help="Treat --npz-root as the parent and evaluate every subfolder")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir",    default="results_npz")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps",        type=float, default=30.0)
    p.add_argument("--hr-low",     type=float, default=48.0)
    p.add_argument("--hr-high",    type=float, default=150.0)
    p.add_argument("--snr-thresh", type=float, default=0.0,
                   help="SNR ratio gate for accepting a spectral peak. "
                        "Default 0.0 = accept every peak, so MAE is measured on "
                        "ALL clips (unbiased algorithm accuracy). Raise it to "
                        "mimic live-app behaviour, but note MAE is then computed "
                        "only on the confident subset.")
    p.add_argument("--gate-snr-db", type=float, default=0.0,
                   help="Reported deployment gate (dB). Does not filter results; "
                        "adds a 'coverage_pct' column showing what fraction of "
                        "clips would clear this SNR in live use.")
    p.add_argument("--max-clips",  type=int,   default=None)
    p.add_argument("--stride",     type=int,   default=1,
                   help="Evaluate every Nth clip (speeds up large datasets)")
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--require-usable-gt", action="store_true",
                   help="Skip clips whose reference BVP is aliased or "
                        "noise-dominated. Strongly recommended: metrics against "
                        "an unusable reference measure noise, not accuracy. "
                        "Run check_npz_gt.py first to see how many clips this "
                        "affects.")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    try:
        ckpt  = resolve_checkpoint(args.checkpoint)
        model = load_model(ckpt, device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    root    = Path(args.npz_root)
    out_dir = Path(args.out_dir)

    # Decide which (name, path) pairs to evaluate
    targets: List[Tuple[str, Path]] = []
    if args.all:
        for d in DATASETS:
            sub = root / d
            if sub.exists() and any(sub.rglob("*.npz")):
                targets.append((d, sub))
        if not targets:
            print(f"No dataset subfolders with .npz under {root}")
            sys.exit(1)
    else:
        name = args.dataset or root.name
        targets.append((name, root))

    print(f"\nDevice     : {args.device}")
    print(f"HR band    : {args.hr_low}-{args.hr_high} BPM  (fps={args.fps})")
    print(f"SNR gate   : accept={args.snr_thresh} (0 = accept all)  "
          f"report-gate={args.gate_snr_db} dB")
    print(f"Datasets   : {', '.join(n for n, _ in targets)}")

    grand: List[Dict] = []

    for name, path in targets:
        rows = evaluate_dataset(path, name, model, device, args)
        if not rows:
            continue

        base = out_dir / f"{name}_npz"
        write_csv(rows, base.with_name(f"{name}_npz_results.csv"))

        summary = [aggregate(rows, t, gate_snr_db=args.gate_snr_db)
                   for t in ("pos", "chrom", "deep", "fusion")]
        write_csv(summary, base.with_name(f"{name}_npz_summary.csv"))

        # Fairness / condition breakdowns where labels exist
        if any(r.get("skin_label") is not None for r in rows):
            skin_rows: List[Dict] = []
            for t in ("pos", "chrom", "deep", "fusion"):
                skin_rows += aggregate_by(rows, t, "skin_label", SKIN_NAMES,
                                          gate_snr_db=args.gate_snr_db)
            write_csv(skin_rows, base.with_name(f"{name}_npz_by_skin.csv"))

        if any(r.get("light_label") is not None for r in rows):
            light_rows: List[Dict] = []
            for t in ("pos", "chrom", "deep", "fusion"):
                light_rows += aggregate_by(rows, t, "light_label", LIGHT_NAMES,
                                           gate_snr_db=args.gate_snr_db)
            write_csv(light_rows, base.with_name(f"{name}_npz_by_light.csv"))

        print_table(summary, f"{name.upper()}  ({len(rows)} clips)")

        for s in summary:
            grand.append({**s, "dataset": name})

    if len(targets) > 1 and grand:
        write_csv(grand, out_dir / "ALL_datasets_summary.csv")
        print("\n" + "=" * 76)
        print("CROSS-DATASET SUMMARY  (MAE in BPM)")
        print("=" * 76)
        hdr = f"{'Dataset':<12}" + "".join(f"{m:>13}"
                                           for m in ["POS", "CHROM", "DEEP", "FUSION"])
        print(hdr); print("-" * len(hdr))
        for name, _ in targets:
            cells = ""
            for m in ["POS", "CHROM", "DEEP", "FUSION"]:
                hit = [g for g in grand if g["dataset"] == name and g["method"] == m]
                cells += f"{str(hit[0].get('mae_mean', '-')) if hit else '-':>13}"
            print(f"{name:<12}{cells}")

    print(f"\nAll CSVs written to: {out_dir}/")


if __name__ == "__main__":
    main()
