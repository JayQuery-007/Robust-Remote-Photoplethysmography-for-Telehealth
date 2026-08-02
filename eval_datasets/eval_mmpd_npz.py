"""eval_mmpd_npz.py — Evaluate Deep + POS + CHROM + Fusion on MMPD NPZ clips.

Folder structure expected
-------------------------
processed_npz/mmpd/
  subject1/
    p1_0_clip_0000.npz
    p1_0_clip_0001.npz
    ...
    p1_19_clip_0000.npz
    ...
  subject2/
    p2_0_clip_0000.npz
    ...
  ...
  subject25/

NPZ schema per clip
-------------------
  video       [150, 64, 64, 3]  float32  in [0,1]
  roi_mask    [64, 64]          float32  in [0,1]
  bvp         [150]             float32  reference signal
  skin_label  scalar int        0-based Fitzpatrick (0=Type I … 5=Type VI)
  light_label scalar int        0=LED-low 1=LED-high 2=Incandescent 3=Natural

Outputs (all in --out-dir)
--------------------------
  mmpd_results.csv          per-clip (hr, MAE, BVP Pearson, skin, light …)
  mmpd_summary.csv          aggregate MAE/RMSE/Pearson/CCC per method
  mmpd_by_skin.csv          MAE breakdown by Fitzpatrick type per method
  mmpd_by_light.csv         MAE breakdown by lighting condition per method
  mmpd_gt_audit.csv         per-clip reference BVP quality flags

Usage
-----
  python eval_mmpd_npz.py --root processed_npz/mmpd \
         --checkpoint checkpoints_v2/equiphys_best.pt
  # quick sanity (20 clips)
  python eval_mmpd_npz.py --root processed_npz/mmpd \
         --checkpoint checkpoints_v2/equiphys_best.pt --max-clips 20
  # skip clips with aliased/noisy reference
  python eval_mmpd_npz.py --root processed_npz/mmpd \
         --checkpoint checkpoints_v2/equiphys_best.pt --require-usable-gt
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── path resolution ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
for _c in [_HERE, _HERE.parent]:
    if (_c / "equiphys_core.py").exists():
        sys.path.insert(0, str(_c))
        break

try:
    from equiphys_v2 import EquiPhysDANNV2 as _ModelCls
    _MODEL_TAG = "v2"
except ImportError:
    from equiphys_core import EquiPhysDANN as _ModelCls   # type: ignore
    _MODEL_TAG = "v1"

from equiphys_core import (
    _extract_patch_weighted_pulse,
    _welch_peak_hz,
    build_clip_tensor_from_buffer,
    compute_signal_quality_index,
    estimate_respiratory_rate,
)

MODEL_FRAMES = 150

SKIN_NAMES  = {0:"Type I", 1:"Type II", 2:"Type III",
               3:"Type IV", 4:"Type V",  5:"Type VI"}
LIGHT_NAMES = {0:"LED-low", 1:"LED-high", 2:"Incandescent", 3:"Natural"}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers shared across the file
# ═════════════════════════════════════════════════════════════════════════════

def _bandpass(sig: np.ndarray, fps: float, f_lo: float, f_hi: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, detrend
    sig = detrend(np.asarray(sig, dtype=np.float64))
    nyq = 0.5 * fps
    lo = max(0.01, min(0.99, f_lo / nyq))
    hi = max(lo + 1e-3, min(0.99, f_hi / nyq))
    try:
        b, a = butter(3, [lo, hi], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 30:
        return None
    x = a[:n] - a[:n].mean()
    y = b[:n] - b[:n].mean()
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-9 or sy < 1e-9:
        return None
    r = float(np.mean(x * y) / (sx * sy))
    return r if np.isfinite(r) else None


def _gt_hr(bvp: np.ndarray, fps: float, f_lo: float, f_hi: float) -> Optional[float]:
    sig = _bandpass(bvp, fps, f_lo, f_hi)
    hz, _ = _welch_peak_hz(sig, fps, f_low=f_lo, f_high=f_hi,
                            snr_ratio_threshold=0.0)
    return float(hz * 60.0) if hz > 0 else None


def _gt_quality(bvp: np.ndarray, fps: float,
                f_lo: float, f_hi: float) -> Tuple[float, float, bool]:
    """Return (inband_pct, above_pct, usable)."""
    from scipy.signal import welch, detrend
    if bvp.size < 60 or np.std(bvp) < 1e-9:
        return 0.0, 0.0, False
    f, P = welch(detrend(bvp.astype(np.float64)), fs=fps,
                 nperseg=min(128, bvp.size), nfft=2048)
    tot = float(P.sum())
    if tot <= 0:
        return 0.0, 0.0, False
    inband = 100.0 * float(P[(f >= f_lo) & (f <= f_hi)].sum()) / tot
    above  = 100.0 * float(P[f > f_hi].sum()) / tot
    return round(inband, 2), round(above, 2), (inband >= 50.0 and above <= 30.0)


def _snr_fusion(candidates: List[Tuple[str, Optional[float], Optional[float]]]
                ) -> Tuple[Optional[float], Optional[float], str]:
    valid = [(l, h, s) for l, h, s in candidates
             if h is not None and h > 0 and s is not None]
    if not valid:
        return None, None, "none"
    if len(valid) == 1:
        return valid[0][1], valid[0][2], valid[0][0]
    hrs  = np.array([h for _, h, _ in valid])
    snrs = np.array([s for _, _, s in valid])
    labs = [l for l, _, _ in valid]
    med  = float(np.median(hrs))
    keep = np.abs(hrs - med) <= 8.0
    if keep.sum() >= 2:
        hrs, snrs, labs = hrs[keep], snrs[keep], [l for l, k in zip(labs, keep) if k]
    w = np.exp(np.clip(snrs / 6.0, -3.0, 3.0))
    w = np.maximum(w, 0.05); w /= w.sum()
    return float((w * hrs).sum()), float(np.mean(snrs)), "+".join(labs)


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device):
    m = _ModelCls(in_channels=3, latent_dim=256,
                  frames=MODEL_FRAMES, lambda_grl=1.0).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    m.load_state_dict(state, strict=False)
    m.eval()
    ep = ckpt.get("epoch")
    print(f"  Loaded ({_MODEL_TAG}): {ckpt_path}" +
          (f"  epoch={ep}" if ep is not None else ""))
    return m


def _resolve(explicit: Optional[str]) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    for base in [_HERE, _HERE.parent]:
        for folder in ["checkpoints_v5","checkpoints_v4","checkpoints_v3",
                       "checkpoints_v2","checkpoints_gpu_mcd",
                       "checkpoints_gpu","checkpoints"]:
            p = base / folder / "equiphys_best.pt"
            if p.exists():
                return str(p)
    raise FileNotFoundError("No checkpoint found. Pass --checkpoint.")


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def _run_deep(video: np.ndarray, model, device, fps, f_lo, f_hi, snr_thr):
    T = video.shape[0]
    if T < MODEL_FRAMES:
        pad = np.repeat(video[-1:], MODEL_FRAMES - T, axis=0)
        video = np.concatenate([video, pad], axis=0)
    elif T > MODEL_FRAMES:
        video = video[:MODEL_FRAMES]
    buf = deque([video[i] for i in range(MODEL_FRAMES)], maxlen=MODEL_FRAMES)
    with torch.no_grad():
        clip = build_clip_tensor_from_buffer(buf, device=device)
        out  = model(clip)
        bvp  = out["bvp_pred"][0].cpu().numpy().astype(np.float64)
    bvp_c = _bandpass(bvp, fps, f_lo, f_hi)
    hz, snr = _welch_peak_hz(bvp_c, fps, f_low=f_lo, f_high=f_hi,
                              snr_ratio_threshold=snr_thr)
    return (float(hz * 60.0) if hz > 0 else None,
            float(snr) if np.isfinite(snr) else None, bvp_c)


def _run_classical(video: np.ndarray, mask: np.ndarray,
                   fps, method, f_lo, f_hi, snr_thr):
    try:
        mask3  = np.repeat((mask > 0.5)[..., None], 3, axis=2).astype(np.float32)
        frames = [video[i] * mask3 for i in range(video.shape[0])]
        pulse, fs = _extract_patch_weighted_pulse(
            frames, timestamps=None, target_fs=fps, method=method)
        hz, snr = _welch_peak_hz(pulse, fs, f_low=f_lo, f_high=f_hi,
                                 snr_ratio_threshold=snr_thr)
        return ((float(hz * 60.0) if hz > 0 else None),
                (float(snr) if np.isfinite(snr) else None), pulse)
    except Exception:
        return None, None, None


# ═════════════════════════════════════════════════════════════════════════════
# Per-clip
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_clip(path: Path, model, device, fps,
                  f_lo, f_hi, snr_thr, require_usable_gt) -> Dict:
    row: Dict = {
        "subject": path.parent.name,
        "clip":    path.stem,
        "status":  "ok",
    }
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as e:
        return {**row, "status": f"load_error:{e}"}

    video = d["video"].astype(np.float32)          # [T,H,W,3]
    mask  = (d["roi_mask"].astype(np.float32)
             if "roi_mask" in d else np.ones(video.shape[1:3], dtype=np.float32))
    row["n_frames"] = int(video.shape[0])

    # Labels
    raw_skin = int(d["skin_label"]) if "skin_label" in d else None
    skin = (raw_skin - 1 if raw_skin is not None and 1 <= raw_skin <= 6
            else raw_skin)                          # MMPD stores 1-based
    light = int(d["light_label"]) if "light_label" in d else None
    row["skin_label"] = skin
    row["light_label"] = light
    row["skin_name"]   = SKIN_NAMES.get(skin)   if skin  is not None else None
    row["light_name"]  = LIGHT_NAMES.get(light) if light is not None else None

    # Ground truth
    if "bvp" not in d:
        return {**row, "status": "no_bvp"}
    gt_bvp = d["bvp"].astype(np.float64)
    inband, above, usable = _gt_quality(gt_bvp, fps, f_lo, f_hi)
    row["gt_inband_pct"] = inband
    row["gt_above_pct"]  = above
    row["gt_usable"]     = int(usable)

    if require_usable_gt and not usable:
        return {**row, "status": "gt_unusable"}

    gt_hr = _gt_hr(gt_bvp, fps, f_lo, f_hi)
    if gt_hr is None:
        return {**row, "status": "gt_hr_undetectable"}
    row["gt_hr"] = round(gt_hr, 2)

    # Inference
    deep_hr, deep_snr, deep_bvp = _run_deep(
        video, model, device, fps, f_lo, f_hi, snr_thr)
    row["deep_hr"]     = round(deep_hr, 2)  if deep_hr  is not None else None
    row["deep_snr_db"] = round(deep_snr, 2) if deep_snr is not None else None
    if deep_bvp is not None:
        row["deep_bvp_pearson"] = _pearson(deep_bvp, gt_bvp)
        row["deep_sqi"] = round(compute_signal_quality_index(deep_bvp, fps), 2)
        rr = estimate_respiratory_rate(deep_bvp, fps)
        row["deep_rr"] = round(rr, 2) if rr > 0 else None

    pos_hr, pos_snr, pos_bvp = _run_classical(
        video, mask, fps, "pos", f_lo, f_hi, snr_thr)
    row["pos_hr"]     = round(pos_hr, 2)  if pos_hr  is not None else None
    row["pos_snr_db"] = round(pos_snr, 2) if pos_snr is not None else None
    if pos_bvp is not None:
        row["pos_bvp_pearson"] = _pearson(pos_bvp, gt_bvp)

    chrom_hr, chrom_snr, chrom_bvp = _run_classical(
        video, mask, fps, "chrom", f_lo, f_hi, snr_thr)
    row["chrom_hr"]     = round(chrom_hr, 2)  if chrom_hr  is not None else None
    row["chrom_snr_db"] = round(chrom_snr, 2) if chrom_snr is not None else None
    if chrom_bvp is not None:
        row["chrom_bvp_pearson"] = _pearson(chrom_bvp, gt_bvp)

    fused_hr, fused_snr, fused_label = _snr_fusion([
        ("POS",   pos_hr,   pos_snr),
        ("CHROM", chrom_hr, chrom_snr),
        ("DEEP",  deep_hr,  deep_snr),
    ])
    row["fusion_hr"]    = round(fused_hr, 2)  if fused_hr  is not None else None
    row["fusion_snr_db"]= round(fused_snr, 2) if fused_snr is not None else None
    row["fusion_label"] = fused_label

    for tag, pred in [("deep", deep_hr), ("pos", pos_hr),
                      ("chrom", chrom_hr), ("fusion", fused_hr)]:
        if pred is not None:
            row[f"{tag}_mae"]  = round(abs(pred - gt_hr), 3)
            row[f"{tag}_sqerr"]= round((pred - gt_hr) ** 2, 4)
            row[f"{tag}_bias"] = round(pred - gt_hr, 3)
            row[f"{tag}_ape"]  = round(100.0 * abs(pred - gt_hr) / max(gt_hr, 1e-6), 3)
        else:
            for s in ("mae", "sqerr", "bias", "ape"):
                row[f"{tag}_{s}"] = None
    return row


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation
# ═════════════════════════════════════════════════════════════════════════════

def aggregate(rows: List[Dict], tag: str, group: str = "overall",
              gate_snr_db: float = 0.0) -> Dict:
    valid = [r for r in rows if r.get(f"{tag}_mae") is not None]
    n = len(valid)
    base = {"method": tag.upper(), "group": group,
            "n_clips": len(rows), "n_valid": n, "n_failed": len(rows) - n}
    if n == 0:
        return base
    maes  = np.array([r[f"{tag}_mae"]    for r in valid])
    sq    = np.array([r[f"{tag}_sqerr"]  for r in valid])
    bias  = np.array([r[f"{tag}_bias"]   for r in valid])
    apes  = np.array([r[f"{tag}_ape"]    for r in valid])
    snrs  = np.array([r[f"{tag}_snr_db"] for r in valid
                      if r.get(f"{tag}_snr_db") is not None])
    preds = np.array([r[f"{tag}_hr"]  for r in valid])
    gts   = np.array([r["gt_hr"]      for r in valid])
    wave  = np.array([r[f"{tag}_bvp_pearson"] for r in valid
                      if r.get(f"{tag}_bvp_pearson") is not None])

    pear = 0.0
    if n > 1 and np.std(preds) > 1e-9 and np.std(gts) > 1e-9:
        pear = float(np.corrcoef(preds, gts)[0, 1])
        if not np.isfinite(pear):
            pear = 0.0
    mx, my = preds.mean(), gts.mean()
    cov = float(np.mean((preds - mx) * (gts - my)))
    den = float(np.var(preds) + np.var(gts) + (mx - my) ** 2)
    ccc = float(2.0 * cov / den) if den > 1e-12 else 0.0
    sd  = float(np.std(bias, ddof=1)) if n > 1 else 0.0

    cov_pct = (round(float(np.mean(snrs >= gate_snr_db) * 100), 1)
               if snrs.size and gate_snr_db > 0 else None)
    base.update({
        "mae_mean":         round(float(maes.mean()), 3),
        "mae_std":          round(float(maes.std()),  3),
        "mae_median":       round(float(np.median(maes)), 3),
        "rmse":             round(float(np.sqrt(sq.mean())), 3),
        "mape_pct":         round(float(apes.mean()), 3),
        "bias":             round(float(bias.mean()), 3),
        "loa_low":          round(float(bias.mean() - 1.96 * sd), 3),
        "loa_high":         round(float(bias.mean() + 1.96 * sd), 3),
        "pearson_r":        round(pear, 4),
        "ccc":              round(ccc, 4),
        "within_2bpm_pct":  round(float(np.mean(maes <= 2.0)  * 100), 1),
        "within_5bpm_pct":  round(float(np.mean(maes <= 5.0)  * 100), 1),
        "within_10bpm_pct": round(float(np.mean(maes <= 10.0) * 100), 1),
        "mean_snr_db":      round(float(snrs.mean()), 2) if snrs.size else None,
        "bvp_pearson":      round(float(wave.mean()), 4) if wave.size else None,
        "coverage_pct":     cov_pct,
    })
    return base


def aggregate_by(rows: List[Dict], tag: str, key: str, names: Dict,
                 gate_snr_db: float = 0.0) -> List[Dict]:
    out = []
    for g in sorted({r.get(key) for r in rows if r.get(key) is not None}):
        subset = [r for r in rows if r.get(key) == g]
        label  = f"{names.get(g, g)} ({key}={g})"
        out.append(aggregate(subset, tag, group=label, gate_snr_db=gate_snr_db))
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"    wrote {len(rows):>5} rows  ->  {path}")


def print_table(summary: List[Dict], title: str) -> None:
    by    = {r["method"]: r for r in summary}
    order = [m for m in ["POS","CHROM","DEEP","FUSION"] if m in by]
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    hdr = f"{'Metric':<26}" + "".join(f"{m:>12}" for m in order)
    print(hdr); print("-" * len(hdr))
    for key, lbl in [
        ("n_valid",          "Valid clips"),
        ("mae_mean",         "MAE (BPM)"),
        ("mae_median",       "MAE median"),
        ("rmse",             "RMSE (BPM)"),
        ("mape_pct",         "MAPE (%)"),
        ("bias",             "Bias (BPM)"),
        ("pearson_r",        "Pearson r"),
        ("ccc",              "CCC"),
        ("within_5bpm_pct",  "Within ±5 BPM %"),
        ("within_10bpm_pct", "Within ±10 BPM %"),
        ("mean_snr_db",      "Mean SNR (dB)"),
        ("bvp_pearson",      "BVP waveform r"),
    ]:
        cells = "".join(f"{str(by[m].get(key,'-')):>12}" for m in order)
        print(f"{lbl:<26}{cells}")


# ═════════════════════════════════════════════════════════════════════════════
# Discovery
# ═════════════════════════════════════════════════════════════════════════════

def discover(root: Path) -> List[Path]:
    """Collect all NPZ files under root regardless of nesting depth."""
    return sorted(root.rglob("*.npz"))


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Deep+POS+CHROM+Fusion on MMPD NPZ clips")
    p.add_argument("--root",       required=True, help="processed_npz/mmpd")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir",    default="results")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps",        type=float, default=30.0)
    p.add_argument("--hr-low",     type=float, default=48.0)
    p.add_argument("--hr-high",    type=float, default=150.0)
    p.add_argument("--snr-thresh", type=float, default=0.0,
                   help="SNR gate for peak acceptance (0 = accept all, unbiased)")
    p.add_argument("--max-clips",  type=int,   default=None)
    p.add_argument("--stride",     type=int,   default=1)
    p.add_argument("--log-every",  type=int,   default=100)
    p.add_argument("--require-usable-gt", action="store_true",
                   help="Skip clips whose reference BVP is aliased/noisy")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    model  = load_model(_resolve(args.checkpoint), device)

    root  = Path(args.root)
    clips = discover(root)
    if not clips:
        print(f"No .npz files found under {root}"); sys.exit(1)
    if args.stride > 1:
        clips = clips[::args.stride]
    if args.max_clips:
        clips = clips[:args.max_clips]

    f_lo  = args.hr_low  / 60.0
    f_hi  = args.hr_high / 60.0
    out   = Path(args.out_dir)

    print(f"\nMMPD NPZ  |  {len(clips)} clips  |  device={args.device}  |  {_MODEL_TAG}")
    print(f"HR band : {args.hr_low}-{args.hr_high} BPM   SNR gate : {args.snr_thresh}")
    print(f"require-usable-gt : {args.require_usable_gt}")

    t0 = time.time()
    results: List[Dict] = []
    for i, c in enumerate(clips, 1):
        try:
            r = evaluate_clip(c, model, device, args.fps,
                              f_lo, f_hi, args.snr_thresh,
                              args.require_usable_gt)
        except Exception as e:
            r = {"subject": c.parent.name, "clip": c.stem,
                 "status": f"crash:{e}"}
        results.append(r)
        if i % args.log_every == 0 or i == len(clips):
            ok = sum(1 for x in results if x.get("status") == "ok")
            elapsed = time.time() - t0
            print(f"  {i:>5}/{len(clips)}  ok={ok:<6} "
                  f"{i/max(elapsed,1e-9):5.1f} clip/s", flush=True)

    usable = sum(1 for r in results if r.get("gt_usable") == 1)
    print(f"\nReference BVP usable: {usable}/{len(results)} clips "
          f"({100.0*usable/max(len(results),1):.1f}%)")
    if usable < len(results) * 0.5:
        print("WARNING: more than half the references are aliased/noisy.")
        print("         Run with --require-usable-gt for reliable metrics.")

    write_csv(results, out / "mmpd_results.csv")

    summary = [aggregate(results, t) for t in ("pos","chrom","deep","fusion")]
    write_csv(summary, out / "mmpd_summary.csv")

    skin_rows: List[Dict] = []
    for t in ("pos","chrom","deep","fusion"):
        skin_rows += aggregate_by(results, t, "skin_label", SKIN_NAMES)
    write_csv(skin_rows, out / "mmpd_by_skin.csv")

    light_rows: List[Dict] = []
    for t in ("pos","chrom","deep","fusion"):
        light_rows += aggregate_by(results, t, "light_label", LIGHT_NAMES)
    write_csv(light_rows, out / "mmpd_by_light.csv")

    print_table(summary, f"MMPD  ({len(results)} clips evaluated)")

    # Skin breakdown — deep only
    deep_skin = [r for r in skin_rows if r["method"] == "DEEP" and r.get("n_valid", 0) > 0]
    if deep_skin:
        print("\nDeep MAE by Fitzpatrick skin type:")
        for r in deep_skin:
            print(f"  {r['group']:<38} n={r['n_valid']:>4}  "
                  f"MAE={r.get('mae_mean','—')}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min   CSVs -> {out}/")


if __name__ == "__main__":
    main()
