"""check_npz_gt.py — Audit ground-truth BVP quality in preprocessed NPZ clips.

Why this exists
---------------
Every supervised metric in training and evaluation is measured against the
`bvp` field stored in each NPZ clip. If that reference signal is aliased,
noise-dominated, or simply the wrong array picked out of the source file,
then:

  * the training loss is computed against noise,
  * validation HR MAE / Pearson are meaningless,
  * and evaluation MAE numbers cannot be trusted.

This script measures, per clip, how much of the reference signal's energy
falls inside the physiological heart-rate band versus outside it, and reports
the fraction of clips whose reference is usable.

What "good" looks like
----------------------
A clean 30 Hz PPG reference from a resting adult should show:

    in-band energy (0.7-3.0 Hz)  : > 50%
    above-band energy (> 3.0 Hz) : < 30%
    dominant peak                : 45-150 BPM

A clip failing these is flagged. A dataset where most clips fail indicates a
preprocessing bug (most commonly: the source signal was sampled far above
30 Hz and was decimated with plain interpolation and no anti-aliasing filter,
or the wrong key was selected from the source .mat/.csv).

Usage
-----
    python check_npz_gt.py --npz-root processed_npz --all
    python check_npz_gt.py --npz-root processed_npz/mmpd --sample 300
    python check_npz_gt.py --npz-root processed_npz --all --csv gt_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import detrend, welch

DATASETS = ["ubfc", "ibvp", "mmpd", "mcd"]

# Physiological band and pass criteria
F_LOW, F_HIGH = 0.7, 3.0          # Hz  (42-180 BPM)
MIN_INBAND_PCT   = 50.0           # at least this much energy in band
MAX_ABOVE_PCT    = 30.0           # at most this much above band
HR_MIN, HR_MAX   = 45.0, 150.0    # plausible dominant peak, BPM


def audit_clip(path: Path, fps: float) -> Dict:
    out: Dict = {"clip": str(path.name), "subject": path.parent.name}
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as e:
        return {**out, "status": f"load_error:{e}"}

    if "bvp" not in d:
        return {**out, "status": "no_bvp"}

    bvp = np.asarray(d["bvp"], dtype=np.float64).ravel()
    n   = bvp.size
    out["n"] = n

    if n < 60:
        return {**out, "status": "too_short"}
    if np.std(bvp) < 1e-9:
        return {**out, "status": "flat"}

    out["mean"] = round(float(bvp.mean()), 3)
    out["std"]  = round(float(bvp.std()), 3)
    # Mean absolute first difference: a smooth 30 Hz PPG has a small value here
    out["mean_abs_diff"] = round(float(np.abs(np.diff(bvp)).mean()), 3)

    f, P = welch(detrend(bvp), fs=fps,
                 nperseg=min(128, n), nfft=2048)
    tot = float(P.sum())
    if tot <= 0:
        return {**out, "status": "zero_power"}

    inband = float(P[(f >= F_LOW) & (f <= F_HIGH)].sum())
    above  = float(P[f > F_HIGH].sum())
    below  = float(P[f < F_LOW].sum())

    out["inband_pct"] = round(100.0 * inband / tot, 2)
    out["above_pct"]  = round(100.0 * above  / tot, 2)
    out["below_pct"]  = round(100.0 * below  / tot, 2)

    band = (f >= F_LOW) & (f <= F_HIGH)
    out["peak_bpm"] = (round(float(f[band][np.argmax(P[band])] * 60.0), 2)
                       if band.any() else None)
    out["global_peak_bpm"] = round(float(f[np.argmax(P)] * 60.0), 2)

    reasons: List[str] = []
    if out["inband_pct"] < MIN_INBAND_PCT:
        reasons.append("low_inband")
    if out["above_pct"] > MAX_ABOVE_PCT:
        reasons.append("aliased_or_noisy")
    if out["peak_bpm"] is None or not (HR_MIN <= out["peak_bpm"] <= HR_MAX):
        reasons.append("implausible_hr")

    out["status"] = "ok" if not reasons else "|".join(reasons)
    out["usable"] = 1 if not reasons else 0
    return out


def audit_dataset(root: Path, name: str, fps: float,
                  sample: Optional[int], seed: int) -> List[Dict]:
    clips = sorted(root.rglob("*.npz"))
    if not clips:
        print(f"  [{name}] no .npz found under {root}")
        return []

    total = len(clips)
    if sample and sample < total:
        random.Random(seed).shuffle(clips)
        clips = clips[:sample]

    print(f"\n[{name.upper()}] auditing {len(clips)} of {total} clips")
    rows = [audit_clip(c, fps) for c in clips]

    usable = [r for r in rows if r.get("usable") == 1]
    scored = [r for r in rows if "inband_pct" in r]
    pct = 100.0 * len(usable) / max(len(rows), 1)

    print(f"  usable            : {len(usable)}/{len(rows)}  ({pct:.1f}%)")
    if scored:
        ib = np.array([r["inband_pct"] for r in scored])
        ab = np.array([r["above_pct"]  for r in scored])
        md = np.array([r["mean_abs_diff"] for r in scored])
        print(f"  in-band energy    : median {np.median(ib):5.1f}%   "
              f"IQR [{np.percentile(ib,25):.1f}, {np.percentile(ib,75):.1f}]")
        print(f"  above-band energy : median {np.median(ab):5.1f}%   "
              f"IQR [{np.percentile(ab,25):.1f}, {np.percentile(ab,75):.1f}]")
        print(f"  mean |diff|       : median {np.median(md):5.2f}")

    # Failure breakdown
    fails: Dict[str, int] = {}
    for r in rows:
        s = r.get("status", "?")
        if s != "ok":
            for part in s.split("|"):
                fails[part] = fails.get(part, 0) + 1
    if fails:
        print("  failure reasons   :")
        for k, v in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"      {k:<22} {v:>6}  ({100.0*v/len(rows):.1f}%)")

    # Verdict
    print("  verdict           : ", end="")
    if pct >= 80:
        print("GOOD — reference signals are usable.")
    elif pct >= 40:
        print("MIXED — a large minority of references are unusable.")
        print("                      Filter to usable clips before trusting metrics.")
    else:
        print("BAD — most references are unusable.")
        print("                      Training and evaluation against these is")
        print("                      measuring noise. Fix preprocessing before")
        print("                      drawing any conclusion from the metrics.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit ground-truth BVP quality in preprocessed NPZ clips")
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--all", action="store_true",
                    help="Treat --npz-root as parent; audit each dataset subfolder")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--sample", type=int, default=500,
                    help="Random subsample per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="Write per-clip audit to this CSV")
    args = ap.parse_args()

    root = Path(args.npz_root)
    targets = []
    if args.all:
        for d in DATASETS:
            sub = root / d
            if sub.exists() and any(sub.rglob("*.npz")):
                targets.append((d, sub))
        if not targets:
            print(f"No dataset subfolders with .npz under {root}")
            return
    else:
        targets.append((root.name, root))

    print("=" * 70)
    print("GROUND-TRUTH BVP AUDIT")
    print("=" * 70)
    print(f"Physiological band : {F_LOW}-{F_HIGH} Hz "
          f"({F_LOW*60:.0f}-{F_HIGH*60:.0f} BPM)")
    print(f"Pass criteria      : in-band >= {MIN_INBAND_PCT}%, "
          f"above-band <= {MAX_ABOVE_PCT}%, peak in {HR_MIN:.0f}-{HR_MAX:.0f} BPM")

    all_rows: List[Dict] = []
    summary: List[tuple] = []
    sample = args.sample if args.sample and args.sample > 0 else None

    for name, path in targets:
        rows = audit_dataset(path, name, args.fps, sample, args.seed)
        for r in rows:
            r["dataset"] = name
        all_rows += rows
        if rows:
            u = sum(1 for r in rows if r.get("usable") == 1)
            summary.append((name, u, len(rows), 100.0 * u / len(rows)))

    if len(summary) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Dataset':<10}{'Usable':>10}{'Audited':>10}{'Usable %':>12}")
        print("-" * 70)
        for name, u, n, pct in summary:
            print(f"{name:<10}{u:>10}{n:>10}{pct:>11.1f}%")

    if args.csv and all_rows:
        keys: List[str] = []
        for r in all_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nPer-clip audit written to {p}")


if __name__ == "__main__":
    main()
