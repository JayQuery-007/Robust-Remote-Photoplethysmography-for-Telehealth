from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.io import loadmat

from equiphys_core import FaceROIExtractor


def _read_video_frames(video_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames


def _read_numeric_txt(path: Path) -> np.ndarray:
    # UBFC text files are numeric; this parser keeps only numeric tokens and flattens to 1D.
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().replace(",", " ").split()
            for tok in tokens:
                try:
                    values.append(float(tok))
                except ValueError:
                    continue
    if not values:
        raise ValueError(f"No numeric values found in text file: {path}")
    return np.asarray(values, dtype=np.float32)


def _read_pw_file(path: Path) -> np.ndarray:
    # MCD .pw can vary by release; try text parse first, then binary float32.
    try:
        sig = _read_numeric_txt(path)
        if sig.size > 0:
            return sig
    except Exception:
        pass

    raw = path.read_bytes()
    if len(raw) % 4 == 0 and len(raw) > 0:
        sig = np.frombuffer(raw, dtype=np.float32)
        if sig.size > 0:
            return sig.astype(np.float32)

    raise ValueError(f"Unsupported .pw format: {path}")


def _read_bvp_csv(path: Path) -> np.ndarray:
    """Read an IBVP BVP CSV and return the BVP column as a 1D float array."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "BVP" in reader.fieldnames:
            values: List[float] = []
            for row in reader:
                try:
                    values.append(float(row["BVP"]))
                except Exception:
                    continue
            if values:
                return np.asarray(values, dtype=np.float32)

    values = _read_numeric_txt(path)
    if values.size == 0:
        raise ValueError(f"No numeric values found in BVP CSV: {path}")
    return values.astype(np.float32)


def _read_image_frames(image_paths: Sequence[Path], max_frames: Optional[int] = None) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for fp in image_paths:
        if max_frames is not None and len(frames) >= max_frames:
            break
        frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)
    return frames


def _extract_roi_sequence(frames_bgr: Sequence[np.ndarray], target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    roi_extractor = FaceROIExtractor(target_size=target_size)
    rois: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for fr in frames_bgr:
        roi_rgb, roi_mask = roi_extractor.extract(fr)
        rois.append(roi_rgb)
        masks.append(roi_mask)

    video_thwc = np.stack(rois, axis=0).astype(np.float32)  # [T,H,W,3]
    roi_mask_hw = np.mean(np.stack(masks, axis=0).astype(np.float32), axis=0)  # [H,W]
    roi_mask_hw = (roi_mask_hw > 0.5).astype(np.float32)
    return video_thwc, roi_mask_hw


def _find_ibvp_csv(rgb_dir: Path) -> Optional[Path]:
    session_name = rgb_dir.name.removesuffix("_rgb")
    candidates = [
        rgb_dir.parent / f"{session_name}_bvp.csv",
        rgb_dir.parent / f"{rgb_dir.parent.name}_bvp.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(rgb_dir.parent.glob("*_bvp.csv"))
    return matches[0] if matches else None


def _resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    if signal.size == target_len:
        return signal.astype(np.float32)

    if signal.size < 2:
        return np.zeros((target_len,), dtype=np.float32)

    x_old = np.linspace(0.0, 1.0, signal.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=True)
    out = np.interp(x_new, x_old, signal.astype(np.float32))
    return out.astype(np.float32)


def _window_indices(total_frames: int, clip_len: int, stride: int) -> List[Tuple[int, int]]:
    if total_frames < clip_len:
        return [(0, total_frames)]
    out: List[Tuple[int, int]] = []
    start = 0
    while start + clip_len <= total_frames:
        out.append((start, start + clip_len))
        start += stride
    if out and out[-1][1] < total_frames:
        out.append((total_frames - clip_len, total_frames))
    return out


def _save_clip_npz(
    out_path: Path,
    video_thwc: np.ndarray,
    roi_mask_hw: np.ndarray,
    bvp_t: Optional[np.ndarray],
    skin_label: Optional[int] = None,
    light_label: Optional[int] = None,
    clinical: Optional[np.ndarray] = None,
) -> None:
    payload = {
        "video": video_thwc.astype(np.float32),
        "roi_mask": roi_mask_hw.astype(np.float32),
    }
    if bvp_t is not None:
        payload["bvp"] = bvp_t.astype(np.float32)
    if skin_label is not None:
        payload["skin_label"] = np.asarray(skin_label, dtype=np.int64)
    if light_label is not None:
        payload["light_label"] = np.asarray(light_label, dtype=np.int64)
    if clinical is not None:
        payload["clinical"] = np.asarray(clinical, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **payload)


def preprocess_ubfc(
    ubfc_root: Path,
    out_root: Path,
    target_size: int,
    clip_len: int,
    stride: int,
    max_frames_per_subject: Optional[int],
) -> None:
    subject_dirs = sorted([p for p in ubfc_root.iterdir() if p.is_dir()])
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found in UBFC root: {ubfc_root}")

    for subj in subject_dirs:
        video_candidates = sorted(subj.rglob("*.avi"))
        gt_candidates = sorted(subj.rglob("*.txt"))
        if not video_candidates or not gt_candidates:
            continue

        video_path = video_candidates[0]
        gt_path = gt_candidates[0]

        frames = _read_video_frames(video_path, max_frames=max_frames_per_subject)
        if not frames:
            continue

        video_thwc, roi_mask = _extract_roi_sequence(frames, target_size=target_size)
        gt_raw = _read_numeric_txt(gt_path)
        bvp_full = _resample_signal(gt_raw, target_len=video_thwc.shape[0])

        windows = _window_indices(video_thwc.shape[0], clip_len=clip_len, stride=stride)
        for i, (s, e) in enumerate(windows):
            clip_video = video_thwc[s:e]
            clip_bvp = _resample_signal(bvp_full[s:e], target_len=clip_len)
            if clip_video.shape[0] < clip_len:
                pad_n = clip_len - clip_video.shape[0]
                clip_video = np.concatenate([clip_video, np.repeat(clip_video[-1:], pad_n, axis=0)], axis=0)

            out_path = out_root / "ubfc" / subj.name / f"clip_{i:04d}.npz"
            _save_clip_npz(out_path, clip_video, roi_mask, clip_bvp)


def _extract_mat_signal_and_labels(mat_dict: dict) -> Tuple[np.ndarray, int, int]:
    # Heuristic keys for pulse signal in MMPD-style .mat files.
    signal_keys = [
        "GT_ppg", "gt_ppg", "bvp", "pulse", "ppg", "rppg", "wave", "gtTrace", "gttrace", "phys",
    ]
    skin_keys = ["skin_color", "skin", "skin_label", "fitzpatrick", "fitz", "skintone"]
    light_keys = ["light", "lighting", "illum", "illumination", "light_label"]

    signal = None
    for k in signal_keys:
        if k in mat_dict:
            arr = np.asarray(mat_dict[k]).squeeze()
            if arr.size > 0:
                signal = arr.astype(np.float32)
                break

    if signal is None:
        # Fallback: first non-trivial numeric array.
        for k, v in mat_dict.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v).squeeze()
            if np.issubdtype(arr.dtype, np.number) and arr.size > 32:
                signal = arr.astype(np.float32)
                break

    if signal is None:
        raise KeyError("No pulse-like signal found in .mat")

    def _find_label(keys: Sequence[str], default: int, map_light_text: bool = False) -> int:
        for k in keys:
            if k in mat_dict:
                arr = np.asarray(mat_dict[k]).reshape(-1)
                if arr.size > 0:
                    v = arr[0]
                    if map_light_text and isinstance(v, (str, np.str_)):
                        light_map = {
                            "low": 0,
                            "normal": 1,
                            "indoor": 1,
                            "bright": 2,
                            "strong": 2,
                            "sunlight": 3,
                            "outdoor": 3,
                        }
                        return light_map.get(str(v).strip().lower(), default)
                    try:
                        return int(v)
                    except Exception:
                        return default
        return default

    skin_label = _find_label(skin_keys, default=0)
    light_label = _find_label(light_keys, default=0, map_light_text=True)
    return signal, skin_label, light_label


def _mat_video_to_frames_bgr(video_arr: np.ndarray, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Convert MAT `video` array to list of BGR uint8 frames for ROI extraction.

    Expected shape [T,H,W,3] (common in MMPD), but supports [H,W,3,T] as fallback.
    """
    arr = np.asarray(video_arr)
    if arr.ndim != 4:
        raise ValueError(f"Unexpected MAT video ndim: {arr.ndim}")

    if arr.shape[-1] == 3:
        thwc = arr
    elif arr.shape[2] == 3:
        # [H,W,3,T] -> [T,H,W,3]
        thwc = np.transpose(arr, (3, 0, 1, 2))
    else:
        raise ValueError(f"Could not infer color axis for MAT video shape: {arr.shape}")

    if max_frames is not None:
        thwc = thwc[:max_frames]

    thwc = thwc.astype(np.float32)
    vmax = float(np.nanmax(thwc)) if thwc.size > 0 else 0.0
    if vmax <= 1.5:
        thwc = thwc * 255.0
    thwc = np.clip(thwc, 0.0, 255.0).astype(np.uint8)

    frames_bgr: List[np.ndarray] = []
    for rgb in thwc:
        frames_bgr.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return frames_bgr


def preprocess_mmpd(
    mmpd_root: Path,
    out_root: Path,
    target_size: int,
    clip_len: int,
    stride: int,
    max_frames_per_subject: Optional[int],
) -> None:
    mat_files = sorted(mmpd_root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in MMPD root: {mmpd_root}")

    n_total = 0
    n_processed = 0
    n_skip_no_video = 0
    n_skip_no_signal = 0

    for mat_fp in mat_files:
        n_total += 1
        subj_name = mat_fp.parent.name

        mat_dict = loadmat(str(mat_fp))
        try:
            signal_raw, skin_label, light_label = _extract_mat_signal_and_labels(mat_dict)
        except Exception:
            n_skip_no_signal += 1
            continue

        # Preferred path: external video next to .mat; fallback path: embedded MAT video.
        local_videos = sorted(mat_fp.parent.glob("*.avi")) + sorted(mat_fp.parent.glob("*.mp4"))
        frames: List[np.ndarray] = []
        if local_videos:
            video_path = local_videos[0]
            frames = _read_video_frames(video_path, max_frames=max_frames_per_subject)
        elif "video" in mat_dict:
            try:
                frames = _mat_video_to_frames_bgr(mat_dict["video"], max_frames=max_frames_per_subject)
            except Exception:
                frames = []

        if not frames:
            n_skip_no_video += 1
            continue

        video_thwc, roi_mask = _extract_roi_sequence(frames, target_size=target_size)
        bvp_full = _resample_signal(signal_raw, target_len=video_thwc.shape[0])

        windows = _window_indices(video_thwc.shape[0], clip_len=clip_len, stride=stride)
        for i, (s, e) in enumerate(windows):
            clip_video = video_thwc[s:e]
            clip_bvp = _resample_signal(bvp_full[s:e], target_len=clip_len)
            if clip_video.shape[0] < clip_len:
                pad_n = clip_len - clip_video.shape[0]
                clip_video = np.concatenate([clip_video, np.repeat(clip_video[-1:], pad_n, axis=0)], axis=0)

            out_path = out_root / "mmpd" / subj_name / f"{mat_fp.stem}_clip_{i:04d}.npz"
            _save_clip_npz(
                out_path,
                clip_video,
                roi_mask,
                clip_bvp,
                skin_label=skin_label,
                light_label=light_label,
            )
            n_processed += 1

    print(
        f"[MMPD] summary: mats={n_total}, clips_written={n_processed}, "
        f"skipped_no_video={n_skip_no_video}, skipped_no_signal={n_skip_no_signal}"
    )


def _parse_mcd_clinical_from_filename(name: str) -> Optional[np.ndarray]:
    # Optional helper when label files are not yet integrated.
    # If filename embeds numbers like ..._SBP120_DBP80_HB13.2..., parse them.
    import re

    vals = re.findall(r"[-+]?\d*\.?\d+", name)
    if len(vals) >= 3:
        try:
            return np.asarray([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)
        except Exception:
            return None
    return None


def _load_mcd_db_map(mcd_root: Path) -> dict:
    db_path = mcd_root / "db.csv"
    if not db_path.exists():
        return {}

    out = {}
    with db_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ppg_key = str(row.get("ppg", "")).replace("\\", "/").lower()
            video_key = str(row.get("video", "")).replace("\\", "/").lower()

            try:
                clinical = np.asarray(
                    [
                        float(row.get("upper_ap", 0.0)),
                        float(row.get("lower_ap", 0.0)),
                        float(row.get("hemoglobin", 0.0)),
                    ],
                    dtype=np.float32,
                )
            except Exception:
                clinical = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

            if ppg_key:
                out[("ppg", ppg_key)] = clinical
            if video_key:
                out[("video", video_key)] = clinical
    return out


def preprocess_mcd(
    mcd_root: Path,
    out_root: Path,
    target_size: int,
    clip_len: int,
    stride: int,
    max_frames_per_subject: Optional[int],
) -> None:
    pw_files = sorted(mcd_root.rglob("*.pw"))
    if not pw_files:
        raise FileNotFoundError(f"No .pw files found in MCD root: {mcd_root}")

    db_map = _load_mcd_db_map(mcd_root)

    # Build global video index by basename for datasets where ppg and video are split folders.
    all_videos = sorted(mcd_root.rglob("*.avi")) + sorted(mcd_root.rglob("*.mp4"))
    videos_by_name = {v.name.lower(): v for v in all_videos}

    n_total = 0
    n_processed = 0
    n_skip_no_video = 0
    n_skip_bad_pw = 0

    for pw_fp in pw_files:
        n_total += 1
        subj_name = pw_fp.stem.split("_")[0]

        # Candidate 1: same folder pairing.
        local_videos = sorted(pw_fp.parent.glob("*.avi")) + sorted(pw_fp.parent.glob("*.mp4"))

        # Candidate 2: split-folder pairing by shared stem prefix (e.g., 1020_after).
        stem = pw_fp.stem
        parts = stem.split("_")
        subj_id = parts[0]
        suffix = parts[-1] if len(parts) > 1 else ""

        # Handles names like: 1020_after.PW -> 1020_FullHDwebcam_after.avi
        if suffix:
            global_candidates = sorted(mcd_root.glob(f"**/{subj_id}_*_{suffix}.avi")) + sorted(mcd_root.glob(f"**/{subj_id}_*_{suffix}.mp4"))
        else:
            global_candidates = sorted(mcd_root.glob(f"**/{subj_id}_*.avi")) + sorted(mcd_root.glob(f"**/{subj_id}_*.mp4"))

        candidates = local_videos if local_videos else global_candidates
        if not candidates:
            # Candidate 3: direct exact-name fallback with common camera suffixes.
            fallback_names = [
                f"{subj_id}_FullHDwebcam_{suffix}.avi",
                f"{subj_id}_USBVideo_{suffix}.avi",
                f"{subj_id}_IriunWebcam_{suffix}.avi",
            ]
            for nm in fallback_names:
                key = nm.lower()
                if key in videos_by_name:
                    candidates = [videos_by_name[key]]
                    break

        if not candidates:
            n_skip_no_video += 1
            continue

        # Prefer FullHD webcam stream if available, else first available camera stream.
        candidates = sorted(candidates, key=lambda p: ("fullhdwebcam" not in p.name.lower(), p.name.lower()))
        video_path = candidates[0]

        frames = _read_video_frames(video_path, max_frames=max_frames_per_subject)
        if not frames:
            continue

        try:
            pw_signal = _read_pw_file(pw_fp)
        except Exception:
            n_skip_bad_pw += 1
            continue

        video_thwc, roi_mask = _extract_roi_sequence(frames, target_size=target_size)
        bvp_full = _resample_signal(pw_signal, target_len=video_thwc.shape[0])

        ppg_rel = str(pw_fp.relative_to(mcd_root)).replace("\\", "/").lower()
        video_rel = str(video_path.relative_to(mcd_root)).replace("\\", "/").lower()
        clinical = db_map.get(("ppg", ppg_rel))
        if clinical is None:
            clinical = db_map.get(("video", video_rel))
        if clinical is None:
            clinical = _parse_mcd_clinical_from_filename(pw_fp.stem)
        if clinical is None:
            clinical = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        windows = _window_indices(video_thwc.shape[0], clip_len=clip_len, stride=stride)
        for i, (s, e) in enumerate(windows):
            clip_video = video_thwc[s:e]
            clip_bvp = _resample_signal(bvp_full[s:e], target_len=clip_len)
            if clip_video.shape[0] < clip_len:
                pad_n = clip_len - clip_video.shape[0]
                clip_video = np.concatenate([clip_video, np.repeat(clip_video[-1:], pad_n, axis=0)], axis=0)

            out_path = out_root / "mcd" / subj_name / f"{pw_fp.stem}_clip_{i:04d}.npz"
            _save_clip_npz(
                out_path,
                clip_video,
                roi_mask,
                clip_bvp,
                clinical=clinical,
            )
            n_processed += 1

    print(
        f"[MCD] summary: pw={n_total}, clips_written={n_processed}, "
        f"skipped_no_video={n_skip_no_video}, skipped_bad_pw={n_skip_bad_pw}"
    )


def preprocess_ibvp(
    ibvp_root: Path,
    out_root: Path,
    target_size: int,
    clip_len: int,
    stride: int,
    max_frames_per_subject: Optional[int],
) -> None:
    rgb_dirs = sorted([p for p in ibvp_root.rglob("*_rgb") if p.is_dir()])
    if not rgb_dirs:
        raise FileNotFoundError(f"No IBVP RGB folders found in root: {ibvp_root}")

    n_total = 0
    n_processed = 0
    n_skip_no_signal = 0
    n_skip_no_frames = 0

    for rgb_dir in rgb_dirs:
        n_total += 1
        csv_path = _find_ibvp_csv(rgb_dir)
        if csv_path is None:
            n_skip_no_signal += 1
            continue

        image_paths = sorted(rgb_dir.glob("*.bmp"))
        if max_frames_per_subject is not None:
            image_paths = image_paths[:max_frames_per_subject]
        frames = _read_image_frames(image_paths, max_frames=max_frames_per_subject)
        if not frames:
            n_skip_no_frames += 1
            continue

        try:
            bvp_raw = _read_bvp_csv(csv_path)
        except Exception:
            n_skip_no_signal += 1
            continue

        video_thwc, roi_mask = _extract_roi_sequence(frames, target_size=target_size)
        bvp_full = _resample_signal(bvp_raw, target_len=video_thwc.shape[0])

        rel_parts = rgb_dir.relative_to(ibvp_root).parts
        subject_name = rel_parts[0] if rel_parts else rgb_dir.parent.parent.name
        session_name = rgb_dir.name.removesuffix("_rgb")

        windows = _window_indices(video_thwc.shape[0], clip_len=clip_len, stride=stride)
        for i, (s, e) in enumerate(windows):
            clip_video = video_thwc[s:e]
            clip_bvp = _resample_signal(bvp_full[s:e], target_len=clip_len)
            if clip_video.shape[0] < clip_len:
                pad_n = clip_len - clip_video.shape[0]
                clip_video = np.concatenate([clip_video, np.repeat(clip_video[-1:], pad_n, axis=0)], axis=0)

            out_path = out_root / "ibvp" / subject_name / session_name / f"clip_{i:04d}.npz"
            _save_clip_npz(out_path, clip_video, roi_mask, clip_bvp)
            n_processed += 1

    print(
        f"[IBVP] summary: sessions={n_total}, clips_written={n_processed}, "
        f"skipped_no_signal={n_skip_no_signal}, skipped_no_frames={n_skip_no_frames}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw UBFC/MMPD/MCD datasets into EquiPhys NPZ clips")
    p.add_argument("--ubfc-root", type=str, default=None, help="Root directory for UBFC raw data (.avi + .txt)")
    p.add_argument("--mmpd-root", type=str, default=None, help="Root directory for MMPD raw data (.mat + video)")
    p.add_argument("--mcd-root", type=str, default=None, help="Root directory for MCD raw data (.pw + video)")
    p.add_argument("--ibvp-root", type=str, default=None, help="Root directory for IBVP raw data (.bmp RGB + .raw thermal + BVP CSV)")
    p.add_argument("--out-root", type=str, required=True, help="Output directory for generated NPZ clips")
    p.add_argument("--target-size", type=int, default=64, help="ROI spatial resolution")
    p.add_argument("--clip-len", type=int, default=150, help="Temporal length per clip")
    p.add_argument("--stride", type=int, default=75, help="Sliding window stride")
    p.add_argument("--max-frames-per-subject", type=int, default=None, help="Optional cap for quick preprocessing tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not any([args.ubfc_root, args.mmpd_root, args.mcd_root, args.ibvp_root]):
        print("Provide at least one dataset root: --ubfc-root / --mmpd-root / --mcd-root / --ibvp-root")
        return

    if args.ubfc_root:
        print(f"[UBFC] preprocessing from {args.ubfc_root}")
        preprocess_ubfc(
            ubfc_root=Path(args.ubfc_root),
            out_root=out_root,
            target_size=args.target_size,
            clip_len=args.clip_len,
            stride=args.stride,
            max_frames_per_subject=args.max_frames_per_subject,
        )
        print("[UBFC] done")

    if args.mmpd_root:
        print(f"[MMPD] preprocessing from {args.mmpd_root}")
        preprocess_mmpd(
            mmpd_root=Path(args.mmpd_root),
            out_root=out_root,
            target_size=args.target_size,
            clip_len=args.clip_len,
            stride=args.stride,
            max_frames_per_subject=args.max_frames_per_subject,
        )
        print("[MMPD] done")

    if args.mcd_root:
        print(f"[MCD] preprocessing from {args.mcd_root}")
        preprocess_mcd(
            mcd_root=Path(args.mcd_root),
            out_root=out_root,
            target_size=args.target_size,
            clip_len=args.clip_len,
            stride=args.stride,
            max_frames_per_subject=args.max_frames_per_subject,
        )
        print("[MCD] done")

    if args.ibvp_root:
        print(f"[IBVP] preprocessing from {args.ibvp_root}")
        preprocess_ibvp(
            ibvp_root=Path(args.ibvp_root),
            out_root=out_root,
            target_size=args.target_size,
            clip_len=args.clip_len,
            stride=args.stride,
            max_frames_per_subject=args.max_frames_per_subject,
        )
        print("[IBVP] done")

    print(f"All preprocessing complete. NPZ saved under: {out_root}")


if __name__ == "__main__":
    main()
