from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset_connectors import NPZClipDataset
from equiphys_core import (
    ClinicalRegressor,
    DANNTrainingLoss,
    EquiPhysDANN,
    LDSConfig,
    LDSWeightedMSELoss,
    PhysiologyInformedLoss,
    build_lds_weights,
    estimate_hr_bpm_from_bvp,
    rppg_collate_fn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EquiPhys on preprocessed NPZ datasets with validation metrics")
    p.add_argument("--ubfc-root", type=str, default=None)
    p.add_argument("--ibvp-root", type=str, default=None)
    p.add_argument("--mmpd-root", type=str, default=None)
    p.add_argument("--mcd-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-steps-ubfc", type=int, default=-1, help="-1 means full train loader")
    p.add_argument("--max-steps-ibvp", type=int, default=-1, help="-1 means full train loader")
    p.add_argument("--max-steps-mmpd", type=int, default=-1, help="-1 means full train loader")
    p.add_argument("--val-max-steps-ubfc", type=int, default=-1, help="-1 means full validation loader")
    p.add_argument("--val-max-steps-ibvp", type=int, default=-1, help="-1 means full validation loader")
    p.add_argument("--val-max-steps-mmpd", type=int, default=-1, help="-1 means full validation loader")
    p.add_argument("--max-steps-mcd", type=int, default=-1, help="-1 means full clinical train loader")
    p.add_argument("--val-max-steps-mcd", type=int, default=-1, help="-1 means full clinical validation loader")
    p.add_argument("--lambda-grl", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-mode", type=str, default="subject", choices=["subject", "clip"], help="Validation split strategy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-fps", type=float, default=30.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ubfc-warmup-epochs", type=int, default=0, help="Train UBFC-only for initial epochs before enabling MMPD")
    p.add_argument("--train-fps", type=float, default=30.0, help="Frame rate used for HR-supervision spectral losses")
    p.add_argument("--lambda-hr", type=float, default=0.3, help="Weight for differentiable HR supervision loss")
    p.add_argument("--mcd-epochs", type=int, default=8, help="Clinical fine-tuning epochs on MCD")
    p.add_argument("--clinical-lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_indices(n: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n > n_val else idx[:]
    return train_idx, val_idx


def _split_indices_by_subject(files: List[Path], dataset_root: Path, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Subject-wise split to reduce leakage and improve clinical validity.

    Subject id is inferred as the first path segment under dataset root, e.g.
    `.../ubfc/subject01/clip_0001.npz` -> `subject01`.
    """
    subject_to_indices: Dict[str, List[int]] = {}
    for i, fp in enumerate(files):
        rel = fp.relative_to(dataset_root)
        subject = rel.parts[0] if len(rel.parts) > 1 else fp.parent.name
        subject_to_indices.setdefault(subject, []).append(i)

    subjects = list(subject_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val_subjects = max(1, int(round(len(subjects) * val_ratio))) if len(subjects) > 1 else 0
    val_subjects = set(subjects[:n_val_subjects])

    tr_idx: List[int] = []
    va_idx: List[int] = []
    for s, idxs in subject_to_indices.items():
        if s in val_subjects:
            va_idx.extend(idxs)
        else:
            tr_idx.extend(idxs)

    # Fallback for tiny datasets.
    if len(va_idx) == 0 and len(tr_idx) > 1:
        va_idx.append(tr_idx.pop())
    if len(tr_idx) == 0 and len(va_idx) > 1:
        tr_idx.append(va_idx.pop())

    return tr_idx, va_idx


def _clinical_hr_metrics(pred_hr: torch.Tensor, gt_hr: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute clinically interpretable HR metrics from sample-wise HR arrays."""
    pred_np = pred_hr.detach().float().cpu().numpy().reshape(-1)
    gt_np = gt_hr.detach().float().cpu().numpy().reshape(-1)

    if pred_np.size == 0:
        return {
            f"{prefix}_hr_mae": 0.0,
            f"{prefix}_hr_rmse": 0.0,
            f"{prefix}_hr_mape": 0.0,
            f"{prefix}_hr_bias": 0.0,
            f"{prefix}_hr_loa_low": 0.0,
            f"{prefix}_hr_loa_high": 0.0,
            f"{prefix}_hr_pearson": 0.0,
            f"{prefix}_hr_ccc": 0.0,
            f"{prefix}_hr_within_5": 0.0,
            f"{prefix}_hr_within_10": 0.0,
        }

    diff = pred_np - gt_np
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mape = float(np.mean(np.abs(diff) / np.clip(np.abs(gt_np), 1e-6, None)) * 100.0)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    loa_low = float(bias - 1.96 * sd)
    loa_high = float(bias + 1.96 * sd)

    if pred_np.size > 1 and float(np.std(pred_np)) > 1e-8 and float(np.std(gt_np)) > 1e-8:
        pearson = float(np.corrcoef(pred_np, gt_np)[0, 1])
        if not np.isfinite(pearson):
            pearson = 0.0
    else:
        pearson = 0.0

    mu_x = float(np.mean(pred_np))
    mu_y = float(np.mean(gt_np))
    var_x = float(np.var(pred_np))
    var_y = float(np.var(gt_np))
    cov_xy = float(np.mean((pred_np - mu_x) * (gt_np - mu_y)))
    ccc_den = var_x + var_y + (mu_x - mu_y) ** 2
    ccc = float((2.0 * cov_xy) / ccc_den) if ccc_den > 1e-12 else 0.0

    within5 = float(np.mean(np.abs(diff) <= 5.0))
    within10 = float(np.mean(np.abs(diff) <= 10.0))

    return {
        f"{prefix}_hr_mae": mae,
        f"{prefix}_hr_rmse": rmse,
        f"{prefix}_hr_mape": mape,
        f"{prefix}_hr_bias": bias,
        f"{prefix}_hr_loa_low": loa_low,
        f"{prefix}_hr_loa_high": loa_high,
        f"{prefix}_hr_pearson": pearson,
        f"{prefix}_hr_ccc": ccc,
        f"{prefix}_hr_within_5": within5,
        f"{prefix}_hr_within_10": within10,
    }


def _build_loader(ds, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=rppg_collate_fn,
        drop_last=False,
    )


def _build_train_val_loaders(args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, int]]:
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    counts: Dict[str, int] = {}

    if args.ubfc_root:
        ds = NPZClipDataset(
            root=args.ubfc_root,
            frames=args.frames,
            require_bvp=True,
            require_domain_labels=False,
            require_clinical=False,
        )
        if args.split_mode == "subject":
            tr_idx, va_idx = _split_indices_by_subject(ds.files, ds.root, args.val_ratio, args.seed)
        else:
            tr_idx, va_idx = _split_indices(len(ds), args.val_ratio, args.seed)
        tr_ds = Subset(ds, tr_idx)
        va_ds = Subset(ds, va_idx)
        train_loaders["ubfc"] = _build_loader(tr_ds, args.batch_size, args.num_workers, shuffle=True)
        val_loaders["ubfc"] = _build_loader(va_ds, args.batch_size, args.num_workers, shuffle=False)
        counts["ubfc_train"] = len(tr_ds)
        counts["ubfc_val"] = len(va_ds)

    if args.ibvp_root:
        ds = NPZClipDataset(
            root=args.ibvp_root,
            frames=args.frames,
            require_bvp=True,
            require_domain_labels=False,
            require_clinical=False,
        )
        if args.split_mode == "subject":
            tr_idx, va_idx = _split_indices_by_subject(ds.files, ds.root, args.val_ratio, args.seed + 3)
        else:
            tr_idx, va_idx = _split_indices(len(ds), args.val_ratio, args.seed + 3)
        tr_ds = Subset(ds, tr_idx)
        va_ds = Subset(ds, va_idx)
        train_loaders["ibvp"] = _build_loader(tr_ds, args.batch_size, args.num_workers, shuffle=True)
        val_loaders["ibvp"] = _build_loader(va_ds, args.batch_size, args.num_workers, shuffle=False)
        counts["ibvp_train"] = len(tr_ds)
        counts["ibvp_val"] = len(va_ds)

    if args.mmpd_root:
        ds = NPZClipDataset(
            root=args.mmpd_root,
            frames=args.frames,
            require_bvp=True,
            require_domain_labels=True,
            require_clinical=False,
        )
        if args.split_mode == "subject":
            tr_idx, va_idx = _split_indices_by_subject(ds.files, ds.root, args.val_ratio, args.seed + 1)
        else:
            tr_idx, va_idx = _split_indices(len(ds), args.val_ratio, args.seed + 1)
        tr_ds = Subset(ds, tr_idx)
        va_ds = Subset(ds, va_idx)
        train_loaders["mmpd"] = _build_loader(tr_ds, args.batch_size, args.num_workers, shuffle=True)
        val_loaders["mmpd"] = _build_loader(va_ds, args.batch_size, args.num_workers, shuffle=False)
        counts["mmpd_train"] = len(tr_ds)
        counts["mmpd_val"] = len(va_ds)

    if args.mcd_root:
        ds = NPZClipDataset(
            root=args.mcd_root,
            frames=args.frames,
            require_bvp=False,
            require_domain_labels=False,
            require_clinical=True,
        )
        if args.split_mode == "subject":
            tr_idx, va_idx = _split_indices_by_subject(ds.files, ds.root, args.val_ratio, args.seed + 2)
        else:
            tr_idx, va_idx = _split_indices(len(ds), args.val_ratio, args.seed + 2)
        tr_ds = Subset(ds, tr_idx)
        va_ds = Subset(ds, va_idx)
        train_loaders["mcd"] = _build_loader(tr_ds, args.batch_size, args.num_workers, shuffle=False)
        val_loaders["mcd"] = _build_loader(va_ds, args.batch_size, args.num_workers, shuffle=False)
        counts["mcd_train"] = len(tr_ds)
        counts["mcd_val"] = len(va_ds)

    return train_loaders, val_loaders, counts


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _mean_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred.pow(2).sum(dim=1) + eps) * (target.pow(2).sum(dim=1) + eps))
    return (num / den).mean()


def _run_ubfc_train_epoch(
    model: EquiPhysDANN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: PhysiologyInformedLoss,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
    grad_clip: float,
    train_fps: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    hr_loss_acc = 0.0
    n = 0

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break

        b = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(b["video"])
            loss_dict = loss_fn(
                bvp_pred=out["bvp_pred"],
                bvp_target=b["bvp"],
                video=b["video"],
                attn_map=out["attn_map"],
                roi_mask=b["roi_mask"],
                fps=train_fps,
            )
            loss = loss_dict["loss_total"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())
        hr_loss_acc += float(loss_dict.get("loss_hr", torch.tensor(0.0, device=loss.device)).detach().item())
        n += 1

    den = max(1, n)
    return {
        "train_ubfc_loss": total / den,
        "train_ubfc_hr_loss": hr_loss_acc / den,
        "train_ubfc_steps": float(n),
    }


def _run_mmpd_train_epoch(
    model: EquiPhysDANN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: DANNTrainingLoss,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
    grad_clip: float,
    train_fps: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    skin = 0.0
    light = 0.0
    n = 0

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break

        b = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(b["video"])
            loss_dict = loss_fn(
                model_out=out,
                bvp_target=b["bvp"],
                video=b["video"],
                roi_mask=b["roi_mask"],
                skin_label=b["skin_label"],
                light_label=b["light_label"],
                fps=train_fps,
            )
            loss = loss_dict["loss_total"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())
        skin += float(loss_dict["loss_skin"].detach().item())
        light += float(loss_dict["loss_light"].detach().item())
        n += 1

    den = max(1, n)
    return {
        "train_mmpd_loss": total / den,
        "train_mmpd_skin_ce": skin / den,
        "train_mmpd_light_ce": light / den,
        "train_mmpd_steps": float(n),
    }


@torch.no_grad()
def _run_ubfc_val_epoch(
    model: EquiPhysDANN,
    loader: DataLoader,
    loss_fn: PhysiologyInformedLoss,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
    eval_fps: float,
) -> Dict[str, float]:
    model.eval()

    n = 0
    l_total = 0.0
    l_wave = 0.0
    l_green = 0.0
    l_roi = 0.0
    l_hr = 0.0
    pear = 0.0
    rmse = 0.0
    pred_hr_all: List[torch.Tensor] = []
    gt_hr_all: List[torch.Tensor] = []

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break

        b = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(b["video"])
            d = loss_fn(
                bvp_pred=out["bvp_pred"],
                bvp_target=b["bvp"],
                video=b["video"],
                attn_map=out["attn_map"],
                roi_mask=b["roi_mask"],
                fps=eval_fps,
            )

        pred = out["bvp_pred"]
        gt = b["bvp"]
        err = pred - gt

        pred_hr = estimate_hr_bpm_from_bvp(pred.float(), fps=eval_fps)
        gt_hr = estimate_hr_bpm_from_bvp(gt.float(), fps=eval_fps)
        hr_diff = pred_hr - gt_hr

        l_total += float(d["loss_total"].item())
        l_wave += float(d["loss_wave"].item())
        l_green += float(d["loss_green"].item())
        l_roi += float(d["loss_roi"].item())
        l_hr += float(d.get("loss_hr", torch.tensor(0.0, device=pred.device)).item())
        pear += float(_mean_pearson(pred, gt).item())
        rmse += float(torch.sqrt(torch.mean(err.pow(2))).item())
        pred_hr_all.append(pred_hr.detach().cpu())
        gt_hr_all.append(gt_hr.detach().cpu())
        n += 1

    den = max(1, n)
    out = {
        "val_ubfc_loss": l_total / den,
        "val_ubfc_wave_loss": l_wave / den,
        "val_ubfc_green_loss": l_green / den,
        "val_ubfc_roi_loss": l_roi / den,
        "val_ubfc_hr_loss": l_hr / den,
        "val_ubfc_pearson": pear / den,
        "val_ubfc_rmse": rmse / den,
        "val_ubfc_steps": float(n),
    }
    if pred_hr_all and gt_hr_all:
        out.update(_clinical_hr_metrics(torch.cat(pred_hr_all), torch.cat(gt_hr_all), prefix="val_ubfc"))
    return out


@torch.no_grad()
def _run_mmpd_val_epoch(
    model: EquiPhysDANN,
    loader: DataLoader,
    loss_fn: DANNTrainingLoss,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
    eval_fps: float,
) -> Dict[str, float]:
    model.eval()

    n = 0
    l_total = 0.0
    l_skin = 0.0
    l_light = 0.0
    skin_correct = 0.0
    light_correct = 0.0
    skin_count = 0.0
    light_count = 0.0
    pred_hr_all: List[torch.Tensor] = []
    gt_hr_all: List[torch.Tensor] = []

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break

        b = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(b["video"])
            d = loss_fn(
                model_out=out,
                bvp_target=b["bvp"],
                video=b["video"],
                roi_mask=b["roi_mask"],
                skin_label=b["skin_label"],
                light_label=b["light_label"],
                fps=eval_fps,
            )

        skin_pred = torch.argmax(out["skin_logits"], dim=1)
        light_pred = torch.argmax(out["light_logits"], dim=1)

        skin_correct += float((skin_pred == b["skin_label"]).sum().item())
        light_correct += float((light_pred == b["light_label"]).sum().item())
        skin_count += float(b["skin_label"].numel())
        light_count += float(b["light_label"].numel())

        pred_hr = estimate_hr_bpm_from_bvp(out["bvp_pred"].float(), fps=eval_fps)
        gt_hr = estimate_hr_bpm_from_bvp(b["bvp"].float(), fps=eval_fps)
        hr_diff = pred_hr - gt_hr

        l_total += float(d["loss_total"].item())
        l_skin += float(d["loss_skin"].item())
        l_light += float(d["loss_light"].item())
        pred_hr_all.append(pred_hr.detach().cpu())
        gt_hr_all.append(gt_hr.detach().cpu())
        n += 1

    den = max(1, n)
    out = {
        "val_mmpd_loss": l_total / den,
        "val_mmpd_skin_ce": l_skin / den,
        "val_mmpd_light_ce": l_light / den,
        "val_mmpd_skin_acc": (skin_correct / max(1.0, skin_count)),
        "val_mmpd_light_acc": (light_correct / max(1.0, light_count)),
        "val_mmpd_steps": float(n),
    }
    if pred_hr_all and gt_hr_all:
        out.update(_clinical_hr_metrics(torch.cat(pred_hr_all), torch.cat(gt_hr_all), prefix="val_mmpd"))
    return out


def _select_score(log: Dict[str, float]) -> float:
    # Lower is better.
    if "val_mcd_mae_mean" in log:
        score = float(log["val_mcd_mae_mean"])
    elif "val_ibvp_hr_mae" in log:
        score = float(log["val_ibvp_hr_mae"])
    elif "val_ubfc_hr_mae" in log:
        score = float(log["val_ubfc_hr_mae"])
    elif "val_mmpd_hr_mae" in log:
        score = float(log["val_mmpd_hr_mae"])
    elif "val_mmpd_loss" in log:
        score = float(log["val_mmpd_loss"])
    else:
        score = float(log.get("train_ubfc_loss", log.get("train_mmpd_loss", 1e9)))
    return score


def _collect_clinical_labels(loader: DataLoader) -> np.ndarray:
    labels: List[np.ndarray] = []
    for batch in loader:
        if "clinical" in batch:
            c = batch["clinical"].numpy().astype(np.float32)
            finite_rows = np.isfinite(c).all(axis=1)
            if np.any(finite_rows):
                labels.append(c[finite_rows])
    if not labels:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(labels, axis=0).astype(np.float32)


def _run_mcd_train_epoch(
    model: ClinicalRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: LDSWeightedMSELoss,
    sample_weights: np.ndarray,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    mae_acc = np.zeros((3,), dtype=np.float64)
    n = 0
    offset = 0

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break

        b = _move_batch(batch, device)
        finite_clin = torch.isfinite(b["clinical"]).all(dim=1)
        finite_video = torch.isfinite(b["video"]).view(b["video"].shape[0], -1).all(dim=1)
        finite_rows = finite_clin & finite_video
        if int(finite_rows.sum().item()) == 0:
            continue
        b_video = b["video"][finite_rows]
        b_clinical = b["clinical"][finite_rows]

        batch_size = int(b_clinical.shape[0])
        w_np = sample_weights[offset: offset + batch_size]
        offset += batch_size
        if w_np.size != batch_size:
            w_np = np.ones((batch_size,), dtype=np.float32)
        w_np = np.where(np.isfinite(w_np), w_np, 1.0).astype(np.float32)
        w = torch.from_numpy(w_np.astype(np.float32)).to(device)

        optimizer.zero_grad(set_to_none=True)
        # Keep clinical regression in full precision for numeric stability.
        with torch.autocast(device_type="cuda", enabled=False):
            pred = model(b_video)
            valid_pred_rows = torch.isfinite(pred).all(dim=1)
            if int(valid_pred_rows.sum().item()) == 0:
                continue
            pred = pred[valid_pred_rows]
            tgt = b_clinical[valid_pred_rows]
            w = w[valid_pred_rows]
            loss = loss_fn(pred, tgt, w)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())
        mae_acc += torch.mean(torch.abs(pred.detach() - tgt), dim=0).cpu().numpy()
        n += 1

    den = max(1, n)
    return {
        "train_mcd_loss": total / den,
        "train_mcd_mae_sbp": float(mae_acc[0] / den),
        "train_mcd_mae_dbp": float(mae_acc[1] / den),
        "train_mcd_mae_hb": float(mae_acc[2] / den),
        "train_mcd_steps": float(n),
    }


@torch.no_grad()
def _run_mcd_val_epoch(
    model: ClinicalRegressor,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_steps: int,
) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for step, batch in enumerate(loader):
        if max_steps >= 0 and step >= max_steps:
            break
        b = _move_batch(batch, device)
        finite_clin = torch.isfinite(b["clinical"]).all(dim=1)
        finite_video = torch.isfinite(b["video"]).view(b["video"].shape[0], -1).all(dim=1)
        finite_rows = finite_clin & finite_video
        if int(finite_rows.sum().item()) == 0:
            continue
        b_video = b["video"][finite_rows]
        b_clinical = b["clinical"][finite_rows]
        with torch.autocast(device_type="cuda", enabled=False):
            pred = model(b_video)
        valid_pred_rows = torch.isfinite(pred).all(dim=1)
        if int(valid_pred_rows.sum().item()) == 0:
            continue
        pred = pred[valid_pred_rows]
        tgt = b_clinical[valid_pred_rows]
        preds.append(pred.detach().float().cpu().numpy())
        targets.append(tgt.detach().float().cpu().numpy())

    if not preds:
        return {
            "val_mcd_mae_sbp": 0.0,
            "val_mcd_mae_dbp": 0.0,
            "val_mcd_mae_hb": 0.0,
            "val_mcd_rmse_sbp": 0.0,
            "val_mcd_rmse_dbp": 0.0,
            "val_mcd_rmse_hb": 0.0,
            "val_mcd_mae_mean": 0.0,
            "val_mcd_steps": 0.0,
        }

    pred_np = np.concatenate(preds, axis=0)
    tgt_np = np.concatenate(targets, axis=0)
    abs_err = np.abs(pred_np - tgt_np)
    rmse = np.sqrt(np.mean((pred_np - tgt_np) ** 2, axis=0))
    mae = np.mean(abs_err, axis=0)

    return {
        "val_mcd_mae_sbp": float(mae[0]),
        "val_mcd_mae_dbp": float(mae[1]),
        "val_mcd_mae_hb": float(mae[2]),
        "val_mcd_rmse_sbp": float(rmse[0]),
        "val_mcd_rmse_dbp": float(rmse[1]),
        "val_mcd_rmse_hb": float(rmse[2]),
        "val_mcd_mae_mean": float(np.mean(mae)),
        "val_mcd_steps": float(pred_np.shape[0]),
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")

    train_loaders, val_loaders, counts = _build_train_val_loaders(args)
    if not train_loaders:
        raise RuntimeError("No datasets provided. Use --ubfc-root and/or --ibvp-root and/or --mmpd-root and/or --mcd-root")

    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]} clips")

    model = EquiPhysDANN(in_channels=3, latent_dim=256, frames=args.frames, lambda_grl=args.lambda_grl).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(args.lr * 0.05, 1e-6),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ubfc_loss = PhysiologyInformedLoss(lambda_green=0.4, lambda_roi=0.2, lambda_hr=args.lambda_hr)
    mmpd_loss = DANNTrainingLoss(alpha_skin=0.2, alpha_light=0.2, lambda_green=0.4, lambda_roi=0.2, lambda_hr=args.lambda_hr)

    start_epoch = 1
    history: List[Dict[str, float]] = []
    best_score = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "history" in ckpt and isinstance(ckpt["history"], list):
            history = ckpt["history"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        log: Dict[str, float] = {"epoch": float(epoch)}

        if "ubfc" in train_loaders:
            log.update(
                _run_ubfc_train_epoch(
                    model=model,
                    loader=train_loaders["ubfc"],
                    optimizer=optimizer,
                    scaler=scaler,
                    loss_fn=ubfc_loss,
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.max_steps_ubfc,
                    grad_clip=args.grad_clip,
                    train_fps=args.train_fps,
                )
            )

        if "ibvp" in train_loaders:
            ibvp_log = _run_ubfc_train_epoch(
                model=model,
                loader=train_loaders["ibvp"],
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=ubfc_loss,
                device=device,
                use_amp=use_amp,
                max_steps=args.max_steps_ibvp,
                grad_clip=args.grad_clip,
                train_fps=args.train_fps,
            )
            log.update({k.replace("ubfc", "ibvp"): v for k, v in ibvp_log.items()})

        if "mmpd" in train_loaders and epoch > max(0, args.ubfc_warmup_epochs):
            log.update(
                _run_mmpd_train_epoch(
                    model=model,
                    loader=train_loaders["mmpd"],
                    optimizer=optimizer,
                    scaler=scaler,
                    loss_fn=mmpd_loss,
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.max_steps_mmpd,
                    grad_clip=args.grad_clip,
                    train_fps=args.train_fps,
                )
            )

        if "ubfc" in val_loaders:
            log.update(
                _run_ubfc_val_epoch(
                    model=model,
                    loader=val_loaders["ubfc"],
                    loss_fn=ubfc_loss,
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.val_max_steps_ubfc,
                    eval_fps=args.eval_fps,
                )
            )

        if "ibvp" in val_loaders:
            ibvp_log = _run_ubfc_val_epoch(
                model=model,
                loader=val_loaders["ibvp"],
                loss_fn=ubfc_loss,
                device=device,
                use_amp=use_amp,
                max_steps=args.val_max_steps_ibvp,
                eval_fps=args.eval_fps,
            )
            log.update({k.replace("ubfc", "ibvp"): v for k, v in ibvp_log.items()})

        if "mmpd" in val_loaders:
            log.update(
                _run_mmpd_val_epoch(
                    model=model,
                    loader=val_loaders["mmpd"],
                    loss_fn=mmpd_loss,
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.val_max_steps_mmpd,
                    eval_fps=args.eval_fps,
                )
            )

        scheduler.step()
        log["lr"] = float(optimizer.param_groups[0]["lr"])
        log["score"] = _select_score(log)

        history.append(log)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "history": history,
            "log": log,
            "best_score": best_score,
        }

        epoch_path = out_dir / f"equiphys_epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)

        if log["score"] < best_score:
            best_score = log["score"]
            best_path = out_dir / "equiphys_best.pt"
            ckpt["best_score"] = best_score
            torch.save(ckpt, best_path)
            print(f"New best checkpoint: {best_path} | score={best_score:.4f}")

        summary = " | ".join([f"{k}={v:.5f}" for k, v in log.items()])
        print(summary)
        print(f"Saved: {epoch_path}")

    clinical_history: List[Dict[str, float]] = []
    if "mcd" in train_loaders and args.mcd_epochs > 0:
        print("Starting MCD clinical transfer stage.")
        clinical_model = ClinicalRegressor(model.extractor, latent_dim=256, out_targets=3).to(device)
        clinical_optimizer = torch.optim.AdamW(clinical_model.head.parameters(), lr=args.clinical_lr, weight_decay=args.weight_decay)
        clinical_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            clinical_optimizer,
            T_max=max(1, args.mcd_epochs),
            eta_min=max(args.clinical_lr * 0.05, 1e-6),
        )
        clinical_loss = LDSWeightedMSELoss()

        mcd_train_labels = _collect_clinical_labels(train_loaders["mcd"])
        lds_cfg = LDSConfig(min_label=0.0, max_label=250.0)
        lds_weights = build_lds_weights(mcd_train_labels[:, 0], lds_cfg) if mcd_train_labels.shape[0] > 0 else np.ones((0,), dtype=np.float32)

        best_mcd = float("inf")
        for clinical_epoch in range(1, args.mcd_epochs + 1):
            log = {"clinical_epoch": float(clinical_epoch)}
            log.update(
                _run_mcd_train_epoch(
                    model=clinical_model,
                    loader=train_loaders["mcd"],
                    optimizer=clinical_optimizer,
                    scaler=scaler,
                    loss_fn=clinical_loss,
                    sample_weights=lds_weights,
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.max_steps_mcd,
                    grad_clip=args.grad_clip,
                )
            )
            log.update(
                _run_mcd_val_epoch(
                    model=clinical_model,
                    loader=val_loaders["mcd"],
                    device=device,
                    use_amp=use_amp,
                    max_steps=args.val_max_steps_mcd,
                )
            )
            clinical_scheduler.step()
            log["clinical_lr"] = float(clinical_optimizer.param_groups[0]["lr"])
            clinical_history.append(log)

            clinical_epoch_path = out_dir / f"equiphys_clinical_epoch_{clinical_epoch:03d}.pt"
            torch.save(
                {
                    "clinical_epoch": clinical_epoch,
                    "clinical_model_state": clinical_model.state_dict(),
                    "extractor_state": model.extractor.state_dict(),
                    "optimizer_state": clinical_optimizer.state_dict(),
                    "scheduler_state": clinical_scheduler.state_dict(),
                    "history": clinical_history,
                    "args": vars(args),
                },
                clinical_epoch_path,
            )

            if log["val_mcd_mae_mean"] < best_mcd:
                best_mcd = log["val_mcd_mae_mean"]
                torch.save(
                    {
                        "clinical_model_state": clinical_model.state_dict(),
                        "extractor_state": model.extractor.state_dict(),
                        "history": clinical_history,
                        "args": vars(args),
                        "best_mcd": best_mcd,
                    },
                    out_dir / "equiphys_clinical_best.pt",
                )
                print(f"New best clinical checkpoint: {out_dir / 'equiphys_clinical_best.pt'} | mae_mean={best_mcd:.4f}")

            print(" | ".join([f"{k}={v:.5f}" for k, v in log.items()]))
            print(f"Saved: {clinical_epoch_path}")

    final_path = out_dir / "equiphys_final.pt"
    torch.save({"model_state": model.state_dict(), "history": history, "args": vars(args), "best_score": best_score}, final_path)

    history_path = out_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if clinical_history:
        clinical_history_path = out_dir / "clinical_history.json"
        clinical_history_path.write_text(json.dumps(clinical_history, indent=2), encoding="utf-8")
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
