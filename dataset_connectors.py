from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from equiphys_core import rppg_collate_fn


@dataclass
class MultiDatasetConfig:
    ubfc_root: Optional[str] = None
    mmpd_root: Optional[str] = None
    mcd_root: Optional[str] = None
    ibvp_root: Optional[str] = None
    frames: int = 150
    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True


# Expected NPZ schema for each clip file:
# - video: [T,H,W,3] float32 in [0,1]
# - roi_mask: [H,W] float32 in [0,1]
# Optional keys based on objective:
# - bvp: [T] float32 (UBFC/MMPD for pulse supervision)
# - skin_label: int in [0..5] (MMPD)
# - light_label: int in [0..n-1] (MMPD)
# - clinical: [D] float32 (MCD, e.g., [SBP, DBP, Hb])


class NPZClipDataset(Dataset):
    def __init__(
        self,
        root: str,
        frames: int = 150,
        require_bvp: bool = False,
        require_domain_labels: bool = False,
        require_clinical: bool = False,
    ) -> None:
        self.root = Path(root)
        self.frames = frames
        self.require_bvp = require_bvp
        self.require_domain_labels = require_domain_labels
        self.require_clinical = require_clinical

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _fix_temporal_length(self, video_thwc: np.ndarray, bvp: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        t = video_thwc.shape[0]
        if bvp is None:
            bvp = np.zeros((t,), dtype=np.float32)

        if t >= self.frames:
            return video_thwc[: self.frames], bvp[: self.frames]

        pad_n = self.frames - t
        video_pad = np.repeat(video_thwc[-1:, ...], pad_n, axis=0)
        bvp_pad = np.repeat(bvp[-1:], pad_n, axis=0)
        return np.concatenate([video_thwc, video_pad], axis=0), np.concatenate([bvp, bvp_pad], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fp = self.files[idx]
        data = np.load(fp, allow_pickle=False)

        if "video" not in data:
            raise KeyError(f"Missing key 'video' in {fp}")

        video_thwc = data["video"].astype(np.float32)
        bvp = data["bvp"].astype(np.float32) if "bvp" in data else None
        roi_mask = data["roi_mask"].astype(np.float32) if "roi_mask" in data else np.ones(video_thwc.shape[1:3], dtype=np.float32)

        if self.require_bvp and bvp is None:
            raise KeyError(f"Missing required key 'bvp' in {fp}")

        if self.require_domain_labels and ("skin_label" not in data or "light_label" not in data):
            raise KeyError(f"Missing required domain labels in {fp}; expected 'skin_label' and 'light_label'")

        if self.require_clinical and "clinical" not in data:
            raise KeyError(f"Missing required key 'clinical' in {fp}")

        video_thwc, bvp_fixed = self._fix_temporal_length(video_thwc=video_thwc, bvp=bvp)

        # [T,H,W,3] -> [T,3,H,W]
        video_tchw = np.transpose(video_thwc, (0, 3, 1, 2))
        # [T,3,H,W] -> [3,T,H,W]
        video_cthw = np.transpose(video_tchw, (1, 0, 2, 3))

        skin_label = int(data["skin_label"]) if "skin_label" in data else 0
        # MMPD commonly stores Fitzpatrick as 1..6; CE expects class ids 0..5.
        if 1 <= skin_label <= 6:
            skin_label -= 1
        skin_label = int(np.clip(skin_label, 0, 5))

        light_label = int(data["light_label"]) if "light_label" in data else 0
        # Current discriminator uses 4 lighting classes -> ids 0..3.
        light_label = int(np.clip(light_label, 0, 3))

        out: Dict[str, torch.Tensor] = {
            "video": torch.from_numpy(video_cthw),
            "bvp": torch.from_numpy(bvp_fixed.astype(np.float32)),
            "roi_mask": torch.from_numpy(roi_mask.astype(np.float32)).unsqueeze(0),
            # Keep labels always present so collate/loss code stays simple.
            "skin_label": torch.tensor(skin_label, dtype=torch.long),
            "light_label": torch.tensor(light_label, dtype=torch.long),
        }

        if "clinical" in data:
            out["clinical"] = torch.from_numpy(np.asarray(data["clinical"], dtype=np.float32))

        return out


def _build_loader(dataset: Dataset, cfg: MultiDatasetConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=rppg_collate_fn,
        drop_last=False,
    )


def build_objective_dataloaders(cfg: MultiDatasetConfig) -> Dict[str, DataLoader]:
    """Create separate loaders aligned to your 4 objectives.

    Returns keys only for provided roots:
    - ubfc_loader: pulse/physics-informed supervision
    - mmpd_loader: adversarial domain disentanglement (skin + lighting)
    - mcd_loader: clinical regression transfer
    """
    out: Dict[str, DataLoader] = {}

    if cfg.ubfc_root:
        ubfc_ds = NPZClipDataset(
            root=cfg.ubfc_root,
            frames=cfg.frames,
            require_bvp=True,
            require_domain_labels=False,
            require_clinical=False,
        )
        out["ubfc_loader"] = _build_loader(ubfc_ds, cfg)

    if cfg.ibvp_root:
        ibvp_ds = NPZClipDataset(
            root=cfg.ibvp_root,
            frames=cfg.frames,
            require_bvp=True,
            require_domain_labels=False,
            require_clinical=False,
        )
        out["ibvp_loader"] = _build_loader(ibvp_ds, cfg)

    if cfg.mmpd_root:
        mmpd_ds = NPZClipDataset(
            root=cfg.mmpd_root,
            frames=cfg.frames,
            require_bvp=True,
            require_domain_labels=True,
            require_clinical=False,
        )
        out["mmpd_loader"] = _build_loader(mmpd_ds, cfg)

    if cfg.mcd_root:
        mcd_ds = NPZClipDataset(
            root=cfg.mcd_root,
            frames=cfg.frames,
            require_bvp=False,
            require_domain_labels=False,
            require_clinical=True,
        )
        out["mcd_loader"] = _build_loader(mcd_ds, cfg)

    return out


def print_dataset_summary(loaders: Dict[str, DataLoader]) -> None:
    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)} clips, batch_size={loader.batch_size}")
