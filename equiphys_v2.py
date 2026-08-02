"""equiphys_v2.py — architecture + loss fixes for EquiPhys.

WHAT WAS BROKEN
---------------
1. `EquiPhysFeatureExtractor` ended with `AdaptiveAvgPool3d((1,1,1))`, collapsing
   the temporal axis to a single value. `PulseHead` was then asked to reconstruct
   a 150-sample waveform from that temporally-flattened vector via one Linear
   layer. That is architecturally impossible: all phase/timing information was
   destroyed before the head ever saw it. This is why waveform Pearson never went
   positive no matter how the loss was tuned.

2. `PhysiologyInformedLoss` included a green-channel term (lambda_green=0.4) whose
   gradient actively fought the waveform term, and an HR term built on
   `smooth_l1_loss` over soft-argmaxed scalars, which carries almost no gradient.

WHAT THIS FIXES
---------------
* `TemporalFeatureExtractor` pools over space only, preserving the time axis.
* `TemporalPulseDecoder` reconstructs the waveform with 1D temporal convolutions
  and learned upsampling — the standard rPPG decoder design.
* `PhysicsInformedLossV2` drops the green term, keeps negative-Pearson as the
  primary signal, and adds spectral-distribution matching + in-band SNR
  maximisation, both of which carry dense gradient.
* `AugmentedNPZClipDataset` adds the augmentations every working rPPG paper uses:
  temporal reversal, gamma/illumination jitter, spatial crop-resize, additive
  noise, and frequency-domain resampling (synthetic HR shifts).

Drop-in usage: import EquiPhysDANNV2 in place of EquiPhysDANN. The forward()
signature and output dict keys are identical, so the training loop needs no
changes beyond the import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from equiphys_core import GradientReversalLayer, DomainDiscriminator


# ═══════════════════════════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════════════════════════


class Conv3DBlock(nn.Module):
    """3D conv + BN + SiLU. Same as v1 but exposed here for clarity."""

    def __init__(self, in_ch: int, out_ch: int,
                 stride: Tuple[int, int, int] = (1, 2, 2)) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalFeatureExtractor(nn.Module):
    """3D extractor that PRESERVES the temporal axis.

    The critical change vs v1: spatial dimensions are downsampled aggressively
    (64 -> 4), but time is only halved once (150 -> 75). Pooling at the end is
    spatial-only, so the output keeps a per-timestep feature vector.

    Input:  x        [B, 3, T, H, W]
    Output: temporal [B, C, T']   per-timestep features  (T' = T // 2)
            latent   [B, D]       time-averaged embedding for the discriminator
            attn_2d  [B, 1, H',W'] spatial attention map for the ROI penalty
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32,
                 latent_dim: int = 256) -> None:
        super().__init__()

        # Spatial downsampling only — time preserved
        self.block1 = Conv3DBlock(in_channels, base_ch,      stride=(1, 2, 2))  # 64->32
        self.block2 = Conv3DBlock(base_ch,     base_ch * 2,  stride=(1, 2, 2))  # 32->16
        # Single temporal downsample here (T -> T/2), keeps enough resolution
        # to resolve a 4 Hz pulse at 30 fps (Nyquist is 7.5 Hz after halving).
        self.block3 = Conv3DBlock(base_ch * 2, base_ch * 4,  stride=(2, 2, 2))  # 16->8
        self.block4 = Conv3DBlock(base_ch * 4, base_ch * 4,  stride=(1, 2, 2))  # 8->4

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 2, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_ch * 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.feat_ch = base_ch * 4

        # Latent for the domain discriminator (time-averaged)
        self.latent_proj = nn.Sequential(
            nn.Linear(self.feat_ch, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)                       # [B, C, T', H', W']

        attn_3d = self.spatial_attention(h)      # [B, 1, T', H', W']
        attn_2d = attn_3d.mean(dim=2)            # [B, 1, H', W']  for ROI penalty

        h = h * attn_3d

        # ── THE FIX: pool over space only, keep time ──────────────────────
        temporal = h.mean(dim=(3, 4))            # [B, C, T']

        latent = self.latent_proj(temporal.mean(dim=2))   # [B, D]

        return temporal, latent, attn_2d


class TemporalPulseDecoder(nn.Module):
    """Reconstruct the BVP waveform from per-timestep features.

    Uses dilated 1D temporal convolutions (growing receptive field over the
    pulse period) followed by learned upsampling back to the input frame count.
    This is the standard rPPG decoder shape — v1's single Linear layer had no
    way to model temporal structure at all.

    Input:  temporal [B, C, T']
    Output: bvp      [B, T]
    """

    def __init__(self, in_ch: int = 128, hidden: int = 64,
                 out_frames: int = 150) -> None:
        super().__init__()
        self.out_frames = out_frames

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, temporal: torch.Tensor) -> torch.Tensor:
        h = self.temporal_conv(temporal)                    # [B, hidden, T']

        # Upsample back to the original frame count
        h = F.interpolate(h, size=self.out_frames,
                          mode="linear", align_corners=False)  # [B, hidden, T]

        bvp = self.refine(h).squeeze(1)                      # [B, T]

        # Instance-normalise: the loss only cares about shape, not scale, and
        # this keeps the output well-conditioned for the spectral terms.
        bvp = bvp - bvp.mean(dim=1, keepdim=True)
        bvp = bvp / (bvp.std(dim=1, keepdim=True) + 1e-6)
        return bvp


class EquiPhysDANNV2(nn.Module):
    """Drop-in replacement for EquiPhysDANN with the temporal path fixed.

    Output dict keys are identical to v1, so the training loop is unchanged.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256,
                 frames: int = 150, lambda_grl: float = 1.0,
                 base_ch: int = 32) -> None:
        super().__init__()
        self.extractor = TemporalFeatureExtractor(
            in_channels=in_channels, base_ch=base_ch, latent_dim=latent_dim
        )
        self.pulse_head = TemporalPulseDecoder(
            in_ch=self.extractor.feat_ch, hidden=64, out_frames=frames
        )
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl)
        self.discriminator = DomainDiscriminator(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        temporal, latent, attn_map = self.extractor(x)
        bvp_pred = self.pulse_head(temporal)
        domain_pred = self.discriminator(self.grl(latent))
        return {
            "latent": latent,
            "bvp_pred": bvp_pred,
            "attn_map": attn_map,
            "skin_logits": domain_pred["skin_logits"],
            "light_logits": domain_pred["light_logits"],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════


def _neg_pearson(pred: torch.Tensor, target: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
    """1 - mean Pearson r. Primary waveform-shape objective."""
    p = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p.pow(2).sum(dim=1) + eps) * (t.pow(2).sum(dim=1) + eps))
    return 1.0 - (num / den).mean()


def _band_psd(x_bt: torch.Tensor, fps: float,
              f_low: float, f_high: float,
              eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Windowed power spectrum restricted to the physiological band.

    Returns (normalised in-band power [B, F_band], band frequencies [F_band]).
    """
    B, T = x_bt.shape
    x = x_bt.float() - x_bt.float().mean(dim=1, keepdim=True)
    win = torch.hann_window(T, periodic=True, device=x.device,
                            dtype=torch.float32).unsqueeze(0)
    spec = torch.fft.rfft(x * win, dim=1)
    power = spec.real.pow(2) + spec.imag.pow(2)

    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fps)).to(x.device)
    band = (freqs >= f_low) & (freqs <= f_high)

    p = power[:, band] + eps
    p = p / p.sum(dim=1, keepdim=True)
    return p, freqs[band]


def _spectral_kl(pred_bt: torch.Tensor, target_bt: torch.Tensor,
                 fps: float, f_low: float = 0.7, f_high: float = 3.0) -> torch.Tensor:
    """KL(target_spectrum || pred_spectrum) inside the HR band.

    This is the term that actually teaches the model *where* to put its spectral
    peak. Unlike smooth_l1 on a soft-argmaxed scalar HR, this has non-zero
    gradient at every frequency bin.
    """
    if pred_bt.shape[1] < 16:
        return torch.tensor(0.0, device=pred_bt.device, dtype=pred_bt.dtype)
    p, _ = _band_psd(pred_bt,   fps, f_low, f_high)
    q, _ = _band_psd(target_bt, fps, f_low, f_high)
    kl = (q * (torch.log(q) - torch.log(p))).sum(dim=1)
    return kl.mean().to(pred_bt.dtype)


def _bandpower_ratio_loss(pred_bt: torch.Tensor, fps: float,
                          f_low: float = 0.7, f_high: float = 3.0,
                          eps: float = 1e-8) -> torch.Tensor:
    """Encourage energy concentration inside the HR band (SNR maximisation).

    Unsupervised — applies even where BVP labels are absent or unreliable, and
    directly rewards the sharp single-peak spectrum a clean pulse produces.
    """
    B, T = pred_bt.shape
    if T < 16:
        return torch.tensor(0.0, device=pred_bt.device, dtype=pred_bt.dtype)

    x = pred_bt.float() - pred_bt.float().mean(dim=1, keepdim=True)
    win = torch.hann_window(T, periodic=True, device=x.device,
                            dtype=torch.float32).unsqueeze(0)
    spec = torch.fft.rfft(x * win, dim=1)
    power = spec.real.pow(2) + spec.imag.pow(2)

    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fps)).to(x.device)
    band = (freqs >= f_low) & (freqs <= f_high)

    in_band = power[:, band].sum(dim=1)
    total = power.sum(dim=1) + eps
    ratio = in_band / total                      # in [0, 1], want high

    # Also sharpen within the band: low entropy = single dominant peak
    p = power[:, band] + eps
    p = p / p.sum(dim=1, keepdim=True)
    entropy = -(p * torch.log(p)).sum(dim=1)
    max_ent = torch.log(torch.tensor(float(p.shape[1]), device=x.device))
    norm_ent = entropy / max_ent                 # in [0, 1], want low

    loss = (1.0 - ratio) + 0.3 * norm_ent
    return loss.mean().to(pred_bt.dtype)


def _temporal_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match first differences — sharpens the systolic upstroke.

    Pearson on the raw waveform is insensitive to how fast the rise is; this term
    supplies that gradient explicitly.
    """
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return _neg_pearson(dp, dt)


def _roi_attention_penalty(attn_map: torch.Tensor,
                           roi_mask: torch.Tensor) -> torch.Tensor:
    """Penalise attention energy falling outside the facial ROI."""
    roi_rs = F.interpolate(roi_mask, size=attn_map.shape[-2:],
                           mode="bilinear", align_corners=False)
    outside = 1.0 - roi_rs
    out_energy = (attn_map * outside).sum(dim=(1, 2, 3))
    total = attn_map.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return (out_energy / total).mean()


class PhysicsInformedLossV2(nn.Module):
    """Physics-informed loss with the green-channel term removed.

    Terms, with default weights:
        1.0  * negative Pearson on the waveform        (primary shape objective)
        0.5  * negative Pearson on first differences   (upstroke sharpness)
        lambda_spec * spectral KL vs ground truth      (peak placement)
        lambda_snr  * in-band power concentration      (SNR / peak sharpness)
        lambda_roi  * attention-outside-ROI penalty    (spatial regularisation)

    The v1 green-channel term is gone entirely. It was injecting a gradient that
    fought the waveform term, which is what drove val Pearson negative.
    """

    def __init__(self,
                 lambda_diff: float = 0.5,
                 lambda_spec: float = 1.0,
                 lambda_snr: float = 0.3,
                 lambda_roi: float = 0.05,
                 f_low: float = 0.7,
                 f_high: float = 3.0) -> None:
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_spec = lambda_spec
        self.lambda_snr = lambda_snr
        self.lambda_roi = lambda_roi
        self.f_low = f_low
        self.f_high = f_high

    def forward(self,
                bvp_pred: torch.Tensor,
                bvp_target: torch.Tensor,
                video: torch.Tensor,
                attn_map: torch.Tensor,
                roi_mask: torch.Tensor,
                fps: float = 30.0) -> Dict[str, torch.Tensor]:

        l_wave = _neg_pearson(bvp_pred, bvp_target)
        l_diff = _temporal_difference_loss(bvp_pred, bvp_target)
        l_spec = _spectral_kl(bvp_pred, bvp_target.detach(), fps,
                              self.f_low, self.f_high)
        l_snr = _bandpower_ratio_loss(bvp_pred, fps, self.f_low, self.f_high)
        l_roi = _roi_attention_penalty(attn_map, roi_mask)

        total = (l_wave
                 + self.lambda_diff * l_diff
                 + self.lambda_spec * l_spec
                 + self.lambda_snr * l_snr
                 + self.lambda_roi * l_roi)

        return {
            "loss_total": total,
            "loss_wave": l_wave,
            "loss_diff": l_diff,
            "loss_spec": l_spec,
            "loss_snr": l_snr,
            "loss_roi": l_roi,
            # Keys the existing training loop logs; kept for compatibility.
            "loss_green": torch.zeros((), device=total.device),
            "loss_hr": l_spec.detach(),
        }


class DANNTrainingLossV2(nn.Module):
    """PhysicsInformedLossV2 + adversarial domain terms.

    Adds class-weighted CE for skin tone, which fixes the discriminator
    collapsing to the majority Fitzpatrick class (v1 sat at chance accuracy for
    all 62 epochs).
    """

    def __init__(self,
                 alpha_skin: float = 0.2,
                 alpha_light: float = 0.2,
                 skin_weights: Optional[torch.Tensor] = None,
                 **phys_kwargs) -> None:
        super().__init__()
        self.phys_loss = PhysicsInformedLossV2(**phys_kwargs)
        self.skin_ce = nn.CrossEntropyLoss(weight=skin_weights)
        self.light_ce = nn.CrossEntropyLoss()
        self.alpha_skin = alpha_skin
        self.alpha_light = alpha_light

    def forward(self,
                model_out: Dict[str, torch.Tensor],
                bvp_target: torch.Tensor,
                video: torch.Tensor,
                roi_mask: torch.Tensor,
                skin_label: torch.Tensor,
                light_label: torch.Tensor,
                fps: float = 30.0) -> Dict[str, torch.Tensor]:

        phys = self.phys_loss(
            bvp_pred=model_out["bvp_pred"],
            bvp_target=bvp_target,
            video=video,
            attn_map=model_out["attn_map"],
            roi_mask=roi_mask,
            fps=fps,
        )

        l_skin = self.skin_ce(model_out["skin_logits"], skin_label)
        l_light = self.light_ce(model_out["light_logits"], light_label)

        total = phys["loss_total"] + self.alpha_skin * l_skin + self.alpha_light * l_light

        out = dict(phys)
        out.update({
            "loss_total": total,
            "loss_phys": phys["loss_total"],
            "loss_skin": l_skin,
            "loss_light": l_light,
        })
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════════════════════


class AugmentedNPZClipDataset(torch.utils.data.Dataset):
    """NPZClipDataset with rPPG-appropriate augmentation.

    Augmentations (train split only — pass augment=False for validation):

    * Temporal reversal      — rPPG is direction-agnostic; doubles effective data
    * Gamma / brightness     — simulates lighting variation across sessions
    * Per-channel gain       — simulates different camera colour responses
    * Random spatial crop    — simulates ROI-detector jitter
    * Additive Gaussian noise— simulates sensor noise
    * Frequency resampling   — synthetically shifts HR by +/- 20%, the single most
                               valuable augmentation for HR generalisation, since
                               it directly expands the label distribution

    Frequency resampling resamples video and BVP together so the label stays
    consistent with the signal.
    """

    def __init__(self, base_dataset, augment: bool = True,
                 p_reverse: float = 0.5,
                 p_gamma: float = 0.5,
                 p_gain: float = 0.5,
                 p_crop: float = 0.3,
                 p_noise: float = 0.3,
                 p_freq: float = 0.4,
                 freq_range: Tuple[float, float] = (0.8, 1.25)) -> None:
        self.base = base_dataset
        self.augment = augment
        self.p_reverse = p_reverse
        self.p_gamma = p_gamma
        self.p_gain = p_gain
        self.p_crop = p_crop
        self.p_noise = p_noise
        self.p_freq = p_freq
        self.freq_range = freq_range

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _resample_time(video_cthw: torch.Tensor, bvp: torch.Tensor,
                       factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resample along time by `factor`, then crop/pad back to original length.

        factor > 1 compresses time (raises apparent HR); factor < 1 stretches it.
        """
        C, T, H, W = video_cthw.shape
        new_T = max(16, int(round(T / factor)))

        # [C,T,H,W] -> [1,C,T,H,W] for interpolate, then back
        v = video_cthw.unsqueeze(0)
        v = F.interpolate(v, size=(new_T, H, W), mode="trilinear",
                          align_corners=False).squeeze(0)

        b = bvp.view(1, 1, -1)
        b = F.interpolate(b, size=new_T, mode="linear",
                          align_corners=False).view(-1)

        if new_T >= T:
            start = torch.randint(0, new_T - T + 1, (1,)).item() if new_T > T else 0
            return v[:, start:start + T], b[start:start + T]

        pad = T - new_T
        v = torch.cat([v, v[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        b = torch.cat([b, b[-1:].repeat(pad)], dim=0)
        return v, b

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]

        if not self.augment:
            return item

        video = item["video"]      # [3, T, H, W]
        bvp = item["bvp"]          # [T]

        # ── Frequency-domain resampling (synthetic HR shift) ───────────────
        if torch.rand(1).item() < self.p_freq:
            lo, hi = self.freq_range
            factor = float(torch.empty(1).uniform_(lo, hi).item())
            video, bvp = self._resample_time(video, bvp, factor)

        # ── Temporal reversal ──────────────────────────────────────────────
        if torch.rand(1).item() < self.p_reverse:
            video = torch.flip(video, dims=[1])
            bvp = torch.flip(bvp, dims=[0])

        # ── Gamma / brightness ─────────────────────────────────────────────
        if torch.rand(1).item() < self.p_gamma:
            gamma = float(torch.empty(1).uniform_(0.7, 1.4).item())
            video = video.clamp_min(1e-6).pow(gamma)

        # ── Per-channel gain (camera colour response) ──────────────────────
        if torch.rand(1).item() < self.p_gain:
            gains = torch.empty(3, 1, 1, 1).uniform_(0.85, 1.15)
            video = video * gains

        # ── Random spatial crop-and-resize (ROI jitter) ────────────────────
        if torch.rand(1).item() < self.p_crop:
            C, T, H, W = video.shape
            scale = float(torch.empty(1).uniform_(0.80, 1.0).item())
            ch, cw = int(H * scale), int(W * scale)
            if ch >= 8 and cw >= 8:
                top = torch.randint(0, H - ch + 1, (1,)).item()
                left = torch.randint(0, W - cw + 1, (1,)).item()
                crop = video[:, :, top:top + ch, left:left + cw]
                video = F.interpolate(crop.unsqueeze(0), size=(T, H, W),
                                      mode="trilinear",
                                      align_corners=False).squeeze(0)

        # ── Additive sensor noise ──────────────────────────────────────────
        if torch.rand(1).item() < self.p_noise:
            sigma = float(torch.empty(1).uniform_(0.002, 0.012).item())
            video = video + torch.randn_like(video) * sigma

        video = video.clamp(0.0, 1.5)

        out = dict(item)
        out["video"] = video
        out["bvp"] = bvp
        return out


def compute_skin_class_weights(npz_root: str, n_classes: int = 6,
                               device: Optional[torch.device] = None) -> torch.Tensor:
    """Inverse-frequency class weights for the Fitzpatrick discriminator.

    v1's skin accuracy sat at chance for the entire run because the CE loss was
    unweighted over an imbalanced label set. This rebalances it.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    root = Path(npz_root)
    for fp in root.rglob("*.npz"):
        try:
            with np.load(fp, allow_pickle=False) as d:
                if "skin_label" in d:
                    lab = int(d["skin_label"])
                    if 1 <= lab <= 6:
                        lab -= 1
                    lab = int(np.clip(lab, 0, n_classes - 1))
                    counts[lab] += 1
        except Exception:
            continue

    if counts.sum() == 0:
        w = torch.ones(n_classes, dtype=torch.float32)
    else:
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        inv = inv / inv.sum() * n_classes
        w = torch.tensor(inv, dtype=torch.float32)

    return w.to(device) if device is not None else w


__all__ = [
    "EquiPhysDANNV2",
    "TemporalFeatureExtractor",
    "TemporalPulseDecoder",
    "PhysicsInformedLossV2",
    "DANNTrainingLossV2",
    "AugmentedNPZClipDataset",
    "compute_skin_class_weights",
]
