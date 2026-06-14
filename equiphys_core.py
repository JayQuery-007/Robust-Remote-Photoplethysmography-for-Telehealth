from __future__ import annotations

import math
import importlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.signal import butter, detrend, filtfilt, hilbert, find_peaks, welch as scipy_welch
from scipy.interpolate import interp1d


# Optional import to keep startup robust in environments where MediaPipe is missing.
try:
    import mediapipe as mp
except Exception:  # pragma: no cover - graceful degradation path
    mp = None


def _resolve_face_landmarker():
    """Resolve FaceLandmarker for MediaPipe Tasks API (>=0.10.x).

    Returns the instantiated FaceLandmarker or None.
    """
    if mp is None:
        return None

    try:
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base_options

        import os
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        if not os.path.exists(model_path):
            return None

        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)
    except Exception:
        pass

    return None


def _resolve_mediapipe_facemesh_class():
    """Resolve legacy FaceMesh class for MediaPipe <0.10."""
    if mp is None:
        return None

    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        face_mesh_mod = getattr(solutions, "face_mesh", None)
        if face_mesh_mod is not None and hasattr(face_mesh_mod, "FaceMesh"):
            return face_mesh_mod.FaceMesh

    try:
        mp_face_mesh = importlib.import_module("mediapipe.python.solutions.face_mesh")
        if hasattr(mp_face_mesh, "FaceMesh"):
            return mp_face_mesh.FaceMesh
    except Exception:
        pass

    return None


# -----------------------------
# Objective 1: DANN + GRL
# -----------------------------


class GradientReversalFunction(torch.autograd.Function):
    """Autograd function for gradient reversal.

    Forward pass is identity, backward pass multiplies incoming gradient by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl: float = 1.0) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_grl)


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int, int] = (1, 2, 2)) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EquiPhysFeatureExtractor(nn.Module):
    """3D spatio-temporal feature extractor.

    Input shape:
        x: [B, C, T, H, W] where C=3 (RGB), T=150 typically.
    Output:
        latent: [B, D] compact pulse-centric representation.
        attn_map: [B, 1, H', W'] spatial map for ROI-aware regularization.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 32, latent_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            Conv3DBlock(in_channels, base_ch, stride=(1, 2, 2)),
            Conv3DBlock(base_ch, base_ch * 2, stride=(2, 2, 2)),
            Conv3DBlock(base_ch * 2, base_ch * 4, stride=(2, 2, 2)),
            Conv3DBlock(base_ch * 4, base_ch * 4, stride=(2, 1, 1)),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 2, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_ch * 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)  # [B, C', T', H', W']

        # Attention is learned over [T', H', W']. We average across time next to keep ROI map 2D.
        attn_3d = self.spatial_attention(feat)  # [B, 1, T', H', W'] in [0, 1]
        attn_2d = attn_3d.mean(dim=2)  # [B, 1, H', W']

        feat_weighted = feat * attn_3d
        pooled = self.pool(feat_weighted)  # [B, C', 1, 1, 1]
        latent = self.proj(pooled)  # Flatten -> [B, C'] then Linear -> [B, latent_dim]
        return latent, attn_2d


class PulseHead(nn.Module):
    """Maps latent vector to temporal BVP waveform.

    Output shape:
        bvp: [B, T_out]
    """

    def __init__(self, latent_dim: int = 256, out_frames: int = 150) -> None:
        super().__init__()
        self.out_frames = out_frames
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(latent_dim, out_frames),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class DomainDiscriminator(nn.Module):
    """Predicts Fitzpatrick skin class and lighting class from latent embedding."""

    def __init__(
        self,
        latent_dim: int = 256,
        n_skin_classes: int = 6,
        n_light_classes: int = 4,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.skin_head = nn.Linear(hidden // 2, n_skin_classes)
        self.light_head = nn.Linear(hidden // 2, n_light_classes)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {
            "skin_logits": self.skin_head(h),
            "light_logits": self.light_head(h),
        }


class EquiPhysDANN(nn.Module):
    """Main model for rPPG + adversarial demographic disentanglement."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256, frames: int = 150, lambda_grl: float = 1.0) -> None:
        super().__init__()
        self.extractor = EquiPhysFeatureExtractor(in_channels=in_channels, latent_dim=latent_dim)
        self.pulse_head = PulseHead(latent_dim=latent_dim, out_frames=frames)
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl)
        self.discriminator = DomainDiscriminator(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, attn_map = self.extractor(x)
        bvp_pred = self.pulse_head(z)
        domain_pred = self.discriminator(self.grl(z))
        return {
            "latent": z,
            "bvp_pred": bvp_pred,
            "attn_map": attn_map,
            "skin_logits": domain_pred["skin_logits"],
            "light_logits": domain_pred["light_logits"],
        }


class RPPGClipDataset(Dataset):
    """Generic clip dataset for UBFC/MMPD/MCD style preprocessed samples.

    Each sample dictionary must contain:
      - video: np.ndarray [T,H,W,3] float32 in [0,1]
      - bvp: np.ndarray [T] float32
      - roi_mask: np.ndarray [H,W] float32 in [0,1]
      - skin_label: int in [0..5]
      - light_label: int in [0..n-1]
      - clinical: np.ndarray [D] optional (BP/Hb targets)
    """

    def __init__(self, samples: List[Dict], frames: int = 150) -> None:
        self.samples = samples
        self.frames = frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        video_thwc = sample["video"].astype(np.float32)
        bvp = sample["bvp"].astype(np.float32)
        roi_mask = sample["roi_mask"].astype(np.float32)

        # Temporal fix to exactly 150 frames: truncate or pad last frame/sample.
        t = video_thwc.shape[0]
        if t >= self.frames:
            video_thwc = video_thwc[: self.frames]
            bvp = bvp[: self.frames]
        else:
            pad_n = self.frames - t
            video_pad = np.repeat(video_thwc[-1:, ...], pad_n, axis=0)
            bvp_pad = np.repeat(bvp[-1:], pad_n, axis=0)
            video_thwc = np.concatenate([video_thwc, video_pad], axis=0)
            bvp = np.concatenate([bvp, bvp_pad], axis=0)

        # [T,H,W,3] -> [T,3,H,W]
        video_tchw = np.transpose(video_thwc, (0, 3, 1, 2))
        # [T,3,H,W] -> [3,T,H,W]
        video_cthw = np.transpose(video_tchw, (1, 0, 2, 3))

        out: Dict[str, torch.Tensor] = {
            "video": torch.from_numpy(video_cthw),
            "bvp": torch.from_numpy(bvp),
            "roi_mask": torch.from_numpy(roi_mask).unsqueeze(0),  # [1,H,W]
            "skin_label": torch.tensor(sample["skin_label"], dtype=torch.long),
            "light_label": torch.tensor(sample["light_label"], dtype=torch.long),
        }

        if "clinical" in sample and sample["clinical"] is not None:
            out["clinical"] = torch.from_numpy(np.asarray(sample["clinical"], dtype=np.float32))

        return out


def rppg_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    video = torch.stack([x["video"] for x in batch], dim=0)  # [B,3,T,H,W]
    bvp = torch.stack([x["bvp"] for x in batch], dim=0)  # [B,T]
    roi_mask = torch.stack([x["roi_mask"] for x in batch], dim=0)  # [B,1,H,W]
    skin_label = torch.stack([x["skin_label"] for x in batch], dim=0)
    light_label = torch.stack([x["light_label"] for x in batch], dim=0)

    out: Dict[str, torch.Tensor] = {
        "video": video,
        "bvp": bvp,
        "roi_mask": roi_mask,
        "skin_label": skin_label,
        "light_label": light_label,
    }
    if "clinical" in batch[0]:
        out["clinical"] = torch.stack([x["clinical"] for x in batch], dim=0)
    return out


# -----------------------------
# Objective 2: Physics-Informed Loss
# -----------------------------


class NegativePearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, T]
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)

        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered.pow(2).sum(dim=1) + self.eps) * (target_centered.pow(2).sum(dim=1) + self.eps))
        corr = numerator / denominator
        return 1.0 - corr.mean()


class PhysiologyInformedLoss(nn.Module):
    """L_phys enforcing
    1) strong waveform agreement (negative Pearson)
    2) strong alignment with green-channel temporal signal
    3) attention concentration inside forehead/cheek ROI
    """

    def __init__(self, lambda_green: float = 0.4, lambda_roi: float = 0.2, lambda_hr: float = 0.3) -> None:
        super().__init__()
        self.neg_pearson = NegativePearsonLoss()
        self.lambda_green = lambda_green
        self.lambda_roi = lambda_roi
        self.lambda_hr = lambda_hr

    @staticmethod
    def soft_hr_bpm_from_signal(
        signal_bt: torch.Tensor,
        fps: float,
        f_low: float = 0.667,
        f_high: float = 4.0,
        temperature: float = 0.08,
    ) -> torch.Tensor:
        """Differentiable HR estimator from temporal waveform using soft spectral expectation.

        signal_bt: [B, T]
        returns: [B] HR in BPM
        """
        bsz, t = signal_bt.shape
        if t < 16:
            return torch.full((bsz,), 75.0, dtype=signal_bt.dtype, device=signal_bt.device)

        # Use float32 for FFT stability under AMP (cuFFT has fp16 constraints for non power-of-two lengths).
        signal = signal_bt.float() - signal_bt.float().mean(dim=1, keepdim=True)
        win = torch.hann_window(t, periodic=True, device=signal_bt.device, dtype=torch.float32).unsqueeze(0)
        signal = signal * win

        spec = torch.fft.rfft(signal, dim=1)
        power = (spec.real.pow(2) + spec.imag.pow(2))

        freqs = torch.fft.rfftfreq(t, d=1.0 / float(fps)).to(signal_bt.device)
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        if int(band_mask.sum().item()) == 0:
            return torch.full((bsz,), 75.0, dtype=signal_bt.dtype, device=signal_bt.device)

        band_power = power[:, band_mask]
        band_freqs = freqs[band_mask].unsqueeze(0)
        logits = band_power / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=1)
        hr_bpm = (probs * band_freqs).sum(dim=1) * 60.0
        return hr_bpm.to(dtype=signal_bt.dtype)

    @staticmethod
    def extract_green_trace(video: torch.Tensor, roi_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build temporal green trace from video tensor.

        video shape: [B, 3, T, H, W]
        roi_mask shape: [B, 1, H, W] in {0,1} or [0,1], optional
        Returns:
            green_trace: [B, T]
        """
        green = video[:, 1, :, :, :]  # [B, T, H, W]

        if roi_mask is not None:
            # Expand ROI from [B,1,H,W] to [B,T,H,W] so each frame uses same spatial mask.
            roi_t = roi_mask.squeeze(1).unsqueeze(1).expand(-1, green.size(1), -1, -1)
            denom = roi_t.sum(dim=(2, 3)).clamp_min(1e-6)
            green_trace = (green * roi_t).sum(dim=(2, 3)) / denom
        else:
            green_trace = green.mean(dim=(2, 3))

        return green_trace

    @staticmethod
    def roi_attention_penalty(attn_map: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        """Penalize energy outside facial ROI.

        attn_map: [B,1,H',W']
        roi_mask: [B,1,H,W]
        """
        roi_rs = F.interpolate(roi_mask, size=attn_map.shape[-2:], mode="bilinear", align_corners=False)

        outside = 1.0 - roi_rs
        outside_energy = (attn_map * outside).sum(dim=(1, 2, 3))
        total_energy = attn_map.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        ratio = outside_energy / total_energy
        return ratio.mean()

    def forward(
        self,
        bvp_pred: torch.Tensor,
        bvp_target: torch.Tensor,
        video: torch.Tensor,
        attn_map: torch.Tensor,
        roi_mask: torch.Tensor,
        fps: float = 30.0,
    ) -> Dict[str, torch.Tensor]:
        l_wave = self.neg_pearson(bvp_pred, bvp_target)

        green_trace = self.extract_green_trace(video=video, roi_mask=roi_mask)
        l_green = self.neg_pearson(bvp_pred, green_trace)

        l_roi = self.roi_attention_penalty(attn_map=attn_map, roi_mask=roi_mask)

        pred_hr = self.soft_hr_bpm_from_signal(bvp_pred, fps=fps)
        gt_hr = self.soft_hr_bpm_from_signal(bvp_target.detach(), fps=fps)
        l_hr = F.smooth_l1_loss(pred_hr, gt_hr)

        total = l_wave + self.lambda_green * l_green + self.lambda_roi * l_roi + self.lambda_hr * l_hr
        return {
            "loss_total": total,
            "loss_wave": l_wave,
            "loss_green": l_green,
            "loss_roi": l_roi,
            "loss_hr": l_hr,
        }


class DANNTrainingLoss(nn.Module):
    """Composite objective for Objective 1 + Objective 2.

    Total loss = L_phys + alpha_skin * CE(skin) + alpha_light * CE(light)
    Because GRL is in the model forward path, minimizing this objective drives
    extractor to increase domain confusion while discriminator learns classification.
    """

    def __init__(
        self,
        alpha_skin: float = 0.2,
        alpha_light: float = 0.2,
        lambda_green: float = 0.4,
        lambda_roi: float = 0.2,
        lambda_hr: float = 0.3,
    ) -> None:
        super().__init__()
        self.phys_loss = PhysiologyInformedLoss(lambda_green=lambda_green, lambda_roi=lambda_roi, lambda_hr=lambda_hr)
        self.skin_ce = nn.CrossEntropyLoss()
        self.light_ce = nn.CrossEntropyLoss()
        self.alpha_skin = alpha_skin
        self.alpha_light = alpha_light

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        bvp_target: torch.Tensor,
        video: torch.Tensor,
        roi_mask: torch.Tensor,
        skin_label: torch.Tensor,
        light_label: torch.Tensor,
        fps: float = 30.0,
    ) -> Dict[str, torch.Tensor]:
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
        return {
            "loss_total": total,
            "loss_phys": phys["loss_total"],
            "loss_wave": phys["loss_wave"],
            "loss_green": phys["loss_green"],
            "loss_roi": phys["loss_roi"],
            "loss_hr": phys["loss_hr"],
            "loss_skin": l_skin,
            "loss_light": l_light,
        }


# -----------------------------
# Objective 3: Source-Free TTA
# -----------------------------


def _bandpower_objective(signal_bt: torch.Tensor, fps: float, f_low: float = 0.7, f_high: float = 3.0) -> torch.Tensor:
    """Maximize normalized spectral power in heart-rate band.

    signal_bt: [B, T]
    Returns negative objective for minimization.
    """
    bsz, t = signal_bt.shape
    if t < 16:
        return torch.tensor(0.0, device=signal_bt.device, dtype=signal_bt.dtype)

    signal = signal_bt - signal_bt.mean(dim=1, keepdim=True)

    # rfft over time dimension; frequencies in Hz.
    spec = torch.fft.rfft(signal, dim=1)
    power = (spec.real.pow(2) + spec.imag.pow(2))  # [B, F]

    freqs = torch.fft.rfftfreq(t, d=1.0 / fps).to(signal_bt.device)  # [F]
    band_mask = (freqs >= f_low) & (freqs <= f_high)

    band_power = power[:, band_mask].sum(dim=1)
    total_power = power.sum(dim=1).clamp_min(1e-8)
    ratio = band_power / total_power

    # entropy-style sharpening: maximize ratio, minimize spectral spread within band.
    band_spec = power[:, band_mask].clamp_min(1e-8)
    band_prob = band_spec / band_spec.sum(dim=1, keepdim=True)
    entropy = -(band_prob * torch.log(band_prob)).sum(dim=1)

    objective = ratio - 0.05 * entropy
    return -objective.mean()


def enable_tta_params(model: nn.Module) -> List[nn.Parameter]:
    """Enable adaptation only on BatchNorm affine params for stable source-free TTA."""
    params: List[nn.Parameter] = []
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()  # update running stats on unlabeled target stream
            if m.weight is not None:
                m.weight.requires_grad_(True)
                params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                params.append(m.bias)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    return params


@torch.no_grad()
def estimate_hr_bpm_from_bvp(bvp: torch.Tensor, fps: float, f_low: float = 0.667, f_high: float = 4.0) -> torch.Tensor:
    """HR estimation from BVP waveform using Welch's periodogram.

    bvp: [B, T]
    returns: [B]
    """
    bvp_np = bvp.float().cpu().numpy()
    B, T = bvp_np.shape
    hrs = np.empty(B, dtype=np.float64)

    for i in range(B):
        peak_hz, _ = _welch_peak_hz(bvp_np[i], fps, f_low=f_low, f_high=f_high)
        hrs[i] = peak_hz * 60.0 if peak_hz > 0 else 75.0

    return torch.tensor(hrs, dtype=bvp.dtype, device=bvp.device)


def test_time_adaptation(
    model: EquiPhysDANN,
    clip_bcthw: torch.Tensor,
    fps: float,
    steps: int = 8,
    lr: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Unsupervised TTA over first 150 frames.

    clip_bcthw shape: [B, C, T, H, W]
    """
    params = enable_tta_params(model)
    if len(params) == 0:
        raise RuntimeError("No BatchNorm parameters found for adaptation.")

    optimizer = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(clip_bcthw)
        bvp = out["bvp_pred"]  # [B, T]
        loss = _bandpower_objective(bvp, fps=fps)
        loss.backward()
        optimizer.step()

    model.eval()
    out_final = model(clip_bcthw)
    hr_bpm = estimate_hr_bpm_from_bvp(out_final["bvp_pred"], fps=fps)
    out_final["hr_bpm"] = hr_bpm
    return out_final


# -----------------------------
# Objective 4: Clinical Transfer + LDS
# -----------------------------


class HemodynamicsHead(nn.Module):
    """Regression head for BP/Hb after freezing rPPG feature extractor."""

    def __init__(self, in_dim: int = 256, hidden: int = 256, out_targets: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_targets),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ClinicalRegressor(nn.Module):
    """Freezes core extractor; trains only clinical regression head."""

    def __init__(self, pretrained_extractor: EquiPhysFeatureExtractor, latent_dim: int = 256, out_targets: int = 3) -> None:
        super().__init__()
        self.extractor = pretrained_extractor
        for p in self.extractor.parameters():
            p.requires_grad_(False)
        self.extractor.eval()

        self.head = HemodynamicsHead(in_dim=latent_dim, out_targets=out_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z, _ = self.extractor(x)
        y_hat = self.head(z)
        return y_hat


@dataclass
class LDSConfig:
    num_bins: int = 100
    kernel_size: int = 9
    sigma: float = 2.0
    min_label: float = 0.0
    max_label: float = 250.0


def gaussian_kernel1d(kernel_size: int, sigma: float) -> np.ndarray:
    half = (kernel_size - 1) / 2.0
    x = np.arange(kernel_size) - half
    kernel = np.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def build_lds_weights(labels: np.ndarray, config: LDSConfig) -> np.ndarray:
    """Compute LDS weights from empirical label distribution.

    labels: [N] continuous scalar labels (e.g., systolic BP)
    returns weights: [N]
    """
    labels = np.asarray(labels, dtype=np.float32)
    if labels.size == 0:
        return np.ones((0,), dtype=np.float32)
    finite = np.isfinite(labels)
    if not finite.any():
        return np.ones_like(labels, dtype=np.float32)

    # Replace invalid values with median finite value to keep per-sample alignment stable.
    labels = labels.copy()
    labels[~finite] = float(np.median(labels[finite]))
    clipped = np.clip(labels, config.min_label, config.max_label)

    hist, bin_edges = np.histogram(
        clipped,
        bins=config.num_bins,
        range=(config.min_label, config.max_label),
        density=False,
    )

    kernel = gaussian_kernel1d(config.kernel_size, config.sigma)
    smooth_hist = np.convolve(hist.astype(np.float32), kernel, mode="same")

    # Map each label to its smoothed bin density.
    bin_ids = np.digitize(clipped, bin_edges[:-1], right=False) - 1
    bin_ids = np.clip(bin_ids, 0, config.num_bins - 1)

    density = smooth_hist[bin_ids]
    inv = 1.0 / np.maximum(density, 1e-6)

    # Normalize to mean=1 for stable optimization.
    mean_inv = float(np.mean(inv))
    if not np.isfinite(mean_inv) or mean_inv <= 0:
        return np.ones_like(inv, dtype=np.float32)
    weights = inv / mean_inv
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return weights.astype(np.float32)


class LDSWeightedMSELoss(nn.Module):
    """Per-sample weighted MSE for imbalanced clinical labels."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, D], sample_weight: [B]
        se = (pred - target).pow(2).mean(dim=1)
        wse = se * sample_weight
        return wse.mean()


# -----------------------------
# Lightweight preprocessing for 30 FPS
# -----------------------------


class FaceROIExtractor:
    """MediaPipe Face Landmarker ROI extractor with lightweight crop for real-time inference.

    Supports both the new Tasks API (>=0.10) and legacy FaceMesh API.
    Returns a normalized ROI frame [H, W, 3] and a binary ROI mask [H, W].
    """

    FOREHEAD_IDXS = [10, 67, 103, 109, 338, 297, 332]
    LEFT_CHEEK_IDXS = [205, 50, 117, 118]
    RIGHT_CHEEK_IDXS = [425, 280, 347, 348]

    def __init__(self, target_size: int = 64, max_faces: int = 1, refine_landmarks: bool = False) -> None:
        self.target_size = target_size
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self._api_mode = "none"  # "tasks" | "legacy" | "cascade" | "none"

        # Try new Tasks API first (MediaPipe >= 0.10)
        self.face_landmarker = _resolve_face_landmarker()
        if self.face_landmarker is not None:
            self._api_mode = "tasks"
            self.has_facemesh = True
            self.face_mesh = None
            self.face_cascade = None
        else:
            # Try legacy FaceMesh
            face_mesh_cls = _resolve_mediapipe_facemesh_class()
            if face_mesh_cls is not None:
                self.face_mesh = face_mesh_cls(
                    static_image_mode=False,
                    max_num_faces=max_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._api_mode = "legacy"
                self.has_facemesh = True
                self.face_landmarker = None
                self.face_cascade = None
            else:
                # Fallback to OpenCV cascade
                self.face_mesh = None
                self.face_landmarker = None
                self.has_facemesh = False
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if not self.face_cascade.empty():
                    self._api_mode = "cascade"

    def _extract_landmarks_tasks(self, rgb: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Extract face landmarks using the new Tasks API."""
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())
        result = self.face_landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        pts = []
        for idx in (self.FOREHEAD_IDXS + self.LEFT_CHEEK_IDXS + self.RIGHT_CHEEK_IDXS):
            pts.append((int(lms[idx].x * w), int(lms[idx].y * h)))
        return pts

    def _extract_landmarks_legacy(self, rgb: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Extract face landmarks using the legacy FaceMesh API."""
        h, w = rgb.shape[:2]
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        pts = []
        for idx in (self.FOREHEAD_IDXS + self.LEFT_CHEEK_IDXS + self.RIGHT_CHEEK_IDXS):
            pts.append((int(lms[idx].x * w), int(lms[idx].y * h)))
        return pts

    def _crop_from_landmarks(self, rgb: np.ndarray, pts: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Crop and mask ROI from landmark points."""
        h, w = rgb.shape[:2]
        pts_np = np.array(pts, dtype=np.int32)

        x_min = max(int(np.min(pts_np[:, 0])) - 16, 0)
        y_min = max(int(np.min(pts_np[:, 1])) - 16, 0)
        x_max = min(int(np.max(pts_np[:, 0])) + 16, w - 1)
        y_max = min(int(np.max(pts_np[:, 1])) + 16, h - 1)

        crop_rgb = rgb[y_min:y_max + 1, x_min:x_max + 1]
        if crop_rgb.size == 0:
            crop_rgb = rgb
            x_min, y_min = 0, 0

        shifted = pts_np.copy()
        shifted[:, 0] -= x_min
        shifted[:, 1] -= y_min

        mask = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(shifted)
        cv2.fillConvexPoly(mask, hull, color=1)

        roi = cv2.resize(crop_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        mask_rs = cv2.resize(mask.astype(np.float32), (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        return roi.astype(np.float32) / 255.0, mask_rs.astype(np.float32)

    def extract_with_landmarks(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w, _ = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Tasks API path
        if self._api_mode == "tasks":
            pts = self._extract_landmarks_tasks(rgb)
            if pts is not None:
                roi, mask = self._crop_from_landmarks(rgb, pts)
                return roi, mask, np.asarray(pts, dtype=np.float32)

        # Legacy FaceMesh path
        elif self._api_mode == "legacy":
            pts = self._extract_landmarks_legacy(rgb)
            if pts is not None:
                roi, mask = self._crop_from_landmarks(rgb, pts)
                return roi, mask, np.asarray(pts, dtype=np.float32)

        # Cascade fallback
        if self._api_mode == "cascade" or (self._api_mode in ("tasks", "legacy")):
            if self.face_cascade is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade is not None and not self.face_cascade.empty():
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

                if len(faces) > 0:
                    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
                    self.prev_bbox = (int(x), int(y), int(bw), int(bh))
                elif self.prev_bbox is not None:
                    x, y, bw, bh = self.prev_bbox
                else:
                    x, y, bw, bh = 0, 0, w, h

                x0, y0 = max(x, 0), max(y, 0)
                x1, y1 = min(x + bw, w), min(y + bh, h)
                crop_rgb = rgb[y0:y1, x0:x1]
                if crop_rgb.size == 0:
                    crop_rgb = rgb

                mask = np.zeros(crop_rgb.shape[:2], dtype=np.float32)
                ch, cw = crop_rgb.shape[:2]
                center = (cw // 2, int(ch * 0.52))
                axes = (max(1, int(cw * 0.42)), max(1, int(ch * 0.40)))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, thickness=-1)

                roi = cv2.resize(crop_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                mask_rs = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
                return roi.astype(np.float32) / 255.0, mask_rs.astype(np.float32), None

        # Last-resort fallback
        roi = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        mask = np.ones((self.target_size, self.target_size), dtype=np.float32)
        return roi.astype(np.float32) / 255.0, mask, None

    def extract(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backwards-compatible ROI extraction without returning landmarks."""
        roi, mask, _ = self.extract_with_landmarks(bgr_frame)
        return roi, mask


def compute_landmark_motion_displacement(
    landmark_window: List[np.ndarray],
    min_frames: int = 8,
) -> float:
    """Estimate landmark motion amplitude in pixels over a sliding window.

    Returns RMS displacement (pixels) derived from coordinate variance across
    time. Larger values indicate likely talking/head-motion artifacts.
    """
    if len(landmark_window) < min_frames:
        return 0.0

    valid = [np.asarray(x, dtype=np.float64) for x in landmark_window if x is not None]
    if len(valid) < min_frames:
        return 0.0

    # Keep a fixed landmark count across frames.
    n_pts = min(v.shape[0] for v in valid if v.ndim == 2 and v.shape[1] == 2)
    if n_pts <= 0:
        return 0.0

    seq = np.stack([v[:n_pts] for v in valid], axis=0)  # [T, K, 2]
    coord_var = np.var(seq, axis=0)  # [K,2] in px^2
    rms_disp = float(np.sqrt(np.mean(coord_var)))
    if not np.isfinite(rms_disp):
        return np.inf
    return rms_disp


def build_clip_tensor_from_buffer(frame_buffer: deque, device: torch.device) -> torch.Tensor:
    """Convert deque of RGB float frames to model tensor [1, 3, T, H, W].

    Each frame expected shape: [H, W, 3], values in [0,1].
    """
    # Stack list of [H,W,3] into [T,H,W,3].
    clip_thwc = np.stack(list(frame_buffer), axis=0).astype(np.float32)

    # Move channels last -> channels first: [T,H,W,3] -> [T,3,H,W].
    clip_tchw = np.transpose(clip_thwc, (0, 3, 1, 2))

    # Add batch then permute to model expected [B,C,T,H,W].
    # Current with batch is [1,T,3,H,W], then permute dims.
    clip_btchw = np.expand_dims(clip_tchw, axis=0)
    clip_bcthw = np.transpose(clip_btchw, (0, 2, 1, 3, 4))

    return torch.from_numpy(clip_bcthw).to(device=device)


def butter_bandpass_filter_1d(signal: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 3.0, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _welch_peak_hz(
    signal_1d: np.ndarray,
    fps: float,
    f_low: float = 0.8,
    f_high: float = 2.5,
    prev_hr_bpm: Optional[float] = None,
    prior_low_hz: float = 1.0,
    prior_high_hz: float = 1.66,
    snr_ratio_threshold: float = 0.35,
) -> Tuple[float, float]:
    """Robust frequency estimation using Welch's periodogram.

    Returns (peak_freq_hz, snr_db). Returns (0, -inf) on failure.
    """
    T = len(signal_1d)
    if T < 30:
        return 0.0, -np.inf

    sig = signal_1d.astype(np.float64)

    # Detrend: remove mean and linear trend
    t_ax = np.arange(T, dtype=np.float64)
    coeffs = np.polyfit(t_ax, sig, 1)
    sig -= np.polyval(coeffs, t_ax)

    # Strict 6th-order physiological bandpass: 48-150 BPM.
    nyq = 0.5 * fps
    lo = max(f_low / nyq, 0.01)
    hi = min(f_high / nyq, 0.99)
    if lo >= hi or T < 15:
        return 0.0, -np.inf
    b, a = butter(6, [lo, hi], btype="band")
    sig = filtfilt(b, a, sig)

    # Welch's periodogram — averages multiple windowed periodograms for
    # much lower variance than a single FFT
    nperseg = min(T, max(int(4 * fps), 64))
    noverlap = nperseg * 3 // 4
    nfft = max(4096, T * 4)
    freqs, psd = scipy_welch(sig, fs=fps, nperseg=nperseg, noverlap=noverlap,
                             nfft=nfft, window="hann", detrend=False)

    band_mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(band_mask):
        return 0.0, -np.inf

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    if band_psd.max() < 1e-15:
        return 0.0, -np.inf

    # Unbiased peak selection via prominence (no Gaussian/prior weighting).
    # Allows free selection of any dominant bin without artificial center-bias.
    bin_hz = band_freqs[1] - band_freqs[0] if len(band_freqs) > 1 else 0.01
    min_dist_bins = max(3, int(0.15 / bin_hz))  # enforce ≥0.15 Hz between candidate peaks
    peaks_idx, props = find_peaks(band_psd, prominence=0.0, distance=min_dist_bins)
    if len(peaks_idx) > 0:
        # Select the peak with the greatest prominence: highest above local noise floor
        best_pi = int(np.argmax(props["prominences"]))
        peak_idx = peaks_idx[best_pi]
    else:
        # Flat/noisy spectrum: fall back to absolute maximum
        peak_idx = int(np.argmax(band_psd))

    peak_freq = band_freqs[peak_idx]

    dom_mask = (band_freqs >= peak_freq - 0.08) & (band_freqs <= peak_freq + 0.08)
    dom_power = float(band_psd[dom_mask].sum()) if np.any(dom_mask) else float(band_psd[peak_idx])
    total_power = max(float(band_psd.sum()), 1e-15)
    snr_ratio = dom_power / total_power
    if snr_ratio < snr_ratio_threshold:
        return 0.0, -np.inf

    # Sub-harmonic rejection: avoid locking at f0/2 after startup artifacts.
    bpm = float(peak_freq * 60.0)
    if bpm < 55.0:
        harmonic = float(peak_freq * 2.0)
        if harmonic <= f_high:
            h_mask = (band_freqs >= harmonic - 0.10) & (band_freqs <= harmonic + 0.10)
            p_mask = (band_freqs >= peak_freq - 0.10) & (band_freqs <= peak_freq + 0.10)
            h_power = float(band_psd[h_mask].sum()) if np.any(h_mask) else 0.0
            p_power = float(band_psd[p_mask].sum()) if np.any(p_mask) else float(band_psd[peak_idx])
            if h_power >= 0.70 * max(p_power, 1e-15):
                peak_freq = harmonic

    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_idx < len(band_psd) - 1:
        lp = np.log(band_psd[peak_idx - 1] + 1e-15)
        cp = np.log(band_psd[peak_idx] + 1e-15)
        rp = np.log(band_psd[peak_idx + 1] + 1e-15)
        denom = lp - 2.0 * cp + rp
        if abs(denom) > 1e-10:
            df = band_freqs[1] - band_freqs[0] if len(band_freqs) > 1 else 0.0
            peak_freq += 0.5 * (lp - rp) / denom * df

    peak_freq = float(np.clip(peak_freq, f_low, f_high))

    # SNR: prominence-selected signal power vs rest of band
    sig_mask = (band_freqs >= peak_freq - 0.2) & (band_freqs <= peak_freq + 0.2)
    sig_power = float(band_psd[sig_mask].sum()) if np.any(sig_mask) else float(band_psd[peak_idx])
    noise_power = max(float(band_psd.sum()) - sig_power, 1e-15)
    snr = 10.0 * np.log10(sig_power / noise_power)
    snr = float(np.clip(snr, -10.0, 30.0))

    return peak_freq, snr


def _extract_rgb_trace(frame_buffer, normalize_size: int = 128) -> np.ndarray:
    """Extract spatially-averaged RGB time series from face crops.

    Normalizes each frame to a fixed spatial resolution before averaging
    to eliminate distance-dependent signal amplitude bias.

    Returns: [T, 3] float64 array
    """
    T = len(frame_buffer)
    rgb = np.zeros((T, 3), dtype=np.float64)
    for i, frame in enumerate(frame_buffer):
        # Resize to fixed resolution to remove distance dependency
        if frame.shape[0] != normalize_size or frame.shape[1] != normalize_size:
            resized = cv2.resize(frame, (normalize_size, normalize_size), interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        rgb[i] = resized.mean(axis=(0, 1))
    return rgb


def _extract_patch_rgb_traces(
    frame_buffer,
    normalize_size: int = 128,
    grid_size: int = 4,
    min_patch_coverage: float = 0.35,
) -> np.ndarray:
    """Extract patch-wise RGB traces from a masked ROI sequence.

    Splits each normalized ROI into a grid and averages only non-zero pixels in
    each patch. Patches that fall mostly outside the face mask are discarded.

    Returns: [T, P, 3] float64 array where P is the number of valid patches.
    """
    T = len(frame_buffer)
    if T == 0:
        return np.zeros((0, 0, 3), dtype=np.float64)

    resized_frames: List[np.ndarray] = []
    for frame in frame_buffer:
        if frame.shape[0] != normalize_size or frame.shape[1] != normalize_size:
            resized = cv2.resize(frame, (normalize_size, normalize_size), interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        resized_frames.append(np.asarray(resized, dtype=np.float64))

    edges = np.linspace(0, normalize_size, grid_size + 1, dtype=np.int32)
    patch_traces: List[np.ndarray] = []

    for gy in range(grid_size):
        y0, y1 = int(edges[gy]), int(edges[gy + 1])
        for gx in range(grid_size):
            x0, x1 = int(edges[gx]), int(edges[gx + 1])
            patch_rgb = np.zeros((T, 3), dtype=np.float64)
            coverages = np.zeros(T, dtype=np.float64)

            for idx, resized in enumerate(resized_frames):
                patch = resized[y0:y1, x0:x1]
                if patch.size == 0:
                    continue

                valid_mask = np.any(patch > 1e-6, axis=2)
                coverages[idx] = float(np.mean(valid_mask)) if valid_mask.size > 0 else 0.0
                if np.any(valid_mask):
                    patch_rgb[idx] = patch[valid_mask].mean(axis=0)

            if float(np.mean(coverages)) >= min_patch_coverage:
                patch_traces.append(patch_rgb)

    if not patch_traces:
        return np.zeros((T, 0, 3), dtype=np.float64)

    return np.stack(patch_traces, axis=1)


def _resample_with_timestamps(
    signal: np.ndarray,
    timestamps: Optional[np.ndarray],
    target_fs: float = 30.0,
) -> Tuple[np.ndarray, float]:
    """Resample temporal signal to a strict uniform timeline with linear interp."""
    x = np.asarray(signal)
    if timestamps is None:
        return x, float(target_fs)

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if x.shape[0] != ts.shape[0] or ts.shape[0] < 2:
        return x, float(target_fs)

    # Enforce strictly increasing timestamps before interpolation.
    keep = np.concatenate(([True], np.diff(ts) > 1e-6))
    ts = ts[keep]
    x = x[keep]
    if ts.shape[0] < 2:
        return x, float(target_fs)

    t0 = float(ts[0])
    t1 = float(ts[-1])
    duration = t1 - t0
    if duration <= 0.1:
        return x, float(target_fs)

    num_frames = int(ts.shape[0])
    t_uniform = np.linspace(t0, t1, num_frames, endpoint=True, dtype=np.float64)

    x_flat = x.reshape(x.shape[0], -1).astype(np.float64)
    interpolator = interp1d(
        ts,
        x_flat,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value=(x_flat[0], x_flat[-1]),
        assume_sorted=True,
    )
    y_flat = interpolator(t_uniform).astype(np.float64)
    y = y_flat.reshape((num_frames,) + x.shape[1:])
    fs_uniform = float((num_frames - 1) / duration) if duration > 1e-8 and num_frames > 1 else float(target_fs)

    return y, fs_uniform


def _detrend_pulse_waveform(signal_1d: np.ndarray) -> np.ndarray:
    """Apply light detrending after color projection, not before it."""
    sig = np.asarray(signal_1d, dtype=np.float64).reshape(-1)
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    if sig.size < 3:
        return sig

    t_ax = np.arange(sig.size, dtype=np.float64)
    coeffs = np.polyfit(t_ax, sig, 1)
    return sig - np.polyval(coeffs, t_ax)


def _project_rgb_to_pulse(rgb: np.ndarray, fs_used: float, method: str) -> np.ndarray:
    """Project raw RGB traces into a 1D rPPG waveform using POS or CHROM."""
    rgb = np.asarray(rgb, dtype=np.float64)
    T = rgb.shape[0]
    win_len = int(1.6 * fs_used)
    win_len = max(win_len, 20)

    pulse = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)

    for start in range(0, T - win_len + 1):
        end = start + win_len
        segment = rgb[start:end]
        mean_seg = segment.mean(axis=0, keepdims=True)
        mean_seg[mean_seg < 1e-6] = 1.0
        cn = segment / mean_seg

        if method == "pos":
            s1 = cn[:, 1] - cn[:, 2]
            s2 = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]
            alpha = np.std(s1) / (np.std(s2) + 1e-8)
            h = s1 + alpha * s2
        elif method == "chrom":
            xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
            ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
            alpha = np.std(xs) / (np.std(ys) + 1e-8)
            h = xs - alpha * ys
        else:
            raise ValueError(f"Unsupported projection method: {method}")

        h = h - np.mean(h)
        pulse[start:end] += h
        count[start:end] += 1.0

    count[count < 1] = 1.0
    pulse /= count
    return _detrend_pulse_waveform(pulse)


def _extract_patch_weighted_pulse(
    frame_buffer,
    timestamps: Optional[np.ndarray],
    target_fs: float,
    method: str,
    normalize_size: int = 128,
    grid_size: int = 4,
) -> Tuple[np.ndarray, float]:
    """Build a patch-weighted pulse waveform from a masked ROI sequence."""
    patch_rgb = _extract_patch_rgb_traces(
        frame_buffer,
        normalize_size=normalize_size,
        grid_size=grid_size,
    )

    if patch_rgb.shape[1] == 0:
        rgb = _extract_rgb_trace(frame_buffer, normalize_size=normalize_size)
        rgb, fs_used = _resample_with_timestamps(rgb, timestamps=timestamps, target_fs=target_fs)
        return _project_rgb_to_pulse(rgb, fs_used, method=method), fs_used

    patch_rgb, fs_used = _resample_with_timestamps(patch_rgb, timestamps=timestamps, target_fs=target_fs)

    patch_waveforms: List[np.ndarray] = []
    patch_variances: List[float] = []
    for patch_idx in range(patch_rgb.shape[1]):
        waveform = _project_rgb_to_pulse(patch_rgb[:, patch_idx, :], fs_used, method=method)
        variance = float(np.var(waveform))
        if not np.all(np.isfinite(waveform)) or variance < 1e-10:
            continue
        patch_waveforms.append(waveform)
        patch_variances.append(variance)

    if not patch_waveforms:
        rgb = _extract_rgb_trace(frame_buffer, normalize_size=normalize_size)
        rgb, fs_used = _resample_with_timestamps(rgb, timestamps=timestamps, target_fs=target_fs)
        return _project_rgb_to_pulse(rgb, fs_used, method=method), fs_used

    weights = 1.0 / np.maximum(np.asarray(patch_variances, dtype=np.float64), 1e-10)
    weights /= np.sum(weights)
    combined = np.sum(np.asarray(patch_waveforms, dtype=np.float64) * weights[:, None], axis=0)
    return _detrend_pulse_waveform(combined), fs_used


def _rolling_zscore(signal: np.ndarray, window: int, eps: float = 1e-8) -> np.ndarray:
    """Rolling mean-centering and normalization for drift-robust preprocessing."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x2 = x[:, None]
    else:
        x2 = x

    n, c = x2.shape
    w = int(max(3, window))
    out = np.zeros_like(x2, dtype=np.float64)

    for ch in range(c):
        v = x2[:, ch]
        for i in range(n):
            s = max(0, i - w + 1)
            seg = v[s:i + 1]
            mu = float(np.mean(seg))
            sd = float(np.std(seg))
            if sd < eps:
                sd = 1.0
            out[i, ch] = (v[i] - mu) / sd

    return out[:, 0] if x.ndim == 1 else out


def estimate_hr_from_autocorr(
    signal_1d: np.ndarray,
    fps: float,
    f_low: float = 0.8,
    f_high: float = 2.5,
    timestamps: Optional[np.ndarray] = None,
    target_fs: float = 30.0,
) -> Tuple[float, float]:
    """Estimate HR with autocorrelation periodicity search on a bandpassed signal.

    Returns: (hr_bpm, periodicity_score)
    """
    sig = np.asarray(signal_1d, dtype=np.float64).reshape(-1)
    sig, fs_used = _resample_with_timestamps(sig, timestamps=timestamps, target_fs=target_fs)
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    sig = _rolling_zscore(sig, window=max(5, int(2.0 * fs_used)))
    T = sig.size
    if T < 45:
        return 0.0, 0.0

    sig = sig - np.mean(sig)
    if float(np.std(sig)) < 1e-8:
        return 0.0, 0.0

    nyq = 0.5 * fs_used
    low = max(f_low / nyq, 0.01)
    high = min(f_high / nyq, 0.99)
    if low >= high:
        return 0.0, 0.0

    b, a = butter(6, [low, high], btype="band")
    sig = filtfilt(b, a, sig)
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    if std < 1e-8:
        return 0.0, 0.0

    sig = sig / std
    ac = np.correlate(sig, sig, mode="full")
    ac = ac[T - 1:]
    ac0 = float(ac[0]) if ac.size > 0 else 0.0
    if ac0 <= 1e-10:
        return 0.0, 0.0
    ac = ac / ac0

    min_lag = max(1, int(np.floor(fs_used / f_high)))
    max_lag = min(T - 1, int(np.ceil(fs_used / f_low)))
    if min_lag >= max_lag:
        return 0.0, 0.0

    search = ac[min_lag:max_lag + 1]
    if search.size == 0:
        return 0.0, 0.0
    peak_rel = int(np.argmax(search))
    peak_lag = min_lag + peak_rel
    periodicity = float(np.clip(search[peak_rel], 0.0, 1.0))
    if peak_lag <= 0:
        return 0.0, 0.0

    hr = 60.0 * fs_used / float(peak_lag)
    hr = float(np.clip(hr, f_low * 60.0, f_high * 60.0))
    return hr, periodicity


def estimate_hr_from_rgb_pos(
    frame_buffer,
    fps: float,
    f_low: float = 0.8,
    f_high: float = 2.5,
    timestamps: Optional[np.ndarray] = None,
    target_fs: float = 30.0,
    prev_hr_bpm: Optional[float] = None,
) -> Tuple[float, float]:
    """POS (Wang et al., 2017) — Plane-Orthogonal-to-Skin.

    Uses Welch's periodogram for robust spectral estimation.
    returns: (HR in BPM, SNR in dB)
    """
    T = len(frame_buffer)
    if T < 30:
        return 0.0, -np.inf

    pulse, fs_used = _extract_patch_weighted_pulse(
        frame_buffer,
        timestamps=timestamps,
        target_fs=target_fs,
        method="pos",
    )

    peak_hz, snr = _welch_peak_hz(pulse, fs_used, f_low=f_low, f_high=f_high, prev_hr_bpm=prev_hr_bpm)
    return peak_hz * 60.0, snr


def estimate_hr_from_rgb_chrom(
    frame_buffer,
    fps: float,
    f_low: float = 0.8,
    f_high: float = 2.5,
    timestamps: Optional[np.ndarray] = None,
    target_fs: float = 30.0,
    prev_hr_bpm: Optional[float] = None,
) -> Tuple[float, float]:
    """CHROM (De Haan & Jeanne, 2013) — chrominance-based rPPG.

    Uses Welch's periodogram for robust spectral estimation.
    returns: (HR in BPM, SNR in dB)
    """
    T = len(frame_buffer)
    if T < 30:
        return 0.0, -np.inf

    pulse, fs_used = _extract_patch_weighted_pulse(
        frame_buffer,
        timestamps=timestamps,
        target_fs=target_fs,
        method="chrom",
    )

    peak_hz, snr = _welch_peak_hz(pulse, fs_used, f_low=f_low, f_high=f_high, prev_hr_bpm=prev_hr_bpm)
    return peak_hz * 60.0, snr


def estimate_hr_from_rgb_green(
    frame_buffer,
    fps: float,
    f_low: float = 0.8,
    f_high: float = 2.5,
    timestamps: Optional[np.ndarray] = None,
    target_fs: float = 30.0,
    prev_hr_bpm: Optional[float] = None,
) -> Tuple[float, float]:
    """Green-channel rPPG — hemoglobin absorption peaks at ~540nm (green).

    Simple but effective baseline. Uses Welch's periodogram.
    returns: (HR in BPM, SNR in dB)
    """
    T = len(frame_buffer)
    if T < 30:
        return 0.0, -np.inf

    rgb = _extract_rgb_trace(frame_buffer)
    rgb, fs_used = _resample_with_timestamps(rgb, timestamps=timestamps, target_fs=target_fs)
    rgb = _rolling_zscore(rgb, window=max(5, int(2.0 * fs_used)))
    green = rgb[:, 1]

    peak_hz, snr = _welch_peak_hz(green, fs_used, f_low=f_low, f_high=f_high, prev_hr_bpm=prev_hr_bpm)
    return peak_hz * 60.0, snr


def estimate_hr_peak_counting(bvp: np.ndarray, fps: float, f_low: float = 0.7, f_high: float = 3.0) -> float:
    """Estimate HR by counting peaks in the bandpass-filtered BVP signal.

    This is independent from FFT-based methods and serves as validation.
    Returns HR in BPM, or 0.0 if insufficient peaks.
    """
    T = len(bvp)
    if T < 30:
        return 0.0

    sig = bvp.astype(np.float64)
    sig = sig - np.mean(sig)

    nyq = 0.5 * fps
    low = max(f_low / nyq, 0.01)
    high = min(f_high / nyq, 0.99)
    if low >= high or T < 15:
        return 0.0

    b, a = butter(3, [low, high], btype="band")
    filtered = filtfilt(b, a, sig)

    # Minimum distance between peaks: based on max HR
    min_dist = max(1, int(fps * 60.0 / (f_high * 60.0)))
    peaks, props = find_peaks(filtered, distance=min_dist, height=0)

    if len(peaks) < 3:
        return 0.0

    # Inter-beat intervals
    ibis = np.diff(peaks) / fps  # seconds
    ibis = ibis[(ibis > 0.33) & (ibis < 1.5)]  # 40-180 BPM range

    if len(ibis) < 2:
        return 0.0

    mean_ibi = float(np.median(ibis))
    hr = 60.0 / mean_ibi
    return float(np.clip(hr, f_low * 60.0, f_high * 60.0))


def compute_pulse_snr(bvp: np.ndarray, fps: float, f_low: float = 0.667, f_high: float = 4.0) -> float:
    """Compute signal-to-noise ratio (dB) using Welch's periodogram."""
    _, snr = _welch_peak_hz(bvp, fps, f_low=f_low, f_high=f_high)
    return snr


def estimate_spo2_from_rgb(frame_buffer, fps: float) -> float:
    """Estimate SpO2 from RGB ROI sequence using robust red/green AC ratio.

    Notes:
    - Uses green (not blue) as the non-red channel proxy for RGB cameras.
    - Computes AC amplitude with percentile peak-to-peak for outlier robustness.
    - Uses POS pulse quality as a patch weight to reduce motion/illumination artifacts.
    - Rejects flat/low-variance windows to avoid sticky constant outputs.
    """
    T = len(frame_buffer)
    if T < 60 or fps <= 1.0:
        return 0.0

    nyq = 0.5 * fps
    low = max(0.8 / nyq, 0.01)
    high = min(2.5 / nyq, 0.99)
    if low >= high:
        return 0.0

    b, a = butter(3, [low, high], btype="band")

    def robust_ptp(x: np.ndarray) -> float:
        return float(np.percentile(x, 95) - np.percentile(x, 5))

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        idx = int(np.searchsorted(cw, 0.5, side="left"))
        idx = int(np.clip(idx, 0, v.size - 1))
        return float(v[idx])

    patch_rgb = _extract_patch_rgb_traces(
        frame_buffer,
        normalize_size=128,
        grid_size=4,
        min_patch_coverage=0.35,
    )

    if patch_rgb.shape[1] == 0:
        rgb = _extract_rgb_trace(frame_buffer, normalize_size=128)
        patch_rgb = rgb[:, None, :]

    r_values: List[float] = []
    weights: List[float] = []

    for p_idx in range(patch_rgb.shape[1]):
        rgb_p = np.asarray(patch_rgb[:, p_idx, :], dtype=np.float64)
        red = rgb_p[:, 0]
        green = rgb_p[:, 1]

        # Keep slow baseline as DC; normalize first to avoid detrend-induced flattening.
        dc_red = float(np.median(red))
        dc_green = float(np.median(green))
        if dc_red <= 1e-6 or dc_green <= 1e-6:
            continue

        red_norm = red / (dc_red + 1e-6) - 1.0
        green_norm = green / (dc_green + 1e-6) - 1.0

        if np.std(red_norm) < 6e-5 or np.std(green_norm) < 6e-5:
            continue

        try:
            ac_red = filtfilt(b, a, red_norm)
            ac_green = filtfilt(b, a, green_norm)
        except Exception:
            continue

        ac_red_amp = robust_ptp(ac_red)
        ac_green_amp = robust_ptp(ac_green)

        # Flat-window rejection to prevent sticky ~constant SpO2.
        if ac_red_amp < 6e-5 or ac_green_amp < 6e-5:
            continue

        r = ac_red_amp / (ac_green_amp + 1e-6)
        if not np.isfinite(r) or r <= 0.0 or r > 2.5:
            continue

        pos_wave = _project_rgb_to_pulse(rgb_p, fs_used=fps, method="pos")
        _, pos_snr = _welch_peak_hz(pos_wave, fps=fps, f_low=0.8, f_high=2.5)
        if not np.isfinite(pos_snr):
            pos_snr = -10.0

        # Weight by pulse quality + channel AC strength.
        w = float(max(0.0, pos_snr + 8.0)) + 60.0 * min(ac_red_amp, ac_green_amp)
        if w <= 0.0:
            continue

        r_values.append(float(r))
        weights.append(w)

    if not r_values:
        rgb = _extract_rgb_trace(frame_buffer, normalize_size=128)
        red = np.asarray(rgb[:, 0], dtype=np.float64)
        green = np.asarray(rgb[:, 1], dtype=np.float64)

        dc_red = float(np.median(red))
        dc_green = float(np.median(green))
        if dc_red <= 1e-6 or dc_green <= 1e-6:
            return 0.0

        red_norm = red / (dc_red + 1e-6) - 1.0
        green_norm = green / (dc_green + 1e-6) - 1.0
        if np.std(red_norm) < 6e-5 or np.std(green_norm) < 6e-5:
            return 0.0

        try:
            ac_red = filtfilt(b, a, red_norm)
            ac_green = filtfilt(b, a, green_norm)
        except Exception:
            return 0.0

        ac_red_amp = robust_ptp(ac_red)
        ac_green_amp = robust_ptp(ac_green)
        if ac_red_amp < 6e-5 or ac_green_amp < 6e-5:
            return 0.0

        r = ac_red_amp / (ac_green_amp + 1e-6)
        if not np.isfinite(r) or r <= 0.0 or r > 2.5:
            return 0.0
        r_values = [float(r)]
        weights = [1.0]

    w_arr = np.asarray(weights, dtype=np.float64)
    w_arr = np.maximum(w_arr, 1e-6)
    w_arr = w_arr / np.sum(w_arr)
    r_arr = np.asarray(r_values, dtype=np.float64)
    # Reject extreme patch ratios that often come from local lighting or motion.
    if r_arr.size >= 3:
        lo = float(np.percentile(r_arr, 15))
        hi = float(np.percentile(r_arr, 85))
        keep = (r_arr >= lo) & (r_arr <= hi)
        if np.sum(keep) >= 2:
            r_arr = r_arr[keep]
            w_arr = w_arr[keep]
            w_arr = np.maximum(w_arr, 1e-6)
            w_arr = w_arr / np.sum(w_arr)

    # Weighted median is less sensitive to high-R outliers that bias SpO2 low.
    r_eff = weighted_median(r_arr, w_arr)
    r_eff = float(np.clip(r_eff, 0.35, 1.25))

    # Bias-corrected calibration for webcam RGB (subject to device-specific tuning).
    spo2 = 109.0 - 15.0 * r_eff
    return float(np.clip(spo2, 90.0, 100.0))


def estimate_respiratory_rate(bvp: np.ndarray, fps: float) -> float:
    """Estimate respiratory rate from BVP amplitude modulation.

    Breathing modulates pulse amplitude via respiratory sinus arrhythmia.
    Returns breaths per minute (0.0 if insufficient data).
    """
    T = len(bvp)
    if T < 90:
        return 0.0

    sig = bvp.astype(np.float64)
    sig = sig - np.mean(sig)

    if np.std(sig) < 1e-8:
        return 0.0

    # Envelope via Hilbert transform
    analytic = hilbert(sig)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)

    # Bandpass in respiratory band: 0.15-0.4 Hz (9-24 breaths/min)
    nyq = 0.5 * fps
    resp_low = max(0.15 / nyq, 0.01)
    resp_high = min(0.4 / nyq, 0.99)
    if resp_low >= resp_high:
        return 0.0

    b, a = butter(2, [resp_low, resp_high], btype="band")
    resp_sig = filtfilt(b, a, envelope)

    nfft = max(1024, T * 4)
    window = np.hanning(T)
    spec = np.abs(np.fft.rfft(resp_sig * window, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)

    mask = (freqs >= 0.15) & (freqs <= 0.4)
    resp_power = spec[mask]
    resp_freqs = freqs[mask]

    if len(resp_power) == 0 or resp_power.max() < 1e-12:
        return 0.0

    peak_idx = int(np.argmax(resp_power))
    resp_freq = resp_freqs[peak_idx]

    return float(np.clip(resp_freq * 60.0, 8.0, 25.0))


def compute_signal_quality_index(bvp: np.ndarray, fps: float) -> float:
    """Compute signal quality index (0-100).

    Combines SNR, temporal periodicity, and peak regularity into
    a single quality score for clinical confidence assessment.
    """
    T = len(bvp)
    if T < 30:
        return 0.0

    sig = bvp.astype(np.float64)
    sig = sig - np.mean(sig)

    # Component 1: SNR (0-40 points)
    snr = compute_pulse_snr(bvp, fps)
    snr_score = float(np.clip((snr + 5.0) * 4.0, 0.0, 40.0))

    # Component 2: Autocorrelation periodicity (0-30 points)
    period_score = 0.0
    if np.std(sig) > 1e-8:
        norm = sig / np.std(sig)
        ac = np.correlate(norm, norm, mode="full")
        ac = ac[T - 1:]  # positive lags
        ac /= max(ac[0], 1e-10)

        min_lag = max(1, int(fps * 0.4))   # 150 BPM
        max_lag = min(len(ac), int(fps * 1.5))  # 40 BPM
        if max_lag > min_lag:
            search = ac[min_lag:max_lag]
            if len(search) > 0:
                period_score = float(np.clip(np.max(search) * 40.0, 0.0, 30.0))

    # Component 3: Peak regularity (0-30 points)
    reg_score = 0.0
    nyq = 0.5 * fps
    low = max(0.7 / nyq, 0.01)
    high = min(3.0 / nyq, 0.99)
    if low < high and T > 15:
        b, a = butter(3, [low, high], btype="band")
        filt = filtfilt(b, a, sig)
        peaks, _ = find_peaks(filt, distance=max(1, int(fps * 0.4)), height=0)
        if len(peaks) >= 3:
            ibi = np.diff(peaks) / fps
            cv = np.std(ibi) / max(np.mean(ibi), 1e-8)
            reg_score = float(np.clip((1.0 - cv) * 40.0, 0.0, 30.0))

    total = snr_score + period_score + reg_score
    return float(np.clip(total, 0.0, 100.0))
