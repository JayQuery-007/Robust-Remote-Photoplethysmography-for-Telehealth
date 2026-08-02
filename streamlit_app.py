from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch

try:
    from equiphys_v2 import EquiPhysDANNV2 as _DeepModelCls
    _MODEL_VERSION = "v2"
except ImportError:
    from equiphys_core import EquiPhysDANN as _DeepModelCls  # type: ignore
    _MODEL_VERSION = "v1"

from equiphys_core import (
    compute_landmark_motion_displacement,
    EquiPhysDANN,
    FaceROIExtractor,
    build_clip_tensor_from_buffer,
    compute_pulse_snr,
    compute_signal_quality_index,
    estimate_hr_from_autocorr,
    estimate_hr_bpm_from_bvp,
    estimate_hr_from_rgb_chrom,
    estimate_hr_from_rgb_pos,
    estimate_respiratory_rate,
    estimate_spo2_from_rgb,
    test_time_adaptation,
)

from facial_vitals import (
    SRC_RATIO_CAL,
    VitalsCalibration,
    clean_frame_mask,
    extract_pulse_morphology,
    frame_diff_motion,
    reconstruct_facial_bvp,
    select_clean_frames,
    spo2_from_faces,
)
from streamlit_vitals_integration import sidebar_calibration_controls
from camera_calibration import (
    CameraCalibration,
    sidebar_camera_calibration_panel,
    apply_camera_calibration,
    feed_calibration_frame,
)
from video_analysis import render_video_analysis_tab


def _inject_css(is_light: bool = False) -> None:
    if is_light:
        root_vars = """
        :root {
            --bg-color: #f8fafc;
            --gradient-1: rgba(59, 130, 246, 0.05);
            --gradient-2: rgba(139, 92, 246, 0.04);
            --text-color: #0f172a;
            --text-muted: #64748b;
            --text-dark: #334155;
            --panel-bg: rgba(255, 255, 255, 0.85);
            --panel-border: rgba(15, 23, 42, 0.08);
            --panel-hover-border: rgba(59, 130, 246, 0.3);
            --card-bg: #ffffff;
            --card-border: rgba(15, 23, 42, 0.08);
            --card-hover-border: rgba(59, 130, 246, 0.25);
            --card-text: #0f172a;
            --card-subtext: #64748b;
            --sidebar-bg: #f1f5f9;
            --sidebar-border: rgba(15, 23, 42, 0.08);
            --tab-color: #64748b;
            --tab-selected: #3b82f6;
            --tab-hover: #0f172a;
            --tab-border: rgba(15, 23, 42, 0.08);
            --expander-bg: #ffffff;
            --expander-border: rgba(15, 23, 42, 0.08);
            --terminal-bg: #f8fafc;
            --terminal-header: #e2e8f0;
            --terminal-border: rgba(15, 23, 42, 0.08);
            --terminal-text: #475569;
            --header-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(241, 245, 249, 0.8) 100%);
            --header-border-top: rgba(15, 23, 42, 0.04);
            --progress-bg: rgba(15, 23, 42, 0.04);
            --alert-bg: rgba(255, 255, 255, 0.95);
            --alert-border: rgba(15, 23, 42, 0.06);
            --scrollbar-thumb: #cbd5e1;
            --scrollbar-thumb-hover: #94a3b8;
            --scrollbar-track: rgba(241, 245, 249, 0.8);
            --input-bg: #ffffff;
            --input-border: rgba(15, 23, 42, 0.12);
        }
        """
    else:
        root_vars = """
        :root {
            --bg-color: #0b0f19;
            --gradient-1: rgba(59, 130, 246, 0.12);
            --gradient-2: rgba(139, 92, 246, 0.08);
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --text-dark: #cbd5e1;
            --panel-bg: rgba(15, 23, 42, 0.45);
            --panel-border: rgba(255, 255, 255, 0.06);
            --panel-hover-border: rgba(59, 130, 246, 0.2);
            --card-bg: rgba(10, 15, 28, 0.7);
            --card-border: rgba(255, 255, 255, 0.05);
            --card-hover-border: rgba(255, 255, 255, 0.12);
            --card-text: #ffffff;
            --card-subtext: #64748b;
            --sidebar-bg: #070a13;
            --sidebar-border: rgba(255, 255, 255, 0.05);
            --tab-color: #64748b;
            --tab-selected: #3b82f6;
            --tab-hover: #f1f5f9;
            --tab-border: rgba(255, 255, 255, 0.05);
            --expander-bg: rgba(15, 23, 42, 0.4);
            --expander-border: rgba(255, 255, 255, 0.06);
            --terminal-bg: #05080f;
            --terminal-header: #0d131f;
            --terminal-border: rgba(255, 255, 255, 0.05);
            --terminal-text: #8b9bb4;
            --header-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            --header-border-top: rgba(255, 255, 255, 0.04);
            --progress-bg: rgba(255, 255, 255, 0.04);
            --alert-bg: rgba(15, 23, 42, 0.6);
            --alert-border: rgba(255, 255, 255, 0.05);
            --scrollbar-thumb: #1e293b;
            --scrollbar-thumb-hover: #334155;
            --scrollbar-track: rgba(15, 23, 42, 0.3);
            --input-bg: rgba(10, 15, 28, 0.8);
            --input-border: rgba(255, 255, 255, 0.08);
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        {root_vars}
        
        /* Global Background and Canvas Overrides */
        .stApp {{
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, var(--gradient-1) 0px, transparent 50%),
                radial-gradient(at 100% 0%, var(--gradient-2) 0px, transparent 50%);
            background-attachment: fixed;
        }}
        
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-color);
        }}
        
        /* Custom Scrollbars */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--scrollbar-track);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--scrollbar-thumb);
            border-radius: 4px;
            border: 2px solid var(--bg-color);
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--scrollbar-thumb-hover);
        }}
        
        .block-container {{ 
            padding-top: 0.4rem; 
            padding-bottom: 0.4rem; 
            max-width: 100% !important; 
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border);
        }}
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            color: var(--text-color);
            letter-spacing: -0.02em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }}

        /* Streamlit Tabs Customization */
        button[data-baseweb="tab"] {{
            background-color: transparent !important;
            color: var(--tab-color) !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            font-size: 13px !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: all 0.2s ease !important;
            letter-spacing: 0.3px;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--tab-selected) !important;
            border-bottom: 2px solid var(--tab-selected) !important;
        }}
        button[data-baseweb="tab"]:hover {{
            color: var(--tab-hover) !important;
        }}
        div[data-testid="stTabBar"] {{
            border-bottom: 1px solid var(--tab-border);
            margin-bottom: 12px;
        }}

        /* Expander Customization */
        div[data-testid="stExpander"] {{
            background-color: var(--expander-bg) !important;
            border: 1px solid var(--expander-border) !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        
        /* Modernized Header Styling */
        .header-shell {{
            background: var(--header-bg);
            backdrop-filter: blur(8px);
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
            padding: 10px 20px;
            margin-bottom: 12px;
            border-top: 1px solid var(--header-border-top);
            border-right: 1px solid var(--panel-border);
            border-bottom: 1px solid var(--panel-border);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        }}
        .title-row {{
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .title-text {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 20px; 
            font-weight: 800; 
            letter-spacing: -0.02em;
            color: var(--text-color);
        }}
        .live-pill {{ 
            color: #10b981; 
            background: rgba(16, 185, 129, 0.1); 
            border: 1px solid rgba(16, 185, 129, 0.2); 
            padding: 4px 10px; 
            border-radius: 9999px; 
            font-size: 11px; 
            font-weight: 700;
            letter-spacing: 0.5px;
            display: inline-flex;
            align-items: center;
        }}
        
        /* CSS Live Indicator Dot (replaces Unicode emoji bullet) */
        .live-indicator-dot {{
            display: inline-block;
            width: 7px;
            height: 7px;
            background-color: #10b981;
            border-radius: 50%;
            margin-right: 6px;
            animation: live-pulse 1.8s infinite alternate;
        }}
        @keyframes live-pulse {{
            0% {{ transform: scale(0.85); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 4px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.85); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        
        /* Modern Glassmorphic Panels */
        .panel-wrap {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
        }}
        .panel-wrap:hover {{
            border-color: var(--panel-hover-border);
        }}
        .panel-title {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-weight: 700;
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 12px;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 8px;
        }}
        
        /* Custom EQUI-SIZED Metric Cards */
        .metric-shell-new {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px 4px;
            text-align: center;
            height: 130px;
            min-height: 130px;
            max-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .metric-shell-new:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px var(--glow-color), 0 3px 8px rgba(0, 0, 0, 0.15);
            border-color: var(--card-hover-border);
        }}
        .metric-label-new {{ 
            color: var(--text-muted) !important; 
            font-size: 9px; 
            font-weight: 700; 
            letter-spacing: 0.4px; 
            text-transform: uppercase;
            margin-bottom: 2px;
        }}
        .metric-value-new {{ 
            font-size: 32px; 
            font-weight: 700; 
            color: var(--card-text) !important; 
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
        }}
        .metric-unit-new {{ 
            color: var(--text-muted) !important; 
            font-size: 11px; 
            font-weight: 600; 
            margin-top: 1px;
        }}
        .metric-status {{
            color: var(--card-subtext) !important;
            font-size: 9px;
            margin-top: 4px;
            font-weight: 500;
            letter-spacing: 0.1px;
        }}
        
        /* Scaled Webcam Image (Viewport-based scaling) */
        div[data-testid="stImage"] img {{
            max-height: 52vh !important;
            object-fit: contain;
            border-radius: 8px;
        }}

        /* Sleek Log Terminal */
        .log-terminal-wrap {{
            border: 1px solid var(--terminal-border);
            border-radius: 6px;
            overflow: hidden;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);
        }}
        .log-terminal-header {{
            background-color: var(--terminal-header);
            border-bottom: 1px solid var(--terminal-border);
            padding: 6px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .log-terminal-dots {{
            display: flex;
            gap: 5px;
        }}
        .log-terminal-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }}
        .dot-red {{ background-color: #ef4444; }}
        .dot-yellow {{ background-color: #f59e0b; }}
        .dot-green {{ background-color: #10b981; }}
        .log-terminal-title {{
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 500;
        }}
        .logbox {{
            background: var(--terminal-bg);
            height: 54vh !important;
            min-height: 54vh !important;
            max-height: 54vh !important;
            overflow-y: auto;
            padding: 10px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            color: var(--terminal-text);
            font-size: 11px;
            line-height: 1.5;
            white-space: pre-wrap;
        }}
        
        /* Custom Progress Styling override */
        div[data-testid="stProgress"] > div:has(div[role="progressbar"]) {{
            background-color: var(--progress-bg) !important;
            border-radius: 9999px;
            height: 6px;
        }}
        div[data-testid="stProgress"] div[role="progressbar"] {{
            background: linear-gradient(90deg, #3b82f6 0%, #a855f7 100%) !important;
            border-radius: 9999px;
        }}
        div[data-testid="stProgress"] > div > div > div {{
            font-size: 10px !important;
            color: var(--text-muted) !important;
            font-weight: 500 !important;
        }}
        
        /* High Contrast Overrides for Sidebar Controls */
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] summary span {{
            color: var(--text-color) !important;
        }}

        /* High Contrast Checkbox Labels */
        div[data-testid="stCheckbox"] span,
        div[data-testid="stCheckbox"] p,
        div[data-testid="stCheckbox"] label {{
            color: var(--text-color) !important;
        }}

        /* Info/Success/Warning/Error Alert Banners text contrast */
        div[data-testid="stNotification"] {{
            background-color: var(--alert-bg) !important;
            border: 1px solid var(--alert-border) !important;
            border-radius: 6px !important;
            backdrop-filter: blur(4px);
            padding: 10px 14px !important;
        }}
        div[data-testid="stNotification"] *,
        div[data-testid="stNotification"] p,
        div[data-testid="stNotification"] span,
        div[data-testid="stNotification"] div,
        div[data-testid="stNotification"] li,
        div[data-testid="stNotification"] ul {{
            color: var(--text-dark) !important;
            font-weight: 600 !important;
        }}

        /* Progress Bar text contrast */
        div[data-testid="stProgress"] *,
        div[data-testid="stProgress"] span,
        div[data-testid="stProgress"] p,
        div[data-testid="stProgress"] div {{
            color: var(--text-color) !important;
            font-weight: 500 !important;
        }}

        /* Dropdowns and select boxes styling */
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 4px !important;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stSelectbox"] span,
        div[data-testid="stSelectbox"] p {{
            color: var(--text-color) !important;
        }}
        div[role="listbox"] li {{
            color: #0f172a !important; /* Keep select options readable */
        }}

        /* Text & Number Inputs styling override */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea {{
            background-color: var(--input-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 4px !important;
        }}
        div[data-testid="stTextInput"] label,
        div[data-testid="stNumberInput"] label {{
            color: var(--text-color) !important;
        }}

        /* Checkbox control background styling */
        div[data-testid="stCheckbox"] div[role="checkbox"] {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
        }}

        /* Expander styling override */
        div[data-testid="stExpander"] {{
            background-color: var(--expander-bg) !important;
            border: 1px solid var(--input-border) !important;
        }}
        div[data-testid="stExpander"] summary {{
            background-color: var(--expander-bg) !important;
            color: var(--text-color) !important;
        }}
        div[data-testid="stExpander"] summary svg {{
            fill: var(--text-color) !important;
        }}
        div[data-testid="stExpander"] * {{
            color: var(--text-color) !important;
        }}
        
        /* Style Streamlit's native bordered containers (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--panel-border) !important;
            border-radius: 10px !important;
            padding: 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease;
            height: 76vh !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: var(--panel-hover-border) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def _load_model(device: torch.device, checkpoint_path: Optional[str] = None):
    model = _DeepModelCls(in_channels=3, latent_dim=256, frames=150, lambda_grl=1.0).to(device)
    loaded = False
    if checkpoint_path:
        try:
            ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            loaded = True
        except Exception as _le:
            import logging
            logging.getLogger(__name__).warning(f"[model] load failed: {_le}")
    model.eval()
    model._checkpoint_loaded = loaded
    model._model_version     = _MODEL_VERSION
    return model

@st.cache_resource
def _load_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _resolve_default_checkpoint() -> str:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "checkpoints_v2" / "equiphys_best.pt",
        root / "checkpoints_gpu_mcd" / "equiphys_best.pt",
        root / "checkpoints_gpu" / "equiphys_best.pt",
        root / "checkpoints" / "equiphys_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""

def _capture_webcam_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def _detect_primary_face(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=6, minSize=(90, 90))
    if len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(bw), int(bh)

def _bbox_iou(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 0 else 0.0

def _draw_roi_overlay(frame_bgr: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw physiological guides only when a face is present."""
    out = frame_bgr.copy()
    if face_bbox is None:
        return out

    x, y, bw, bh = face_bbox
    c = (59, 130, 246) # Swapped to a clinical blue (#3b82f6) from neon green
    cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 2)

    # Forehead band
    fx0, fy0 = x + int(0.15 * bw), y + int(0.05 * bh)
    fx1, fy1 = x + int(0.85 * bw), y + int(0.25 * bh)
    cv2.rectangle(out, (fx0, fy0), (fx1, fy1), c, 2)

    # Cheek strips
    left_pts = np.array([
        [x + int(0.12 * bw), y + int(0.45 * bh)],
        [x + int(0.30 * bw), y + int(0.40 * bh)],
        [x + int(0.28 * bw), y + int(0.72 * bh)],
        [x + int(0.10 * bw), y + int(0.75 * bh)],
    ], dtype=np.int32)
    right_pts = np.array([
        [x + int(0.70 * bw), y + int(0.40 * bh)],
        [x + int(0.88 * bw), y + int(0.45 * bh)],
        [x + int(0.90 * bw), y + int(0.75 * bh)],
        [x + int(0.72 * bw), y + int(0.72 * bh)],
    ], dtype=np.int32)
    cv2.polylines(out, [left_pts], isClosed=True, color=c, thickness=2)
    cv2.polylines(out, [right_pts], isClosed=True, color=c, thickness=2)
    return out

def _render_metric_card(label: str, value: str, unit: str, color_theme: str = "blue", status: str = "") -> None:
    theme_colors = {
        "red": {"border": "#f43f5e", "glow": "rgba(244, 63, 94, 0.15)"},
        "cyan": {"border": "#06b6d4", "glow": "rgba(6, 182, 212, 0.15)"},
        "emerald": {"border": "#10b981", "glow": "rgba(16, 185, 129, 0.15)"},
        "purple": {"border": "#8b5cf6", "glow": "rgba(139, 92, 246, 0.15)"},
        "blue": {"border": "#3b82f6", "glow": "rgba(59, 130, 246, 0.15)"},
    }
    c = theme_colors.get(color_theme, theme_colors["blue"])
    status_html = f'<div class="metric-status">{status}</div>' if status else ""
    st.markdown(
        f"""
        <div class="metric-shell-new" style="border-left: 4px solid {c['border']}; --glow-color: {c['glow']};">
          <div class="metric-label-new">{label}</div>
          <div class="metric-value-new">{value}</div>
          <div class="metric-unit-new">{unit}</div>
          {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_diagnostics_log(slot, logs_deque) -> None:
    logs_content = chr(10).join(list(logs_deque)[:26])
    slot.markdown(
        f"""
        <div class="log-terminal-wrap">
          <div class="log-terminal-header">
            <div class="log-terminal-dots">
              <span class="log-terminal-dot dot-red"></span>
              <span class="log-terminal-dot dot-yellow"></span>
              <span class="log-terminal-dot dot-green"></span>
            </div>
            <div class="log-terminal-title">diagnostics.log</div>
          </div>
          <div class="logbox">{logs_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# NOTE: the old `_estimate_bp_hb_from_hr` heuristic (BP/Hb as a fixed linear
# function of heart rate) has been REMOVED. BP now comes from facial pulse-wave
# morphology via facial_vitals.PersonalizedBPEstimator, which needs a cuff
# baseline or trained model and refuses to fabricate a number otherwise.

def _estimate_camera_fps(ts_buffer: deque, fallback: float = 30.0) -> float:
    if len(ts_buffer) < 10:
        return fallback
    ts = np.asarray(list(ts_buffer), dtype=np.float64)
    d = np.diff(ts)
    d = d[d > 1e-4]
    if d.size == 0:
        return fallback
    d_sorted = np.sort(d)
    trim = max(1, len(d_sorted) // 10)
    d_trimmed = d_sorted[trim:-trim] if len(d_sorted) > 2 * trim else d_sorted
    fps = 1.0 / float(np.mean(d_trimmed))
    return float(np.clip(fps, 10.0, 120.0))

def _effective_fs_from_timestamps(ts_buffer: deque, fallback: float = 30.0) -> float:
    if len(ts_buffer) < 2:
        return fallback
    ts = np.asarray(list(ts_buffer), dtype=np.float64)
    dt = float(ts[-1] - ts[0])
    if dt <= 1e-6:
        return fallback
    fs = float(len(ts) / dt)
    return float(np.clip(fs, 8.0, 120.0))

def _green_trace_from_faces(raw_face_buffer: deque) -> Optional[np.ndarray]:
    if len(raw_face_buffer) < 16:
        return None
    arr = np.asarray(list(raw_face_buffer), dtype=np.float32)
    g = arr[:, :, :, 1].mean(axis=(1, 2))
    g = g - np.mean(g)
    if np.std(g) < 1e-6:
        return None
    return g.astype(np.float32)

def _confidence_from_snr_sqi(snr_db: float, sqi: float) -> float:
    snr_term = float(np.clip((snr_db + 6.0) / 18.0, 0.0, 1.0))
    sqi_term = float(np.clip(sqi / 100.0, 0.0, 1.0))
    return 0.6 * snr_term + 0.4 * sqi_term

class HRTemporalTracker:
    def __init__(self, max_hist: int = 25) -> None:
        self.value: Optional[float] = None
        self.hist: deque = deque(maxlen=max_hist)

    def reset(self) -> None:
        self.value = None
        self.hist.clear()

    def update(self, measurement: float, snr_db: float, sqi: float) -> float:
        if not np.isfinite(measurement):
            return float(self.value if self.value is not None else 0.0)

        conf = _confidence_from_snr_sqi(snr_db, sqi)
        gate = 14.0 if conf >= 0.7 else 9.0

        if self.value is not None:
            delta = measurement - self.value
            if abs(delta) > gate:
                measurement = self.value + float(np.clip(delta, -gate, gate))

        alpha = 0.10 + 0.55 * conf
        if self.value is None:
            self.value = float(measurement)
        else:
            self.value = float((1.0 - alpha) * self.value + alpha * measurement)

        self.hist.append(self.value)
        if len(self.hist) >= 6:
            w = np.asarray(self.hist, dtype=np.float32)
            med = float(np.median(w))
            mad = float(np.median(np.abs(w - med))) + 1e-6
            if abs(self.value - med) > max(3.5, 3.0 * mad):
                self.value = float(0.75 * self.value + 0.25 * med)

        return float(self.value)

def _configure_camera_normal(cap: cv2.VideoCapture, logs: deque) -> None:
    try:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except Exception:
        pass

    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Camera running in normal auto mode.")


class _DeepResidualTracker:
    """Tracks deep model HR EMA to extract signed deltas.

    The v2 model has ~30 BPM offset but positive HR Pearson (+0.10).
    Its *changes* are more reliable than its absolute value.
    """
    def __init__(self, alpha: float = 0.12, min_samples: int = 8) -> None:
        self.alpha = alpha; self.min_samples = min_samples
        self._ema: Optional[float] = None; self._n: int = 0

    def reset(self) -> None:
        self._ema = None; self._n = 0

    def update(self, model_hr: float) -> Optional[float]:
        if not np.isfinite(model_hr) or model_hr <= 0:
            return None
        self._ema = model_hr if self._ema is None else (1-self.alpha)*self._ema + self.alpha*model_hr
        self._n += 1
        return None if self._n < self.min_samples else float(model_hr - self._ema)

    @property
    def ready(self) -> bool:
        return self._n >= self.min_samples


def _fuse_hr(traditional_candidates, model_hr, model_snr,
             use_model, hr_band_low, hr_band_high, deep_tracker):
    """Classical-anchored fusion with deep residual delta nudge.

    Classical methods (POS+CHROM) always set the absolute HR anchor.
    The deep model contributes only its *delta from its own baseline*,
    scaled by DELTA_WEIGHT=0.18 and capped at MAX_NUDGE=3.5 BPM.
    This means the model always contributes once warmed up (8 ticks),
    regardless of how biased its absolute HR value is.
    """
    DELTA_WEIGHT = 0.18
    MAX_NUDGE    = 3.5

    if not traditional_candidates:
        if use_model and np.isfinite(model_hr) and hr_band_low <= model_hr <= hr_band_high:
            deep_tracker.update(model_hr)
            return model_hr, model_snr, "DEEP_ONLY", True
        return 0.0, -np.inf, "none", False

    hrs     = np.array([c[1] for c in traditional_candidates], dtype=np.float64)
    snrs    = np.array([c[2] for c in traditional_candidates], dtype=np.float64)
    weights = np.exp(np.clip(snrs / 6.0, -3.0, 3.0))
    weights = np.maximum(weights, 0.05); weights /= weights.sum()
    classical_hr  = float((weights * hrs).sum())
    classical_snr = float(np.mean(snrs))
    label = "+".join(c[0] for c in traditional_candidates)

    if hr_band_low <= classical_hr <= 120.0:
        doubled = classical_hr * 2.0
        if max(70.0, hr_band_low) <= doubled <= hr_band_high:
            sb = sum(1 for _,h,_ in traditional_candidates if abs(h-classical_hr)<10)
            sd = sum(1 for _,h,_ in traditional_candidates if abs(h-doubled)<12)
            if sd > sb:
                classical_hr = doubled; label = label + "x2"

    deep_contributed = False
    fused_hr = classical_hr

    if use_model and np.isfinite(model_hr) and model_hr > 0:
        delta = deep_tracker.update(model_hr)
        if delta is not None and deep_tracker.ready and model_snr > -1.0:
            nudge     = float(np.clip(DELTA_WEIGHT * delta, -MAX_NUDGE, MAX_NUDGE))
            candidate = classical_hr + nudge
            if hr_band_low <= candidate <= hr_band_high and abs(nudge) > 0.2:
                fused_hr = candidate; deep_contributed = True; label = label + "+Dδ"

    return fused_hr, classical_snr, label, deep_contributed

def main() -> None:
    st.set_page_config(page_title="rPPG Telehealth Portal", layout="wide", initial_sidebar_state="expanded")
    
    # Sidebar theme selection
    st.sidebar.markdown("### Theme Control")
    theme_mode = st.sidebar.selectbox("Theme Mode", ["Dark Theme", "Light Theme"], index=0, key="theme_mode")
    is_light = (theme_mode == "Light Theme")
    _inject_css(is_light=is_light)

    now_str = datetime.now().strftime("%a, %b %d, %Y | %H:%M")
    st.markdown(
        f"""
        <div class="header-shell">
          <div class="title-row">
            <div class="title-text">Robust Remote Photoplethysmography Telehealth Portal <span style="font-weight: 400; color: var(--text-muted); font-size: 15px;">| Live Inference</span></div>
            <div><span class="live-pill"><span class="live-indicator-dot"></span>LIVE</span> &nbsp; <span style="font-size: 14px; font-weight: 500; color: var(--text-muted);">{now_str}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown("### Model Settings")
    use_tta       = st.sidebar.checkbox("Enable test-time adaptation", value=True)
    use_locking   = st.sidebar.checkbox("Lock final HR reading", value=True)
    use_model     = st.sidebar.checkbox("Use deep model in fusion", value=True)
    manual_freeze = st.sidebar.checkbox("Freeze displayed HR (manual)", value=False)
    st.sidebar.caption(
        f"Mode: **{'Classical+Deep' if use_model else 'Classical only'}** · arch `{_MODEL_VERSION}`"
    )
    
    st.sidebar.markdown("---")
    default_ckpt = _resolve_default_checkpoint()
    checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)

    # SpO2 device calibration + BP cuff-baseline capture live here.
    calib = sidebar_calibration_controls(st, st.session_state)

    # Camera colour, lighting, motion-floor calibration.
    cam_cal = sidebar_camera_calibration_panel(st, st.session_state)
    # Wire the SpO2 device calibration result from the camera panel back into calib,
    # so both sidebar sections stay in sync.
    _spo2_cal_from_cam = st.session_state.get("spo2_cal_result")
    if _spo2_cal_from_cam is not None:
        calib.spo2 = _spo2_cal_from_cam
    calib.allow_provisional_spo2 = st.session_state.get("spo2_show_prov", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device, checkpoint_path=checkpoint_path if checkpoint_path else None)
    
    roi_extractor = FaceROIExtractor(target_size=64)
    api_mode = getattr(roi_extractor, "_api_mode", "none")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Diagnostics")
    if api_mode == "tasks":
        st.sidebar.success("FaceLandmarker (Tasks API)")
    elif api_mode == "legacy":
        st.sidebar.success("FaceMesh (Legacy API)")
    elif api_mode == "cascade":
        st.sidebar.warning("OpenCV Cascade Fallback")
    else:
        st.sidebar.error("No face detector available")
    if getattr(model, "_checkpoint_loaded", False):
        st.sidebar.success(f"Deep model loaded ({_MODEL_VERSION})")
    elif checkpoint_path:
        st.sidebar.warning("Deep model: load failed")
    else:
        st.sidebar.info("Deep model: no checkpoint")

    # ── State variables ──
    TARGET_FS = 30.0
    CALIBRATION_SECONDS = 10.0
    HR_EMA_ALPHA = 0.25
    SNR_SNAP_THRESHOLD = 3.0
    SNR_NOISY_THRESHOLD = -2.0
    BPM_SNAP_DELTA = 6.0
    TACHY_SNAP_SNR_DB = 7.5
    TACHY_SNAP_DELTA_BPM = 15.0
    MOTION_GATE_PX = 3.5
    FACE_SWITCH_VOTES_REQUIRED = 12  # sustained IoU drop before declaring a new face
    MOTION_MAD_K = 3.5               # per-frame motion rejection sensitivity
    MOTION_CLEAN_MIN_FRAC = 0.35     # below this fraction of clean frames -> hold
    NORM_CROP_SIZE = 128
    MODEL_FRAMES = 150
    TRAD_BUFFER_LEN = int(TARGET_FS * 10.0)
    MIN_TRAD_FRAMES = int(TARGET_FS * CALIBRATION_SECONDS)
    frame_buffer: deque = deque(maxlen=MODEL_FRAMES)
    trad_roi_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    trad_ts_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    trad_landmark_buffer: deque = deque(maxlen=TRAD_BUFFER_LEN)
    ts_buffer: deque = deque(maxlen=300)
    hr_buffer: deque = deque(maxlen=30)
    hr_long_buffer: deque = deque(maxlen=60)
    logs: deque = deque(maxlen=200)
    face_miss_count = 0
    face_switch_votes = 0
    prev_face_bbox: Optional[Tuple[int, int, int, int]] = None
    no_face_grace = 45  # ~1.5 s: coast through brief detector misses, don't reset
    last_infer_ts = 0.0
    inference_interval_s = 1.0
    last_hr = 0.0
    last_bvp: Optional[np.ndarray] = None
    ema_hr = 0.0
    ema_initialized = False
    display_hr = 0.0
    warmup_count = 0
    warmup_needed = 2
    hr_stability_window: deque = deque(maxlen=12)
    locked_hr: Optional[float] = None
    manual_locked_hr: Optional[float] = None
    unlock_votes = 0
    hr_tracker   = HRTemporalTracker(max_hist=25)
    deep_tracker = _DeepResidualTracker()

    LOCK_MIN_POINTS = 8
    LOCK_MAX_SPREAD = 8.0
    LOCK_MAX_MAD = 3.0
    UNLOCK_DRIFT_BPM = 22.0
    UNLOCK_VOTES_REQUIRED = 12
    HR_BAND_LOW = 60.0
    HR_BAND_HIGH = 120.0

    last_spo2 = 0.0
    last_rr = 0.0
    last_sqi = 0.0
    ema_spo2 = 0.0
    ema_rr = 0.0
    spo2_initialized = False
    rr_initialized = False
    best_snr = -np.inf

    # Provenance-aware SpO2 / BP state (None until a real facial measurement).
    last_spo2_vs = None
    last_bp_sbp_vs = None
    last_bp_dbp_vs = None
    bvp_wave = None
    bvp_fs = float(TARGET_FS)

    cascade = _load_cascade()
    if cascade.empty():
        st.error("OpenCV face cascade failed to load.")
        return

    # ── Top-level mode tabs ──────────────────────────────────────────────────────────────────
    tab_live, tab_video = st.tabs(["Live Inference", "Video Analysis"])

    # ── Video Analysis tab (self-contained, no webcam needed) ─────────────────
    with tab_video:
        render_video_analysis_tab(st, cam_cal=cam_cal)

    # ── Live Inference tab ─────────────────────────────────────────────────────────────────────
    with tab_live:
        left_col, mid_col, right_col = st.columns([1.1, 2.0, 0.9])

        with left_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Video Feed</div>', unsafe_allow_html=True)
                frame_slot = st.empty()
                status_slot = st.empty()
                run = st.checkbox("Start Live Inference", value=False)

        with mid_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Clinical Vitals</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                with m1: hr_slot = st.empty()
                with m2: spo2_slot = st.empty()
                with m3: rr_slot = st.empty()
                with m4: bp_slot = st.empty()
                sqi_slot = st.empty()
                wave_slot = st.empty()
                
                # Render initial placeholders so the dashboard is populated immediately on startup
                with hr_slot: _render_metric_card("HEART RATE", "--", "BPM", "red", "Waiting...")
                with spo2_slot: _render_metric_card("SpO₂ ESTIMATE", "--", "%", "cyan", "Waiting...")
                with rr_slot: _render_metric_card("RESP RATE", "--", "br/min", "emerald", "Waiting...")
                with bp_slot: _render_metric_card("BLOOD PRESSURE", "--", "mmHg", "purple", "Needs baseline")
                
                sqi_slot.progress(0.0, text="Signal Quality: Waiting for data...")
                import pandas as pd
                placeholder_df = pd.DataFrame({"BVP": [0.0] * 100})
                wave_slot.line_chart(placeholder_df, height=140)

        with right_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Diagnostics Log</div>', unsafe_allow_html=True)
                log_slot = st.empty()
                _render_diagnostics_log(log_slot, logs)

    if not run:
        with tab_live:
            with left_col:
                st.info("Toggle 'Start Live Inference' to connect webcam.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    _configure_camera_normal(cap, logs)
    adapted = False

    try:
        while run:
            frame = _capture_webcam_frame(cap)
            if frame is None:
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Frame capture failed.")
                break

            frame_ts = time.time()
            ts_buffer.append(frame_ts)
            fps_est = _estimate_camera_fps(ts_buffer, fallback=30.0)

            face_bbox = _detect_primary_face(frame, cascade)
            if face_bbox is None:
                face_miss_count += 1
            else:
                face_miss_count = 0

            if face_bbox is not None and prev_face_bbox is not None:
                if _bbox_iou(face_bbox, prev_face_bbox) < 0.20:
                    face_switch_votes += 1
                else:
                    face_switch_votes = max(0, face_switch_votes - 2)
            else:
                face_switch_votes = max(0, face_switch_votes - 2)

            if face_switch_votes >= FACE_SWITCH_VOTES_REQUIRED:
                face_switch_votes = 0
                frame_buffer.clear(); trad_roi_buffer.clear(); trad_ts_buffer.clear(); trad_landmark_buffer.clear()
                hr_buffer.clear(); hr_long_buffer.clear()
                adapted = False; ema_initialized = False; last_bvp = None; last_infer_ts = 0.0; last_hr = 0.0
                display_hr = 0.0; warmup_count = 0; hr_stability_window.clear(); locked_hr = None
                manual_locked_hr = None; unlock_votes = 0; hr_tracker.reset(); deep_tracker.reset(); spo2_initialized = False
                rr_initialized = False; last_spo2 = 0.0; last_rr = 0.0; last_sqi = 0.0; best_snr = -np.inf
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Different face \u2014 recalibrating.")

            prev_face_bbox = face_bbox if face_bbox is not None else prev_face_bbox

            framed = _draw_roi_overlay(frame, face_bbox)
            frame_slot.image(framed[:, :, ::-1], channels="RGB", width="stretch")

            if face_miss_count >= no_face_grace:
                frame_buffer.clear(); trad_roi_buffer.clear(); trad_ts_buffer.clear(); trad_landmark_buffer.clear()
                hr_buffer.clear(); hr_long_buffer.clear()
                adapted = False; ema_initialized = False; last_bvp = None; last_infer_ts = 0.0; last_hr = 0.0
                prev_face_bbox = None; display_hr = 0.0; warmup_count = 0; hr_stability_window.clear()
                locked_hr = None; manual_locked_hr = None; unlock_votes = 0; hr_tracker.reset()
                spo2_initialized = False; rr_initialized = False; last_spo2 = 0.0; last_rr = 0.0
                last_sqi = 0.0; best_snr = -np.inf

                with hr_slot: _render_metric_card("HEART RATE", "--", "BPM", "red", "No Face Detected")
                with spo2_slot: _render_metric_card("SpO₂ ESTIMATE", "--", "%", "cyan", "No Face Detected")
                with rr_slot: _render_metric_card("RESP RATE", "--", "br/min", "emerald", "No Face Detected")
                with bp_slot: _render_metric_card("BLOOD PRESSURE", "--", "mmHg", "purple", "No Face Detected")
                sqi_slot.progress(0.0, text="Signal Quality: -- / No face detected")
                import pandas as pd
                placeholder_df = pd.DataFrame({"BVP": [0.0] * 100})
                wave_slot.line_chart(placeholder_df, height=140)
                status_slot.warning("No face detected. Please align face in frame.")
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] No face detected.")
                _render_diagnostics_log(log_slot, logs)
                continue

            if face_bbox is not None:
                roi_rgb, roi_mask, roi_landmarks = roi_extractor.extract_with_landmarks(frame)

                # ── Camera calibration: feed raw frame into active capture window,
                #    then apply colour + lighting corrections before buffering. ──
                feed_calibration_frame(roi_rgb, st.session_state, logs)
                roi_rgb_cal = apply_camera_calibration(roi_rgb, cam_cal)

                frame_buffer.append(roi_rgb_cal)

                mask3 = np.repeat((roi_mask > 0.5).astype(np.float32)[..., None], 3, axis=2)
                masked_roi = roi_rgb_cal.astype(np.float32) * mask3
                trad_crop_norm = cv2.resize(masked_roi, (NORM_CROP_SIZE, NORM_CROP_SIZE), interpolation=cv2.INTER_AREA)
                trad_roi_buffer.append(trad_crop_norm.astype(np.float32))
                trad_ts_buffer.append(frame_ts)
                if roi_landmarks is not None:
                    trad_landmark_buffer.append(np.asarray(roi_landmarks, dtype=np.float32))

            if len(trad_roi_buffer) < MIN_TRAD_FRAMES:
                pct = len(trad_roi_buffer) / float(MIN_TRAD_FRAMES)
                status_slot.progress(pct, text=f"Calibrating... Buffering {len(trad_roi_buffer)}/{MIN_TRAD_FRAMES} frames")
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Calibrating... buffering {len(trad_roi_buffer)}/{MIN_TRAD_FRAMES}")
                _render_diagnostics_log(log_slot, logs)
                continue

            now_ts = time.time()
            should_infer = (not ema_initialized) or ((now_ts - last_infer_ts) >= inference_interval_s)

            if should_infer:
                last_infer_ts = now_ts
                # -- Per-frame motion masking: drop only shaky frames, keep the rest --
                trad_ts = np.asarray(list(trad_ts_buffer), dtype=np.float64)
                motion = frame_diff_motion(list(trad_roi_buffer))
                _motion_k = cam_cal.motion.motion_mad_k if cam_cal.motion.ready else MOTION_MAD_K
                keep_mask = clean_frame_mask(motion, k=_motion_k)
                roi_est, ts_est = select_clean_frames(list(trad_roi_buffer), trad_ts, keep_mask)
                clean_frac = float(np.mean(keep_mask)) if keep_mask.size else 0.0

                if len(roi_est) < 30 or clean_frac < MOTION_CLEAN_MIN_FRAC:
                    # Too much motion this tick -- HOLD the last reading; never clear the buffer.
                    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] High motion (clean {clean_frac*100:.0f}%) \u2014 holding last reading.")
                    status_slot.info("Motion high \u2014 holding last reading (no reset). Settle for a moment.")
                    _render_diagnostics_log(log_slot, logs)
                    continue

                fs_effective = _effective_fs_from_timestamps(trad_ts_buffer, fallback=fps_est)

                pos_hr, pos_snr = estimate_hr_from_rgb_pos(
                    roi_est, fps=fs_effective, f_low=1.0, f_high=2.0, 
                    timestamps=ts_est, target_fs=fs_effective, prev_hr_bpm=None,
                )
                chrom_hr, chrom_snr = estimate_hr_from_rgb_chrom(
                    roi_est, fps=fs_effective, f_low=1.0, f_high=2.0,
                    timestamps=ts_est, target_fs=fs_effective, prev_hr_bpm=None,
                )
                green_trace = _green_trace_from_faces(roi_est)
                ac_hr, ac_score = estimate_hr_from_autocorr(
                    green_trace, fps=fs_effective, f_low=0.8, f_high=2.5,
                    timestamps=ts_est, target_fs=fs_effective,
                ) if green_trace is not None else (0.0, 0.0)

                # ── Deep model (always run for delta tracking) ──────────────
                model_hr  = 0.0
                model_snr = -np.inf
                last_bvp  = None
                if len(frame_buffer) >= MODEL_FRAMES:
                    try:
                        clip = build_clip_tensor_from_buffer(frame_buffer, device=device)
                        with torch.enable_grad() if (use_tta and not adapted) else torch.no_grad():
                            if use_tta and not adapted:
                                out = test_time_adaptation(
                                    model=model, clip_bcthw=clip, fps=fps_est, steps=8, lr=1e-4)
                                adapted = True
                                logs.appendleft(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] TTA complete.")
                            else:
                                out = model(clip)
                                out["hr_bpm"] = estimate_hr_bpm_from_bvp(
                                    out["bvp_pred"], fps=fps_est)
                        model_hr  = float(out["hr_bpm"][0].item())
                        last_bvp  = out["bvp_pred"][0].detach().float().cpu().numpy()
                        model_snr = compute_pulse_snr(last_bvp, fps=fps_est)
                    except Exception as _me:
                        logs.appendleft(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Model err: {_me}")

                quality_sig = last_bvp if last_bvp is not None else green_trace
                sqi = compute_signal_quality_index(
                    quality_sig, fps=fs_effective) if quality_sig is not None else 0.0
                bvp_wave, bvp_fs, _bvp_snr, _ = reconstruct_facial_bvp(
                    roi_est, timestamps=ts_est, target_fs=fs_effective)
                last_spo2_vs = spo2_from_faces(roi_est, fs_effective, calib.spo2)
                spo2_raw = last_spo2_vs.value if last_spo2_vs.available else 0.0
                rr_raw = estimate_respiratory_rate(
                    quality_sig, fps=fs_effective) if quality_sig is not None else 0.0

                # ── Classical candidates ─────────────────────────────────────
                traditional_candidates = []
                if np.isfinite(pos_hr) and HR_BAND_LOW <= pos_hr <= HR_BAND_HIGH:
                    traditional_candidates.append(
                        ("POS", pos_hr, pos_snr if np.isfinite(pos_snr) else -10.0))
                if np.isfinite(chrom_hr) and HR_BAND_LOW <= chrom_hr <= HR_BAND_HIGH:
                    traditional_candidates.append(
                        ("CHROM", chrom_hr, chrom_snr if np.isfinite(chrom_snr) else -10.0))

                # ── Residual-delta fusion ────────────────────────────────────
                hr_raw, best_snr, best_label, deep_used = _fuse_hr(
                    traditional_candidates=traditional_candidates,
                    model_hr=model_hr,
                    model_snr=model_snr,
                    use_model=use_model,
                    hr_band_low=HR_BAND_LOW,
                    hr_band_high=HR_BAND_HIGH,
                    deep_tracker=deep_tracker,
                )
                if deep_used and traditional_candidates:
                    logs.appendleft(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Deepδ: model={model_hr:.0f} → fused={hr_raw:.0f} "
                        f"({best_label})")
                best_hr_val = hr_raw

                hr_buffer.append(hr_raw)
                hr_long_buffer.append(hr_raw)
                warmup_count += 1

                tachy_snap_applied = False
                if not ema_initialized:
                    ema_hr = hr_raw
                    display_hr = hr_raw
                    ema_initialized = True
                else:
                    if (
                        np.isfinite(best_snr)
                        and best_snr >= TACHY_SNAP_SNR_DB
                        and hr_raw >= 100.0
                        and abs(hr_raw - ema_hr) > TACHY_SNAP_DELTA_BPM
                    ):
                        ema_hr = hr_raw
                        display_hr = hr_raw
                        tachy_snap_applied = True
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Tachycardia snap: {hr_raw:.0f} BPM (SNR={best_snr:.1f}dB).")
                    elif best_snr < SNR_NOISY_THRESHOLD:
                        ema_alpha = 0.05
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Signal noisy – Recalculating...")
                    elif best_snr >= SNR_SNAP_THRESHOLD and abs(hr_raw - ema_hr) >= BPM_SNAP_DELTA:
                        ema_alpha = 0.8
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] High-SNR snap (α=0.80): {hr_raw:.0f} BPM.")
                    else:
                        ema_alpha = HR_EMA_ALPHA
                    
                    if not tachy_snap_applied:
                        ema_hr = (ema_alpha * hr_raw) + ((1.0 - ema_alpha) * ema_hr)
                        display_hr = ema_hr

                last_hr = float(display_hr)

                if tachy_snap_applied and use_locking:
                    locked_hr = None
                    unlock_votes = 0
                    hr_stability_window.clear()

                if use_locking:
                    quality_ok = bool(sqi >= 50.0 and np.isfinite(best_snr) and best_snr >= -1.0)
                    if quality_ok:
                        hr_stability_window.append(hr_raw)

                    if locked_hr is None:
                        if len(hr_stability_window) >= LOCK_MIN_POINTS:
                            w = np.asarray(list(hr_stability_window), dtype=np.float32)
                            med = float(np.median(w))
                            mad = float(np.median(np.abs(w - med)))
                            spread = float(np.max(w) - np.min(w))
                            if mad <= LOCK_MAX_MAD and spread <= LOCK_MAX_SPREAD:
                                if len(hr_long_buffer) >= 8:
                                    locked_hr = float(np.median(np.asarray(list(hr_long_buffer)[-20:], dtype=np.float32)))
                                else:
                                    locked_hr = med
                                unlock_votes = 0
                                display_hr = locked_hr
                                last_hr = float(locked_hr)
                                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR locked at {locked_hr:.0f} BPM.")
                    else:
                        display_hr = locked_hr
                        if quality_ok and abs(hr_raw - locked_hr) > UNLOCK_DRIFT_BPM and best_snr >= 3.0:
                            unlock_votes += 1
                        elif quality_ok:
                            unlock_votes = max(0, unlock_votes - 1)

                        if unlock_votes >= UNLOCK_VOTES_REQUIRED:
                            recent_lock = list(hr_long_buffer)[-20:] if len(hr_long_buffer) >= 8 else [hr_raw]
                            locked_hr = float(np.median(np.asarray(recent_lock, dtype=np.float32)))
                            display_hr = locked_hr
                            last_hr = float(locked_hr)
                            unlock_votes = 0
                            logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR re-locked at {locked_hr:.0f} BPM.")
                if not (traditional_candidates or (use_model and model_hr > 0)):
                    if last_hr > 0:
                        display_hr = float(last_hr)
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] Window rejected by weighted PSD/SQI. Holding {last_hr:.0f} BPM.")

                if manual_freeze:
                    if manual_locked_hr is None and warmup_count >= warmup_needed:
                        manual_locked_hr = float(display_hr)
                    if manual_locked_hr is not None:
                        display_hr = manual_locked_hr
                else:
                    manual_locked_hr = None

                if spo2_raw > 0:
                    if not spo2_initialized:
                        ema_spo2 = spo2_raw
                        spo2_initialized = True
                    else:
                        ema_spo2 = 0.8 * ema_spo2 + 0.2 * spo2_raw
                    last_spo2 = ema_spo2

                if rr_raw > 0:
                    if not rr_initialized:
                        ema_rr = rr_raw
                        rr_initialized = True
                    else:
                        ema_rr = 0.1 * rr_raw + 0.9 * ema_rr
                    last_rr = ema_rr

                last_sqi = sqi

                # Blood pressure from facial pulse-wave morphology.
                morph = extract_pulse_morphology(bvp_wave, bvp_fs) if bvp_wave is not None else {"morph_quality": 0.0}
                if display_hr > 0:
                    morph["heart_rate"] = float(display_hr)
                st.session_state["last_morph"] = morph
                # Auto-anchor to the entered cuff reading on the first stable tick
                # (and re-anchor if the entered values change). Works inside the
                # blocking loop, unlike a sidebar button.
                if calib.bp_anchor is not None:
                    want_sbp, want_dbp = calib.bp_anchor
                    b = calib.bp.baseline
                    if b is None or abs(b.sbp - want_sbp) > 0.5 or abs(b.dbp - want_dbp) > 0.5:
                        calib.bp.set_baseline(morph, want_sbp, want_dbp)
                        logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] BP anchored to cuff {want_sbp:.0f}/{want_dbp:.0f} mmHg.")
                last_bp_sbp_vs, last_bp_dbp_vs = calib.bp.estimate(morph)

                # Build log string from fusion result
                val_str = f"{best_label}={hr_raw:.0f}BPM snr={best_snr:.1f}dB" if hr_raw > 0 else "none"
                logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] HR={display_hr:.0f} [{val_str}] sqi={sqi:.0f} fps={fs_effective:.1f}")

                # Log the raw SpO2 ratio-R so users can pair it with an oximeter reading
                # and paste the pair into the SpO2 calibration box in the sidebar.
                if last_spo2_vs is not None and last_spo2_vs.extra:
                    _ratio_r = last_spo2_vs.extra.get("ratio_R")
                    if _ratio_r is not None and np.isfinite(_ratio_r):
                        logs.appendleft(
                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                            f"SpO₂ ratio-R={_ratio_r:.4f}  "
                            f"(paste as '<oximeter %>  , {_ratio_r:.4f}' in sidebar)"
                        )

                # ── Display metrics ──
            with hr_slot:
                if warmup_count < warmup_needed:
                    _render_metric_card("HEART RATE", "--", "BPM", "red", "Stabilizing...")
                else:
                    snr_str = f"SNR: {best_snr:.1f} dB" if np.isfinite(best_snr) else "Normal"
                    _render_metric_card("HEART RATE", f"{int(round(display_hr))}", "BPM", "red", snr_str)
            with spo2_slot:
                if last_spo2 > 0:
                    _spo2_unit = "%"
                    _spo2_status = "Calibrated" if (last_spo2_vs is not None and last_spo2_vs.source == SRC_RATIO_CAL) else "Provisional"
                    _render_metric_card("SpO₂ ESTIMATE", f"{last_spo2:.0f}", _spo2_unit, "cyan", _spo2_status)
                else:
                    _render_metric_card("SpO₂ ESTIMATE", "--", "%", "cyan", "Waiting...")
            with rr_slot:
                if last_rr > 0:
                    _render_metric_card("RESP RATE", f"{last_rr:.0f}", "br/min", "emerald", "Spectral")
                else:
                    _render_metric_card("RESP RATE", "--", "br/min", "emerald", "Waiting...")
            with bp_slot:
                if (warmup_count >= warmup_needed and last_bp_sbp_vs is not None
                        and last_bp_sbp_vs.available and last_bp_dbp_vs is not None
                        and last_bp_dbp_vs.available):
                    _bp_status = "Trend" if last_bp_sbp_vs.source == "rppg_morphology_personalized" else "Estimated"
                    _render_metric_card("BLOOD PRESSURE",
                                        f"{last_bp_sbp_vs.value:.0f}/{last_bp_dbp_vs.value:.0f}", "mmHg", "purple", _bp_status)
                else:
                    _render_metric_card("BLOOD PRESSURE", "--", "mmHg", "purple", "Needs cuff baseline")
 
            if last_sqi > 0:
                q_label = "Good Signal" if last_sqi >= 60 else ("Fair Signal" if last_sqi >= 30 else "Poor — Keep Still")
                sqi_slot.progress(min(last_sqi / 100.0, 1.0), text=f"Signal Quality: {last_sqi:.0f}/100 — {q_label}")
            else:
                sqi_slot.progress(0.0, text="Signal Quality: Waiting for data...")
 
            if last_bvp is not None:
                wave_slot.line_chart(last_bvp, height=140)
 
            if warmup_count < warmup_needed:
                status_slot.info(f"Stabilizing readings... ({warmup_count}/{warmup_needed})")
            elif manual_freeze and manual_locked_hr is not None:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.success(f"Manual lock: {manual_locked_hr:.0f} BPM | fps={fps_est:.0f}{snr_str}")
            elif use_locking and locked_hr is not None:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.success(f"Locked HR: {locked_hr:.0f} BPM | fps={fps_est:.0f}{snr_str}")
            elif ema_initialized:
                snr_str = f" SNR={best_snr:.1f}dB" if np.isfinite(best_snr) else ""
                status_slot.info(f"Measuring... fps={fps_est:.0f}{snr_str}")
            else:
                status_slot.info("Initializing...")
 
            _render_diagnostics_log(log_slot, logs)

    finally:
        cap.release()

if __name__ == "__main__":
    main()