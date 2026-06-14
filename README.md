# EquiPhys rPPG

EquiPhys is a webcam-based remote photoplethysmography (rPPG) project for estimating heart rate, respiratory rate, and approximate SpO2 from face video. It includes:

- a live Streamlit app for webcam inference
- preprocessing scripts for raw rPPG datasets
- training code for the EquiPhys deep model
- pre-generated checkpoints for quick experimentation

This project is research code, not a medical device.

## Overview

The live app uses a classical rPPG pipeline by default and can optionally fuse a learned model output. The main pipeline:

1. captures webcam frames
2. extracts a face ROI with MediaPipe landmarks or an OpenCV fallback
3. buffers a rolling time window of ROI frames
4. estimates heart rate from POS, CHROM, and related spectral signals
5. optionally fuses a deep BVP prediction from EquiPhysDANN

## Repository Layout

- `streamlit_app.py` - live webcam app and UI
- `equiphys_core.py` - ROI extraction, signal processing, and model definitions
- `preprocess_raw_datasets.py` - converts raw datasets into NPZ clips
- `prepare_dataloaders.py` - builds dataset loaders and prints a summary
- `train_equiphys.py` - training and checkpoint generation
- `dataset_connectors.py` - NPZ dataset loading and batching helpers
- `processed_npz/` - expected location for generated training data
- `checkpoints*/` - training outputs and saved model weights

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Suggested environment:

- Python 3.10 or newer
- a webcam for live inference
- PyTorch with CUDA if you want GPU training or faster inference

## Quick Start

### 1. Run the live app

```bash
streamlit run streamlit_app.py
```

The app looks for a checkpoint in this order:

1. `checkpoints_v2/equiphys_best.pt`
2. `checkpoints_gpu_mcd/equiphys_best.pt`
3. `checkpoints_gpu/equiphys_best.pt`
4. `checkpoints/equiphys_best.pt`

If one of those files exists, it will be loaded automatically.

### 2. Preprocess raw datasets

Use this script to convert raw UBFC, MMPD, MCD, or IBVP data into NPZ clips.

```bash
python preprocess_raw_datasets.py --out-root processed_npz --ubfc-root C:\path\to\ubfc
```

You can pass any combination of dataset roots:

- `--ubfc-root` for UBFC raw data
- `--mmpd-root` for MMPD raw data
- `--mcd-root` for MCD raw data
- `--ibvp-root` for IBVP raw data

Common options:

- `--target-size 64`
- `--clip-len 150`
- `--stride 75`

The script writes clips under subfolders like `processed_npz/ubfc/...`, `processed_npz/mmpd/...`, `processed_npz/mcd/...`, and `processed_npz/ibvp/...`.

### 3. Inspect prepared loaders

```bash
python prepare_dataloaders.py --ubfc-root processed_npz\ubfc
```

You can add `--ibvp-root`, `--mmpd-root`, and `--mcd-root` as needed. The script prints a compact summary of the resulting loaders.

### 4. Train EquiPhys

```bash
python train_equiphys.py --ubfc-root processed_npz\ubfc --out-dir checkpoints
```

Typical training flags:

- `--ibvp-root`, `--mmpd-root`, `--mcd-root` to include more datasets
- `--epochs` to change the number of epochs
- `--batch-size` and `--num-workers` for throughput tuning
- `--device cuda` or `--device cpu`
- `--amp` to enable mixed precision on CUDA
- `--resume path\to\checkpoint.pt` to continue training

The trainer writes:

- `equiphys_epoch_###.pt`
- `equiphys_best.pt`
- `equiphys_final.pt`
- `train_history.json`
- `clinical_history.json` when the clinical stage runs

## Expected NPZ Format

Training data is stored as `.npz` clips. The loader expects at least:

- `video` with shape `[T, H, W, 3]` in float32
- `roi_mask` with shape `[H, W]`

Optional keys depend on the objective:

- `bvp` for pulse supervision
- `skin_label` and `light_label` for domain adaptation
- `clinical` for clinical regression targets

## Checkpoints

Several checkpoint folders are already included in the repo, so you can test the app immediately. The most relevant file for live inference is usually `equiphys_best.pt`.

## Notes

- Webcam SpO2 is approximate and should not be treated as a clinical measurement.
- Live HR estimates are strongest when the face is well lit, mostly frontal, and relatively still.
- If MediaPipe is unavailable, the ROI extractor falls back to an OpenCV face detector.

## License

No license file is included in this repository. Add one before distributing the code externally.