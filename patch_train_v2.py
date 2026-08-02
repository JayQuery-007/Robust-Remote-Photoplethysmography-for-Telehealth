"""patch_train_v2.py — rewire train_equiphys.py onto the v2 architecture.

Run this once, in the same directory as train_equiphys.py and equiphys_v2.py.
It is idempotent: running twice is harmless.

Changes applied:
  1. Import EquiPhysDANNV2 / PhysicsInformedLossV2 / DANNTrainingLossV2
  2. Swap the model constructor
  3. Swap both loss constructors, with class-weighted skin CE
  4. Wrap training subsets in AugmentedNPZClipDataset (validation stays clean)
  5. Add a ClinicalRegressor shim (the v2 extractor returns 3 values, not 2)
"""
from __future__ import annotations

import sys
from pathlib import Path

TRAIN_PY = Path("train_equiphys.py")
if not TRAIN_PY.exists():
    print(f"ERROR: {TRAIN_PY} not found in {Path.cwd()}")
    sys.exit(1)

src = TRAIN_PY.read_text()
applied = []
skipped = []

# ─────────────────────────────────────────────────────────────────────────
# 1. Add the v2 imports
# ─────────────────────────────────────────────────────────────────────────
IMPORT_ANCHOR = "from dataset_connectors import NPZClipDataset"
V2_IMPORTS = """from dataset_connectors import NPZClipDataset
from equiphys_v2 import (
    AugmentedNPZClipDataset,
    DANNTrainingLossV2,
    EquiPhysDANNV2,
    PhysicsInformedLossV2,
    compute_skin_class_weights,
)"""

if "from equiphys_v2 import" in src:
    skipped.append("imports (already present)")
elif IMPORT_ANCHOR in src:
    src = src.replace(IMPORT_ANCHOR, V2_IMPORTS, 1)
    applied.append("v2 imports added")
else:
    print("ERROR: could not find the dataset_connectors import anchor")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────
# 2. Swap the model constructor
# ─────────────────────────────────────────────────────────────────────────
OLD_MODEL = ("model = EquiPhysDANN(in_channels=3, latent_dim=256, "
             "frames=args.frames, lambda_grl=args.lambda_grl).to(device)")
NEW_MODEL = ("model = EquiPhysDANNV2(in_channels=3, latent_dim=256, "
             "frames=args.frames, lambda_grl=args.lambda_grl).to(device)")

if NEW_MODEL in src:
    skipped.append("model constructor (already v2)")
elif OLD_MODEL in src:
    src = src.replace(OLD_MODEL, NEW_MODEL, 1)
    applied.append("model -> EquiPhysDANNV2")
else:
    print("WARNING: model constructor line not found verbatim; check manually")

# ─────────────────────────────────────────────────────────────────────────
# 3. Swap the loss constructors
# ─────────────────────────────────────────────────────────────────────────
OLD_LOSSES_A = ("    ubfc_loss = PhysiologyInformedLoss(lambda_green=0.4, "
                "lambda_roi=0.2, lambda_hr=args.lambda_hr)\n"
                "    mmpd_loss = DANNTrainingLoss(alpha_skin=0.2, alpha_light=0.2, "
                "lambda_green=0.4, lambda_roi=0.2, lambda_hr=args.lambda_hr)")

# tolerate the earlier 0.05/0.1 patch too
OLD_LOSSES_B = ("    ubfc_loss = PhysiologyInformedLoss(lambda_green=0.05, "
                "lambda_roi=0.1, lambda_hr=args.lambda_hr)\n"
                "    mmpd_loss = DANNTrainingLoss(alpha_skin=0.2, alpha_light=0.2, "
                "lambda_green=0.05, lambda_roi=0.1, lambda_hr=args.lambda_hr)")

NEW_LOSSES = """    # ── v2 losses: green-channel term removed, spectral + SNR terms added ──
    _phys_kwargs = dict(
        lambda_diff=0.5,
        lambda_spec=float(args.lambda_hr),   # --lambda-hr now weights the spectral term
        lambda_snr=0.3,
        lambda_roi=0.05,
        f_low=0.7,
        f_high=3.0,
    )
    ubfc_loss = PhysicsInformedLossV2(**_phys_kwargs)

    # Class-weighted skin CE fixes the discriminator collapsing to the
    # majority Fitzpatrick class (v1 sat at chance accuracy for 62 epochs).
    _skin_w = None
    if args.mmpd_root:
        try:
            _skin_w = compute_skin_class_weights(args.mmpd_root, n_classes=6, device=device)
            print(f"Skin class weights: {_skin_w.cpu().numpy().round(3).tolist()}")
        except Exception as _e:
            print(f"Could not compute skin class weights ({_e}); using uniform.")

    mmpd_loss = DANNTrainingLossV2(
        alpha_skin=0.2, alpha_light=0.2, skin_weights=_skin_w, **_phys_kwargs
    )"""

if "PhysicsInformedLossV2(**_phys_kwargs)" in src:
    skipped.append("losses (already v2)")
elif OLD_LOSSES_A in src:
    src = src.replace(OLD_LOSSES_A, NEW_LOSSES, 1)
    applied.append("losses -> v2 (from original 0.4/0.2)")
elif OLD_LOSSES_B in src:
    src = src.replace(OLD_LOSSES_B, NEW_LOSSES, 1)
    applied.append("losses -> v2 (from patched 0.05/0.1)")
else:
    print("WARNING: loss constructor lines not found; check manually")

# ─────────────────────────────────────────────────────────────────────────
# 4. Wrap training subsets in the augmented dataset
# ─────────────────────────────────────────────────────────────────────────
# Each dataset block has the same shape:
#     tr_ds = Subset(ds, tr_idx)
#     va_ds = Subset(ds, va_idx)
#     train_loaders["X"] = _build_loader(tr_ds, ...)
# We insert an augmentation wrapper between the Subset and the loader.

OLD_WRAP = """        tr_ds = Subset(ds, tr_idx)
        va_ds = Subset(ds, va_idx)"""
NEW_WRAP = """        tr_ds = AugmentedNPZClipDataset(Subset(ds, tr_idx), augment=not args.no_augment)
        va_ds = Subset(ds, va_idx)"""

n_wrapped = src.count(OLD_WRAP)
if "AugmentedNPZClipDataset(Subset(" in src:
    skipped.append("augmentation wrapper (already applied)")
elif n_wrapped > 0:
    src = src.replace(OLD_WRAP, NEW_WRAP)
    applied.append(f"augmentation wrapper applied to {n_wrapped} dataset block(s)")
else:
    print("WARNING: Subset() blocks not found; augmentation not wired")

# The MCD block uses `shuffle=False` and clinical labels; augmentation there
# would break the LDS weight alignment, so revert that one specifically.
MCD_MARKER = '''        train_loaders["mcd"] = _build_loader(tr_ds, args.batch_size, args.num_workers, shuffle=False)'''
if MCD_MARKER in src and "AugmentedNPZClipDataset" in src:
    # Find the MCD block and un-wrap it
    mcd_start = src.rfind("if args.mcd_root:")
    if mcd_start > 0:
        mcd_block = src[mcd_start:]
        fixed = mcd_block.replace(
            "tr_ds = AugmentedNPZClipDataset(Subset(ds, tr_idx), augment=not args.no_augment)",
            "tr_ds = Subset(ds, tr_idx)  # no augmentation: LDS weights are index-aligned",
            1,
        )
        src = src[:mcd_start] + fixed
        applied.append("MCD block excluded from augmentation (LDS alignment)")

# ─────────────────────────────────────────────────────────────────────────
# 5. Add the --no-augment flag
# ─────────────────────────────────────────────────────────────────────────
ARG_ANCHOR = '    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")'
ARG_NEW = ('    p.add_argument("--no-augment", action="store_true", '
           'help="Disable training-time augmentation")\n' + ARG_ANCHOR)

if '"--no-augment"' in src:
    skipped.append("--no-augment flag (already present)")
elif ARG_ANCHOR in src:
    src = src.replace(ARG_ANCHOR, ARG_NEW, 1)
    applied.append("--no-augment flag added")

# ─────────────────────────────────────────────────────────────────────────
# 6. ClinicalRegressor shim — v2 extractor returns 3 values, not 2
# ─────────────────────────────────────────────────────────────────────────
CLIN_ANCHOR = "        clinical_model = ClinicalRegressor(model.extractor, latent_dim=256, out_targets=3).to(device)"
CLIN_NEW = """        # v2 extractor returns (temporal, latent, attn); ClinicalRegressor expects
        # a 2-tuple. Wrap it so the clinical stage keeps working unchanged.
        class _ExtractorShim(torch.nn.Module):
            def __init__(self, ext):
                super().__init__()
                self.ext = ext
            def forward(self, x):
                _temporal, latent, attn = self.ext(x)
                return latent, attn

        clinical_model = ClinicalRegressor(_ExtractorShim(model.extractor), latent_dim=256, out_targets=3).to(device)"""

if "_ExtractorShim" in src:
    skipped.append("clinical shim (already present)")
elif CLIN_ANCHOR in src:
    src = src.replace(CLIN_ANCHOR, CLIN_NEW, 1)
    applied.append("ClinicalRegressor shim added")

# ─────────────────────────────────────────────────────────────────────────
# Write out + verify
# ─────────────────────────────────────────────────────────────────────────
TRAIN_PY.write_text(src)

import py_compile
try:
    py_compile.compile(str(TRAIN_PY), doraise=True)
    compile_ok = True
except py_compile.PyCompileError as e:
    compile_ok = False
    print(f"\nCOMPILE ERROR after patching:\n{e}")

print("\n" + "=" * 62)
print("PATCH SUMMARY")
print("=" * 62)
for a in applied:
    print(f"  [applied] {a}")
for s in skipped:
    print(f"  [skipped] {s}")
print()
print(f"  py_compile: {'OK' if compile_ok else 'FAILED'}")

# Assertions so a broken patch fails loudly rather than silently training v1
assert "EquiPhysDANNV2(" in src, "model swap did not take"
assert "PhysicsInformedLossV2(" in src, "loss swap did not take"
assert "AugmentedNPZClipDataset(" in src, "augmentation wrapper did not take"
assert compile_ok, "patched file does not compile"
print("  all assertions passed")
