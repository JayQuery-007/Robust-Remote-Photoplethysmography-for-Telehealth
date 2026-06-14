from __future__ import annotations

import argparse

from dataset_connectors import MultiDatasetConfig, build_objective_dataloaders, print_dataset_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EquiPhys dataloaders for UBFC/IBVP/MMPD/MCD")
    p.add_argument("--ubfc-root", type=str, default=None, help="Path to UBFC preprocessed NPZ clips")
    p.add_argument("--ibvp-root", type=str, default=None, help="Path to IBVP preprocessed NPZ clips")
    p.add_argument("--mmpd-root", type=str, default=None, help="Path to MMPD preprocessed NPZ clips")
    p.add_argument("--mcd-root", type=str, default=None, help="Path to MCD preprocessed NPZ clips")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MultiDatasetConfig(
        ubfc_root=args.ubfc_root,
        ibvp_root=args.ibvp_root,
        mmpd_root=args.mmpd_root,
        mcd_root=args.mcd_root,
        frames=args.frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    loaders = build_objective_dataloaders(cfg)
    if not loaders:
        print("No dataset roots provided. Pass one or more of --ubfc-root --ibvp-root --mmpd-root --mcd-root")
        return

    print_dataset_summary(loaders)


if __name__ == "__main__":
    main()
