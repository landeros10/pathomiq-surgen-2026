"""Generate synthetic WSI embeddings for end-to-end pipeline testing.

Creates N slides, each saved as a (patches_per_slide, embedding_dim) float32
tensor at <output_dir>/<slide_id>.pt, then writes train/val/test CSV splits
that match the real dataset schema.

Usage:
    python scripts/etl/synthetic.py                        # uses configs/config.yaml
    python scripts/etl/synthetic.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def generate_synthetic_data(
    output_dir: str,
    data_dir: str,
    num_slides: int = 10,
    patches_per_slide: int = 100,
    embedding_dim: int = 1024,
    msi_fraction: float = 0.20,
    seed: int = 42,
) -> None:
    """Write synthetic embeddings and CSV splits to disk.

    Args:
        output_dir:       Where to write .pt embedding files.
        data_dir:         Where to write train/val/test CSV splits.
        num_slides:       Number of fake slides to generate.
        patches_per_slide: Patches per slide (all the same for synthetic data).
        embedding_dim:    Feature dimension — must match real UNI embeddings (1024).
        msi_fraction:     Fraction of slides labelled MSI/dMMR (real rate ~8%;
                          higher default here gives both classes in tiny splits).
        seed:             Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    emb_dir = Path(output_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(data_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    slide_ids = [f"SYNTHETIC_{i:04d}" for i in range(num_slides)]

    # Assign MSI labels
    labels = np.zeros(num_slides, dtype=int)
    n_msi = max(1, round(num_slides * msi_fraction))
    msi_idx = np.random.choice(num_slides, size=n_msi, replace=False)
    labels[msi_idx] = 1

    # Write one .pt file per slide
    for slide_id in slide_ids:
        emb = torch.randn(patches_per_slide, embedding_dim)
        torch.save(emb, emb_dir / f"{slide_id}.pt")

    # Build metadata frame and split 60 : 20 : 20
    df = pd.DataFrame({
        "case_id":  slide_ids,
        "slide_id": slide_ids,
        "label":    labels,
    })

    n_train = round(0.6 * num_slides)
    n_val   = round(0.2 * num_slides)
    n_test  = num_slides - n_train - n_val

    df.iloc[:n_train].to_csv(split_dir / "train.csv", index=False)
    df.iloc[n_train : n_train + n_val].to_csv(split_dir / "val.csv", index=False)
    df.iloc[n_train + n_val :].to_csv(split_dir / "test.csv", index=False)

    print("Synthetic data generated successfully.")
    print(f"  Embeddings : {emb_dir}/")
    print(f"               {num_slides} slides × {patches_per_slide} patches × {embedding_dim}-dim")
    print(f"  Splits     : {split_dir}/  (train={n_train}, val={n_val}, test={n_test})")
    print(f"  MSI slides : {n_msi} / {num_slides}  ({msi_fraction:.0%})")
    print()
    print("To train on synthetic data, run:")
    print(f"  python scripts/train.py --data-dir {split_dir} --embeddings-dir {emb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic WSI embeddings")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    syn = cfg["synthetic"]
    generate_synthetic_data(
        output_dir=syn["output_dir"],
        data_dir=syn["data_dir"],
        num_slides=syn["num_slides"],
        patches_per_slide=syn["patches_per_slide"],
        embedding_dim=syn["embedding_dim"],
        msi_fraction=syn["msi_fraction"],
        seed=syn["seed"],
    )
