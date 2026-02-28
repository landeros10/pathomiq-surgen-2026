"""Utilities for creating and validating train/val/test dataset splits.

Typical usage:
    # Build stratified splits from a master metadata CSV
    from scripts.etl.splits import create_splits
    create_splits("data/metadata.csv", "data/")

    # Verify all slides have embedding files before training
    from scripts.etl.splits import validate_splits
    missing = validate_splits("data/", "embeddings/")
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    metadata_csv: str,
    output_dir: str,
    label_col: str = "mmr_status",
    slide_id_col: str = "slide_id",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits from a master metadata CSV.

    Stratification is applied on label_col to preserve the MSI class ratio
    (~8%) across all three splits.

    Args:
        metadata_csv: Path to CSV with at least slide_id and label columns.
        output_dir:   Directory to write train.csv / val.csv / test.csv.
        label_col:    Binary label column name.
        slide_id_col: Slide identifier column name.
        train_frac:   Proportion of data for training.
        val_frac:     Proportion of data for validation.
        seed:         Random seed.

    Returns:
        (train_df, val_df, test_df) DataFrames.
    """
    df = pd.read_csv(metadata_csv)
    test_frac = 1.0 - train_frac - val_frac

    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df[label_col], random_state=seed
    )
    val_frac_adj = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_frac_adj, stratify=train_val[label_col], random_state=seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv",   index=False)
    test.to_csv(output_dir / "test.csv",  index=False)

    print(f"Splits written to {output_dir}/")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: n={len(split):4d},  MSI rate={split[label_col].mean():.3f}")

    return train, val, test


def validate_splits(
    data_dir: str,
    embeddings_dir: str,
    slide_id_col: str = "slide_id",
) -> List[Tuple[str, str]]:
    """Check every slide in the CSV splits has a corresponding .pt embedding.

    Prints a warning for each missing file and returns the full list so the
    caller can decide how to handle failures.

    Args:
        data_dir:      Directory containing train/val/test CSV files.
        embeddings_dir: Root directory for .pt embedding files.
        slide_id_col:  Slide identifier column name.

    Returns:
        List of (split_filename, slide_id) tuples for missing embeddings.
    """
    data_dir = Path(data_dir)
    emb_dir  = Path(str(embeddings_dir).rstrip("/"))
    missing: List[Tuple[str, str]] = []

    for split_file in ["train.csv", "val.csv", "test.csv"]:
        path = data_dir / split_file
        if not path.exists():
            print(f"WARNING: {split_file} not found in {data_dir}")
            continue
        for slide_id in pd.read_csv(path)[slide_id_col]:
            if not (emb_dir / f"{slide_id}.pt").exists():
                missing.append((split_file, slide_id))

    if missing:
        print(f"WARNING: {len(missing)} missing embedding(s):")
        for fname, sid in missing[:10]:
            print(f"  {fname}: {sid}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("All embeddings present.")

    return missing
