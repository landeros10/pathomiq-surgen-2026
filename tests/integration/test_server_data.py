"""Integration tests — require /mnt/data-surgen to be mounted.

Run automatically on the GCP server. Skipped silently on any machine
where the mount is absent (e.g. local Mac dev environment).
"""
import pandas as pd
import pytest

from pathlib import Path

SERVER_EMB    = Path("/mnt/data-surgen/embeddings")
SERVER_SPLITS = Path("data/splits")

skip_if_no_mount = pytest.mark.skipif(
    not SERVER_EMB.exists(),
    reason="/mnt/data-surgen not mounted — skipping integration tests",
)


@skip_if_no_mount
def test_sr386_slide_loads():
    """First SR386 slide in the combined train split loads with shape (N, 1024)."""
    from scripts.etl.dataset import MILDataset
    df = pd.read_csv(SERVER_SPLITS / "SurGen_msi_train.csv")
    sr386_idx = int(df[df["case_id"].str.startswith("SR386")].index[0])

    ds = MILDataset(
        str(SERVER_SPLITS / "SurGen_msi_train.csv"),
        str(SERVER_EMB),
        label_col="label",
        slide_id_col="case_id",
    )
    emb, label, sid = ds[sr386_idx]
    assert emb.ndim == 2
    assert emb.shape[1] == 1024


@skip_if_no_mount
def test_sr1482_slide_loads():
    """First SR1482 slide in the combined train split loads with shape (N, 1024)."""
    from scripts.etl.dataset import MILDataset
    df = pd.read_csv(SERVER_SPLITS / "SurGen_msi_train.csv")
    sr1482_idx = int(df[df["case_id"].str.startswith("SR1482")].index[0])

    ds = MILDataset(
        str(SERVER_SPLITS / "SurGen_msi_train.csv"),
        str(SERVER_EMB),
        label_col="label",
        slide_id_col="case_id",
    )
    emb, label, sid = ds[sr1482_idx]
    assert emb.ndim == 2
    assert emb.shape[1] == 1024


@skip_if_no_mount
def test_validate_splits_server():
    """Every slide in all three SurGen MSI splits resolves to a real Zarr file."""
    from scripts.etl.splits import validate_splits
    missing = validate_splits(str(SERVER_SPLITS), str(SERVER_EMB))
    assert missing == [], f"{len(missing)} embedding(s) missing — run validate_splits manually to see which."
