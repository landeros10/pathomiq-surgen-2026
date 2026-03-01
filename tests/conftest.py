"""Shared pytest fixtures for the SurGen test suite."""
import numpy as np
import pandas as pd
import pytest
import torch
import zarr


@pytest.fixture
def zarr_emb_dir(tmp_path):
    """Temp dir with three Zarr embedding stores:

    - SLIDE_DIRECT: straightforward name match (SR1482-style)
    - SR386_40X_HE_T005_01: zero-padded server filename for un-padded CSV ID 'SR386_40X_HE_T5'
    - SLIDE_BAD_KEY: store that uses old 'feats' key instead of 'features'
    """
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    root = zarr.open_group(str(emb_dir / "SLIDE_DIRECT.zarr"), mode="w")
    root.create_array("features", data=np.random.randn(50, 1024).astype(np.float32))

    root = zarr.open_group(str(emb_dir / "SR386_40X_HE_T005_01.zarr"), mode="w")
    root.create_array("features", data=np.random.randn(30, 1024).astype(np.float32))

    root = zarr.open_group(str(emb_dir / "SLIDE_BAD_KEY.zarr"), mode="w")
    root.create_array("feats", data=np.random.randn(10, 1024).astype(np.float32))

    return emb_dir


@pytest.fixture
def pt_emb_dir(tmp_path):
    """Temp dir with a single .pt embedding file."""
    emb_dir = tmp_path / "embeddings_pt"
    emb_dir.mkdir()
    torch.save(torch.randn(40, 1024), emb_dir / "SLIDE_PT.pt")
    return emb_dir


@pytest.fixture
def split_csv(tmp_path):
    """Minimal split CSV with case_id, slide_id (wrong sequential), and label columns."""
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({
        "case_id":  ["SLIDE_DIRECT", "SR386_40X_HE_T5"],
        "slide_id": ["SLIDE_DIRECT", "slide_0"],   # slide_id intentionally wrong
        "label":    [0, 1],
    }).to_csv(csv_path, index=False)
    return csv_path
