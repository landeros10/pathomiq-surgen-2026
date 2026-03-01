"""Tests for validate_splits and _embedding_exists."""
import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from scripts.etl.splits import _embedding_exists, validate_splits


@pytest.fixture
def mixed_emb_dir(tmp_path):
    """Dir containing one .pt, one .zarr, and one SR386 zero-padded .zarr."""
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()

    torch.save(torch.randn(10, 1024), emb_dir / "SLIDE_PT.pt")

    root = zarr.open_group(str(emb_dir / "SLIDE_ZARR.zarr"), mode="w")
    root.create_array("features", data=np.zeros((10, 1024), dtype=np.float32))

    root = zarr.open_group(str(emb_dir / "SR386_40X_HE_T003_01.zarr"), mode="w")
    root.create_array("features", data=np.zeros((10, 1024), dtype=np.float32))

    return emb_dir


def _write_splits(split_dir, train_ids, labels=None):
    if labels is None:
        labels = [0] * len(train_ids)
    pd.DataFrame({"case_id": train_ids, "label": labels}).to_csv(
        split_dir / "train.csv", index=False
    )
    pd.DataFrame({"case_id": [], "label": []}).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame({"case_id": [], "label": []}).to_csv(split_dir / "test.csv", index=False)


class TestEmbeddingExists:

    def test_pt_found(self, mixed_emb_dir):
        assert _embedding_exists(mixed_emb_dir, "SLIDE_PT")

    def test_zarr_found(self, mixed_emb_dir):
        assert _embedding_exists(mixed_emb_dir, "SLIDE_ZARR")

    def test_sr386_padded_found(self, mixed_emb_dir):
        """Un-padded CSV ID 'SR386_40X_HE_T3' must resolve to SR386_40X_HE_T003_01.zarr."""
        assert _embedding_exists(mixed_emb_dir, "SR386_40X_HE_T3")

    def test_missing_not_found(self, mixed_emb_dir):
        assert not _embedding_exists(mixed_emb_dir, "DOES_NOT_EXIST")


class TestValidateSplits:

    def test_all_present_returns_empty(self, tmp_path, mixed_emb_dir):
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        _write_splits(split_dir, ["SLIDE_PT", "SLIDE_ZARR", "SR386_40X_HE_T3"])
        missing = validate_splits(str(split_dir), str(mixed_emb_dir))
        assert missing == []

    def test_missing_slide_reported(self, tmp_path, mixed_emb_dir):
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        _write_splits(split_dir, ["SLIDE_PT", "GHOST_SLIDE"])
        missing = validate_splits(str(split_dir), str(mixed_emb_dir))
        assert any("GHOST_SLIDE" in sid for _, sid in missing)

    def test_missing_csv_warned(self, tmp_path, mixed_emb_dir, capsys):
        """validate_splits prints a warning when a split CSV file is absent."""
        split_dir = tmp_path / "empty_splits"
        split_dir.mkdir()
        validate_splits(str(split_dir), str(mixed_emb_dir))
        assert "WARNING" in capsys.readouterr().out
