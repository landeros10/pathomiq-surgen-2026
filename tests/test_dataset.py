"""Tests for MILDataset — loading, ID resolution, error handling."""
import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from scripts.etl.dataset import MILDataset


class TestZarrLoading:

    def test_direct_slide_loads(self, zarr_emb_dir, split_csv, tmp_path):
        """SLIDE_DIRECT resolves immediately without any ID transform."""
        ds = MILDataset(str(split_csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        emb, label, sid, coords = ds[0]
        assert sid == "SLIDE_DIRECT"
        assert emb.shape == (50, 1024)
        assert emb.dtype == torch.float32

    def test_sr386_padding_fallback(self, zarr_emb_dir, split_csv):
        """Un-padded SR386 CSV ID resolves to the zero-padded server Zarr filename."""
        ds = MILDataset(str(split_csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        emb, label, sid, coords = ds[1]
        assert sid == "SR386_40X_HE_T5"
        assert emb.shape == (30, 1024)

    def test_wrong_zarr_key_raises(self, zarr_emb_dir, tmp_path):
        """Store with old 'feats' key raises KeyError — catches regression to pre-fix behaviour."""
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"case_id": ["SLIDE_BAD_KEY"], "label": [0]}).to_csv(csv, index=False)
        ds = MILDataset(str(csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        with pytest.raises(KeyError):
            ds[0]

    def test_missing_slide_raises(self, zarr_emb_dir, tmp_path):
        """Slide with no matching file on disk raises FileNotFoundError."""
        csv = tmp_path / "missing.csv"
        pd.DataFrame({"case_id": ["DOES_NOT_EXIST"], "label": [0]}).to_csv(csv, index=False)
        ds = MILDataset(str(csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        with pytest.raises(FileNotFoundError):
            ds[0]

    def test_label_dtype_and_value(self, zarr_emb_dir, split_csv):
        """Label is float32 tensor with the correct value from the CSV."""
        ds = MILDataset(str(split_csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        _, label, _, _ = ds[1]   # MSI slide, label=1
        assert label.dtype == torch.float32
        assert label.item() == 1.0

    def test_case_id_column_used_not_slide_id(self, zarr_emb_dir, split_csv):
        """slide_id column contains wrong sequential values; case_id must be used."""
        ds = MILDataset(str(split_csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        _, _, sid, _ = ds[0]
        assert sid == "SLIDE_DIRECT"   # not "slide_0"

    def test_len(self, zarr_emb_dir, split_csv):
        ds = MILDataset(str(split_csv), str(zarr_emb_dir), label_col="label", slide_id_col="case_id")
        assert len(ds) == 2


class TestPtLoading:

    def test_pt_slide_loads(self, pt_emb_dir, tmp_path):
        csv = tmp_path / "pt.csv"
        pd.DataFrame({"case_id": ["SLIDE_PT"], "label": [0]}).to_csv(csv, index=False)
        ds = MILDataset(str(csv), str(pt_emb_dir), label_col="label", slide_id_col="case_id")
        emb, label, sid, coords = ds[0]
        assert sid == "SLIDE_PT"
        assert emb.shape == (40, 1024)
        assert emb.dtype == torch.float32
        assert coords is None  # .pt files have no coords
