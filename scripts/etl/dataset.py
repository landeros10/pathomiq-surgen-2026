"""Lazy-loading MIL dataset for WSI patch embeddings.

Supports two embedding formats, auto-detected from the path in embeddings_dir:
  - Zarr  (.zarr) — Zenodo pre-extracted UNI embeddings; reads ['features'] array
                    of shape (N_patches, 1024) stored as one directory per slide.
  - PyTorch (.pt) — locally saved tensors of shape (N_patches, embedding_dim).

Files are loaded one at a time so the full dataset never sits in RAM.
GCP (gs://) paths are supported for .pt files via gcsfs; set
DataLoader(num_workers=0) when using GCS to avoid multiprocessing issues.
"""

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def get_grid_shape(zarr_path: str) -> tuple[int, int]:
    """Return (H, W) grid dimensions from a Zarr store's coords array.

    Args:
        zarr_path: Path to a .zarr directory containing a 'coords' array of
                   shape (N, 2) with (row, col) patch indices.
    Returns:
        (H, W) tuple where H = max_row + 1 and W = max_col + 1.
    """
    import zarr
    store = zarr.open(zarr_path, mode="r")
    coords = store["coords"][:]  # (N, 2) int64
    H = int(coords[:, 0].max()) + 1
    W = int(coords[:, 1].max()) + 1
    return H, W


def mil_collate_fn(batch):
    """Collate function for MIL DataLoaders (always batch_size=1).

    Identical to default_collate except that None coords (from .pt files)
    are preserved as None rather than causing a collate error.
    Tensors are unsqueezed to add the batch dimension.
    Strings are wrapped in a list to match default_collate behaviour.
    """
    assert len(batch) == 1, "MIL training requires batch_size=1"
    item = batch[0]
    result = []
    for elem in item:
        if isinstance(elem, torch.Tensor):
            result.append(elem.unsqueeze(0))
        elif elem is None:
            result.append(None)
        else:
            result.append([elem])  # strings → list, matches default_collate
    return tuple(result)


class MILDataset(Dataset):
    """MIL dataset that lazily loads one slide embedding per __getitem__ call.

    Args:
        split_csv:      Path to a CSV with at least slide_id and label columns.
        embeddings_dir: Root directory containing per-slide embedding files.
                        Each slide is expected at <embeddings_dir>/<slide_id>.zarr
                        or <embeddings_dir>/<slide_id>.pt — format is auto-detected
                        from the suffix of the first resolved path.
        label_col:      Column name for the binary MMR label (0=MSS, 1=MSI).
        slide_id_col:   Column name for the slide identifier.

    Returns per item:
        embeddings: (N_patches, embedding_dim) float32 tensor
        label:      scalar float32 tensor
        slide_id:   str
    """

    def __init__(
        self,
        split_csv: str,
        embeddings_dir: str,
        label_col: str = "mmr_status",
        slide_id_col: str = "slide_id",
    ):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.embeddings_dir = str(embeddings_dir).rstrip("/")
        self.label_col = label_col
        self.slide_id_col = slide_id_col
        self._gcs_fs = None  # lazily initialised on first GCS access

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, object]:
        row = self.df.iloc[idx]
        slide_id = str(row[self.slide_id_col])
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        embeddings, coords = self._load_embedding(slide_id)
        return embeddings, label, slide_id, coords

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_embedding(self, slide_id: str) -> Tuple[torch.Tensor, object]:
        """Resolve path and dispatch to the correct loader by file extension.

        Returns:
            (embeddings, coords) where coords is a (N, 2) long tensor for Zarr
            files, or None for .pt files.
        """
        zarr_path = Path(f"{self.embeddings_dir}/{slide_id}.zarr")
        pt_path   = Path(f"{self.embeddings_dir}/{slide_id}.pt")

        if zarr_path.exists():
            return self._load_zarr(str(zarr_path))
        if pt_path.exists():
            pt = str(pt_path)
            if pt.startswith("gs://"):
                emb = self._load_gcs(pt)
            else:
                emb = torch.load(pt, map_location="cpu", weights_only=True)
            return emb, None

        # SR386 fallback: CSV IDs like 'SR386_40X_HE_T1' map to server Zarr
        # filenames 'SR386_40X_HE_T001_01.zarr' (zero-padded, _01 suffix).
        resolved = self._pad_sr386_id(slide_id)
        if resolved != slide_id:
            zarr_path = Path(f"{self.embeddings_dir}/{resolved}.zarr")
            if zarr_path.exists():
                return self._load_zarr(str(zarr_path))

        raise FileNotFoundError(
            f"No embedding found for slide '{slide_id}' in {self.embeddings_dir}. "
            f"Expected '{zarr_path}' or '{pt_path}'."
        )

    @staticmethod
    def _pad_sr386_id(slide_id: str) -> str:
        """Map un-padded SR386 CSV IDs to zero-padded server Zarr filenames.

        e.g. 'SR386_40X_HE_T1' → 'SR386_40X_HE_T001_01'
             'SR386_40X_HE_T10' → 'SR386_40X_HE_T010_01'
        Returns slide_id unchanged if it doesn't match the pattern.
        """
        m = re.match(r'^(.*_T)(\d{1,3})$', slide_id)
        if m:
            return f"{m.group(1)}{int(m.group(2)):03d}_01"
        return slide_id

    def _load_zarr(self, path: str) -> Tuple[torch.Tensor, object]:
        """Load pre-extracted UNI embeddings from a Zenodo-format Zarr store.

        The store must contain a 'features' array of shape (N_patches, 1024)
        and optionally a 'coords' array of shape (N_patches, 2).

        Returns:
            (features, coords) where coords is a (N, 2) long tensor or None.
        """
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "zarr is required for .zarr embeddings. "
                "Install with: pip install zarr"
            )
        store = zarr.open(path, mode="r")
        features = torch.from_numpy(store["features"][:]).to(torch.float32)
        coords = torch.from_numpy(store["coords"][:]).to(torch.long) if "coords" in store else None
        return features, coords

    def _load_gcs(self, gcs_path: str) -> torch.Tensor:
        """Load a .pt file directly from GCS without a local temp file.

        Returns only the embedding tensor (no coords for .pt files).
        """
        try:
            import gcsfs  # noqa: F401
        except ImportError:
            raise ImportError(
                "gcsfs is required for GCP paths. "
                "Install with: pip install gcsfs"
            )
        if self._gcs_fs is None:
            import gcsfs
            self._gcs_fs = gcsfs.GCSFileSystem()
        with self._gcs_fs.open(gcs_path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=True)


class MultitaskMILDataset(MILDataset):
    """MILDataset variant for joint multitask learning.

    Expects CSV with columns label_{task} for each task in tasks list.
    Returns a valid_mask tensor alongside labels so the training loop can
    compute loss only where a label is present (NaN → mask=0).

    Returns per item:
        embeddings:  (N_patches, embedding_dim) float32 tensor
        labels:      (num_tasks,) float32 tensor — 0.0 where NaN
        valid_mask:  (num_tasks,) float32 tensor — 1.0 where label present
        slide_id:    str
    """

    def __init__(self, split_csv, embeddings_dir, tasks, slide_id_col="slide_id"):
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        self.embeddings_dir = str(embeddings_dir).rstrip("/")
        self.tasks = tasks
        self.label_cols = [f"label_{t}" for t in tasks]
        self.slide_id_col = slide_id_col
        self._gcs_fs = None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row[self.slide_id_col])
        embeddings, coords = self._load_embedding(slide_id)
        labels, mask = [], []
        for col in self.label_cols:
            val = row[col]
            if pd.isna(val):
                labels.append(0.0); mask.append(0.0)
            else:
                labels.append(float(val)); mask.append(1.0)
        return (
            embeddings,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(mask,   dtype=torch.float32),
            slide_id,
            coords,
        )
