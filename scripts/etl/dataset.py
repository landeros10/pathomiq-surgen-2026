"""Lazy-loading MIL dataset for WSI patch embeddings.

Each .pt file is expected to contain a (N_patches, embedding_dim) float32
tensor for a single slide. Files are loaded one at a time so the full
dataset never sits in memory simultaneously.

Supports both local paths and GCP (gs://) via gcsfs. When using GCP, set
DataLoader(num_workers=0) to avoid multiprocessing issues with the GCS client.
"""

from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """MIL dataset that lazily loads one slide embedding per __getitem__ call.

    Args:
        split_csv:      Path to a CSV with at least slide_id and label columns.
        embeddings_dir: Root directory (local) or GCS prefix (gs://…) for .pt files.
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        slide_id = str(row[self.slide_id_col])
        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        embeddings = self._load_embedding(slide_id)
        return embeddings, label, slide_id

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_embedding(self, slide_id: str) -> torch.Tensor:
        path = f"{self.embeddings_dir}/{slide_id}.pt"
        if path.startswith("gs://"):
            return self._load_gcs(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_gcs(self, gcs_path: str) -> torch.Tensor:
        """Load a .pt file directly from GCS without a local temp file."""
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
