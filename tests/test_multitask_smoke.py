"""Smoke test for MultitaskMILDataset and MultiMILTransformer.

Migrated from scripts/tests/test_multitask_smoke.py.
Run with: python -m pytest tests/test_multitask_smoke.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.etl.dataset import MultitaskMILDataset
from scripts.models.mil_transformer import MultiMILTransformer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def _collate(batch):
    """Collate that tolerates None coords (returned when loading .pt files)."""
    embeddings, labels, masks, slide_ids, coords = zip(*batch)
    return (
        default_collate(list(embeddings)),
        default_collate(list(labels)),
        default_collate(list(masks)),
        list(slide_ids),
        list(coords),
    )


def _run_smoke_test():
    tasks   = ["mmr", "ras", "braf"]
    n_tasks = len(tasks)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        slide_ids = ["slide_001", "slide_002"]
        for sid in slide_ids:
            emb = torch.randn(10, 1024)
            torch.save(emb, tmpdir / f"{sid}.pt")

        df = pd.DataFrame({
            "slide_id":   slide_ids,
            "label_mmr":  [1.0, 0.0],
            "label_ras":  [0.0, 1.0],
            "label_braf": [1.0, float("nan")],
        })
        csv_path = tmpdir / "splits.csv"
        df.to_csv(csv_path, index=False)

        ds = MultitaskMILDataset(
            split_csv=str(csv_path),
            embeddings_dir=str(tmpdir),
            tasks=tasks,
        )
        assert len(ds) == 2, f"Expected 2 rows, got {len(ds)}"

        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate)

        model = MultiMILTransformer(
            input_dim=1024, hidden_dim=64, num_layers=1, num_heads=2,
            ffn_dim=128, dropout=0.0, output_classes=n_tasks,
        )
        model.eval()

        criterion = nn.BCEWithLogitsLoss()

        task_probs  = {t: [] for t in tasks}
        task_labels = {t: [] for t in tasks}

        for embeddings, labels, valid_mask, slide_id, coords in loader:
            logits = model(embeddings)  # (1, 3)
            assert logits.shape == (1, n_tasks), \
                f"Expected logits shape (1, {n_tasks}), got {logits.shape}"

            step_loss = torch.tensor(0.0)
            for i, t in enumerate(tasks):
                mask_i = valid_mask[:, i].bool()
                if mask_i.any():
                    tl        = criterion(logits[:, i][mask_i], labels[:, i][mask_i])
                    step_loss = step_loss + tl

            assert step_loss.requires_grad or step_loss.item() >= 0

            probs_all = torch.sigmoid(logits.detach())
            for i, t in enumerate(tasks):
                if valid_mask[0, i].item() == 1.0:
                    task_probs[t].append(probs_all[0, i].item())
                    task_labels[t].append(int(labels[0, i].item()))

        assert len(task_probs["braf"]) == 1, \
            f"Expected 1 braf prob (NaN slide masked), got {len(task_probs['braf'])}"
        assert len(task_probs["mmr"]) == 2, \
            f"Expected 2 mmr probs, got {len(task_probs['mmr'])}"
        assert len(task_probs["ras"]) == 2, \
            f"Expected 2 ras probs, got {len(task_probs['ras'])}"

        model.train()
        embeddings, labels, valid_mask, _, _coords = next(iter(loader))
        logits = model(embeddings)
        loss   = torch.tensor(0.0)
        for i, t in enumerate(tasks):
            mask_i = valid_mask[:, i].bool()
            if mask_i.any():
                loss = loss + criterion(logits[:, i][mask_i], labels[:, i][mask_i])
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), \
            "No gradients computed"


def test_multitask_smoke():
    _run_smoke_test()
