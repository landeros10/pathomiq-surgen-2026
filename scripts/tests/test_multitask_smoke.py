"""Smoke test for MultitaskMILDataset and MultiMILTransformer.

Run with:
    source surgen-env/bin/activate
    python scripts/tests/test_multitask_smoke.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.etl.dataset import MultitaskMILDataset
from scripts.models.mil_transformer import MultiMILTransformer
from torch.utils.data import DataLoader


def main():
    tasks = ["mmr", "ras", "braf"]
    n_tasks = len(tasks)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Build two synthetic slides
        slide_ids = ["slide_001", "slide_002"]
        for sid in slide_ids:
            emb = torch.randn(10, 1024)
            torch.save(emb, tmpdir / f"{sid}.pt")

        # CSV: slide_002 has NaN for braf
        df = pd.DataFrame({
            "slide_id":   slide_ids,
            "label_mmr":  [1.0, 0.0],
            "label_ras":  [0.0, 1.0],
            "label_braf": [1.0, float("nan")],
        })
        csv_path = tmpdir / "splits.csv"
        df.to_csv(csv_path, index=False)

        # Dataset
        ds = MultitaskMILDataset(
            split_csv=str(csv_path),
            embeddings_dir=str(tmpdir),
            tasks=tasks,
        )
        assert len(ds) == 2, f"Expected 2 rows, got {len(ds)}"

        loader = DataLoader(ds, batch_size=1, shuffle=False)

        # Model
        model = MultiMILTransformer(
            input_dim=1024, hidden_dim=64, num_layers=1, num_heads=2,
            ffn_dim=128, dropout=0.0, output_classes=n_tasks,
        )
        model.eval()

        criterion = nn.BCEWithLogitsLoss()

        task_probs = {t: [] for t in tasks}
        task_labels = {t: [] for t in tasks}

        for embeddings, labels, valid_mask, slide_id in loader:
            # Forward pass
            logits = model(embeddings)  # (1, 3)
            assert logits.shape == (1, n_tasks), \
                f"Expected logits shape (1, {n_tasks}), got {logits.shape}"

            # Masked BCE loss
            step_loss = torch.tensor(0.0)
            for i, t in enumerate(tasks):
                mask_i = valid_mask[:, i].bool()
                if mask_i.any():
                    tl = criterion(logits[:, i][mask_i], labels[:, i][mask_i])
                    step_loss = step_loss + tl

            assert step_loss.requires_grad or step_loss.item() >= 0, \
                "step_loss should be a valid scalar"

            # Collect probs per task
            probs_all = torch.sigmoid(logits.detach())
            for i, t in enumerate(tasks):
                if valid_mask[0, i].item() == 1.0:
                    task_probs[t].append(probs_all[0, i].item())
                    task_labels[t].append(int(labels[0, i].item()))

        # slide_002 has NaN for braf — its braf entry should not be collected
        # slide_001 has braf=1.0 — should appear once
        assert len(task_probs["braf"]) == 1, \
            f"Expected 1 braf prob (NaN slide masked), got {len(task_probs['braf'])}"
        assert len(task_probs["mmr"]) == 2, \
            f"Expected 2 mmr probs, got {len(task_probs['mmr'])}"
        assert len(task_probs["ras"]) == 2, \
            f"Expected 2 ras probs, got {len(task_probs['ras'])}"

        # Verify gradient flows
        model.train()
        embeddings, labels, valid_mask, _ = next(iter(loader))
        logits = model(embeddings)
        loss = torch.tensor(0.0)
        for i, t in enumerate(tasks):
            mask_i = valid_mask[:, i].bool()
            if mask_i.any():
                loss = loss + criterion(logits[:, i][mask_i], labels[:, i][mask_i])
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), \
            "No gradients computed"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
