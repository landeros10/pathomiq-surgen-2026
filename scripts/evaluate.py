"""Evaluate a trained MIL checkpoint on a dataset split.

Prints AUROC, AUPRC, and per-class classification reports at four thresholds,
then logs all metrics and confusion-matrix figures to MLflow.

Usage:
    python scripts/evaluate.py --checkpoint models/best_model.pt
    python scripts/evaluate.py --checkpoint models/best_model.pt --split test
    python scripts/evaluate.py --checkpoint models/best_model.pt --split test \\
        --data-dir data/synthetic --embeddings-dir embeddings/synthetic
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.etl.dataset import MILDataset
from scripts.models.mil_transformer import MILTransformer
from scripts.utils.metrics import full_report
from scripts.utils.mlflow_utils import log_metrics_at_thresholds


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(
    config_path: str,
    checkpoint_path: str,
    split: str = "test",
    data_dir_override: str = None,
    emb_dir_override: str = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    embeddings_dir = emb_dir_override or cfg["paths"]["embeddings_dir"]
    data_dir       = data_dir_override or cfg["paths"]["data_dir"]
    split_key      = f"{split}_split"

    loader = DataLoader(
        MILDataset(
            split_csv=os.path.join(data_dir, cfg["data"][split_key]),
            embeddings_dir=embeddings_dir,
            label_col=cfg["data"]["label_column"],
            slide_id_col=cfg["data"]["slide_id_column"],
        ),
        batch_size=1,
        shuffle=False,
    )

    mc = cfg["model"]
    model = MILTransformer(
        input_dim=mc["input_dim"],
        hidden_dim=mc["hidden_dim"],
        num_layers=mc["transformer_layers"],
        num_heads=mc["num_heads"],
        ffn_dim=mc["ffn_dim"],
        dropout=mc["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            logits = model(embeddings.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())

    thresholds = cfg["evaluation"]["thresholds"]
    report     = full_report(all_labels, all_probs, thresholds)

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {split.upper()} SET  —  {checkpoint_path}")
    print(f"{'='*55}")
    print(f"  AUROC : {report['auroc']:.4f}")
    print(f"  AUPRC : {report['auprc']:.4f}")

    for t in thresholds:
        preds = (np.array(all_probs) >= t).astype(int)
        print(f"\n  ── threshold = {t} ──")
        print(
            classification_report(
                all_labels,
                preds,
                target_names=["MSS/pMMR", "MSI/dMMR"],
                zero_division=0,
            )
        )

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"eval_{split}"):
        mlflow.log_metrics({
            f"{split}_auroc": report["auroc"],
            f"{split}_auprc": report["auprc"],
        })
        log_metrics_at_thresholds(all_probs, all_labels, thresholds, prefix=split)
        mlflow.log_artifact(checkpoint_path)

    print(f"\nMetrics logged to MLflow experiment: {cfg['mlflow']['experiment_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--checkpoint",     required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split",          default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir",       default=None)
    parser.add_argument("--embeddings-dir", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split, args.data_dir, args.embeddings_dir)
