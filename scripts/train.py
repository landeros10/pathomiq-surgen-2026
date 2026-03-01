"""Train the transformer-based MIL classifier for MMR/MSI prediction.

Usage:
    # Real data (paths from config.yaml)
    python scripts/train.py

    # Synthetic data (quick pipeline test)
    python scripts/etl/synthetic.py
    python scripts/train.py --data-dir data/synthetic --embeddings-dir embeddings/synthetic

    # Custom config
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.etl.dataset import MILDataset
from scripts.models.mil_transformer import MILTransformer
from scripts.utils.mlflow_utils import log_metrics_at_thresholds


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Training / evaluation loops ───────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> float:
    model.train()
    losses = []
    amp_enabled = device.type == "cuda"
    for embeddings, labels, _ in loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            loss = criterion(model(embeddings), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Returns (mean_loss, auroc, probs, labels)."""
    model.eval()
    losses, probs, labels_out = [], [], []
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for embeddings, labels, _ in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(embeddings)
            losses.append(criterion(logits, labels).item())
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels_out.extend(labels.cpu().tolist())
    auroc = roc_auc_score(labels_out, probs) if len(set(labels_out)) > 1 else 0.0
    return float(np.mean(losses)), float(auroc), probs, labels_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    config_path: str,
    data_dir_override: str = None,
    emb_dir_override: str = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    embeddings_dir = emb_dir_override or cfg["paths"]["embeddings_dir"]
    data_dir       = data_dir_override or cfg["paths"]["data_dir"]

    def make_loader(split_key: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            MILDataset(
                split_csv=os.path.join(data_dir, cfg["data"][split_key]),
                embeddings_dir=embeddings_dir,
                label_col=cfg["data"]["label_column"],
                slide_id_col=cfg["data"]["slide_id_column"],
            ),
            batch_size=1,
            shuffle=shuffle,
        )

    train_loader = make_loader("train_split", shuffle=True)
    val_loader   = make_loader("val_split",   shuffle=False)

    mc = cfg["model"]
    model = MILTransformer(
        input_dim=mc["input_dim"],
        hidden_dim=mc["hidden_dim"],
        num_layers=mc["transformer_layers"],
        num_heads=mc["num_heads"],
        ffn_dim=mc["ffn_dim"],
        dropout=mc["dropout"],
        layer_norm_eps=mc["layer_norm_eps"],
    ).to(device)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    tc        = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    criterion = nn.BCEWithLogitsLoss()
    thresholds = cfg["evaluation"]["thresholds"]

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "input_dim":          mc["input_dim"],
            "hidden_dim":         mc["hidden_dim"],
            "transformer_layers": mc["transformer_layers"],
            "num_heads":          mc["num_heads"],
            "ffn_dim":            mc["ffn_dim"],
            "dropout":            mc["dropout"],
            "layer_norm_eps":     mc["layer_norm_eps"],
            "lr":                 tc["lr"],
            "epochs":             tc["epochs"],
            "device":             str(device),
            "amp":                scaler is not None,
            "embeddings_dir":     embeddings_dir,
        })

        best_auroc = -1.0
        no_improve = 0
        patience   = tc.get("early_stopping_patience", 30)
        best_path  = models_dir / "best_model.pt"

        for epoch in range(tc["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_auroc, val_probs, val_labels = evaluate(
                model, val_loader, criterion, device
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "val_auroc":  val_auroc,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1:>3}/{tc['epochs']}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_auroc={val_auroc:.4f}"
            )

            # Periodic checkpoint
            if (epoch + 1) % tc.get("save_every", 20) == 0:
                torch.save(model.state_dict(), models_dir / f"epoch_{epoch+1:04d}.pt")

            # Best-model checkpoint
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        # Log final threshold metrics on the validation set
        log_metrics_at_thresholds(val_probs, val_labels, thresholds, prefix="val")
        mlflow.log_metric("best_val_auroc", best_auroc)
        if best_path.exists():
            mlflow.log_artifact(str(best_path))

    print(f"\nDone.  Best val AUROC: {best_auroc:.4f}  →  {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="configs/config.yaml")
    parser.add_argument("--data-dir",        default=None, help="Override data_dir from config")
    parser.add_argument("--embeddings-dir",  default=None, help="Override embeddings_dir from config")
    args = parser.parse_args()
    main(args.config, args.data_dir, args.embeddings_dir)
