"""Train the transformer-based MIL classifier for MMR/MSI prediction.

Usage:
    # Real data (paths from config.yaml)
    python scripts/train.py

    # Synthetic data (quick pipeline test)
    python scripts/etl/synthetic.py
    python scripts/train.py --config configs/config_gate_test.yaml

    # Custom config
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.etl.dataset import MILDataset, MultitaskMILDataset
from scripts.models.mil_transformer import MILTransformer, MultiMILTransformer
from scripts.utils.metrics import compute_auprc, metrics_at_threshold
from scripts.utils.mlflow_utils import log_confusion_matrix, log_metrics_at_thresholds


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict using dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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
    accum_steps: int = 1,
) -> tuple:
    """Returns (mean_loss, auroc, probs, labels)."""
    model.train()
    losses, probs, labels_out = [], [], []
    amp_enabled = device.type == "cuda"
    optimizer.zero_grad()
    for step, (embeddings, labels, _) in enumerate(loader):
        embeddings, labels = embeddings.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(embeddings)
            loss = criterion(logits, labels) / accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        losses.append(loss.item() * accum_steps)  # log unscaled loss
        probs.extend(torch.sigmoid(logits.detach()).cpu().tolist())
        labels_out.extend(labels.cpu().tolist())
        is_last = (step + 1) == len(loader)
        if (step + 1) % accum_steps == 0 or is_last:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
    auroc = roc_auc_score(labels_out, probs) if len(set(labels_out)) > 1 else 0.0
    return float(np.mean(losses)), float(auroc), probs, labels_out


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


def train_one_epoch_multitask(model, loader, optimizer, criterion, device,
                               scaler, accum_steps, tasks):
    """Returns (mean_total_loss, {task: mean_loss}, {task: auroc},
                {task: probs}, {task: labels}, mean_grad_norm)."""
    model.train()
    n = len(tasks)
    task_probs  = {t: [] for t in tasks}
    task_labels = {t: [] for t in tasks}
    task_losses = {t: [] for t in tasks}
    total_losses = []
    grad_norms   = []
    optimizer.zero_grad()
    amp_enabled = device.type == "cuda"

    for step, (embeddings, labels, valid_mask, _) in enumerate(loader):
        embeddings = embeddings.to(device)
        labels     = labels.to(device)       # (1, n_tasks)
        valid_mask = valid_mask.to(device)   # (1, n_tasks)

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(embeddings)  # (1, n_tasks)
            step_loss = torch.tensor(0.0, device=device)
            for i, t in enumerate(tasks):
                mask_i = valid_mask[:, i].bool()
                if mask_i.any():
                    tl = criterion(logits[:, i][mask_i], labels[:, i][mask_i])
                    task_losses[t].append(tl.item())
                    step_loss = step_loss + tl
            step_loss = step_loss / n / accum_steps

        if scaler is not None:
            scaler.scale(step_loss).backward()
        else:
            step_loss.backward()

        total_losses.append(step_loss.item() * accum_steps * n)

        probs_all = torch.sigmoid(logits.detach()).cpu()
        for i, t in enumerate(tasks):
            if valid_mask[0, i].item() == 1.0:
                task_probs[t].append(probs_all[0, i].item())
                task_labels[t].append(int(labels[0, i].item()))

        is_last = (step + 1) == len(loader)
        if (step + 1) % accum_steps == 0 or is_last:
            total_norm = torch.sqrt(torch.stack([
                p.grad.detach().norm() ** 2
                for p in model.parameters() if p.grad is not None
            ]).sum())
            grad_norms.append(total_norm.item())
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    task_auroc = {}
    task_mean_loss = {}
    for t in tasks:
        p, l = task_probs[t], task_labels[t]
        task_auroc[t] = roc_auc_score(l, p) if len(set(l)) > 1 else 0.0
        task_mean_loss[t] = float(np.mean(task_losses[t])) if task_losses[t] else 0.0

    mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return float(np.mean(total_losses)), task_mean_loss, task_auroc, task_probs, task_labels, mean_grad_norm


def evaluate_multitask(model, loader, criterion, device, tasks):
    """Returns (mean_total_loss, {task: mean_loss}, {task: auroc},
                {task: probs}, {task: labels})."""
    model.eval()
    n = len(tasks)
    task_probs  = {t: [] for t in tasks}
    task_labels = {t: [] for t in tasks}
    task_losses = {t: [] for t in tasks}
    total_losses = []
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for embeddings, labels, valid_mask, _ in loader:
            embeddings = embeddings.to(device)
            labels     = labels.to(device)
            valid_mask = valid_mask.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits    = model(embeddings)
                step_loss = torch.tensor(0.0, device=device)
                for i, t in enumerate(tasks):
                    mask_i = valid_mask[:, i].bool()
                    if mask_i.any():
                        tl = criterion(logits[:, i][mask_i], labels[:, i][mask_i])
                        task_losses[t].append(tl.item())
                        step_loss = step_loss + tl
                step_loss = step_loss / n
            total_losses.append(step_loss.item())
            probs_all = torch.sigmoid(logits).cpu()
            for i, t in enumerate(tasks):
                if valid_mask[0, i].item() == 1.0:
                    task_probs[t].append(probs_all[0, i].item())
                    task_labels[t].append(int(labels[0, i].item()))

    task_auroc = {}
    task_mean_loss = {}
    for t in tasks:
        p, l = task_probs[t], task_labels[t]
        task_auroc[t] = roc_auc_score(l, p) if len(set(l)) > 1 else 0.0
        task_mean_loss[t] = float(np.mean(task_losses[t])) if task_losses[t] else 0.0

    return float(np.mean(total_losses)), task_mean_loss, task_auroc, task_probs, task_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    config_path: str,
    data_dir_override: str = None,
    emb_dir_override: str = None,
    run_name_override: str = None,
    run_suffix: str = None,
    max_epochs: int = None,
    patience_override: int = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Seeding ───────────────────────────────────────────────────────────────
    seed = cfg["training"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = get_device()
    print(f"Device: {device}")

    embeddings_dir = emb_dir_override or cfg["paths"]["embeddings_dir"]
    data_dir       = data_dir_override or cfg["paths"]["data_dir"]

    mc = cfg["model"]
    output_classes = mc.get("output_classes", 1)
    multitask = output_classes > 1
    tasks = cfg["data"].get("tasks", []) if multitask else []

    def make_loader(split_key: str, shuffle: bool) -> DataLoader:
        if multitask:
            ds = MultitaskMILDataset(
                split_csv=os.path.join(data_dir, cfg["data"][split_key]),
                embeddings_dir=embeddings_dir,
                tasks=tasks,
                slide_id_col=cfg["data"]["slide_id_column"],
            )
        else:
            ds = MILDataset(
                split_csv=os.path.join(data_dir, cfg["data"][split_key]),
                embeddings_dir=embeddings_dir,
                label_col=cfg["data"]["label_column"],
                slide_id_col=cfg["data"]["slide_id_column"],
            )
        return DataLoader(ds, batch_size=1, shuffle=shuffle)

    train_loader = make_loader("train_split", shuffle=True)
    val_loader   = make_loader("val_split",   shuffle=False)
    test_loader  = make_loader("test_split",  shuffle=False)

    if multitask:
        model = MultiMILTransformer(
            input_dim=mc["input_dim"],
            hidden_dim=mc["hidden_dim"],
            num_layers=mc["transformer_layers"],
            num_heads=mc["num_heads"],
            ffn_dim=mc["ffn_dim"],
            dropout=mc["dropout"],
            layer_norm_eps=mc["layer_norm_eps"],
            output_classes=output_classes,
        ).to(device)
    else:
        model = MILTransformer(
            input_dim=mc["input_dim"],
            hidden_dim=mc["hidden_dim"],
            num_layers=mc["transformer_layers"],
            num_heads=mc["num_heads"],
            ffn_dim=mc["ffn_dim"],
            dropout=mc["dropout"],
            layer_norm_eps=mc["layer_norm_eps"],
        ).to(device)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    tc        = cfg["training"]
    n_epochs  = max_epochs if max_epochs is not None else tc["epochs"]
    lr_scheduler_type = tc.get("lr_scheduler", "none")
    class_weighting   = tc.get("class_weighting", False)
    accum_steps       = tc.get("grad_accum_steps", 1)
    early_stopping_patience = (
        patience_override if patience_override is not None
        else tc.get("early_stopping_patience", 0)
    )

    weight_decay = tc.get("weight_decay", 0.0)
    opt_type = tc.get("optimizer", "adam").lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=weight_decay)

    if multitask:
        pos_weight = None
    elif class_weighting:
        train_csv = os.path.join(data_dir, cfg["data"]["train_split"])
        _df = pd.read_csv(train_csv)
        _n_pos = int(_df[cfg["data"]["label_column"]].sum())
        _n_neg = len(_df) - _n_pos
        pos_weight = torch.tensor([_n_neg / _n_pos], dtype=torch.float32).to(device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr_warmup_epochs = tc.get("lr_scheduler_warmup_epochs", 0)
    lr_eta_min       = tc.get("lr_scheduler_eta_min", 0.0)
    lr_T_max         = tc.get("lr_scheduler_T_max", max(1, n_epochs - lr_warmup_epochs))

    if lr_scheduler_type == "cosine":
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=lr_T_max, eta_min=lr_eta_min
        )
        if lr_warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=lr_warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[lr_warmup_epochs]
            )
        else:
            scheduler = cosine_sched
    else:
        scheduler = None

    thresholds = cfg["evaluation"]["thresholds"]

    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow.enable_system_metrics_logging()

    run_name = run_name_override or cfg["mlflow"].get("run_name")
    if run_suffix is not None:
        run_name = f"{run_name}-{run_suffix}"
    with mlflow.start_run(run_name=run_name):
        # ── Full config + runtime-derived params ──────────────────────────────
        mlflow.log_params(_flatten_dict(cfg))
        extra = {
            "loss_fn":                   "BCEWithLogitsLoss(weighted)" if (class_weighting and not multitask) else "BCEWithLogitsLoss",
            "aggregator":                mc.get("aggregation", "mean"),
            "arch_variant":              "MILTransformer",
            "device":                    str(device),
            "amp":                       scaler is not None,
            "lr_scheduler":              lr_scheduler_type,
            "lr_warmup_epochs":          lr_warmup_epochs,
            "lr_eta_min":                lr_eta_min,
            "lr_T_max":                  lr_T_max,
            "class_weighting":           class_weighting,
            "grad_accum_steps":          accum_steps,
            "early_stopping_patience":   early_stopping_patience,
        }
        if class_weighting and not multitask:
            extra["pos_weight"] = round(_n_neg / _n_pos, 4)
        if run_name is not None:
            extra["run_name"] = run_name
        if multitask:
            extra["model_type"] = "MultiMILTransformer"
            extra["output_classes"] = output_classes
            extra["tasks"] = ",".join(tasks)
        mlflow.log_params(extra)
        mlflow.set_tag("git_commit", _git_commit())

        if multitask:
            for split_key, split_name in [
                ("train_split", "train"), ("val_split", "val"), ("test_split", "test")
            ]:
                split_csv = os.path.join(data_dir, cfg["data"][split_key])
                df_split = pd.read_csv(split_csv)
                for t in tasks:
                    mlflow.log_param(
                        f"n_valid_{t}_{split_name}",
                        int(df_split[f"label_{t}"].notna().sum())
                    )

        # Log the config file itself as an artifact
        mlflow.log_artifact(config_path)

        best_auroc = -1.0
        best_epoch = -1
        best_val_probs: list = []
        best_val_labels: list = []
        no_improve = 0
        # 0 = disabled; treat as "never trigger" sentinel
        patience   = early_stopping_patience if early_stopping_patience > 0 else 10 ** 9
        best_path  = models_dir / "best_model.pt"

        # Multitask best-val tracking
        best_val_probs_d:  dict = {t: [] for t in tasks}
        best_val_labels_d: dict = {t: [] for t in tasks}
        best_val_auroc_d:  dict = {t: 0.0 for t in tasks}
        best_val_auprc_mean: float = 0.0

        val_probs, val_labels = [], []
        for epoch in range(n_epochs):
            if multitask:
                train_loss, train_loss_d, train_auroc_d, train_probs_d, train_labels_d, grad_norm = \
                    train_one_epoch_multitask(
                        model, train_loader, optimizer, criterion, device, scaler, accum_steps, tasks
                    )
                val_loss, val_loss_d, val_auroc_d, val_probs_d, val_labels_d = \
                    evaluate_multitask(model, val_loader, criterion, device, tasks)

                val_auroc_mean = float(np.mean(list(val_auroc_d.values())))
                val_auprc_mean = float(np.mean([
                    compute_auprc(val_labels_d[t], val_probs_d[t]) for t in tasks
                ]))

                metrics = {
                    "train_loss":       train_loss,
                    "val_loss":         val_loss,
                    "val_auroc_mean":   val_auroc_mean,
                    "val_auprc_mean":   val_auprc_mean,
                    "global_grad_norm": grad_norm,
                    "learning_rate":    optimizer.param_groups[0]["lr"],
                }
                for i, t in enumerate(tasks):
                    metrics[f"head_weight_var_{t}"] = model.classifier.weight[i].var().item()
                for t in tasks:
                    metrics[f"train_loss_{t}"]        = train_loss_d[t]
                    metrics[f"val_loss_{t}"]           = val_loss_d[t]
                    metrics[f"train_auroc_{t}"]        = train_auroc_d[t]
                    metrics[f"val_auroc_{t}"]          = val_auroc_d[t]
                    metrics[f"train_auprc_{t}"]        = compute_auprc(train_labels_d[t], train_probs_d[t])
                    metrics[f"val_auprc_{t}"]          = compute_auprc(val_labels_d[t], val_probs_d[t])
                    tm = metrics_at_threshold(train_labels_d[t], train_probs_d[t], 0.5)
                    vm = metrics_at_threshold(val_labels_d[t], val_probs_d[t], 0.5)
                    metrics[f"train_f1_{t}"]          = tm["f1"]
                    metrics[f"train_sensitivity_{t}"] = tm["recall"]
                    metrics[f"train_specificity_{t}"] = tm["specificity"]
                    metrics[f"val_f1_{t}"]            = vm["f1"]
                    metrics[f"val_sensitivity_{t}"]   = vm["recall"]
                    metrics[f"val_specificity_{t}"]   = vm["specificity"]
                mlflow.log_metrics(metrics, step=epoch)

                print(
                    f"Epoch {epoch+1:>3}/{n_epochs}  "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_auroc_mean={val_auroc_mean:.4f}  val_auprc_mean={val_auprc_mean:.4f}"
                )

                monitor_auroc = val_auroc_mean
            else:
                train_loss, train_auroc, train_probs, train_labels = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, scaler, accum_steps
                )
                val_loss, val_auroc, val_probs, val_labels = evaluate(
                    model, val_loader, criterion, device
                )

                train_m     = metrics_at_threshold(train_labels, train_probs, 0.5)
                train_auprc = compute_auprc(train_labels, train_probs)
                val_m       = metrics_at_threshold(val_labels, val_probs, 0.5)
                val_auprc   = compute_auprc(val_labels, val_probs)

                mlflow.log_metrics(
                    {
                        "train_loss":        train_loss,
                        "train_auroc":       train_auroc,
                        "train_auprc":       train_auprc,
                        "train_f1":          train_m["f1"],
                        "train_sensitivity": train_m["recall"],
                        "train_specificity": train_m["specificity"],
                        "val_loss":          val_loss,
                        "val_auroc":         val_auroc,
                        "val_auprc":         val_auprc,
                        "val_f1":            val_m["f1"],
                        "val_sensitivity":   val_m["recall"],
                        "val_specificity":   val_m["specificity"],
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch+1:>3}/{n_epochs}  "
                    f"train_loss={train_loss:.4f}  train_auroc={train_auroc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_auroc={val_auroc:.4f}  "
                    f"val_f1={val_m['f1']:.4f}  val_sens={val_m['recall']:.4f}  val_spec={val_m['specificity']:.4f}"
                )

                monitor_auroc = val_auroc

            if scheduler is not None:
                scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % tc.get("save_every", 20) == 0:
                ckpt = {"model": model.state_dict()}
                if scheduler is not None:
                    ckpt["scheduler"] = scheduler.state_dict()
                torch.save(ckpt, models_dir / f"epoch_{epoch+1:04d}.pt")

            # Best-model checkpoint
            if monitor_auroc > best_auroc:
                best_auroc = monitor_auroc
                best_epoch = epoch + 1
                if multitask:
                    best_val_probs_d  = {t: list(val_probs_d[t]) for t in tasks}
                    best_val_labels_d = {t: list(val_labels_d[t]) for t in tasks}
                    best_val_auroc_d  = dict(val_auroc_d)
                    best_val_auprc_mean = val_auprc_mean
                else:
                    best_val_probs  = list(val_probs)
                    best_val_labels = list(val_labels)
                no_improve = 0
                ckpt = {"model": model.state_dict()}
                if scheduler is not None:
                    ckpt["scheduler"] = scheduler.state_dict()
                torch.save(ckpt, best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        # ── Final val metrics + confusion matrix ──────────────────────────────
        if multitask:
            mlflow.log_metric("best_val_auroc_mean", best_auroc)
            mlflow.log_metric("best_val_auprc_mean", best_val_auprc_mean)
            mlflow.log_metric("best_epoch", best_epoch)
        else:
            log_metrics_at_thresholds(val_probs, val_labels, thresholds, prefix="val")
            mlflow.log_metric("best_val_auroc", best_auroc)
            mlflow.log_metric("best_epoch", best_epoch)
            log_metrics_at_thresholds(best_val_probs, best_val_labels, thresholds, prefix="best_val", step=best_epoch)
            log_confusion_matrix(val_probs, val_labels, threshold=0.5, prefix="val")

        # ── Test set evaluation ───────────────────────────────────────────────
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)  # backward compat with pre-Phase-4 checkpoints
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="best_model",
                step=best_epoch,
            )

        if multitask:
            test_loss, test_loss_d, test_auroc_d, test_probs_d, test_labels_d = \
                evaluate_multitask(model, test_loader, criterion, device, tasks)
            test_auprc_mean = float(np.mean([
                compute_auprc(test_labels_d[t], test_probs_d[t]) for t in tasks
            ]))
            mlflow.log_metric("test_auprc_mean", test_auprc_mean)
            for t in tasks:
                mlflow.log_metric(f"best_val_auroc_{t}", best_val_auroc_d[t])
                mlflow.log_metric(f"best_val_auprc_{t}", compute_auprc(best_val_labels_d[t], best_val_probs_d[t]))
                mlflow.log_metric(f"test_auroc_{t}", test_auroc_d[t])
                mlflow.log_metric(f"test_auprc_{t}", compute_auprc(test_labels_d[t], test_probs_d[t]))
                log_metrics_at_thresholds(best_val_probs_d[t], best_val_labels_d[t],
                                          thresholds, prefix=f"best_val_{t}", step=best_epoch)
                log_metrics_at_thresholds(test_probs_d[t], test_labels_d[t],
                                          thresholds, prefix=f"test_{t}")
                log_confusion_matrix(test_probs_d[t], test_labels_d[t], 0.5, prefix=f"test_{t}")
            print(f"Test AUROC: " + "  ".join(f"{t}={test_auroc_d[t]:.4f}" for t in tasks))
            print(f"\nDone.  Best val AUROC mean: {best_auroc:.4f}  (epoch {best_epoch})  →  {best_path}")
        else:
            _, test_auroc, test_probs, test_labels = evaluate(
                model, test_loader, criterion, device
            )
            mlflow.log_metric(
                "test_auroc", test_auroc,
                model_id=model_info.model_id if best_path.exists() else None,
            )
            log_confusion_matrix(test_probs, test_labels, threshold=0.5, prefix="test")
            print(f"Test  AUROC: {test_auroc:.4f}")
            print(f"\nDone.  Best val AUROC: {best_auroc:.4f}  (epoch {best_epoch})  →  {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="configs/config.yaml")
    parser.add_argument("--data-dir",        default=None, help="Override data_dir from config")
    parser.add_argument("--embeddings-dir",  default=None, help="Override embeddings_dir from config")
    parser.add_argument("--run-name",         default=None, help="Override mlflow run_name from config")
    parser.add_argument("--run-suffix",       default=None, help="Append -{suffix} to the run name")
    parser.add_argument("--max-epochs",       type=int, default=None, help="Cap training epochs (for preflight/debug)")
    parser.add_argument("--patience",         type=int, default=None, help="Override early_stopping_patience from config")
    args = parser.parse_args()
    main(args.config, args.data_dir, args.embeddings_dir, args.run_name, args.run_suffix, args.max_epochs, args.patience)
