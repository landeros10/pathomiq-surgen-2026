"""MLP-RPB diagnostic functions — model loading, inference, and signal reports.

Extracted from scripts/studies/phase8_rpb_diagnostic.py so these functions
can be imported without pulling in the CLI orchestration layer.

Signal 1 — Weight Magnitude:
  Compare L2 norm of rpb.mlp.weight to kaiming-init scale.

Signal 2 — Bias Activation Scale:
  Hook rpb forward output during test inference; report mean/std.

Signal 3 — Counterfactual AUROC:
  Full test inference with RPB enabled, then zeroed weights.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mlflow  # noqa: E402

from utils.eval_utils import mlflow_run_id  # noqa: E402
from etl.dataset import MultitaskMILDataset, mil_collate_fn  # noqa: E402

TASKS = ["mmr", "ras", "braf"]


def load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(run_name: str, mlflow_db: Path, device: torch.device):
    """Load best_model checkpoint for *run_name* from MLflow."""
    run_id = mlflow_run_id(run_name, mlflow_db=mlflow_db)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model",
                                      map_location=device)
    model.to(device)
    model.eval()
    return model


def build_loader(cfg: dict, root: Path) -> DataLoader:
    paths = cfg["paths"]
    data  = cfg["data"]
    split_csv = root / paths["data_dir"] / data["test_split"]
    ds = MultitaskMILDataset(
        split_csv=str(split_csv),
        embeddings_dir=paths["embeddings_dir"],
        tasks=data["tasks"],
        slide_id_col=data.get("slide_id_column", "slide_id"),
    )
    return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)


def infer(model, loader, device, max_slides=None):
    """Run inference; return (all_logits, all_labels, all_masks) as numpy arrays."""
    logits_list, labels_list, masks_list = [], [], []
    for i, (embeddings, labels, valid_mask, slide_id, coords) in enumerate(loader):
        if max_slides is not None and i >= max_slides:
            break
        embeddings = embeddings.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)
        if isinstance(coords, list):
            coords = None
        elif coords is not None:
            coords = coords.to(device)

        with torch.no_grad():
            out = model(embeddings, coords=coords)
            if isinstance(out, tuple):
                out = out[0]
        logits_list.append(out.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        masks_list.append(valid_mask.cpu().numpy())

    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(masks_list,  axis=0),
    )


def auroc_per_task(logits, labels, masks):
    """Compute per-task AUROC and macro mean, skipping tasks with <2 classes."""
    from sklearn.metrics import roc_auc_score
    aurocs = {}
    for i, t in enumerate(TASKS):
        m = masks[:, i].astype(bool)
        if m.sum() < 2:
            aurocs[t] = float("nan")
            continue
        y_true = labels[m, i]
        y_prob  = torch.sigmoid(torch.tensor(logits[m, i])).numpy()
        if len(np.unique(y_true)) < 2:
            aurocs[t] = float("nan")
        else:
            aurocs[t] = roc_auc_score(y_true, y_prob)
    valid = [v for v in aurocs.values() if not math.isnan(v)]
    aurocs["mean"] = float(np.mean(valid)) if valid else float("nan")
    return aurocs


def weight_report(model) -> None:
    """Signal 1: compare RPB weight L2 norm to kaiming-init expected scale."""
    print("\n" + "=" * 60)
    print("Signal 1 — RPB Weight Magnitude")
    print("=" * 60)

    if not hasattr(model, "rpb") or model.rpb is None:
        print("  [SKIP] model.rpb is None — not an mlp_rpb model.")
        return

    w = model.rpb.mlp.weight.detach().cpu()
    b = model.rpb.mlp.bias.detach().cpu()
    fan_in      = w.shape[1]
    init_scale  = 1.0 / math.sqrt(fan_in)

    w_norm  = w.norm().item()
    b_norm  = b.norm().item()
    expected_w_norm = init_scale * math.sqrt(w.numel())
    ratio = w_norm / expected_w_norm

    print(f"  weight shape : {list(w.shape)}")
    print(f"  weight values: {w.tolist()}")
    print(f"  bias values  : {b.tolist()}")
    print(f"  ||w||₂       : {w_norm:.6f}")
    print(f"  ||b||₂       : {b_norm:.6f}")
    print(f"  kaiming init expected ||w||₂ ≈ {expected_w_norm:.4f}  "
          f"(1/√{fan_in} × √{w.numel()} params)")
    print(f"  ratio = ||w||₂ / expected : {ratio:.4f}")

    if ratio < 0.1:
        verdict = "COLLAPSED — weights near zero; RPB is a near-no-op."
    elif ratio < 0.5:
        verdict = "SUPPRESSED — weights shrank significantly below init; weak contribution."
    elif ratio < 1.5:
        verdict = "INCONCLUSIVE — weights near init magnitude; may not have learned."
    else:
        verdict = "ACTIVE — weights grew beyond init; RPB is being used."

    print(f"\n  Verdict: {verdict}")


def bias_stats_report(model, loader, device, n_slides: int = 30) -> None:
    """Signal 2: hook RPB forward output and report activation mean/std."""
    print("\n" + "=" * 60)
    print(f"Signal 2 — RPB Bias Activation Scale  (first {n_slides} slides)")
    print("=" * 60)

    if not hasattr(model, "rpb") or model.rpb is None:
        print("  [SKIP] model.rpb is None — not an mlp_rpb model.")
        return

    captured = []

    def _hook(module, input, output):
        captured.append(output.detach().cpu().float())

    handle = model.rpb.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for i, (embeddings, labels, valid_mask, slide_id, coords) in enumerate(loader):
            if i >= n_slides:
                break
            embeddings = embeddings.to(device)
            if isinstance(coords, list):
                coords = None
            elif coords is not None:
                coords = coords.to(device)
            _ = model(embeddings, coords=coords)

    handle.remove()

    if not captured:
        print("  [ERROR] No bias tensors captured.")
        return

    all_vals = torch.cat([t.flatten() for t in captured])
    n_slides_actual = len(captured)
    mean_ = all_vals.mean().item()
    std_  = all_vals.std().item()
    min_  = all_vals.min().item()
    max_  = all_vals.max().item()

    print(f"  Slides captured : {n_slides_actual}")
    print(f"  Total bias values: {all_vals.numel():,}")
    print(f"  mean  : {mean_:+.6f}")
    print(f"  std   : {std_:.6f}")
    print(f"  min   : {min_:+.6f}")
    print(f"  max   : {max_:+.6f}")

    if std_ < 0.02:
        verdict = "NEAR-ZERO — std < 0.02; bias is effectively constant (zero-op)."
    elif std_ < 0.10:
        verdict = f"SMALL — std={std_:.4f} < 0.10; marginal effect on softmax."
    else:
        verdict = f"MEANINGFUL — std={std_:.4f} ≥ 0.10; bias meaningfully shifts attention."

    print(f"\n  Verdict: {verdict}")


def counterfactual_report(model, loader, device) -> None:
    """Signal 3: compare full test AUROC with RPB enabled vs zeroed in-place."""
    print("\n" + "=" * 60)
    print("Signal 3 — Counterfactual AUROC  (RPB enabled vs zeroed)")
    print("=" * 60)

    if not hasattr(model, "rpb") or model.rpb is None:
        print("  [SKIP] model.rpb is None — not an mlp_rpb model.")
        return

    print("  Running test inference with RPB enabled...")
    logits_on, labels, masks = infer(model, loader, device)
    aurocs_on = auroc_per_task(logits_on, labels, masks)

    with torch.no_grad():
        model.rpb.mlp.weight.zero_()
        model.rpb.mlp.bias.zero_()

    print("  Running test inference with RPB zeroed...")
    logits_off, _, _ = infer(model, loader, device)
    aurocs_off = auroc_per_task(logits_off, labels, masks)

    print(f"\n  {'Task':<8}  {'AUROC (on)':>10}  {'AUROC (off)':>11}  {'Δ':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}")
    for t in TASKS + ["mean"]:
        on  = aurocs_on[t]
        off = aurocs_off[t]
        delta = on - off if not (math.isnan(on) or math.isnan(off)) else float("nan")
        marker = "  ←" if t == "mean" else ""
        print(f"  {t:<8}  {on:>10.4f}  {off:>11.4f}  {delta:>+8.4f}{marker}")

    delta_mean = aurocs_on["mean"] - aurocs_off["mean"]

    if abs(delta_mean) < 0.001:
        verdict = (f"|Δ mean AUROC| = {abs(delta_mean):.4f} < 0.001 — "
                   f"RPB FUNCTIONALLY IGNORED; removing it has no measurable effect.")
    elif abs(delta_mean) < 0.005:
        verdict = (f"|Δ mean AUROC| = {abs(delta_mean):.4f} in [0.001, 0.005) — "
                   f"MARGINAL contribution (below winner threshold).")
    else:
        verdict = (f"|Δ mean AUROC| = {abs(delta_mean):.4f} ≥ 0.005 — "
                   f"MEANINGFUL contribution; RPB is actively used.")

    print(f"\n  Verdict: {verdict}")
