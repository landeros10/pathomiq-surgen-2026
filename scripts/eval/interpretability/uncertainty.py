"""MC Dropout uncertainty functions — Phase 8.

T=50 stochastic forward passes with dropout p=0.15 (transformer in .train() mode).
Handles both single-task (1D) and multi-task (2D) logit output from MultiMILTransformer.
"""

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def run_mc_dropout(
    model,
    embeddings,
    n_passes: int = 50,
    dropout_rate: float = 0.15,
    task_idx: int = 0,
):
    """Run MC Dropout uncertainty estimation for one slide.

    Puts the model in .train() mode to activate dropout stochasticity, then runs
    n_passes stochastic forward passes under torch.no_grad(). Handles tuple output
    (logits, weights) from multi-task models, and both 1D/2D logit tensors.

    Args:
        model:        Loaded MultiMILTransformer.
        embeddings:   (N, D) float array or tensor of patch embeddings.
        n_passes:     Number of stochastic forward passes.
        dropout_rate: Dropout probability (informational; model must already be configured).
        task_idx:     Which task head to use for output.

    Returns:
        dict with keys:
            mean_prob  : float — mean predicted probability across passes
            std_prob   : float — standard deviation across passes
            all_probs  : list[float] — per-pass probabilities
    """
    device = next(model.parameters()).device
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, D)

    model.train()  # activate dropout stochasticity — critical: NOT .eval()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(emb)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.ndim == 2:
                task_logits = logits[0]      # (B, T) → (T,)
            elif logits.ndim == 1:
                task_logits = logits         # (T,) or scalar-wrapped
            else:
                task_logits = logits.unsqueeze(0)
            all_probs.append(float(torch.sigmoid(task_logits[task_idx]).item()))

    return {
        "mean_prob": float(np.mean(all_probs)),
        "std_prob":  float(np.std(all_probs)),
        "all_probs": all_probs,
    }


def compute_ece(
    probs,
    labels,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        probs:  Predicted probabilities (N,).
        labels: Ground-truth binary labels (N,).
        n_bins: Number of calibration bins.

    Returns:
        ECE scalar.
    """
    import numpy as np
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=float)
    ece    = 0.0
    n      = len(probs)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)
