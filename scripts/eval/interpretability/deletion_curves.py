"""Deletion/Insertion curve functions — Phase 8.

Ranks patches by attribution method; computes deletion/insertion AUC curves
to assess causal importance of high-attribution patches.

Primary metric: SRG = AUPC_ins − AUPC_del (higher is better).
Canonical attribution method: InputXGrad (gradnorm excluded; degenerate for mean-pool models).
Insertion baseline: N copies of per-slide mean embedding (approximation of global test-set mean).
"""

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def _get_prob(model, bag_np, device, task_idx: int) -> float:
    """Forward pass on a single bag; return sigmoid probability for task_idx."""
    t = torch.tensor(bag_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim <= 1:
            logit = out.flatten()[0]
        else:
            logit = out[0, task_idx]
        return float(torch.sigmoid(logit).item())


def compute_deletion_curve(
    model,
    embeddings,
    scores,
    n_steps: int = 20,
    task_idx: int = 0,
    baseline: str = "mean",
):
    """Compute deletion curve: remove top-k% patches and measure probability drop.

    Args:
        model:      Loaded MultiMILTransformer.
        embeddings: (N, D) float array or tensor.
        scores:     (N,) attribution scores for patch ranking.
        n_steps:    Number of deletion steps.
        task_idx:   Which task head to evaluate.
        baseline:   Unused (kept for API symmetry with insertion curve).

    Returns:
        list of (frac_removed, prob) tuples.
    """
    device = next(model.parameters()).device
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
    N = emb_np.shape[0]
    k_levels = np.linspace(0, 1, n_steps + 1)
    rank_desc = np.argsort(scores)[::-1]

    curve = []
    for k in k_levels:
        n_remove = int(k * N)
        keep = rank_desc[n_remove:]
        bag  = emb_np[keep] if len(keep) > 0 else emb_np[:1]
        curve.append((float(k), _get_prob(model, bag, device, task_idx)))
    return curve


def compute_insertion_curve(
    model,
    embeddings,
    scores,
    n_steps: int = 20,
    task_idx: int = 0,
    baseline: str = "mean",
):
    """Compute insertion curve: add top-k% patches to baseline and measure recovery.

    Baseline bag: N copies of the per-slide mean embedding ("mean") or zeros ("zero").
    This approximates phase 8's precomputed global test-set mean when unavailable.

    Args:
        model:      Loaded MultiMILTransformer.
        embeddings: (N, D) float array or tensor.
        scores:     (N,) attribution scores for patch ranking.
        n_steps:    Number of insertion steps.
        task_idx:   Which task head to evaluate.
        baseline:   Starting representation ("mean" or "zero").

    Returns:
        list of (frac_inserted, prob) tuples.
    """
    device = next(model.parameters()).device
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
    N = emb_np.shape[0]
    k_levels = np.linspace(0, 1, n_steps + 1)
    rank_desc = np.argsort(scores)[::-1]

    if baseline == "mean":
        base_patch = emb_np.mean(axis=0)
    else:
        base_patch = np.zeros(emb_np.shape[1], dtype=np.float32)
    baseline_bag = np.tile(base_patch, (N, 1))

    curve = []
    for k in k_levels:
        n_reveal = max(1, int(k * N)) if k > 0 else 0
        bag = baseline_bag.copy()
        if n_reveal > 0:
            bag[rank_desc[:n_reveal]] = emb_np[rank_desc[:n_reveal]]
        curve.append((float(k), _get_prob(model, bag, device, task_idx)))
    return curve


def score_deletion_auc(curve: list) -> float:
    """Compute AUC of a deletion/insertion curve using the trapezoidal rule.

    Args:
        curve: list of (frac, prob) tuples.

    Returns:
        AUC scalar (lower is better for deletion; higher for insertion).
    """
    import numpy as np
    fracs = [p[0] for p in curve]
    probs = [p[1] for p in curve]
    return float(np.trapezoid(probs, fracs))


def select_canonical_method(
    deletion_aucs: dict,
    insertion_aucs: dict = None,
) -> str:
    """Select canonical attribution method by SRG = ins_AUC − del_AUC (higher is better).

    Falls back to lowest del_AUC when insertion_aucs is not provided.

    Args:
        deletion_aucs:  {method_name: del_auc_scalar}
        insertion_aucs: {method_name: ins_auc_scalar} (optional)

    Returns:
        Name of the best method.
    """
    if insertion_aucs is not None:
        srg = {m: insertion_aucs[m] - deletion_aucs[m] for m in deletion_aucs}
        return max(srg, key=srg.get)
    return min(deletion_aucs, key=deletion_aucs.get)
