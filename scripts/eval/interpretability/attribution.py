"""Gradient attribution functions — Phase 7 Step 2.

Computes per-patch GradNorm (||∇||₂) and InputXGrad (||∇⊙emb||₂) attribution.
Requires GCP model inference. SSH orchestration uses utils/gcp_utils.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))


def compute_gradnorm(
    embeddings,
    model,
    slide_idx: int,
    task_idx: int = 0,
):
    """Compute per-patch gradient norm attribution for one slide.

    Args:
        embeddings: (N, D) float tensor of patch embeddings.
        model:      Loaded MultiMILTransformer in eval mode.
        slide_idx:  Slide index (for logging).
        task_idx:   Which task head to backprop through.

    Returns:
        np.ndarray of shape (N,) — per-patch GradNorm scores.
    """
    raise NotImplementedError(
        "compute_gradnorm() requires model inference on GCP. "
        "Run phase7_attribution.py on the GCP server."
    )


def compute_input_x_grad(
    embeddings,
    model,
    slide_idx: int,
    task_idx: int = 0,
):
    """Compute per-patch Input×Gradient attribution for one slide.

    Args:
        embeddings: (N, D) float tensor of patch embeddings.
        model:      Loaded MultiMILTransformer in eval mode.
        slide_idx:  Slide index (for logging).
        task_idx:   Which task head to backprop through.

    Returns:
        np.ndarray of shape (N,) — per-patch IxG scores.
    """
    raise NotImplementedError(
        "compute_input_x_grad() requires model inference on GCP. "
        "Run phase7_attribution.py on the GCP server."
    )


def compute_spearman_correlation(attn_weights, gradnorm_scores) -> dict:
    """Compute Spearman correlation between attention weights and GradNorm scores.

    Args:
        attn_weights:   (N,) float array of attention weights.
        gradnorm_scores: (N,) float array of GradNorm scores.

    Returns:
        dict with keys: rho, p_value.
    """
    from scipy.stats import spearmanr
    import numpy as np
    rho, pval = spearmanr(attn_weights, gradnorm_scores)
    return {"rho": float(rho), "p_value": float(pval)}
