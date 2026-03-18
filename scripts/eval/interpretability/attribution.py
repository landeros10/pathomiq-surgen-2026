"""Gradient attribution functions — Phase 7 Step 2.

Computes per-patch GradNorm (||∇||₂) and InputXGrad (||∇⊙emb||₂) attribution.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))


def _extract_logit(out, task_idx: int):
    """Extract scalar logit from model output (handles ST and MT output shapes)."""
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 0:
        return out
    if out.ndim == 1:
        return out[0]
    return out[0, task_idx]


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
    device = next(model.parameters()).device
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
    emb_leaf = torch.tensor(emb_np, dtype=torch.float32, device=device, requires_grad=True)
    out = model(emb_leaf.unsqueeze(0))
    logit = _extract_logit(out, task_idx)
    logit.backward()
    grad = emb_leaf.grad.detach().cpu().numpy()  # (N, D)
    return np.linalg.norm(grad, axis=1).astype(np.float32)


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
    device = next(model.parameters()).device
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)
    emb_leaf = torch.tensor(emb_np, dtype=torch.float32, device=device, requires_grad=True)
    out = model(emb_leaf.unsqueeze(0))
    logit = _extract_logit(out, task_idx)
    logit.backward()
    grad    = emb_leaf.grad.detach().cpu().numpy()  # (N, D)
    emb_val = emb_leaf.detach().cpu().numpy()        # (N, D)
    return np.linalg.norm(grad * emb_val, axis=1).astype(np.float32)


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
