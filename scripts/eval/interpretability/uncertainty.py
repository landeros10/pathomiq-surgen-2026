"""MC Dropout uncertainty functions — Phase 7 Step 3.

T=50 stochastic forward passes with dropout p=0.15 (transformer in .train() mode).
Requires GCP model inference. SSH orchestration uses utils/gcp_utils.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_mc_dropout(
    model,
    embeddings,
    n_passes: int = 50,
    dropout_rate: float = 0.15,
    task_idx: int = 0,
):
    """Run MC Dropout uncertainty estimation for one slide.

    Args:
        model:        Loaded MultiMILTransformer.
        embeddings:   (N, D) float tensor of patch embeddings.
        n_passes:     Number of stochastic forward passes.
        dropout_rate: Dropout probability (model put in .train() mode).
        task_idx:     Which task head to use for output.

    Returns:
        dict with keys:
            mean_prob  : float — mean predicted probability across passes
            std_prob   : float — standard deviation across passes
            all_probs  : list[float] — per-pass probabilities
    """
    raise NotImplementedError(
        "run_mc_dropout() requires model inference on GCP. "
        "Run phase7_mc_dropout.py on the GCP server."
    )


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
