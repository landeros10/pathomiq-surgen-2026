"""Deletion/Insertion curve functions — Phase 7 Step 4.

Ranks patches by attribution method; computes deletion/insertion AUC curves
to assess causal importance of high-attention patches.
SSH orchestration uses utils/gcp_utils.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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
        embeddings: (N, D) float tensor.
        scores:     (N,) attribution scores for patch ranking.
        n_steps:    Number of deletion steps.
        task_idx:   Which task head to evaluate.
        baseline:   Replacement strategy ("mean" or "zero").

    Returns:
        list of (frac_removed, prob) tuples.
    """
    raise NotImplementedError(
        "compute_deletion_curve() requires model inference on GCP. "
        "Run phase7_deletion.py on the GCP server."
    )


def compute_insertion_curve(
    model,
    embeddings,
    scores,
    n_steps: int = 20,
    task_idx: int = 0,
    baseline: str = "mean",
):
    """Compute insertion curve: add top-k% patches to baseline and measure recovery.

    Args:
        model:      Loaded MultiMILTransformer.
        embeddings: (N, D) float tensor.
        scores:     (N,) attribution scores for patch ranking.
        n_steps:    Number of insertion steps.
        task_idx:   Which task head to evaluate.
        baseline:   Starting representation ("mean" or "zero").

    Returns:
        list of (frac_inserted, prob) tuples.
    """
    raise NotImplementedError(
        "compute_insertion_curve() requires model inference on GCP. "
        "Run phase7_deletion.py on the GCP server."
    )


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
) -> str:
    """Select canonical attribution method with lowest deletion AUC.

    Args:
        deletion_aucs: {method_name: auc_scalar}

    Returns:
        Name of the best (lowest AUC) method.
    """
    return min(deletion_aucs, key=deletion_aucs.get)
