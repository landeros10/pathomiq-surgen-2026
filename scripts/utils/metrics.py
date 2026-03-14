"""Evaluation metrics for binary MMR/MSI classification.

Primary metric: AUROC (class-imbalance robust).
Secondary: AUPRC, and per-threshold precision / recall / specificity / F1.
Bootstrap CI helpers for test-sample uncertainty quantification.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_auroc(labels: List[int], probs: List[float]) -> float:
    """AUROC. Returns 0.0 if only one class is present in labels."""
    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probs))


def compute_auprc(labels: List[int], probs: List[float]) -> float:
    """Area under precision-recall curve (average precision)."""
    if len(set(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probs))


def metrics_at_threshold(
    labels: List[int],
    probs: List[float],
    threshold: float,
) -> Dict[str, float]:
    """Classification metrics for a given probability cut-off.

    Args:
        labels:    Ground-truth binary labels (0 = MSS/pMMR, 1 = MSI/dMMR).
        probs:     Predicted probabilities for the positive class.
        threshold: Decision boundary in [0, 1].

    Returns:
        Dict with keys: threshold, precision, recall, specificity, f1, tp, tn, fp, fn.
    """
    preds  = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "threshold":   threshold,
        "precision":   precision,
        "recall":      recall,
        "specificity": spec,
        "f1":          f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def full_report(
    labels: List[int],
    probs: List[float],
    thresholds: List[float],
) -> Dict:
    """Compute AUROC, AUPRC, and per-threshold metrics in one call."""
    return {
        "auroc": compute_auroc(labels, probs),
        "auprc": compute_auprc(labels, probs),
        "by_threshold": {
            t: metrics_at_threshold(labels, probs, t) for t in thresholds
        },
    }


# ── Bootstrap CI helpers ───────────────────────────────────────────────────────

def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Test-sample bootstrap 95% CI for *metric_fn(labels, probs)*.

    Args:
        labels:      Ground-truth binary labels array.
        probs:       Predicted probabilities array.
        metric_fn:   Callable (labels, probs) → float.
        n_bootstrap: Number of bootstrap resamples.
        rng:         Optional numpy Generator; seeded at 42 if None.

    Returns:
        (lo, hi) — 2.5th and 97.5th percentile of bootstrapped scores.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n      = len(labels)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(labels[idx], probs[idx])
            if not math.isnan(s):
                scores.append(s)
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def pooled_bootstrap_ci(
    inference: dict,
    run_names: List[str],
    task: str,
    metric_fn: Callable,
    n: int = 2000,
) -> Tuple[float, float]:
    """Bootstrap CI on probs/labels concatenated across seed runs for one task.

    Args:
        inference:  {run_name: {"labels": {task: arr}, "probs": {task: arr}}}
        run_names:  Which runs to pool.
        task:       Task key (e.g. "mmr").
        metric_fn:  Callable (labels, probs) → float.
        n:          Bootstrap resamples.

    Returns:
        (lo, hi) — pooled 95% CI.
    """
    all_labels: List[np.ndarray] = []
    all_probs:  List[np.ndarray] = []
    for rn in run_names:
        if rn not in inference:
            continue
        lbl  = inference[rn]["labels"].get(task, np.array([]))
        prob = inference[rn]["probs"].get(task, np.array([]))
        if len(lbl) > 0:
            all_labels.append(lbl)
            all_probs.append(prob)
    if not all_labels:
        return float("nan"), float("nan")
    labels_pool = np.concatenate(all_labels)
    probs_pool  = np.concatenate(all_probs)
    rng = np.random.default_rng(42)
    return bootstrap_ci(labels_pool, probs_pool, metric_fn, n_bootstrap=n, rng=rng)
