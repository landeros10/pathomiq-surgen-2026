"""Evaluation metrics for binary MMR/MSI classification.

Primary metric: AUROC (class-imbalance robust).
Secondary: AUPRC, and per-threshold precision / recall / specificity / F1.
"""

from typing import Dict, List

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
