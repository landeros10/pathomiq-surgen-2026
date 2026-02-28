"""MLflow helper utilities for logging confusion matrices and threshold metrics."""

from typing import List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_confusion_matrix(
    probs: List[float],
    labels: List[int],
    threshold: float,
    prefix: str = "val",
) -> None:
    """Compute and log a confusion matrix figure to the active MLflow run.

    Args:
        probs:     Predicted probabilities for the positive class.
        labels:    Ground-truth binary labels (0 = MSS/pMMR, 1 = MSI/dMMR).
        threshold: Decision boundary used to binarise probabilities.
        prefix:    Artefact name prefix (e.g. "val", "test").
    """
    preds = (np.array(probs) >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["MSS/pMMR", "MSI/dMMR"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{prefix.upper()} — threshold {threshold}")
    plt.tight_layout()

    tag = str(threshold).replace(".", "")
    mlflow.log_figure(fig, f"{prefix}_cm_t{tag}.png")
    plt.close(fig)


def log_metrics_at_thresholds(
    probs: List[float],
    labels: List[int],
    thresholds: List[float],
    prefix: str = "val",
    step: Optional[int] = None,
) -> None:
    """Log scalar metrics and confusion matrix figures for every threshold.

    Calls log_confusion_matrix for each threshold so figures appear as
    MLflow artefacts alongside the scalar metrics.

    Args:
        probs:      Predicted probabilities for the positive class.
        labels:     Ground-truth binary labels.
        thresholds: List of decision thresholds to evaluate.
        prefix:     Metric/artefact name prefix (e.g. "val", "test").
        step:       MLflow step (epoch) to associate with scalar metrics.
    """
    from scripts.utils.metrics import metrics_at_threshold

    for t in thresholds:
        m   = metrics_at_threshold(labels, probs, t)
        tag = str(t).replace(".", "")
        mlflow.log_metrics(
            {
                f"{prefix}_precision_t{tag}":   m["precision"],
                f"{prefix}_recall_t{tag}":      m["recall"],
                f"{prefix}_f1_t{tag}":          m["f1"],
                f"{prefix}_specificity_t{tag}": m["specificity"],
            },
            step=step,
        )
        log_confusion_matrix(probs, labels, t, prefix=prefix)
