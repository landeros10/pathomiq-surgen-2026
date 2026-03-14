"""MLflow helper utilities.

Logging helpers (confusion matrices, threshold metrics) for training scripts,
plus SQLite query helpers for reading MLflow experiment data in report scripts.
"""

import sqlite3
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
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


# ── SQLite query helpers for report scripts ────────────────────────────────────

def get_experiment_id(conn: sqlite3.Connection, name: str) -> "str | None":
    """Return the experiment_id for experiment *name*, or None if not found."""
    row = pd.read_sql(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn, params=(name,)
    )
    if row.empty:
        return None
    return str(row.iloc[0]["experiment_id"])


def load_runs(
    conn: sqlite3.Connection,
    exp_id: str,
    tasks: Optional[List[str]] = None,
    exclude_pattern: str = "preflight",
) -> Dict:
    """Return {run_name: {run_id, params, scalars}} for FINISHED non-preflight runs.

    Args:
        conn:            Open SQLite connection to mlflow.db.
        exp_id:          Experiment ID string.
        tasks:           Task names for per-task scalar keys; defaults to
                         ["mmr", "ras", "braf"].
        exclude_pattern: Run names containing this string are filtered out.

    Returns:
        Dict keyed by run name; each value has:
            run_id  : str
            params  : {key: value}
            scalars : {key: float}  — last logged value per scalar key
    """
    if tasks is None:
        tasks = ["mmr", "ras", "braf"]

    runs_df = pd.read_sql(
        """
        SELECT run_uuid, name
        FROM runs
        WHERE experiment_id = ? AND status = 'FINISHED'
        ORDER BY start_time ASC
        """,
        conn, params=(exp_id,)
    )
    runs_df = runs_df[
        ~runs_df["name"].str.contains(exclude_pattern, case=False, na=False)
    ]

    result: Dict = {}
    for _, row in runs_df.iterrows():
        uid  = row["run_uuid"]
        name = row["name"]
        result[name] = {"run_id": uid, "params": {}, "scalars": {}}

    if not result:
        return result

    run_uuids    = [v["run_id"] for v in result.values()]
    uid_to_name  = {v["run_id"]: k for k, v in result.items()}
    placeholders = ",".join("?" * len(run_uuids))

    # params
    params_df = pd.read_sql(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({placeholders})",
        conn, params=run_uuids
    )
    for _, row in params_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["params"][row["key"]] = row["value"]

    # scalars — last logged value per key
    scalar_keys = (
        ["best_val_auroc_mean", "best_val_auprc_mean", "best_epoch",
         "test_auroc_mean", "test_auprc_mean"] +
        [f"best_val_auroc_{t}" for t in tasks] +
        [f"best_val_auprc_{t}" for t in tasks] +
        [f"test_auroc_{t}"     for t in tasks] +
        [f"test_auprc_{t}"     for t in tasks]
    )
    placeholders_k = ",".join("?" * len(scalar_keys))
    scalars_df = pd.read_sql(
        f"""
        SELECT run_uuid, key, MAX(step) as step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key IN ({placeholders_k})
        GROUP BY run_uuid, key
        """,
        conn, params=run_uuids + scalar_keys
    )
    for _, row in scalars_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["scalars"][row["key"]] = float(row["value"])

    return result


def load_trajectories(
    conn: sqlite3.Connection,
    runs: Dict,
    traj_keys: Optional[List[str]] = None,
) -> Dict:
    """Return {run_name: {metric: [val, ...]}} per epoch.

    Args:
        conn:      Open SQLite connection to mlflow.db.
        runs:      Output of load_runs() — {run_name: {run_id, ...}}.
        traj_keys: Metric keys to fetch; defaults to
                   ["val_auroc_mean", "train_loss", "val_loss"].

    Returns:
        {run_name: {metric_key: [float, ...]}} ordered by step.
    """
    if not runs:
        return {}
    if traj_keys is None:
        traj_keys = ["val_auroc_mean", "train_loss", "val_loss"]

    run_uuids    = [v["run_id"] for v in runs.values()]
    uid_to_name  = {v["run_id"]: k for k, v in runs.items()}
    placeholders = ",".join("?" * len(run_uuids))
    placeholders_k = ",".join("?" * len(traj_keys))

    df = pd.read_sql(
        f"""
        SELECT run_uuid, key, step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key IN ({placeholders_k})
        ORDER BY run_uuid, key, step
        """,
        conn, params=run_uuids + traj_keys
    )

    result: Dict = {name: {} for name in runs}
    for (uid, key), grp in df.groupby(["run_uuid", "key"]):
        name = uid_to_name[uid]
        result[name][key] = grp.sort_values("step")["value"].tolist()
    return result
