"""Compute post-hoc training stability metrics from val_auroc epoch trajectories.

Connects to the local mlflow.db and outputs a CSV + printed table.

Usage:
    python scripts/stability_metrics.py
    python scripts/stability_metrics.py --experiment mmr-surgen-stability
    python scripts/stability_metrics.py --experiment mmr-surgen-stability --last-n 8
    python scripts/stability_metrics.py --db /path/to/mlflow.db --out results/stability.csv
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["std", "cv_%", "post_peak_drop", "max_swing", "reversals", "early_peak_ratio"]


def stability_stats(auroc_by_epoch: list[float]) -> dict:
    a = np.array(auroc_by_epoch)
    n = len(a)
    best_idx = int(np.argmax(a))
    best_val = float(a[best_idx])

    std = float(np.std(a))
    cv  = float(std / np.mean(a) * 100)

    post = a[best_idx + 1:] if best_idx + 1 < n else np.array([])
    post_peak_drop = float(np.mean(best_val - post)) if len(post) else 0.0

    diffs      = np.abs(np.diff(a))
    max_swing  = float(diffs.max()) if len(diffs) else 0.0

    signs      = np.sign(np.diff(a))
    reversals  = int(np.sum(signs[1:] != signs[:-1]))

    early_peak = (best_idx + 1) / n

    return {
        "n_epochs":         n,
        "best_epoch":       best_idx + 1,
        "best_auroc":       round(best_val, 4),
        "std":              round(std, 4),
        "cv_%":             round(cv, 2),
        "post_peak_drop":   round(post_peak_drop, 4),
        "max_swing":        round(max_swing, 4),
        "reversals":        reversals,
        "early_peak_ratio": round(early_peak, 2),
    }


def main(db_path: str, experiment: str, last_n: int, out_path: str) -> None:
    conn = sqlite3.connect(db_path)

    exp_row = pd.read_sql(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn, params=(experiment,)
    )
    if exp_row.empty:
        raise SystemExit(f"Experiment '{experiment}' not found in {db_path}")
    exp_id = str(exp_row.iloc[0]["experiment_id"])

    runs = pd.read_sql(
        """
        SELECT run_uuid, name, start_time
        FROM runs
        WHERE experiment_id = ? AND status = 'FINISHED'
        ORDER BY start_time DESC
        """,
        conn, params=(exp_id,)
    )
    if last_n:
        runs = runs.head(last_n)

    if runs.empty:
        raise SystemExit(f"No finished runs found for experiment '{experiment}'")

    placeholders = ",".join(f"'{r}'" for r in runs["run_uuid"])
    auroc_rows = pd.read_sql(
        f"""
        SELECT run_uuid, step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key = 'val_auroc'
        ORDER BY run_uuid, step
        """,
        conn
    )

    records = []
    for _, row in runs.iterrows():
        rid  = row["run_uuid"]
        name = row["name"]
        vals = auroc_rows[auroc_rows["run_uuid"] == rid].sort_values("step")["value"].tolist()
        if not vals:
            continue
        stats = stability_stats(vals)
        records.append({"run_name": name, **stats})

    df = pd.DataFrame(records).sort_values("best_auroc", ascending=False)

    # ── Print ──────────────────────────────────────────────────────────────────
    print(f"\nExperiment : {experiment}")
    print(f"Runs       : {len(df)}  (last {last_n or 'all'} finished)\n")
    print(df.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc training stability metrics")
    parser.add_argument("--db",         default="mlflow.db",              help="Path to mlflow SQLite DB")
    parser.add_argument("--experiment", default="mmr-surgen-stability",   help="MLflow experiment name")
    parser.add_argument("--last-n",     type=int, default=0,              help="Only analyse the N most recent runs (0 = all)")
    parser.add_argument("--out",        default="results/stability.csv",  help="Output CSV path")
    args = parser.parse_args()
    main(args.db, args.experiment, args.last_n, args.out)
