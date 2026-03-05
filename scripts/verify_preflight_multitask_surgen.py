#!/usr/bin/env python3
"""Verify Phase 5 multitask preflight runs completed successfully.

Checks MLflow for both preflight runs and validates:
  - Run exists with FINISHED status
  - Exactly 1 epoch of metrics logged
  - All expected per-task metrics present and finite
  - Losses are positive and finite
  - Per-task AUROCs are in [0, 1]
  - Log file tail shows no Python traceback

Usage:
    python3 scripts/verify_preflight_multitask_surgen.py
    python3 scripts/verify_preflight_multitask_surgen.py --db mlflow.db
"""

import argparse
import math
import os
import re
import sys

import mlflow
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────

EXPERIMENT  = "multitask-surgen"
RUN_NAMES   = ["multitask-base-preflight", "multitask-cosine-accum16-preflight"]
TASKS       = ["mmr", "ras", "braf"]
LOG_FILES   = {
    "multitask-base-preflight":
        "logs/phase5/config_multitask_base_preflight.log",
    "multitask-cosine-accum16-preflight":
        "logs/phase5/config_multitask_cosine_accum16_preflight.log",
}

REQUIRED_METRICS = (
    ["train_loss", "val_loss", "val_auroc_mean", "val_auprc_mean"]
    + [f"train_loss_{t}"   for t in TASKS]
    + [f"val_loss_{t}"     for t in TASKS]
    + [f"train_auroc_{t}"  for t in TASKS]
    + [f"val_auroc_{t}"    for t in TASKS]
    + [f"val_f1_{t}"       for t in TASKS]
    + [f"val_sensitivity_{t}" for t in TASKS]
    + [f"val_specificity_{t}" for t in TASKS]
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def ok(msg):  print(f"  \033[32m✓\033[0m {msg}")
def fail(msg):print(f"  \033[31m✗\033[0m {msg}")
def warn(msg):print(f"  \033[33m!\033[0m {msg}")

def check(cond, pass_msg, fail_msg):
    if cond:
        ok(pass_msg)
    else:
        fail(fail_msg)
    return cond


def verify_run(run_name: str, client: mlflow.MlflowClient, exp_id: str) -> bool:
    print(f"\n{'─'*60}")
    print(f"Run: {run_name}")
    print(f"{'─'*60}")
    passed = True

    # ── 1. Run exists ────────────────────────────────────────────────────────
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        fail(f"No MLflow run found with name '{run_name}'")
        return False
    run = runs[0]
    ok(f"Run found  (id={run.info.run_id[:8]}…)")

    # ── 2. Status = FINISHED ─────────────────────────────────────────────────
    status = run.info.status
    passed &= check(status == "FINISHED",
                    f"Status = FINISHED",
                    f"Status = {status}  (expected FINISHED)")

    # ── 3. Fetch metric history ──────────────────────────────────────────────
    run_id = run.info.run_id
    metric_steps: dict[str, list] = {}
    for key in REQUIRED_METRICS:
        history = client.get_metric_history(run_id, key)
        metric_steps[key] = history

    # ── 4. Exactly 1 epoch logged ────────────────────────────────────────────
    epoch_counts = {k: len(v) for k, v in metric_steps.items() if v}
    unique_counts = set(epoch_counts.values())
    if unique_counts:
        n_epochs = max(epoch_counts.values())
        passed &= check(n_epochs == 1,
                        f"1 epoch of metrics logged",
                        f"Expected 1 epoch, got {n_epochs}")
    else:
        fail("No epoch metrics found at all")
        return False

    # ── 5. All required metrics present ─────────────────────────────────────
    missing = [k for k in REQUIRED_METRICS if not metric_steps.get(k)]
    if missing:
        fail(f"Missing metrics: {missing}")
        passed = False
    else:
        ok(f"All {len(REQUIRED_METRICS)} expected metrics present")

    # ── 6. Losses positive + finite ─────────────────────────────────────────
    loss_keys = (["train_loss", "val_loss"]
                 + [f"train_loss_{t}" for t in TASKS]
                 + [f"val_loss_{t}"   for t in TASKS])
    bad_losses = []
    for k in loss_keys:
        hist = metric_steps.get(k, [])
        if hist:
            v = hist[-1].value
            if not (math.isfinite(v) and v > 0):
                bad_losses.append(f"{k}={v:.4f}")
    passed &= check(not bad_losses,
                    "All losses positive and finite",
                    f"Bad losses: {bad_losses}")

    # ── 7. Per-task AUROCs in [0, 1] ────────────────────────────────────────
    auroc_keys = (["val_auroc_mean"]
                  + [f"train_auroc_{t}" for t in TASKS]
                  + [f"val_auroc_{t}"   for t in TASKS])
    bad_auroc = []
    for k in auroc_keys:
        hist = metric_steps.get(k, [])
        if hist:
            v = hist[-1].value
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                bad_auroc.append(f"{k}={v:.4f}")
    passed &= check(not bad_auroc,
                    "All AUROCs in [0, 1]",
                    f"Out-of-range AUROCs: {bad_auroc}")

    # ── 8. Print metric snapshot ─────────────────────────────────────────────
    print()
    print("  Metric snapshot (epoch 1):")
    snapshot_keys = (
        ["train_loss", "val_loss", "val_auroc_mean", "val_auprc_mean"]
        + [f"val_auroc_{t}" for t in TASKS]
        + [f"val_f1_{t}"    for t in TASKS]
    )
    for k in snapshot_keys:
        hist = metric_steps.get(k, [])
        v = hist[-1].value if hist else float("nan")
        print(f"    {k:<35} {v:.4f}")

    # ── 9. Log file tail — no traceback ─────────────────────────────────────
    log_path = LOG_FILES.get(run_name)
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            tail = f.readlines()[-50:]
        tail_text = "".join(tail)
        has_tb = bool(re.search(r"Traceback \(most recent call last\)", tail_text))
        has_error = bool(re.search(r"^(Error|Exception):", tail_text, re.MULTILINE))
        passed &= check(not has_tb and not has_error,
                        "Log tail: no traceback or Error",
                        "Log tail: Python traceback or Error detected — check the log")
    elif log_path:
        warn(f"Log file not found: {log_path}")
    else:
        warn("No log path configured for this run")

    return passed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mlflow.db",
                        help="Path to MLflow SQLite DB (default: mlflow.db)")
    args = parser.parse_args()

    tracking_uri = f"sqlite:///{args.db}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"\033[31mERROR\033[0m: Experiment '{EXPERIMENT}' not found in {args.db}")
        sys.exit(1)

    print(f"\nPhase 5 Preflight Verification")
    print(f"Experiment : {EXPERIMENT}")
    print(f"DB         : {args.db}")
    print(f"Runs       : {RUN_NAMES}")

    results = {}
    for run_name in RUN_NAMES:
        results[run_name] = verify_run(run_name, client, exp.experiment_id)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("SUMMARY")
    print(f"{'═'*60}")
    all_passed = True
    for run_name, passed in results.items():
        status = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        print(f"  {status}  {run_name}")
        all_passed &= passed

    print()
    if all_passed:
        print("\033[32mAll preflight checks passed. Safe to launch full runs.\033[0m")
        sys.exit(0)
    else:
        print("\033[31mOne or more checks failed. Investigate before launching full runs.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
