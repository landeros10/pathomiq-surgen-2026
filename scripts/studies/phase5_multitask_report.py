#!/usr/bin/env python3
"""Generate Phase 5 multitask results report.

Queries the multitask-surgen MLflow experiment from a local SQLite DB.
All FINISHED runs whose name does NOT contain "preflight" are auto-discovered.

Outputs:
  reports/YYYY-MM-DD-phase5-results.md
  reports/figures/phase5_*.png

Usage:
    python scripts/studies/phase5_multitask_report.py
"""

import math
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Import compute_stability_metrics from stability_ablation ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from stability_ablation import compute_stability_metrics  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
TASKS        = ["mmr", "ras", "braf"]
PREVALENCES  = {"mmr": 0.10, "ras": 0.44, "braf": 0.14}   # train split positive rates
PHASE4_MMR_VAL  = 0.9002
PHASE4_MMR_TEST = 0.8640
EXPERIMENT_NAME = "multitask-surgen"
MLFLOW_DB_PATH  = Path("mlflow.db")
REPORT_DATE     = datetime.now().strftime("%Y-%m-%d")
OUT_DIR         = Path("reports")
FIGURES_DIR     = OUT_DIR / "figures"

COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def fmt(value, decimals: int = 4) -> str:
    """None/NaN-safe float formatter."""
    if value is None:
        return "—"
    try:
        if math.isnan(float(value)):
            return "—"
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value) -> str:
    if value is None:
        return "—"
    try:
        if math.isnan(float(value)):
            return "—"
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


# ── SQLite data loading ───────────────────────────────────────────────────────

def get_experiment_id(conn: sqlite3.Connection, name: str) -> str | None:
    row = pd.read_sql(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn, params=(name,)
    )
    if row.empty:
        return None
    return str(row.iloc[0]["experiment_id"])


def load_runs(conn: sqlite3.Connection, exp_id: str) -> dict:
    """Return {run_name: {run_id, params, scalars}} for non-preflight FINISHED runs."""
    runs_df = pd.read_sql(
        """
        SELECT run_uuid, name
        FROM runs
        WHERE experiment_id = ? AND status = 'FINISHED'
        ORDER BY start_time ASC
        """,
        conn, params=(exp_id,)
    )
    # Exclude preflight runs
    runs_df = runs_df[~runs_df["name"].str.contains("preflight", case=False, na=False)]

    result: dict = {}
    for _, row in runs_df.iterrows():
        uid  = row["run_uuid"]
        name = row["name"]
        result[name] = {"run_id": uid, "params": {}, "scalars": {}}

    if not result:
        return result

    run_uuids  = [v["run_id"] for v in result.values()]
    uid_to_name = {v["run_id"]: k for k, v in result.items()}

    # Load params
    placeholders = ",".join("?" * len(run_uuids))
    params_df = pd.read_sql(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({placeholders})",
        conn, params=run_uuids
    )
    for _, row in params_df.iterrows():
        name = uid_to_name[row["run_uuid"]]
        result[name]["params"][row["key"]] = row["value"]

    # Load final scalars
    scalar_keys = (
        ["best_val_auroc_mean", "best_val_auprc_mean", "best_epoch", "test_auprc_mean"] +
        [f"best_val_auroc_{t}" for t in TASKS] +
        [f"best_val_auprc_{t}" for t in TASKS] +
        [f"test_auroc_{t}" for t in TASKS] +
        [f"test_auprc_{t}" for t in TASKS]
    )
    placeholders_keys = ",".join("?" * len(scalar_keys))
    scalars_df = pd.read_sql(
        f"""
        SELECT run_uuid, key, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key IN ({placeholders_keys})
        """,
        conn, params=run_uuids + scalar_keys
    )
    for _, row in scalars_df.iterrows():
        name = uid_to_name[row["run_uuid"]]
        result[name]["scalars"][row["key"]] = float(row["value"])

    return result


def load_trajectories(conn: sqlite3.Connection, runs: dict) -> dict:
    """Return {run_name: {metric_name: [val_epoch0, val_epoch1, ...]}}."""
    if not runs:
        return {}

    run_uuids  = [v["run_id"] for v in runs.values()]
    uid_to_name = {v["run_id"]: k for k, v in runs.items()}

    traj_keys = (
        ["train_loss", "val_loss", "val_auroc_mean"] +
        [f"val_auroc_{t}"       for t in TASKS] +
        [f"train_auroc_{t}"     for t in TASKS] +
        [f"val_loss_{t}"        for t in TASKS] +
        [f"train_loss_{t}"      for t in TASKS] +
        [f"val_f1_{t}"          for t in TASKS] +
        [f"val_sensitivity_{t}" for t in TASKS] +
        [f"val_specificity_{t}" for t in TASKS] +
        [f"head_weight_var_{t}" for t in TASKS]
    )

    placeholders_uuids = ",".join("?" * len(run_uuids))
    placeholders_keys  = ",".join("?" * len(traj_keys))

    rows = pd.read_sql(
        f"""
        SELECT run_uuid, key, step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders_uuids}) AND key IN ({placeholders_keys})
        ORDER BY run_uuid, key, step
        """,
        conn, params=run_uuids + traj_keys
    )

    result: dict = {name: {} for name in runs}
    for name, run_info in runs.items():
        uid = run_info["run_id"]
        sub = rows[rows["run_uuid"] == uid]
        for key in traj_keys:
            series = sub[sub["key"] == key].sort_values("step")["value"].tolist()
            result[name][key] = series

    return result


# ── Markdown sections ─────────────────────────────────────────────────────────

def section_header(git_sha: str, runs: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_names = ", ".join(f"`{n}`" for n in sorted(runs.keys()))
    return (
        f"# Phase 5 Multitask Results\n\n"
        f"**Generated**: {timestamp}  **Git**: `{git_sha}`  \n"
        f"**Experiment**: `{EXPERIMENT_NAME}`  \n"
        f"**Runs**: {run_names}\n\n"
        f"---\n\n"
    )


def section_dataset_summary(runs: dict) -> str:
    lines = ["## Section 1 — Dataset Summary\n\n"]

    header = "| Task | Split | N valid | Prevalence (train %) |\n"
    sep    = "|------|-------|---------|----------------------|\n"
    lines.append(header)
    lines.append(sep)

    # Use first run's params (dataset is shared across both runs)
    first_run = next(iter(runs.values()))
    params = first_run["params"]

    for task in TASKS:
        prev = PREVALENCES.get(task)
        prev_str = f"{prev*100:.1f}" if prev is not None else "—"
        for split in ["train", "val", "test"]:
            n = params.get(f"n_valid_{task}_{split}")
            # Only show prevalence for train split
            p_col = prev_str if split == "train" else "—"
            lines.append(f"| {task} | {split} | {fmt_int(n)} | {p_col} |\n")

    lines.append("\n_Prevalence = positive rate in the training split (used as AUPRC random baseline)._\n\n")
    return "".join(lines)


def section_phase4_reference() -> str:
    return (
        f"## Section 2 — Phase 4 Reference (Single-Task MMR)\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Best val AUROC (MMR) | {fmt(PHASE4_MMR_VAL)} |\n"
        f"| Test AUROC (MMR)     | {fmt(PHASE4_MMR_TEST)} |\n"
        f"\n"
    )


def section_results_table(runs: dict) -> str:
    lines = ["## Section 3 — Per-Task Final Results\n\n"]

    header = "| Config | Task | Best Val AUROC | Test AUROC | Test AUPRC | AUPRC Random | Best Epoch |\n"
    sep    = "|--------|------|---------------|------------|------------|--------------|------------|\n"
    lines.append(header)
    lines.append(sep)

    for run_name, run_info in sorted(runs.items()):
        scalars    = run_info["scalars"]
        best_epoch = scalars.get("best_epoch")

        for task in TASKS:
            val_auroc    = scalars.get(f"best_val_auroc_{task}")
            test_auroc   = scalars.get(f"test_auroc_{task}")
            test_auprc   = scalars.get(f"test_auprc_{task}")
            random_auprc = PREVALENCES.get(task)
            lines.append(
                f"| {run_name} | {task} "
                f"| {fmt(val_auroc)} | {fmt(test_auroc)} | {fmt(test_auprc)} "
                f"| {fmt(random_auprc, 2)} | {fmt_int(best_epoch)} |\n"
            )

        # Mean row
        val_auroc_mean  = scalars.get("best_val_auroc_mean")
        test_auroc_vals = [scalars.get(f"test_auroc_{t}") for t in TASKS]
        test_auroc_mean = float(np.mean(test_auroc_vals)) if all(v is not None for v in test_auroc_vals) else None
        test_auprc_mean = scalars.get("test_auprc_mean")
        lines.append(
            f"| {run_name} | **mean** "
            f"| {fmt(val_auroc_mean)} | {fmt(test_auroc_mean)} | {fmt(test_auprc_mean)} "
            f"| 0.50 (AUROC) | {fmt_int(best_epoch)} |\n"
        )
        lines.append("| | | | | | | |\n")

    lines.append(
        "\n_AUROC random baseline = 0.50. AUPRC random baseline = prevalence (training split)._\n\n"
    )
    return "".join(lines)


def section_stability_table(runs: dict, trajectories: dict) -> str:
    lines = ["## Section 4 — Per-Task Stability\n\n"]

    metric_groups = (
        [(task, f"val_auroc_{task}", f"val_loss_{task}") for task in TASKS] +
        [("mean", "val_auroc_mean", "val_loss")]
    )

    for label, auroc_key, loss_key in metric_groups:
        lines.append(f"### Stability — `{auroc_key}`\n\n")
        header = "| Config | reversals | mono_ratio | max_swing | cv_% | vl_rise | post_peak_drop |\n"
        sep    = "|--------|-----------|------------|-----------|------|---------|----------------|\n"
        lines.append(header)
        lines.append(sep)

        for run_name in sorted(runs.keys()):
            traj         = trajectories.get(run_name, {})
            auroc_series = traj.get(auroc_key, [])
            loss_series  = traj.get(loss_key, [])
            stab         = compute_stability_metrics(auroc_series, loss_series)
            lines.append(
                f"| {run_name} "
                f"| {fmt_int(stab.get('reversals'))} "
                f"| {fmt(stab.get('monotonicity_ratio'), 3)} "
                f"| {fmt(stab.get('max_swing'), 4)} "
                f"| {fmt(stab.get('cv_pct'), 2)} "
                f"| {fmt(stab.get('val_loss_rise'), 4)} "
                f"| {fmt(stab.get('post_peak_drop'), 4)} |\n"
            )
        lines.append("\n")

    return "".join(lines)


def section_imbalance(runs: dict, trajectories: dict) -> str:
    lines = ["## Section 5 — Class Imbalance / Trivial Guessing\n\n"]

    # 5a: AUPRC vs random baseline
    lines.append("### 5a — AUPRC vs Random Baseline\n\n")
    header = "| Config | Task | Test AUPRC | Random Baseline (prevalence) | Delta |\n"
    sep    = "|--------|------|------------|------------------------------|-------|\n"
    lines.append(header)
    lines.append(sep)

    for run_name, run_info in sorted(runs.items()):
        scalars = run_info["scalars"]
        for task in TASKS:
            test_auprc   = scalars.get(f"test_auprc_{task}")
            random_auprc = PREVALENCES.get(task)
            delta = (test_auprc - random_auprc) if (test_auprc is not None and random_auprc is not None) else None
            lines.append(
                f"| {run_name} | {task} "
                f"| {fmt(test_auprc)} | {fmt(random_auprc, 2)} | {fmt(delta)} |\n"
            )
    lines.append("\n")

    # 5b: Sensitivity / Specificity at best epoch
    lines.append("### 5b — Sensitivity / Specificity at Best Epoch\n\n")
    lines.append("_Sensitivity near 0 = model predicts all-negative (trivial solution)._\n\n")
    header = "| Config | Task | Val Sensitivity | Val Specificity |\n"
    sep    = "|--------|------|----------------|----------------|\n"
    lines.append(header)
    lines.append(sep)

    for run_name, run_info in sorted(runs.items()):
        scalars = run_info["scalars"]
        traj    = trajectories.get(run_name, {})
        best_epoch = scalars.get("best_epoch")
        for task in TASKS:
            sens_series = traj.get(f"val_sensitivity_{task}", [])
            spec_series = traj.get(f"val_specificity_{task}", [])
            sens = spec = None
            if best_epoch is not None and sens_series:
                idx  = min(max(int(best_epoch) - 1, 0), len(sens_series) - 1)
                sens = sens_series[idx] if idx < len(sens_series) else None
                spec = spec_series[idx] if (spec_series and idx < len(spec_series)) else None
            lines.append(
                f"| {run_name} | {task} "
                f"| {fmt(sens)} | {fmt(spec)} |\n"
            )
    lines.append("\n")

    # 5c: Head weight variance pointer
    lines.append("### 5c — Head Weight Variance over Training\n\n")
    lines.append(
        "Figure 6 (`phase5_head_weight_var.png`) shows `head_weight_var_{task}` trajectories. "
        "Near-zero variance = dead head (model ignores that output).\n\n"
    )
    lines.append(
        "> **Gap**: output logit std (std of sigmoid(logits)) is not logged in `train.py`. "
        "Sensitivity collapse (Section 5b) is the available proxy for trivial prediction.\n\n"
    )

    return "".join(lines)


def section_figures(figures: dict) -> str:
    lines = ["## Section 6 — Figures\n\n"]
    for label, path in figures.items():
        rel = path.relative_to(OUT_DIR)
        lines.append(f"### {label}\n\n![{label}]({rel})\n\n")
    return "".join(lines)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_val_auroc_per_task(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    run_names = sorted(runs.keys())

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(run_names):
            vals = trajectories.get(run_name, {}).get(f"val_auroc_{task}", [])
            if not vals:
                continue
            color  = COLORS[i % len(COLORS)]
            epochs = list(range(1, len(vals) + 1))
            ax.plot(epochs, vals, linewidth=1.5, color=color, label=run_name)
        ax.set_title(f"Val AUROC — {task.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val AUROC")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 5 — Val AUROC per Task", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "phase5_val_auroc_per_task.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_val_auroc_mean(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    run_names = sorted(runs.keys())

    for i, run_name in enumerate(run_names):
        vals = trajectories.get(run_name, {}).get("val_auroc_mean", [])
        if not vals:
            continue
        color  = COLORS[i % len(COLORS)]
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, linewidth=1.5, color=color, label=run_name)

    ax.set_title("Phase 5 — Val AUROC Mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC Mean")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "phase5_val_auroc_mean.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_loss_per_task(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    run_names = sorted(runs.keys())

    for row_idx, task in enumerate(TASKS):
        for col_idx, loss_type in enumerate(["train_loss", "val_loss"]):
            ax         = axes[row_idx][col_idx]
            metric_key = f"{loss_type}_{task}"
            for i, run_name in enumerate(run_names):
                vals = trajectories.get(run_name, {}).get(metric_key, [])
                if not vals:
                    continue
                color  = COLORS[i % len(COLORS)]
                epochs = list(range(1, len(vals) + 1))
                ax.plot(epochs, vals, linewidth=1.5, color=color, label=run_name)
            ax.set_title(f"{task.upper()} — {loss_type.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 5 — Loss per Task", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "phase5_loss_per_task.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_sensitivity_specificity(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    run_names = sorted(runs.keys())

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(run_names):
            traj  = trajectories.get(run_name, {})
            color = COLORS[i % len(COLORS)]
            sens  = traj.get(f"val_sensitivity_{task}", [])
            spec  = traj.get(f"val_specificity_{task}", [])
            if sens:
                epochs = list(range(1, len(sens) + 1))
                ax.plot(epochs, sens, linewidth=1.5, color=color, linestyle="-",
                        label=f"{run_name} sens")
            if spec:
                epochs = list(range(1, len(spec) + 1))
                ax.plot(epochs, spec, linewidth=1.5, color=color, linestyle="--",
                        label=f"{run_name} spec")
        ax.set_title(f"Sens / Spec — {task.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 5 — Val Sensitivity / Specificity per Task", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "phase5_sensitivity_specificity.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_f1(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    run_names = sorted(runs.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles = ["-", "--", ":"]

    for i, run_name in enumerate(run_names):
        traj  = trajectories.get(run_name, {})
        color = COLORS[i % len(COLORS)]
        for j, task in enumerate(TASKS):
            vals = traj.get(f"val_f1_{task}", [])
            if vals:
                epochs = list(range(1, len(vals) + 1))
                ax.plot(epochs, vals, linewidth=1.2, color=color,
                        linestyle=linestyles[j], label=f"{run_name} — {task}")

        # Mean F1 across tasks
        f1_series_list = [traj.get(f"val_f1_{t}", []) for t in TASKS]
        valid_series   = [s for s in f1_series_list if s]
        if valid_series:
            min_len  = min(len(s) for s in valid_series)
            mean_f1  = [float(np.mean([s[ep] for s in valid_series])) for ep in range(min_len)]
            epochs   = list(range(1, min_len + 1))
            ax.plot(epochs, mean_f1, linewidth=2.0, color=color, linestyle="-.",
                    label=f"{run_name} — mean")

    ax.set_title("Phase 5 — Val F1 per Task + Mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "phase5_f1.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def plot_head_weight_var(trajectories: dict, runs: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    run_names = sorted(runs.keys())

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(run_names):
            vals = trajectories.get(run_name, {}).get(f"head_weight_var_{task}", [])
            if not vals:
                continue
            color  = COLORS[i % len(COLORS)]
            epochs = list(range(1, len(vals) + 1))
            ax.plot(epochs, vals, linewidth=1.5, color=color, label=run_name)
        ax.set_title(f"Head Weight Var — {task.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 5 — Head Weight Variance per Task", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "phase5_head_weight_var.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not MLFLOW_DB_PATH.exists():
        raise SystemExit(f"MLflow DB not found: {MLFLOW_DB_PATH}  (run scripts/mlflow_local.sh first)")

    conn    = sqlite3.connect(str(MLFLOW_DB_PATH))
    git_sha = get_git_commit()

    exp_id = get_experiment_id(conn, EXPERIMENT_NAME)
    if exp_id is None:
        raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found in {MLFLOW_DB_PATH}")

    runs = load_runs(conn, exp_id)
    if not runs:
        raise SystemExit(f"No FINISHED non-preflight runs found in '{EXPERIMENT_NAME}'")
    print(f"Found {len(runs)} run(s): {list(runs.keys())}")

    trajectories = load_trajectories(conn, runs)
    conn.close()

    # ── Figures ───────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures: dict = {}
    figures["Figure 1 — Val AUROC per Task"]      = plot_val_auroc_per_task(trajectories, runs, FIGURES_DIR)
    figures["Figure 2 — Val AUROC Mean"]           = plot_val_auroc_mean(trajectories, runs, FIGURES_DIR)
    figures["Figure 3 — Loss per Task"]            = plot_loss_per_task(trajectories, runs, FIGURES_DIR)
    figures["Figure 4 — Sensitivity/Specificity"]  = plot_sensitivity_specificity(trajectories, runs, FIGURES_DIR)
    figures["Figure 5 — Val F1"]                   = plot_f1(trajectories, runs, FIGURES_DIR)
    figures["Figure 6 — Head Weight Variance"]     = plot_head_weight_var(trajectories, runs, FIGURES_DIR)

    # ── Assemble report ───────────────────────────────────────────────────────
    report = (
        section_header(git_sha, runs) +
        section_dataset_summary(runs) +
        section_phase4_reference() +
        section_results_table(runs) +
        section_stability_table(runs, trajectories) +
        section_imbalance(runs, trajectories) +
        section_figures(figures) +
        "\n---\n*Generated by `scripts/studies/phase5_multitask_report.py`*\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{REPORT_DATE}-phase5-results.md"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    main()
