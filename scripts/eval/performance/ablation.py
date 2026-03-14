#!/usr/bin/env python3
"""Phase 8 ablation analysis functions — MLP-RPB × Patch Dropout.

Pure function library. No __main__ block. Imported by
scripts/studies/phase8_ablation.py which provides the CLI entry point.
"""

import math
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project setup ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "reports"
FIG_DIR    = REPORT_DIR / "figures" / "phase8"

TASKS       = ["mmr", "ras", "braf"]
EXPERIMENT  = "multitask-surgen-phase8"

# Phase 6 ABMIL-joined reference (baseline gate)
PHASE6_MEAN_TEST = 0.7763
PHASE6_MMR_TEST  = 0.8351
PHASE6_RAS_TEST  = 0.6543
PHASE6_BRAF_TEST = 0.8394

# Config display metadata — order matches the report table
CONFIGS = [
    {
        "stem":    "config_phase8_baseline",
        "pe":      "none",
        "dropout": 0.0,
        "label":   "baseline",
    },
    {
        "stem":    "config_phase8_mlp_rpb",
        "pe":      "mlp_rpb",
        "dropout": 0.0,
        "label":   "mlp_rpb",
    },
    {
        "stem":    "config_phase8_dropout10",
        "pe":      "none",
        "dropout": 0.1,
        "label":   "dropout10",
    },
    {
        "stem":    "config_phase8_dropout25",
        "pe":      "none",
        "dropout": 0.25,
        "label":   "dropout25",
    },
    {
        "stem":    "config_phase8_mlp_rpb_dropout10",
        "pe":      "mlp_rpb",
        "dropout": 0.1,
        "label":   "mlp_rpb_dropout10",
    },
    {
        "stem":    "config_phase8_mlp_rpb_dropout25",
        "pe":      "mlp_rpb",
        "dropout": 0.25,
        "label":   "mlp_rpb_dropout25",
    },
]

SEEDS = [0, 1, 2]


# ── Helpers ───────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def fmt(v, d: int = 4) -> str:
    if v is None:
        return "—"
    try:
        if math.isnan(float(v)):
            return "—"
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_delta(v) -> str:
    if v is None:
        return "—"
    try:
        val = float(v)
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.4f}"
    except (TypeError, ValueError):
        return "—"


def _auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def bootstrap_ci(
    labels: np.ndarray,
    probs:  np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """Test-sample bootstrap CI. Identical to utils/metrics.py:bootstrap_ci."""
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
    run_names: list,
    task: str,
    metric_fn,
    n: int = 2000,
) -> tuple:
    """Bootstrap CI on probs/labels concatenated across seed runs for one task."""
    all_labels = []
    all_probs  = []
    for rn in run_names:
        if rn not in inference:
            continue
        labels = inference[rn]["labels"].get(task, np.array([]))
        probs  = inference[rn]["probs"].get(task, np.array([]))
        if len(labels) > 0:
            all_labels.append(labels)
            all_probs.append(probs)
    if not all_labels:
        return float("nan"), float("nan")
    labels_pool = np.concatenate(all_labels)
    probs_pool  = np.concatenate(all_probs)
    rng = np.random.default_rng(42)
    return bootstrap_ci(labels_pool, probs_pool, metric_fn, n_bootstrap=n, rng=rng)


# ── Inference data loading ────────────────────────────────────────────────────

def load_inference(data_dir: Path) -> dict:
    """Load per-slide predictions. Errors out if any expected run dir is missing."""
    infer_dir = data_dir / "inference"

    missing = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            run_name = f"{cfg['stem']}-s{seed}"
            if not (infer_dir / run_name).exists():
                missing.append(run_name)

    if missing:
        print(f"ERROR: Missing {len(missing)} expected inference directories in {infer_dir}:",
              file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    result = {}
    for cfg in CONFIGS:
        for seed in SEEDS:
            run_name = f"{cfg['stem']}-s{seed}"
            run_dir  = infer_dir / run_name
            entry    = {"probs": {}, "labels": {}}
            for task in TASKS:
                p_path = run_dir / f"probs_{task}.npy"
                l_path = run_dir / f"labels_{task}.npy"
                if p_path.exists() and l_path.exists():
                    entry["probs"][task]  = np.load(p_path)
                    entry["labels"][task] = np.load(l_path)
                else:
                    entry["probs"][task]  = np.array([], dtype=np.float32)
                    entry["labels"][task] = np.array([], dtype=np.float32)
            result[run_name] = entry
    return result


# ── SQLite data loading ───────────────────────────────────────────────────────

def get_experiment_id(conn: sqlite3.Connection, name: str) -> "str | None":
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
    runs_df = runs_df[~runs_df["name"].str.contains("preflight", case=False, na=False)]

    result: dict = {}
    for _, row in runs_df.iterrows():
        uid  = row["run_uuid"]
        name = row["name"]
        result[name] = {"run_id": uid, "params": {}, "scalars": {}}

    if not result:
        return result

    run_uuids   = [v["run_id"] for v in result.values()]
    uid_to_name = {v["run_id"]: k for k, v in result.items()}
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
        [f"best_val_auroc_{t}" for t in TASKS] +
        [f"best_val_auprc_{t}" for t in TASKS] +
        [f"test_auroc_{t}"     for t in TASKS] +
        [f"test_auprc_{t}"     for t in TASKS]
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


def load_trajectories(conn: sqlite3.Connection, runs: dict) -> dict:
    """Return {run_name: {metric: [val, ...]}} per epoch."""
    if not runs:
        return {}
    run_uuids   = [v["run_id"] for v in runs.values()]
    uid_to_name = {v["run_id"]: k for k, v in runs.items()}
    placeholders = ",".join("?" * len(run_uuids))
    traj_keys = ["val_auroc_mean", "train_loss", "val_loss"]
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

    result: dict = {name: {} for name in runs}
    for (uid, key), grp in df.groupby(["run_uuid", "key"]):
        name = uid_to_name[uid]
        result[name][key] = grp.sort_values("step")["value"].tolist()
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_config(cfg: dict, runs: dict) -> dict:
    """Collect seed runs for a config and return per-metric mean/std."""
    stem = cfg["stem"]
    seed_runs = []
    for seed in SEEDS:
        run_name = f"{stem}-s{seed}"
        if run_name in runs:
            seed_runs.append(runs[run_name])

    n_complete = len(seed_runs)
    result = {"n_complete": n_complete, "seeds": seed_runs}

    for key in (
        ["test_auroc_mean", "test_auprc_mean", "best_val_auroc_mean", "best_epoch"] +
        [f"test_auroc_{t}" for t in TASKS] +
        [f"test_auprc_{t}" for t in TASKS]
    ):
        vals = [r["scalars"].get(key) for r in seed_runs if r["scalars"].get(key) is not None]
        if vals:
            result[f"{key}_mean"] = float(np.mean(vals))
            result[f"{key}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"]  = None

    return result


# ── Figures ───────────────────────────────────────────────────────────────────

def _short_label(cfg: dict) -> str:
    pe = "RPB" if cfg["pe"] == "mlp_rpb" else "no-PE"
    dr = f"drop{int(cfg['dropout']*100)}" if cfg["dropout"] > 0 else "no-drop"
    return f"{pe}\n{dr}"


def plot_test_auroc_grouped(agg: dict, out_path: Path) -> None:
    """Bar chart: mean test AUROC per task per config, with seed scatter."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    metrics = ["test_auroc_mean"] + [f"test_auroc_{t}" for t in TASKS]
    titles  = ["Mean", "MMR", "RAS", "BRAF"]
    x       = np.arange(len(CONFIGS))
    width   = 0.6

    for ax, metric, title in zip(axes, metrics, titles):
        means = [agg[c["label"]][f"{metric}_mean"] or 0.0 for c in CONFIGS]
        errs  = [agg[c["label"]][f"{metric}_std"]  or 0.0 for c in CONFIGS]
        ax.bar(x, means, width=width, yerr=errs, capsize=4,
               color=plt.cm.tab10.colors[:len(CONFIGS)], alpha=0.8)

        # Scatter individual seeds
        for i, cfg in enumerate(CONFIGS):
            seed_vals = [r["scalars"].get(metric) for r in agg[cfg["label"]]["seeds"]
                         if r["scalars"].get(metric) is not None]
            ax.scatter([i] * len(seed_vals), seed_vals, color="black", s=18, zorder=5)

        # Phase 6 reference line
        ref = {
            "test_auroc_mean": PHASE6_MEAN_TEST,
            "test_auroc_mmr":  PHASE6_MMR_TEST,
            "test_auroc_ras":  PHASE6_RAS_TEST,
            "test_auroc_braf": PHASE6_BRAF_TEST,
        }.get(metric)
        if ref:
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"Phase 6 ref ({ref:.4f})")
            ax.legend(fontsize=7)

        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(c) for c in CONFIGS], fontsize=8)
        ax.set_ylabel("Test AUROC")
        ax.set_ylim(0.55, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 8 — Test AUROC by Config (mean ± std, seed scatter)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pe_vs_nope(agg: dict, out_path: Path) -> None:
    """Paired bar: PE effect for each dropout level."""
    dropout_levels = [0.0, 0.1, 0.25]
    nope_labels    = ["baseline",    "dropout10",    "dropout25"]
    rpb_labels     = ["mlp_rpb",     "mlp_rpb_dropout10", "mlp_rpb_dropout25"]

    x = np.arange(len(dropout_levels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    nope_means = [agg[l]["test_auroc_mean_mean"] or 0.0 for l in nope_labels]
    rpb_means  = [agg[l]["test_auroc_mean_mean"] or 0.0 for l in rpb_labels]
    nope_errs  = [agg[l]["test_auroc_mean_std"]  or 0.0 for l in nope_labels]
    rpb_errs   = [agg[l]["test_auroc_mean_std"]  or 0.0 for l in rpb_labels]

    ax.bar(x - w/2, nope_means, w, yerr=nope_errs, capsize=4, label="no-PE", alpha=0.85)
    ax.bar(x + w/2, rpb_means,  w, yerr=rpb_errs,  capsize=4, label="MLP-RPB", alpha=0.85)
    ax.axhline(PHASE6_MEAN_TEST, color="red", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Phase 6 ref ({PHASE6_MEAN_TEST:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"dropout={d}" for d in dropout_levels])
    ax.set_ylabel("Mean Test AUROC")
    ax.set_title("PE Effect: no-PE vs MLP-RPB at each dropout level")
    ax.legend()
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(trajs: dict, out_path: Path) -> None:
    """Val AUROC mean learning curves, one line per config (seed 0 only for clarity)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for i, cfg in enumerate(CONFIGS):
        run_name = f"{cfg['stem']}-s0"
        traj = trajs.get(run_name, {})
        vals = traj.get("val_auroc_mean", [])
        if vals:
            ax.plot(vals, label=_short_label(cfg).replace("\n", " "),
                    color=colors[i], linewidth=1.5)

    ax.axhline(PHASE6_MEAN_TEST, color="red", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Phase 6 ref ({PHASE6_MEAN_TEST:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC (mean)")
    ax.set_title("Phase 8 — Validation Learning Curves (seed 0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Gate evaluation ───────────────────────────────────────────────────────────

def evaluate_gate(agg: dict, inference: Optional[dict] = None, n_bootstrap: int = 2000) -> dict:
    """Return gate verdict dict.

    When inference is provided, uses test-sample pooled CIs for non-overlap check.
    When inference is None, falls back to delta-only gate (no CI check).
    """
    baseline      = agg["baseline"]
    baseline_mean = baseline.get("test_auroc_mean_mean")

    gate_pass = (
        baseline_mean is not None and
        baseline_mean >= PHASE6_MEAN_TEST
    )

    base_run_names = [f"{CONFIGS[0]['stem']}-s{seed}" for seed in SEEDS]

    winner       = None
    winner_delta = None
    for cfg in CONFIGS[1:]:  # skip baseline itself
        label     = cfg["label"]
        this_mean = agg[label].get("test_auroc_mean_mean")
        base_mean = baseline_mean
        if this_mean is None or base_mean is None:
            continue
        delta = this_mean - base_mean
        if delta <= 0.005:
            continue

        if inference is not None:
            # Test-sample pooled CIs: average per-task CI bounds
            cand_run_names = [f"{cfg['stem']}-s{seed}" for seed in SEEDS]
            task_lo_winner   = []
            task_hi_baseline = []
            for task in TASKS:
                w_lo, _   = pooled_bootstrap_ci(inference, cand_run_names, task, _auroc, n=n_bootstrap)
                _,    b_hi = pooled_bootstrap_ci(inference, base_run_names,  task, _auroc, n=n_bootstrap)
                if not math.isnan(w_lo) and not math.isnan(b_hi):
                    task_lo_winner.append(w_lo)
                    task_hi_baseline.append(b_hi)

            if not task_lo_winner:
                continue
            pooled_lo_winner   = float(np.mean(task_lo_winner))
            pooled_hi_baseline = float(np.mean(task_hi_baseline))
            ci_nonoverlap = pooled_lo_winner > pooled_hi_baseline
        else:
            ci_nonoverlap = True  # no inference data: skip CI check

        if ci_nonoverlap:
            if winner is None or delta > winner_delta:
                winner       = cfg
                winner_delta = delta

    return {
        "baseline_mean": baseline_mean,
        "gate_pass":     gate_pass,
        "winner":        winner,
        "winner_delta":  winner_delta,
        "ci_used":       inference is not None,
    }


# ── Bootstrap CI report section ───────────────────────────────────────────────

def section_bootstrap_ci(inference: dict, n_bootstrap: int) -> list:
    """Return markdown lines for per-run, per-task test-sample CI table."""
    lines = []
    lines.append("## Bootstrap 95% CIs (Test-Sample)")
    lines.append("")
    lines.append(
        f"Per-slide bootstrap ({n_bootstrap} resamples, 95% CI). "
        "Half-width ±hw = (hi − lo) / 2. "
        "MMR ≈15 positives → expected ±hw ≈ 0.08–0.12 (cf. Phase 6: 0.8533 ±0.0939)."
    )
    lines.append("")
    lines.append("| Run | Task | AUROC 95% CI | AUPRC 95% CI |")
    lines.append("|-----|------|--------------|--------------|")

    rng = np.random.default_rng(42)
    for cfg in CONFIGS:
        for seed in SEEDS:
            run_name = f"{cfg['stem']}-s{seed}"
            if run_name not in inference:
                continue
            inf = inference[run_name]
            for task in TASKS:
                labels = inf["labels"].get(task, np.array([]))
                probs  = inf["probs"].get(task, np.array([]))
                if len(labels) < 10:
                    lines.append(f"| {run_name} | {task} | — | — |")
                    continue
                auroc = _auroc(labels, probs)
                auprc = _auprc(labels, probs)
                a_lo, a_hi = bootstrap_ci(labels, probs, _auroc, n_bootstrap, rng)
                p_lo, p_hi = bootstrap_ci(labels, probs, _auprc, n_bootstrap, rng)
                hw_a = (a_hi - a_lo) / 2
                hw_p = (p_hi - p_lo) / 2
                lines.append(
                    f"| {run_name} | {task} "
                    f"| {fmt(auroc)} \u00b1{fmt(hw_a)} "
                    f"| {fmt(auprc)} \u00b1{fmt(hw_p)} |"
                )
            lines.append("| | | | |")
    lines.append("")
    return lines


# ── Markdown report ───────────────────────────────────────────────────────────

def render_report(runs: dict, agg: dict, trajs: dict, gate: dict,
                  n_bootstrap: int, out_path: Path,
                  inference: Optional[dict] = None) -> None:
    date  = datetime.now().strftime("%Y-%m-%d")
    sha   = git_commit()
    lines = []

    def w(s=""):
        lines.append(s)

    w(f"# Phase 8 Results — MLP-RPB + Patch Dropout Ablation")
    w(f"**Date:** {date}  |  **Commit:** `{sha}`")
    w()

    # ── Main results table ────────────────────────────────────────────────────
    n_complete = {cfg["label"]: agg[cfg["label"]]["n_complete"] for cfg in CONFIGS}
    complete_configs   = [cfg["label"] for cfg in CONFIGS if n_complete[cfg["label"]] == 3]
    incomplete_configs = [cfg["label"] for cfg in CONFIGS if n_complete[cfg["label"]] < 3]

    ci_method = (
        f"test-sample bootstrap ({n_bootstrap} resamples over ~165 test slides)"
        if inference is not None
        else f"seed-level mean delta (Δ > +0.005 vs Phase 8 baseline) with 95% bootstrap CIs on the mean for non-overlap confirmation"
    )
    completeness = (
        f"{len(complete_configs)}/6 configurations have completed all 3 seeds at the time "
        f"of this report"
    )
    if incomplete_configs:
        completeness += f" (`{'`, `'.join(incomplete_configs)}` are pending)."
    else:
        completeness += "."
    w(
        f"Each Phase 8 configuration was trained with 3 independent random seeds "
        f"(varying `training.random_seed`); reported values are mean ± standard deviation "
        f"across seeds. {completeness}"
    )
    w()
    w(
        "Two complementary metrics are reported. **AUROC** (Area Under the ROC Curve) measures "
        "rank discrimination between positive and negative cases regardless of operating threshold; "
        "it is the primary gate metric and is well-suited to the class-imbalanced tasks here "
        "(particularly BRAF, ~10% positive rate). **AUPRC** (Area Under the Precision-Recall Curve) "
        "is more sensitive to performance on the minority class: a random classifier scores equal "
        "to the prevalence rate rather than 0.5, so AUPRC gains above that baseline reflect genuine "
        "positive-class recall. Together the two metrics provide a fuller picture — a model can "
        "improve AUROC by better separating easy negatives while AUPRC reveals whether it also "
        f"recovers hard positives. Winner determination uses {ci_method}."
    )
    w()
    w("For context, the tables include the best result from Phase 5 (mean-pooling multitask "
      "baseline, single run) and the Phase 6 ABMIL-joined best (single run), which serves "
      "as the gate threshold for Phase 8.")
    w()

    # ── Build shared row dicts ─────────────────────────────────────────────────
    TOTAL_EPOCHS = 150

    def make_ref(phase, cosine, accum, agg_str, pe, dropout,
                 mmr_roc, ras_roc, braf_roc, mean_roc,
                 mmr_prc="—", ras_prc="—", braf_prc="—", mean_prc="—"):
        return dict(
            phase=phase, cosine=cosine, accum=accum, agg=agg_str, pe=pe, dropout=dropout,
            mean_auroc=float(mean_roc) if mean_roc != "—" else None,
            mean_auprc=float(mean_prc) if mean_prc != "—" else None,
            mmr_roc=mmr_roc, ras_roc=ras_roc, braf_roc=braf_roc, mean_roc=mean_roc, delta="—",
            mmr_prc=mmr_prc, ras_prc=ras_prc, braf_prc=braf_prc, mean_prc=mean_prc,
            epoch="—",
        )

    all_rows = [
        make_ref("4",              "no",  "1",  "mean",  "none", "0.0", "0.8422", "—",      "—",      "—"),
        make_ref("4",              "yes", "16", "mean",  "none", "0.0", "0.8640", "—",      "—",      "—"),
        make_ref("5",              "no",  "1",  "mean",  "none", "0.0", "0.8931", "0.6645", "0.8342", "0.7973"),
        make_ref("6",              "yes", "16", "abmil", "none", "0.0", "0.8222", "—",      "—",      "—"),
        make_ref("6 *(gate ref)*", "yes", "16", "abmil", "none", "0.0", "0.8351", "0.6543", "0.8394", "0.7763"),
    ]

    baseline_mean = agg["baseline"].get("test_auroc_mean_mean")

    for cfg in CONFIGS:
        label = cfg["label"]
        a = agg[label]

        def ms(key):
            m = a.get(f"{key}_mean")
            s = a.get(f"{key}_std")
            if m is None:
                return "—"
            if s is None or s == 0.0:
                return fmt(m)
            return f"{fmt(m)}±{fmt(s)}"

        mean_val  = a.get("test_auroc_mean_mean")
        mean_pval = a.get("test_auprc_mean_mean")
        epoch_m   = a.get("best_epoch_mean")

        if baseline_mean is not None and mean_val is not None:
            delta_str = fmt_delta(mean_val - baseline_mean) if label != "baseline" else "—"
        else:
            delta_str = "—"

        epoch_str = f"{epoch_m:.0f}/{TOTAL_EPOCHS}" if epoch_m is not None else "—"

        all_rows.append(dict(
            phase="8", cosine="yes", accum="16", agg="abmil",
            pe=cfg["pe"], dropout=str(cfg["dropout"]),
            mean_auroc=mean_val, mean_auprc=mean_pval,
            mmr_roc=ms("test_auroc_mmr"), ras_roc=ms("test_auroc_ras"),
            braf_roc=ms("test_auroc_braf"), mean_roc=ms("test_auroc_mean"), delta=delta_str,
            mmr_prc=ms("test_auprc_mmr"), ras_prc=ms("test_auprc_ras"),
            braf_prc=ms("test_auprc_braf"), mean_prc=ms("test_auprc_mean"),
            epoch=epoch_str,
        ))

    def _bold_row(row: dict, fields: list) -> dict:
        return {k: (f"**{v}**" if k in fields else v) for k, v in row.items()}

    PREFIX  = "| {phase} | {cosine} | {accum} | {agg} | {pe} | {dropout} |"
    HDR_PRE = "| Phase | Cosine LR | Accum | Agg | PE | Dropout |"
    SEP_PRE = "|---|---|---|---|---|---"

    w("## Test Performance — Mean ± Std across Seeds")
    w()

    # ── AUROC table ───────────────────────────────────────────────────────────
    best_auroc = max((r["mean_auroc"] for r in all_rows if r["mean_auroc"] is not None), default=None)

    w("### Test AUROC")
    w()
    w(HDR_PRE + " MMR | RAS | BRAF | Mean±std | Δ vs Ph8 baseline | Peak epoch/Total |")
    w(SEP_PRE + "|---|---|---|---|---|---|")
    for row in all_rows:
        r = _bold_row(row, ["mmr_roc", "ras_roc", "braf_roc", "mean_roc", "delta", "epoch"]) \
            if best_auroc is not None and row["mean_auroc"] == best_auroc else row
        w((PREFIX + " {mmr_roc} | {ras_roc} | {braf_roc} | {mean_roc} | {delta} | {epoch} |").format(**r))
    w()

    # ── AUPRC table ───────────────────────────────────────────────────────────
    best_auprc = max((r["mean_auprc"] for r in all_rows if r["mean_auprc"] is not None), default=None)

    w("### Test AUPRC")
    w()
    w(HDR_PRE + " MMR | RAS | BRAF | Mean±std | Peak epoch/Total |")
    w(SEP_PRE + "|---|---|---|---|---|")
    for row in all_rows:
        r = _bold_row(row, ["mmr_prc", "ras_prc", "braf_prc", "mean_prc", "epoch"]) \
            if best_auprc is not None and row["mean_auprc"] == best_auprc else row
        w((PREFIX + " {mmr_prc} | {ras_prc} | {braf_prc} | {mean_prc} | {epoch} |").format(**r))
    w()

    # ── Bootstrap CI section (test-sample, only when inference data available) ─
    if inference is not None:
        ci_lines = section_bootstrap_ci(inference, n_bootstrap)
        for line in ci_lines:
            w(line)

    # ── Gate ──────────────────────────────────────────────────────────────────
    w("## Gate")
    w()
    bm       = fmt(gate["baseline_mean"])
    gate_sym = "✅ PASS" if gate["gate_pass"] else "❌ FAIL"
    ci_note  = " (CI non-overlap via test-sample bootstrap)" if gate["ci_used"] else " (delta-only; run with --data-dir for CI gate)"
    w(f"**Baseline gate** (mean test AUROC ≥ {PHASE6_MEAN_TEST}): `{bm}` — **{gate_sym}**")
    w()
    winner_cfg = gate["winner"]
    if winner_cfg:
        w(f"**Winner:** `{winner_cfg['label']}` — Δ = {fmt_delta(gate['winner_delta'])} vs baseline"
          f"{ci_note}. "
          f"Carry `positional_encoding={winner_cfg['pe']}`, "
          f"`patch_dropout_rate={winner_cfg['dropout']}` to Phase 9.")
    else:
        w(f"**No winner** (no config clears Δ > +0.005{ci_note}).")
        w("Carry `positional_encoding=none`, `patch_dropout_rate=0.0` forward to Phase 9.")
    w()

    # ── Figures ───────────────────────────────────────────────────────────────
    w("## Figures")
    w()
    w("- `reports/figures/phase8/test_auroc_grouped.png` — grouped bar chart per task")
    w("- `reports/figures/phase8/pe_vs_nope.png` — PE effect at each dropout level")
    w("- `reports/figures/phase8/learning_curves.png` — val AUROC trajectories (seed 0)")
    w()

    # ── Analysis ──────────────────────────────────────────────────────────────
    w("## Analysis")
    w()
    w("### PE Effect (MLP-RPB vs no-PE)")
    for dr in [0.0, 0.1, 0.25]:
        nope_label = {0.0: "baseline", 0.1: "dropout10", 0.25: "dropout25"}[dr]
        rpb_label  = {0.0: "mlp_rpb", 0.1: "mlp_rpb_dropout10", 0.25: "mlp_rpb_dropout25"}[dr]
        nope_m = agg[nope_label].get("test_auroc_mean_mean")
        rpb_m  = agg[rpb_label].get("test_auroc_mean_mean")
        if nope_m is not None and rpb_m is not None:
            delta = rpb_m - nope_m
            sign = "+" if delta >= 0 else ""
            w(f"- dropout={dr}: no-PE = {fmt(nope_m)}, MLP-RPB = {fmt(rpb_m)}, Δ = {sign}{fmt(delta)}")
        else:
            w(f"- dropout={dr}: data missing")
    w()
    w("### Patch Dropout Effect (no-PE configs)")
    for dr, label in [(0.0, "baseline"), (0.1, "dropout10"), (0.25, "dropout25")]:
        m = agg[label].get("test_auroc_mean_mean")
        w(f"- dropout={dr}: mean test AUROC = {fmt(m)}")
    w()
    w("### Patch Dropout Effect (MLP-RPB configs)")
    for dr, label in [(0.0, "mlp_rpb"), (0.1, "mlp_rpb_dropout10"), (0.25, "mlp_rpb_dropout25")]:
        m = agg[label].get("test_auroc_mean_mean")
        w(f"- dropout={dr}: mean test AUROC = {fmt(m)}")
    w()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Report written → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8 ablation report")
    parser.add_argument("--mlflow-db", default="mlflow.db",
                        help="Path to MLflow SQLite DB (default: mlflow.db)")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for test-sample CIs (default: 2000)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to phase8_extract.py output dir. "
                             "Enables test-sample bootstrap CIs and CI-based gate.")
    args = parser.parse_args()

    db_path = Path(args.mlflow_db)
    if not db_path.exists():
        print(f"ERROR: mlflow.db not found at {db_path}. "
              f"Run from project root or pass --mlflow-db.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    exp_id = get_experiment_id(conn, EXPERIMENT)
    if exp_id is None:
        print(f"ERROR: experiment '{EXPERIMENT}' not found in {db_path}.", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(conn, exp_id)
    print(f"Found {len(runs)} FINISHED non-preflight runs in '{EXPERIMENT}'.")
    if runs:
        print("  " + ", ".join(sorted(runs.keys())))
    print()

    trajs = load_trajectories(conn, runs)

    agg = {}
    for cfg in CONFIGS:
        agg[cfg["label"]] = aggregate_config(cfg, runs)
        n = agg[cfg["label"]]["n_complete"]
        print(f"  {cfg['label']:30s}  {n}/3 seeds complete")

    # Load inference data if requested
    inference = None
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"ERROR: --data-dir not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"\nLoading inference data from {data_dir} …")
        inference = load_inference(data_dir)
        print(f"  Loaded {len(inference)} run(s).")

    gate = evaluate_gate(agg, inference=inference, n_bootstrap=args.n_bootstrap)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_test_auroc_grouped(agg, FIG_DIR / "test_auroc_grouped.png")
    plot_pe_vs_nope(agg, FIG_DIR / "pe_vs_nope.png")
    plot_learning_curves(trajs, FIG_DIR / "learning_curves.png")
    print(f"Figures written → {FIG_DIR}/")

    out_path = REPORT_DIR / "phase8-results.md"
    render_report(runs, agg, trajs, gate, args.n_bootstrap, out_path, inference=inference)

    conn.close()

    print()
    print("Gate summary:")
    print(f"  Baseline mean test AUROC: {fmt(gate['baseline_mean'])}")
    print(f"  Gate (≥ {PHASE6_MEAN_TEST}): {'PASS' if gate['gate_pass'] else 'FAIL'}")
    ci_method = "test-sample pooled CI" if gate["ci_used"] else "delta-only (no inference data)"
    print(f"  CI method: {ci_method}")
    winner_cfg = gate["winner"]
    if winner_cfg:
        print(f"  Winner: {winner_cfg['label']}  Δ={fmt_delta(gate['winner_delta'])}")
    else:
        print("  No winner — carry none/0.0 to Phase 9")


if __name__ == "__main__":
    main()
