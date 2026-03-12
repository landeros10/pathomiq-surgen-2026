#!/usr/bin/env python3
"""Phase 8 ablation report — MLP-RPB × Patch Dropout.

Reads the `multitask-surgen-phase8` MLflow experiment from the local SQLite DB.
Groups 18 runs (6 configs × 3 seeds) and computes mean ± std per config.
Outputs a markdown report and figures.

Usage (run from project root after scp of mlflow.db):
    python scripts/studies/phase8_ablation.py
    python scripts/studies/phase8_ablation.py --mlflow-db /path/to/mlflow.db
    python scripts/studies/phase8_ablation.py --n-bootstrap 5000
"""

import argparse
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

# ── Project setup ─────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
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


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Return (low, high) bootstrap CI for the mean."""
    if len(values) < 2:
        m = float(np.mean(values)) if values else float("nan")
        return m, m
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boot = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


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

def aggregate_config(cfg: dict, runs: dict, n_bootstrap: int = 2000) -> dict:
    """Collect seed runs for a config and return per-metric stats."""
    stem = cfg["stem"]
    seed_runs = []
    for seed in SEEDS:
        # run_name matches what run_phase8.sh generates: {stem}-s{seed}
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
            lo, hi = bootstrap_ci(vals, n=n_bootstrap)
            result[f"{key}_ci_lo"] = lo
            result[f"{key}_ci_hi"] = hi
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_std"]  = None
            result[f"{key}_ci_lo"] = None
            result[f"{key}_ci_hi"] = None

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
        bars  = ax.bar(x, means, width=width, yerr=errs, capsize=4,
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

def evaluate_gate(agg: dict) -> dict:
    """Return gate verdict dict."""
    baseline = agg["baseline"]
    baseline_mean = baseline.get("test_auroc_mean_mean")

    gate_pass = (
        baseline_mean is not None and
        baseline_mean >= PHASE6_MEAN_TEST
    )

    # Find winner: Δ_mean > +0.005 AND CIs non-overlapping
    winner = None
    winner_delta = None
    for cfg in CONFIGS[1:]:  # skip baseline itself
        label = cfg["label"]
        this_mean = agg[label].get("test_auroc_mean_mean")
        base_mean = baseline_mean
        if this_mean is None or base_mean is None:
            continue
        delta = this_mean - base_mean
        this_lo = agg[label].get("test_auroc_mean_ci_lo")
        base_hi = baseline.get("test_auroc_mean_ci_hi")
        ci_nonoverlap = (this_lo is not None and base_hi is not None and this_lo > base_hi)
        if delta > 0.005 and ci_nonoverlap:
            if winner is None or delta > winner_delta:
                winner = cfg
                winner_delta = delta

    return {
        "baseline_mean": baseline_mean,
        "gate_pass":     gate_pass,
        "winner":        winner,
        "winner_delta":  winner_delta,
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def render_report(runs: dict, agg: dict, trajs: dict, gate: dict,
                  n_bootstrap: int, out_path: Path) -> None:
    date  = datetime.now().strftime("%Y-%m-%d")
    sha   = git_commit()
    lines = []

    def w(s=""):
        lines.append(s)

    w(f"# Phase 8 Results — MLP-RPB + Patch Dropout Ablation")
    w(f"**Date:** {date}  |  **Commit:** `{sha}`  |  **Bootstrap N:** {n_bootstrap}")
    w()
    w("## Overview")
    w()
    w(f"**Experiment:** `{EXPERIMENT}`  |  **Runs found:** {len(runs)}")
    w()
    n_complete = {cfg["label"]: agg[cfg["label"]]["n_complete"] for cfg in CONFIGS}
    missing = [(cfg["label"], 3 - n_complete[cfg["label"]]) for cfg in CONFIGS
               if n_complete[cfg["label"]] < 3]
    if missing:
        w("**⚠ Incomplete runs:**")
        for label, n_miss in missing:
            w(f"  - `{label}`: {n_miss} seed(s) missing")
        w()

    # ── Main results table ────────────────────────────────────────────────────
    w("## Test AUROC — Mean ± Std across Seeds")
    w()
    w("Phase 6 ABMIL-joined reference: MMR=0.8351  RAS=0.6543  BRAF=0.8394  Mean=**0.7763**")
    w()
    header = "| Config | PE | Dropout | MMR mean±std | RAS mean±std | BRAF mean±std | Mean±std | Δ vs baseline |"
    sep    = "|---|---|---|---|---|---|---|---|"
    w(header)
    w(sep)

    baseline_mean = agg["baseline"].get("test_auroc_mean_mean")

    for cfg in CONFIGS:
        label = cfg["label"]
        a = agg[label]
        n = a["n_complete"]

        def ms(key):
            m = a.get(f"{key}_mean")
            s = a.get(f"{key}_std")
            if m is None:
                return "—"
            if s is None or s == 0.0:
                return fmt(m)
            return f"{fmt(m)}±{fmt(s)}"

        if baseline_mean is not None and a.get("test_auroc_mean_mean") is not None:
            delta = a["test_auroc_mean_mean"] - baseline_mean
            delta_str = fmt_delta(delta) if label != "baseline" else "—"
        else:
            delta_str = "—"

        seeds_str = f"({n}/3 seeds)"
        w(f"| `{label}` {seeds_str} | {cfg['pe']} | {cfg['dropout']} | "
          f"{ms('test_auroc_mmr')} | {ms('test_auroc_ras')} | {ms('test_auroc_braf')} | "
          f"{ms('test_auroc_mean')} | {delta_str} |")

    w()

    # ── AUPRC table ────────────────────────────────────────────────────────────
    w("## Test AUPRC — Mean ± Std across Seeds")
    w()
    w("| Config | MMR | RAS | BRAF | Mean |")
    w("|---|---|---|---|---|")
    for cfg in CONFIGS:
        label = cfg["label"]
        a = agg[label]
        def ms2(key):
            m = a.get(f"{key}_mean")
            s = a.get(f"{key}_std")
            if m is None:
                return "—"
            if s is None or s == 0.0:
                return fmt(m)
            return f"{fmt(m)}±{fmt(s)}"
        w(f"| `{label}` | {ms2('test_auprc_mmr')} | {ms2('test_auprc_ras')} | "
          f"{ms2('test_auprc_braf')} | {ms2('test_auprc_mean')} |")
    w()

    # ── Best epoch / val AUROC ─────────────────────────────────────────────────
    w("## Val AUROC + Best Epoch")
    w()
    w("| Config | Val AUROC mean±std | Best epoch mean±std |")
    w("|---|---|---|")
    for cfg in CONFIGS:
        label = cfg["label"]
        a = agg[label]
        def ms3(key):
            m = a.get(f"{key}_mean")
            s = a.get(f"{key}_std")
            if m is None:
                return "—"
            if s is None or s == 0.0:
                return fmt(m)
            return f"{fmt(m)}±{fmt(s)}"
        w(f"| `{label}` | {ms3('best_val_auroc_mean')} | {ms3('best_epoch')} |")
    w()

    # ── Per-seed detail ────────────────────────────────────────────────────────
    w("## Per-Seed Detail")
    w()
    w("| Run name | MMR | RAS | BRAF | Mean | Best epoch |")
    w("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        for seed in SEEDS:
            run_name = f"{cfg['stem']}-s{seed}"
            if run_name not in runs:
                w(f"| `{run_name}` | — | — | — | — | — |")
                continue
            s = runs[run_name]["scalars"]
            w(f"| `{run_name}` | "
              f"{fmt(s.get('test_auroc_mmr'))} | "
              f"{fmt(s.get('test_auroc_ras'))} | "
              f"{fmt(s.get('test_auroc_braf'))} | "
              f"{fmt(s.get('test_auroc_mean'))} | "
              f"{fmt(s.get('best_epoch'), d=0)} |")
    w()

    # ── Gate ──────────────────────────────────────────────────────────────────
    w("## Gate")
    w()
    bm = fmt(gate["baseline_mean"])
    gate_sym = "✅ PASS" if gate["gate_pass"] else "❌ FAIL"
    w(f"**Baseline gate** (mean test AUROC ≥ {PHASE6_MEAN_TEST}): `{bm}` — **{gate_sym}**")
    w()
    if gate["winner"]:
        w = gate["winner"]
        w(f"**Winner:** `{w['label']}` — Δ = {fmt_delta(gate['winner_delta'])} vs baseline, "
          f"CI non-overlapping. "
          f"Carry `positional_encoding={w['pe']}`, `patch_dropout_rate={w['dropout']}` to Phase 9.")
    else:
        w("**No winner** (no config clears Δ > +0.005 with non-overlapping CIs).")
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
    parser = argparse.ArgumentParser(description="Phase 8 ablation report")
    parser.add_argument("--mlflow-db", default="mlflow.db",
                        help="Path to MLflow SQLite DB (default: mlflow.db)")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for CIs (default: 2000)")
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
        agg[cfg["label"]] = aggregate_config(cfg, runs, n_bootstrap=args.n_bootstrap)
        n = agg[cfg["label"]]["n_complete"]
        print(f"  {cfg['label']:30s}  {n}/3 seeds complete")

    gate = evaluate_gate(agg)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_test_auroc_grouped(agg, FIG_DIR / "test_auroc_grouped.png")
    plot_pe_vs_nope(agg, FIG_DIR / "pe_vs_nope.png")
    plot_learning_curves(trajs, FIG_DIR / "learning_curves.png")
    print(f"Figures written → {FIG_DIR}/")

    date = datetime.now().strftime("%Y-%m-%d")
    out_path = REPORT_DIR / f"{date}-phase8-results.md"
    render_report(runs, agg, trajs, gate, args.n_bootstrap, out_path)

    conn.close()

    print()
    print("Gate summary:")
    print(f"  Baseline mean test AUROC: {fmt(gate['baseline_mean'])}")
    print(f"  Gate (≥ {PHASE6_MEAN_TEST}): {'PASS' if gate['gate_pass'] else 'FAIL'}")
    if gate["winner"]:
        w = gate["winner"]
        print(f"  Winner: {w['label']}  Δ={fmt_delta(gate['winner_delta'])}")
    else:
        print("  No winner — carry none/0.0 to Phase 9")


if __name__ == "__main__":
    main()
