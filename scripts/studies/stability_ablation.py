#!/usr/bin/env python3
"""Generate a Markdown + PNG training stability ablation report.

Queries all stability study MLflow runs from a local SQLite DB and merges
with a pre-computed stability CSV.  Gracefully handles partially-complete
or missing run groups.

Usage:
    python scripts/studies/stability_ablation.py
    python scripts/studies/stability_ablation.py \\
        --db mlflow.db \\
        --stability-csv results/stability.csv \\
        --experiment mmr-surgen-stability \\
        --baseline-run mmr-uni-surgen-mean-bce-s1-baseline \\
        --out-dir reports
"""

import argparse
import math
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── SQLite data loading ────────────────────────────────────────────────────────

def get_experiment_id(conn: sqlite3.Connection, name: str) -> str | None:
    row = pd.read_sql(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn, params=(name,)
    )
    if row.empty:
        return None
    return str(row.iloc[0]["experiment_id"])


def load_runs(conn: sqlite3.Connection, exp_id: str) -> pd.DataFrame:
    """Return all FINISHED runs for an experiment."""
    return pd.read_sql(
        """
        SELECT run_uuid, name, start_time
        FROM runs
        WHERE experiment_id = ? AND status = 'FINISHED'
        ORDER BY start_time DESC
        """,
        conn, params=(exp_id,)
    )


def load_val_auroc_trajectories(conn: sqlite3.Connection,
                                run_uuids: list[str]) -> dict[str, list[float]]:
    """Return {run_uuid: [auroc_epoch0, auroc_epoch1, ...]}."""
    if not run_uuids:
        return {}
    placeholders = ",".join("?" * len(run_uuids))
    rows = pd.read_sql(
        f"""
        SELECT run_uuid, step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key = 'val_auroc'
        ORDER BY run_uuid, step
        """,
        conn, params=run_uuids
    )
    result: dict[str, list[float]] = {}
    for uid in run_uuids:
        sub = rows[rows["run_uuid"] == uid].sort_values("step")
        result[uid] = sub["value"].tolist()
    return result


def load_final_scalars(conn: sqlite3.Connection,
                       run_uuids: list[str],
                       keys: list[str]) -> dict[str, dict[str, float]]:
    """Return {run_uuid: {key: value}} for the requested scalar metric keys."""
    if not run_uuids:
        return {}
    placeholders_uuids = ",".join("?" * len(run_uuids))
    placeholders_keys  = ",".join("?" * len(keys))
    rows = pd.read_sql(
        f"""
        SELECT run_uuid, key, value
        FROM metrics
        WHERE run_uuid IN ({placeholders_uuids}) AND key IN ({placeholders_keys})
        """,
        conn, params=run_uuids + keys
    )
    result: dict[str, dict[str, float]] = {uid: {} for uid in run_uuids}
    for _, row in rows.iterrows():
        result[row["run_uuid"]][row["key"]] = row["value"]
    return result


# ── Run grouping ──────────────────────────────────────────────────────────────

def assign_group(run_name: str) -> str:
    n = run_name.lower()
    if "-s2-accum" in n:
        return "S2 Accum Sweep"
    if "-round_2" in n or "-r2-" in n:
        return "Round 2"
    if "-s1-" in n:
        return "Round 1"
    return "Other"


# ── Chart generation ──────────────────────────────────────────────────────────

COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]


def plot_curves(
    trajectories: dict[str, list[float]],  # run_name → epochs
    title: str,
    out_path: Path,
    baseline_auroc: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, vals) in enumerate(trajectories.items()):
        color = COLORS[i % len(COLORS)]
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals, linewidth=1.2, color=color, label=name)

    if baseline_auroc is not None:
        ax.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1,
                   label=f"Phase 3 baseline ({baseline_auroc:.4f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Report assembly ───────────────────────────────────────────────────────────

STABILITY_COLS = ["std", "cv_%", "post_peak_drop", "max_swing", "reversals", "early_peak_ratio"]


def build_condition_table(rows: list[dict], stability_df: pd.DataFrame | None) -> str:
    """Build markdown table sorted by best_val_auroc desc."""
    header = ("| run_name | group | best_val_auroc | best_epoch | test_auroc"
              " | std | cv_% | max_swing | reversals |\n"
              "|----------|-------|---------------|------------|------------"
              "|-----|------|-----------|----------|\n")
    lines = [header]
    for r in rows:
        name = r["run_name"]
        stab: dict = {}
        if stability_df is not None and not stability_df.empty:
            match = stability_df[stability_df["run_name"] == name]
            if not match.empty:
                stab = match.iloc[0].to_dict()

        lines.append(
            f"| {name} "
            f"| {r.get('group', '—')} "
            f"| {fmt(r.get('best_val_auroc'))} "
            f"| {fmt_int(r.get('best_epoch'))} "
            f"| {fmt(r.get('test_auroc'))} "
            f"| {fmt(stab.get('std'), 4)} "
            f"| {fmt(stab.get('cv_%'), 2)} "
            f"| {fmt(stab.get('max_swing'), 4)} "
            f"| {fmt_int(stab.get('reversals'))} |\n"
        )
    return "".join(lines)


def build_conclusions(rows: list[dict], baseline_auroc: float | None,
                      stability_df: pd.DataFrame | None) -> str:
    if not rows:
        return "_No runs available to draw conclusions._\n"

    by_auroc = sorted(rows, key=lambda r: r.get("best_val_auroc") or 0, reverse=True)
    best_auroc_row = by_auroc[0]

    lines = []

    # Best by AUROC
    lines.append(
        f"- **Best val AUROC**: `{best_auroc_row['run_name']}` "
        f"({fmt(best_auroc_row.get('best_val_auroc'))})"
    )

    # Best by stability (cv_%)
    if stability_df is not None and not stability_df.empty:
        merged = []
        for r in rows:
            match = stability_df[stability_df["run_name"] == r["run_name"]]
            if not match.empty:
                merged.append({**r, "cv_%": match.iloc[0].get("cv_%")})
        if merged:
            best_stab = min(merged, key=lambda r: r.get("cv_%") or float("inf"))
            lines.append(
                f"- **Most stable (lowest CV%)**: `{best_stab['run_name']}` "
                f"(CV% = {fmt(best_stab.get('cv_%'), 2)})"
            )

    # Did we match/exceed baseline?
    if baseline_auroc is not None:
        exceeded = [r for r in rows if (r.get("best_val_auroc") or 0) >= baseline_auroc]
        if exceeded:
            lines.append(
                f"- **Matched/exceeded baseline** ({fmt(baseline_auroc)}): "
                + ", ".join(f"`{r['run_name']}`" for r in exceeded[:5])
            )
        else:
            lines.append(
                f"- No condition matched the Phase 3 baseline "
                f"({fmt(baseline_auroc)}) — further tuning needed."
            )

    # Heuristic comparisons (cosine, weighting, accum)
    def avg_auroc(keyword: str) -> float | None:
        matched = [r.get("best_val_auroc") for r in rows
                   if keyword in r["run_name"].lower() and r.get("best_val_auroc") is not None]
        return float(np.mean(matched)) if matched else None

    cosine_avg  = avg_auroc("cosine")
    no_cosine   = [r.get("best_val_auroc") for r in rows
                   if "cosine" not in r["run_name"].lower() and r.get("best_val_auroc") is not None]
    no_cosine_avg = float(np.mean(no_cosine)) if no_cosine else None

    if cosine_avg is not None and no_cosine_avg is not None:
        diff = cosine_avg - no_cosine_avg
        direction = "helped" if diff > 0 else "did not help"
        lines.append(
            f"- **Cosine scheduling**: {direction} on average "
            f"(cosine avg = {cosine_avg:.4f} vs no-cosine avg = {no_cosine_avg:.4f})"
        )

    weighted_avg = avg_auroc("weighted")
    no_weighted  = [r.get("best_val_auroc") for r in rows
                    if "weighted" not in r["run_name"].lower() and r.get("best_val_auroc") is not None]
    no_weighted_avg = float(np.mean(no_weighted)) if no_weighted else None

    if weighted_avg is not None and no_weighted_avg is not None:
        diff = weighted_avg - no_weighted_avg
        direction = "helped" if diff > 0 else "did not help"
        lines.append(
            f"- **Class weighting**: {direction} on average "
            f"(weighted avg = {weighted_avg:.4f} vs unweighted avg = {no_weighted_avg:.4f})"
        )

    accum_avg   = avg_auroc("accum")
    no_accum    = [r.get("best_val_auroc") for r in rows
                   if "accum" not in r["run_name"].lower() and r.get("best_val_auroc") is not None]
    no_accum_avg = float(np.mean(no_accum)) if no_accum else None

    if accum_avg is not None and no_accum_avg is not None:
        diff = accum_avg - no_accum_avg
        direction = "helped" if diff > 0 else "did not help"
        lines.append(
            f"- **Gradient accumulation**: {direction} on average "
            f"(accum avg = {accum_avg:.4f} vs no-accum avg = {no_accum_avg:.4f})"
        )

    lines.append("")
    lines.append("*(Edit this section manually as needed.)*")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Training stability ablation report")
    parser.add_argument("--db",             default="mlflow.db",
                        help="Path to MLflow SQLite DB")
    parser.add_argument("--stability-csv",  default="results/stability.csv",
                        help="Pre-computed stability CSV")
    parser.add_argument("--experiment",     default="mmr-surgen-stability",
                        help="Stability MLflow experiment name")
    parser.add_argument("--baseline-exp",   default="mmr-prediction-baseline",
                        help="Phase 3 baseline experiment name")
    parser.add_argument("--baseline-run",   default="",
                        help="Phase 3 baseline run name (exact match; empty = skip)")
    parser.add_argument("--out-dir",        default="reports",
                        help="Report output directory")
    args = parser.parse_args()

    db_path  = Path(args.db)
    out_dir  = Path(args.out_dir)
    fig_dir  = out_dir / "figures"
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    git_sha  = get_git_commit()

    # ── Connect ────────────────────────────────────────────────────────────────
    if not db_path.exists():
        raise SystemExit(f"MLflow DB not found: {db_path}  (run scripts/mlflow_local.sh first)")
    conn = sqlite3.connect(str(db_path))

    # ── Load stability CSV ─────────────────────────────────────────────────────
    csv_path = Path(args.stability_csv)
    if csv_path.exists():
        stability_df = pd.read_csv(csv_path)
        print(f"Loaded stability CSV: {csv_path} ({len(stability_df)} rows)")
    else:
        stability_df = None
        print(f"WARNING: stability CSV not found at {csv_path} — stability columns will be '—'")

    # ── Load baseline ──────────────────────────────────────────────────────────
    baseline_auroc: float | None = None
    baseline_epoch: int | None = None
    baseline_test_auroc: float | None = None
    baseline_name = args.baseline_run

    if baseline_name:
        bl_exp_id = get_experiment_id(conn, args.baseline_exp)
        if bl_exp_id is None:
            print(f"WARNING: Baseline experiment '{args.baseline_exp}' not found — skipping baseline")
        else:
            bl_runs = pd.read_sql(
                "SELECT run_uuid, name FROM runs WHERE experiment_id = ? AND name = ? AND status = 'FINISHED'",
                conn, params=(bl_exp_id, baseline_name)
            )
            if bl_runs.empty:
                print(f"WARNING: Baseline run '{baseline_name}' not found — skipping baseline")
            else:
                bl_uuid = bl_runs.iloc[0]["run_uuid"]
                bl_scalars = load_final_scalars(conn, [bl_uuid],
                                                ["best_val_auroc", "val_auroc", "best_epoch", "test_auroc"])
                s = bl_scalars[bl_uuid]
                baseline_auroc = s.get("best_val_auroc") or s.get("val_auroc")
                baseline_epoch = s.get("best_epoch")
                baseline_test_auroc = s.get("test_auroc")
                print(f"Baseline: {baseline_name}  val_auroc={baseline_auroc}")

    # ── Load stability experiment runs ─────────────────────────────────────────
    stab_exp_id = get_experiment_id(conn, args.experiment)
    if stab_exp_id is None:
        print(f"WARNING: Experiment '{args.experiment}' not found — no stability runs to report")
        all_runs = pd.DataFrame(columns=["run_uuid", "name", "start_time"])
    else:
        all_runs = load_runs(conn, stab_exp_id)
        print(f"Found {len(all_runs)} FINISHED runs in '{args.experiment}'")

    # ── Load trajectories + scalars ────────────────────────────────────────────
    run_uuids = all_runs["run_uuid"].tolist()
    trajectories_by_uuid = load_val_auroc_trajectories(conn, run_uuids)
    scalars_by_uuid = load_final_scalars(conn, run_uuids,
                                         ["best_val_auroc", "val_auroc", "best_epoch", "test_auroc"])

    # ── Build per-run summary rows ─────────────────────────────────────────────
    summary_rows: list[dict] = []
    for _, run_row in all_runs.iterrows():
        uid  = run_row["run_uuid"]
        name = run_row["name"]
        s    = scalars_by_uuid.get(uid, {})
        best_val = s.get("best_val_auroc") or s.get("val_auroc")
        summary_rows.append({
            "run_uuid":      uid,
            "run_name":      name,
            "group":         assign_group(name),
            "best_val_auroc": best_val,
            "best_epoch":    s.get("best_epoch"),
            "test_auroc":    s.get("test_auroc"),
        })

    summary_rows.sort(key=lambda r: r.get("best_val_auroc") or 0, reverse=True)

    # ── Build trajectory dicts keyed by run_name per group ─────────────────────
    def trajectories_for_group(group: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in summary_rows:
            if r["group"] != group:
                continue
            vals = trajectories_by_uuid.get(r["run_uuid"], [])
            if vals:
                out[r["run_name"]] = vals
        return out

    # ── Generate PNG figures ───────────────────────────────────────────────────
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    group_fig_map = {
        "Round 1":       ("round1_val_auroc.png",  "Round 1 — 8 Conditions"),
        "Round 2":       ("round2_val_auroc.png",  "Round 2 — Top 4"),
        "S2 Accum Sweep": ("s2_val_auroc.png",     "S2 — Accum Sweep"),
    }

    for group, (fname, title) in group_fig_map.items():
        traj = trajectories_for_group(group)
        if not traj:
            print(f"  No trajectory data for group '{group}' — skipping figure")
            continue
        out_path = fig_dir / fname
        plot_curves(traj, title, out_path, baseline_auroc)
        figures[group] = out_path

    # ── Markdown sections ──────────────────────────────────────────────────────

    # Baseline section
    if baseline_name and baseline_auroc is not None:
        baseline_section = f"""## Phase 3 Baseline (reference)

| Metric | Value |
|--------|-------|
| Run name | `{baseline_name}` |
| Val AUROC | {fmt(baseline_auroc)} |
| Test AUROC | {fmt(baseline_test_auroc)} |
| Best epoch | {fmt_int(baseline_epoch)} |

"""
    elif baseline_name:
        baseline_section = f"## Phase 3 Baseline (reference)\n\n_Run `{baseline_name}` not found._\n\n"
    else:
        baseline_section = "## Phase 3 Baseline (reference)\n\n_No baseline run specified (`--baseline-run`)._\n\n"

    # Condition table
    condition_table = build_condition_table(summary_rows, stability_df)

    # Curve sections
    def curve_block(group: str, heading: str) -> str:
        if group not in figures:
            return f"### {heading}\n\n_No data available._\n\n"
        rel = figures[group].relative_to(out_dir)
        return f"### {heading}\n\n![{heading}]({rel})\n\n"

    curves_section = (
        curve_block("Round 1",        "Round 1 — 8 Conditions") +
        curve_block("Round 2",        "Round 2 — Top 4") +
        curve_block("S2 Accum Sweep", "S2 — Accum Sweep")
    )

    # Conclusions
    conclusions = build_conclusions(summary_rows, baseline_auroc, stability_df)

    # ── Assemble full report ───────────────────────────────────────────────────
    report = f"""# Training Stability Ablation Report

**Generated**: {timestamp}  **Git**: `{git_sha}`

---

{baseline_section}## Condition Table (all stability runs)

{condition_table}
## Val AUROC Curves

{curves_section}## Conclusions

{conclusions}
---
*Generated by `scripts/studies/stability_ablation.py`*
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}-training-stability.md"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
