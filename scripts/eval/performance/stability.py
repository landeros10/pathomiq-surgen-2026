"""Training stability analysis functions — extracted from stability_ablation.py.

Pure function library of stability metrics, chart generation, and report
assembly. Imported by scripts/studies/stability_ablation.py.
"""

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

def get_experiment_id(conn: sqlite3.Connection, name: str) -> "str | None":
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
                                run_uuids: "list[str]") -> "dict[str, list[float]]":
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
    result: "dict[str, list[float]]" = {}
    for uid in run_uuids:
        sub = rows[rows["run_uuid"] == uid].sort_values("step")
        result[uid] = sub["value"].tolist()
    return result


def load_loss_trajectories(conn: sqlite3.Connection,
                           run_uuids: "list[str]") -> "dict[str, dict[str, list[float]]]":
    """Return {run_uuid: {'train_loss': [...], 'val_loss': [...]}} per epoch."""
    if not run_uuids:
        return {}
    placeholders = ",".join("?" * len(run_uuids))
    rows = pd.read_sql(
        f"""
        SELECT run_uuid, key, step, value
        FROM metrics
        WHERE run_uuid IN ({placeholders}) AND key IN ('train_loss', 'val_loss')
        ORDER BY run_uuid, key, step
        """,
        conn, params=run_uuids
    )
    result: "dict[str, dict[str, list[float]]]" = {
        uid: {"train_loss": [], "val_loss": []} for uid in run_uuids
    }
    for uid in run_uuids:
        sub = rows[rows["run_uuid"] == uid]
        for metric_key in ("train_loss", "val_loss"):
            series = sub[sub["key"] == metric_key].sort_values("step")
            result[uid][metric_key] = series["value"].tolist()
    return result


def load_final_scalars(conn: sqlite3.Connection,
                       run_uuids: "list[str]",
                       keys: "list[str]") -> "dict[str, dict[str, float]]":
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
    result: "dict[str, dict[str, float]]" = {uid: {} for uid in run_uuids}
    for _, row in rows.iterrows():
        result[row["run_uuid"]][row["key"]] = row["value"]
    return result


# ── Stability metric computation ───────────────────────────────────────────────

def compute_stability_metrics(val_auroc_series: "list[float]",
                              val_loss_series: "list[float] | None" = None) -> dict:
    """Compute stability metrics from per-epoch trajectories.

    Returns a dict with keys:
      reversals, monotonicity_ratio, max_swing, cv_pct,
      post_peak_drop, auc_lc, convergence_epoch_95,
      val_loss_rise, val_loss_divergence_pct
    All values are None if the series has fewer than 2 points.
    val_loss_rise / val_loss_divergence_pct are None if val_loss_series not provided.
    """
    empty = {
        "reversals": None, "monotonicity_ratio": None, "max_swing": None,
        "cv_pct": None, "post_peak_drop": None, "auc_lc": None,
        "convergence_epoch_95": None,
        "val_loss_rise": None, "val_loss_divergence_pct": None,
    }
    if len(val_auroc_series) < 2:
        return empty

    vals = np.array(val_auroc_series, dtype=float)
    n = len(vals)

    diffs = np.diff(vals)
    reversals = int(np.sum(diffs < 0))
    monotonicity_ratio = float(np.sum(diffs > 0)) / (n - 1)
    max_swing = float(vals.max() - vals.min())

    mean_val = float(vals.mean())
    std_val  = float(vals.std())
    cv_pct   = (std_val / mean_val * 100) if mean_val != 0 else None

    best_val       = float(vals.max())
    final_val      = float(vals[-1])
    post_peak_drop = best_val - final_val
    auc_lc         = float(np.trapezoid(vals) / n)

    threshold = 0.95 * best_val
    convergence_epoch_95 = None
    for i, v in enumerate(vals):
        if v >= threshold:
            convergence_epoch_95 = i + 1
            break

    val_loss_rise = None
    val_loss_divergence_pct = None
    if val_loss_series and len(val_loss_series) >= 2:
        vl    = np.array(val_loss_series, dtype=float)
        min_vl   = float(vl.min())
        final_vl = float(vl[-1])
        val_loss_rise = final_vl - min_vl
        val_loss_divergence_pct = (val_loss_rise / min_vl * 100) if min_vl != 0 else None

    return {
        "reversals": reversals,
        "monotonicity_ratio": monotonicity_ratio,
        "max_swing": max_swing,
        "cv_pct": cv_pct,
        "post_peak_drop": post_peak_drop,
        "auc_lc": auc_lc,
        "convergence_epoch_95": convergence_epoch_95,
        "val_loss_rise": val_loss_rise,
        "val_loss_divergence_pct": val_loss_divergence_pct,
    }


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
    trajectories: "dict[str, list[float]]",
    title: str,
    out_path: Path,
    baseline_auroc: "float | None" = None,
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


def plot_loss_curves(
    auroc_trajectories: "dict[str, list[float]]",
    loss_trajectories: "dict[str, dict[str, list[float]]]",
    out_path: Path,
    title: str = "Round 1 — Learning Curves",
    baseline_auroc: "float | None" = None,
) -> None:
    """Dual-panel figure: val_auroc (top) + train/val loss (bottom)."""
    fig, (ax_auroc, ax_loss) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

    run_names = list(auroc_trajectories.keys())

    for i, name in enumerate(run_names):
        vals = auroc_trajectories.get(name, [])
        if not vals:
            continue
        color = COLORS[i % len(COLORS)]
        epochs = list(range(1, len(vals) + 1))
        ax_auroc.plot(epochs, vals, linewidth=1.2, color=color, label=name)

    if baseline_auroc is not None:
        ax_auroc.axhline(baseline_auroc, color="black", linestyle="--", linewidth=1,
                         label=f"Baseline ({baseline_auroc:.4f})")

    ax_auroc.set_ylabel("Val AUROC")
    ax_auroc.set_title(title)
    ax_auroc.legend(fontsize=6, loc="lower right", ncol=2)
    ax_auroc.grid(True, alpha=0.3)

    for i, name in enumerate(run_names):
        loss_data  = loss_trajectories.get(name, {})
        color      = COLORS[i % len(COLORS)]
        train_loss = loss_data.get("train_loss", [])
        val_loss   = loss_data.get("val_loss", [])
        if train_loss:
            epochs_t = list(range(1, len(train_loss) + 1))
            ax_loss.plot(epochs_t, train_loss, linewidth=1.0, color=color,
                         linestyle="-", alpha=0.85)
        if val_loss:
            epochs_v = list(range(1, len(val_loss) + 1))
            ax_loss.plot(epochs_v, val_loss, linewidth=1.0, color=color,
                         linestyle="--", alpha=0.85)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="grey", linestyle="-",  linewidth=1, label="train_loss"),
        Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="val_loss"),
    ]
    ax_loss.legend(handles=style_handles, fontsize=7, loc="upper right")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / Val Loss")
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Report assembly ───────────────────────────────────────────────────────────

def build_condition_table(rows: "list[dict]") -> str:
    header = (
        "| run_name | group | best_val_auroc | best_epoch | test_auroc"
        " | reversals | mono_ratio | max_swing | cv_% | post_peak_drop | auc_lc | conv_ep95"
        " | vl_rise | vl_div_% |\n"
        "|----------|-------|---------------|------------|------------"
        "|-----------|------------|-----------|------|----------------|--------|----------"
        "|---------|----------|\n"
    )
    lines = [header]
    for r in rows:
        stab = r.get("stability", {}) or {}
        lines.append(
            f"| {r['run_name']} "
            f"| {r.get('group', '—')} "
            f"| {fmt(r.get('best_val_auroc'))} "
            f"| {fmt_int(r.get('best_epoch'))} "
            f"| {fmt(r.get('test_auroc'))} "
            f"| {fmt_int(stab.get('reversals'))} "
            f"| {fmt(stab.get('monotonicity_ratio'), 3)} "
            f"| {fmt(stab.get('max_swing'), 4)} "
            f"| {fmt(stab.get('cv_pct'), 2)} "
            f"| {fmt(stab.get('post_peak_drop'), 4)} "
            f"| {fmt(stab.get('auc_lc'), 4)} "
            f"| {fmt_int(stab.get('convergence_epoch_95'))} "
            f"| {fmt(stab.get('val_loss_rise'), 4)} "
            f"| {fmt(stab.get('val_loss_divergence_pct'), 1)} |\n"
        )
    return "".join(lines)


VL_RISE_THRESHOLD = 0.10


def build_stable_candidates_table(
    rows: "list[dict]",
    loss_trajectories: "dict[str, dict[str, list[float]]]",
    threshold: float = VL_RISE_THRESHOLD,
) -> str:
    stable = [r for r in rows
              if (r.get("stability") or {}).get("val_loss_rise") is not None
              and r["stability"]["val_loss_rise"] <= threshold]
    if not stable:
        return f"_No runs with vl_rise ≤ {threshold}._\n"

    header = (
        "| run_name | best_val_auroc | best_epoch | test_auroc"
        " | mono_ratio | reversals | cv_% | post_peak_drop | auc_lc | conv_ep95"
        " | train_loss@best | val_loss@best | gap@best | vl_rise |\n"
        "|----------|---------------|------------|------------"
        "|------------|-----------|------|----------------|--------|----------"
        "|-----------------|---------------|----------|--------|\n"
    )
    lines = [header]
    for r in sorted(stable, key=lambda r: r.get("best_val_auroc") or 0, reverse=True):
        stab = r.get("stability", {}) or {}
        name = r["run_name"]
        best_ep = r.get("best_epoch")
        loss_data = loss_trajectories.get(name, {})
        train_loss_series = loss_data.get("train_loss", [])
        val_loss_series   = loss_data.get("val_loss", [])
        tl = vl = gap = None
        if best_ep is not None and train_loss_series and val_loss_series:
            idx = min(max(int(best_ep) - 1, 0), len(train_loss_series) - 1)
            tl  = train_loss_series[idx] if idx < len(train_loss_series) else None
            vl  = val_loss_series[idx]   if idx < len(val_loss_series) else None
            gap = (vl - tl) if (tl is not None and vl is not None) else None
        lines.append(
            f"| {name} "
            f"| {fmt(r.get('best_val_auroc'))} "
            f"| {fmt_int(best_ep)} "
            f"| {fmt(r.get('test_auroc'))} "
            f"| {fmt(stab.get('monotonicity_ratio'), 3)} "
            f"| {fmt_int(stab.get('reversals'))} "
            f"| {fmt(stab.get('cv_pct'), 2)} "
            f"| {fmt(stab.get('post_peak_drop'), 4)} "
            f"| {fmt(stab.get('auc_lc'), 4)} "
            f"| {fmt_int(stab.get('convergence_epoch_95'))} "
            f"| {fmt(tl, 4)} "
            f"| {fmt(vl, 4)} "
            f"| {fmt(gap, 4)} "
            f"| {fmt(stab.get('val_loss_rise'), 4)} |\n"
        )
    return "".join(lines)


def build_conclusions(rows: "list[dict]", baseline_auroc: "float | None",
                      loss_trajectories: "dict[str, dict[str, list[float]]] | None") -> str:
    if not rows:
        return "_No runs available to draw conclusions._\n"

    by_auroc         = sorted(rows, key=lambda r: r.get("best_val_auroc") or 0, reverse=True)
    best_auroc_row   = by_auroc[0]
    lines = []

    lines.append(
        f"- **Best val AUROC**: `{best_auroc_row['run_name']}` "
        f"({fmt(best_auroc_row.get('best_val_auroc'))})"
    )

    rows_with_stab = [r for r in rows if r.get("stability") and
                      r["stability"].get("monotonicity_ratio") is not None]
    if rows_with_stab:
        by_stability = sorted(
            rows_with_stab,
            key=lambda r: (
                -(r["stability"].get("monotonicity_ratio") or 0),
                r["stability"].get("reversals") or float("inf"),
            )
        )
        best_stab_row = by_stability[0]
        lines.append("")
        lines.append("### Stability Ranking (monotonicity_ratio DESC, reversals ASC)")
        lines.append("")
        lines.append("| rank | run_name | mono_ratio | reversals | cv_% | post_peak_drop |")
        lines.append("|------|----------|------------|-----------|------|----------------|")
        for rank, r in enumerate(by_stability, 1):
            s = r["stability"]
            lines.append(
                f"| {rank} | `{r['run_name']}` "
                f"| {fmt(s.get('monotonicity_ratio'), 3)} "
                f"| {fmt_int(s.get('reversals'))} "
                f"| {fmt(s.get('cv_pct'), 2)} "
                f"| {fmt(s.get('post_peak_drop'), 4)} |"
            )
        lines.append("")

        if best_stab_row["run_name"] != best_auroc_row["run_name"]:
            lines.append(
                f"- **Top stability pick**: `{best_stab_row['run_name']}` "
                f"(mono_ratio={fmt(best_stab_row['stability'].get('monotonicity_ratio'), 3)}) "
                f"differs from top AUROC pick: `{best_auroc_row['run_name']}` "
                f"(AUROC={fmt(best_auroc_row.get('best_val_auroc'))})"
            )
        else:
            lines.append(
                f"- **Top stability and AUROC both point to**: `{best_auroc_row['run_name']}`"
            )

        if loss_trajectories:
            lines.append("")
            lines.append("### Overfitting Signals")
            lines.append("")
            lines.append(
                "**`vl_rise`** = final_val_loss − min_val_loss. "
                "A large rise means val loss spiraled after its optimum — "
                "the model continued training into overfitting. "
                "**`gap@best`** = val_loss − train_loss at best AUROC epoch."
            )
            lines.append("")
            lines.append("| run_name | best_epoch | train_loss@best | val_loss@best | gap@best | vl_rise | vl_div_% |")
            lines.append("|----------|------------|-----------------|---------------|----------|---------|----------|")
            SPIRAL_THRESHOLD = 0.10
            spiraling_runs = []
            for r in by_auroc:
                name = r["run_name"]
                best_ep = r.get("best_epoch")
                loss_data = loss_trajectories.get(name, {})
                train_loss_series = loss_data.get("train_loss", [])
                val_loss_series   = loss_data.get("val_loss", [])
                stab    = r.get("stability", {}) or {}
                vl_rise = stab.get("val_loss_rise")
                vl_div  = stab.get("val_loss_divergence_pct")
                flag = " ⚠" if (vl_rise is not None and vl_rise > SPIRAL_THRESHOLD) else ""
                if flag:
                    spiraling_runs.append(name)
                if best_ep is not None and train_loss_series and val_loss_series:
                    idx = min(int(best_ep) - 1, len(train_loss_series) - 1)
                    idx = max(idx, 0)
                    tl  = train_loss_series[idx] if idx < len(train_loss_series) else None
                    vl  = val_loss_series[idx]   if idx < len(val_loss_series) else None
                    gap = (vl - tl) if (tl is not None and vl is not None) else None
                    lines.append(
                        f"| `{name}`{flag} | {fmt_int(best_ep)} "
                        f"| {fmt(tl, 4)} | {fmt(vl, 4)} | {fmt(gap, 4)} "
                        f"| {fmt(vl_rise, 4)} | {fmt(vl_div, 1)} |"
                    )
                else:
                    lines.append(
                        f"| `{name}`{flag} | {fmt_int(best_ep)} | — | — | — "
                        f"| {fmt(vl_rise, 4)} | {fmt(vl_div, 1)} |"
                    )
            lines.append("")
            if spiraling_runs:
                lines.append(
                    f"- **Val-loss spiraling** (vl_rise > {SPIRAL_THRESHOLD}): "
                    + ", ".join(f"`{n}`" for n in spiraling_runs)
                    + " — these runs overfit after their best epoch and should not be "
                    "selected even if val AUROC appears competitive."
                )

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

    def avg_auroc(keyword: str) -> "float | None":
        matched = [r.get("best_val_auroc") for r in rows
                   if keyword in r["run_name"].lower() and r.get("best_val_auroc") is not None]
        return float(np.mean(matched)) if matched else None

    cosine_avg   = avg_auroc("cosine")
    no_cosine    = [r.get("best_val_auroc") for r in rows
                    if "cosine" not in r["run_name"].lower() and r.get("best_val_auroc") is not None]
    no_cosine_avg = float(np.mean(no_cosine)) if no_cosine else None

    if cosine_avg is not None and no_cosine_avg is not None:
        diff      = cosine_avg - no_cosine_avg
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
        diff      = weighted_avg - no_weighted_avg
        direction = "helped" if diff > 0 else "did not help"
        lines.append(
            f"- **Class weighting**: {direction} on average "
            f"(weighted avg = {weighted_avg:.4f} vs unweighted avg = {no_weighted_avg:.4f})"
        )

    accum_avg  = avg_auroc("accum")
    no_accum   = [r.get("best_val_auroc") for r in rows
                  if "accum" not in r["run_name"].lower() and r.get("best_val_auroc") is not None]
    no_accum_avg = float(np.mean(no_accum)) if no_accum else None

    if accum_avg is not None and no_accum_avg is not None:
        diff      = accum_avg - no_accum_avg
        direction = "helped" if diff > 0 else "did not help"
        lines.append(
            f"- **Gradient accumulation**: {direction} on average "
            f"(accum avg = {accum_avg:.4f} vs no-accum avg = {no_accum_avg:.4f})"
        )

    lines.append("")
    lines.append("*(Edit this section manually as needed.)*")
    return "\n".join(lines) + "\n"


def build_round2_conclusions(
    all_rows: "list[dict]",
    loss_trajectories: "dict[str, dict[str, list[float]]]",
) -> str:
    r2_rows = [r for r in all_rows if r["group"] == "Round 2"]
    if not r2_rows:
        return "_No Round 2 runs available._\n"

    r1_by_base = {r["run_name"]: r for r in all_rows if r["group"] == "Round 1"}
    lines = []

    best = max(r2_rows, key=lambda r: r.get("best_val_auroc") or 0)
    lines.append(f"- **Best R2 val AUROC**: `{best['run_name']}` ({fmt(best.get('best_val_auroc'))})")

    rows_with_stab = [r for r in r2_rows if (r.get("stability") or {}).get("monotonicity_ratio") is not None]
    if rows_with_stab:
        by_stab = sorted(rows_with_stab,
                         key=lambda r: (-(r["stability"].get("monotonicity_ratio") or 0),
                                         r["stability"].get("reversals") or float("inf")))
        lines += ["", "### Round 2 — Stability Ranking", "",
                  "| rank | run_name | mono_ratio | reversals | cv_% | post_peak_drop |",
                  "|------|----------|------------|-----------|------|----------------|"]
        for rank, r in enumerate(by_stab, 1):
            s = r["stability"]
            lines.append(
                f"| {rank} | `{r['run_name']}` | {fmt(s.get('monotonicity_ratio'), 3)} "
                f"| {fmt_int(s.get('reversals'))} | {fmt(s.get('cv_pct'), 2)} "
                f"| {fmt(s.get('post_peak_drop'), 4)} |"
            )
        lines.append("")

    by_auroc = sorted(r2_rows, key=lambda r: r.get("best_val_auroc") or 0, reverse=True)
    lines += ["", "### Round 2 — Overfitting Signals", "",
              "**`vl_rise`** = final_val_loss − min_val_loss. "
              "A large rise means val loss spiraled after its optimum. "
              "**`gap@best`** = val_loss − train_loss at best AUROC epoch.",
              "",
              "| run_name | best_epoch | train_loss@best | val_loss@best | gap@best | vl_rise | vl_div_% |",
              "|----------|------------|-----------------|---------------|----------|---------|----------|"]
    SPIRAL_THRESHOLD = 0.10
    spiraling_runs = []
    for r in by_auroc:
        name = r["run_name"]
        best_ep = r.get("best_epoch")
        loss_data = loss_trajectories.get(name, {})
        train_loss_series = loss_data.get("train_loss", [])
        val_loss_series   = loss_data.get("val_loss", [])
        stab    = r.get("stability", {}) or {}
        vl_rise = stab.get("val_loss_rise")
        vl_div  = stab.get("val_loss_divergence_pct")
        flag = " ⚠" if (vl_rise is not None and vl_rise > SPIRAL_THRESHOLD) else ""
        if flag:
            spiraling_runs.append(name)
        if best_ep is not None and train_loss_series and val_loss_series:
            idx = min(max(int(best_ep) - 1, 0), len(train_loss_series) - 1)
            tl  = train_loss_series[idx] if idx < len(train_loss_series) else None
            vl  = val_loss_series[idx]   if idx < len(val_loss_series) else None
            gap = (vl - tl) if (tl is not None and vl is not None) else None
            lines.append(
                f"| `{name}`{flag} | {fmt_int(best_ep)} "
                f"| {fmt(tl, 4)} | {fmt(vl, 4)} | {fmt(gap, 4)} "
                f"| {fmt(vl_rise, 4)} | {fmt(vl_div, 1)} |"
            )
        else:
            lines.append(
                f"| `{name}`{flag} | {fmt_int(best_ep)} | — | — | — "
                f"| {fmt(vl_rise, 4)} | {fmt(vl_div, 1)} |"
            )
    lines.append("")
    if spiraling_runs:
        lines.append(
            f"- **Val-loss spiraling** (vl_rise > {SPIRAL_THRESHOLD}): "
            + ", ".join(f"`{n}`" for n in spiraling_runs)
            + " — these runs overfit after their best epoch."
        )

    lines += ["", "### Round 1 → Round 2 Delta (same conditions, more epochs)", "",
              "| condition | r1_val_auroc | r2_val_auroc | Δauroc | r1_vl_rise | r2_vl_rise | Δvl_rise |",
              "|-----------|-------------|-------------|--------|-----------|-----------|---------|"]
    for r2 in sorted(r2_rows, key=lambda r: r.get("best_val_auroc") or 0, reverse=True):
        base    = r2["run_name"].replace("-round_2", "")
        r1      = r1_by_base.get(base)
        r2_auroc = r2.get("best_val_auroc")
        r1_auroc = r1.get("best_val_auroc") if r1 else None
        d_auroc  = (r2_auroc - r1_auroc) if (r2_auroc is not None and r1_auroc is not None) else None
        r2_vl = (r2.get("stability") or {}).get("val_loss_rise")
        r1_vl = (r1.get("stability") or {}).get("val_loss_rise") if r1 else None
        d_vl  = (r2_vl - r1_vl) if (r2_vl is not None and r1_vl is not None) else None
        lines.append(
            f"| {base} | {fmt(r1_auroc)} | {fmt(r2_auroc)} | {fmt(d_auroc)} "
            f"| {fmt(r1_vl, 4)} | {fmt(r2_vl, 4)} | {fmt(d_vl, 4)} |"
        )
    lines.append("")
    lines.append("*(Δ = R2 − R1; negative vl_rise delta = less overfitting)*")
    return "\n".join(lines) + "\n"
