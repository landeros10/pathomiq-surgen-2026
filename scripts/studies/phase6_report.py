#!/usr/bin/env python3
"""Phase 6 local report generator.

Reads data extracted by phase6_extract.py (after scp from the GCP server) and
produces the full Phase 6 results report. No MLflow, GPU, or embeddings access
required — everything runs from the downloaded JSON + numpy files.

Usage:
    python scripts/studies/phase6_report.py --data-dir /path/to/downloaded/phase6-data
    python scripts/studies/phase6_report.py --data-dir /path/to/phase6-data --n-bootstrap 1000
"""

import argparse
import glob
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from stability_ablation import compute_stability_metrics  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
TASKS              = ["mmr", "ras", "braf"]
SINGLETASK_RUN     = "singletask-mmr-abmil-cosine-accum16"
ABMIL_PATTERN      = "abmil"
PREVALENCES        = {"mmr": 0.10, "ras": 0.44, "braf": 0.14}
PHASE4_MMR_VAL     = 0.9002
PHASE4_MMR_TEST    = 0.8640
PHASE4_MMR_VL_RISE = 0.021
PHASE5_MEAN_VAL    = 0.821
REPORT_DATE        = datetime.now().strftime("%Y-%m-%d")
OUT_DIR            = ROOT / "reports"
FIGURES_DIR        = OUT_DIR / "figures"
COLORS             = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]

ABLATION_AXES = [
    ("ABMIL vs mean (no PE)",
     ["multitask-abmil-nope-cosine-accum16"],
     ["multitask-mean-cosine-accum16"]),
    ("PE vs no-PE (ABMIL split)",
     ["multitask-abmil-cosine-accum16"],
     ["multitask-abmil-nope-cosine-accum16"]),
    ("split vs joined attn (no PE)",
     ["multitask-abmil-nope-cosine-accum16"],
     ["multitask-abmil-joined-cosine-accum16"]),
    ("split vs joined attn (with PE)",
     ["multitask-abmil-cosine-accum16"],
     ["multitask-abmil-joined-pe-cosine-accum16"]),
    ("PE vs no-PE (joined attn)",
     ["multitask-abmil-joined-pe-cosine-accum16"],
     ["multitask-abmil-joined-cosine-accum16"]),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> dict:
    """Load everything saved by phase6_extract.py."""
    def _load_json(name):
        p = data_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    runs         = _load_json("runs.json") or {}
    trajectories = _load_json("trajectories.json") or {}
    p5_baseline  = _load_json("phase5_baseline.json")

    inference: dict = {}
    infer_dir = data_dir / "inference"
    if infer_dir.exists():
        for run_dir in sorted(infer_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            sid_path = run_dir / "slide_ids.json"
            if not sid_path.exists():
                continue
            slide_ids = json.loads(sid_path.read_text())

            probs:  dict = {}
            labels: dict = {}
            for task in TASKS + ["mmr"]:
                pp = run_dir / f"probs_{task}.npy"
                lp = run_dir / f"labels_{task}.npy"
                if pp.exists() and lp.exists():
                    probs[task]  = np.load(pp)
                    labels[task] = np.load(lp)

            attn_weights: dict = {}
            coords_map:   dict = {}
            attn_dir   = run_dir / "attn"
            coords_dir = run_dir / "coords"
            for i, sid in enumerate(slide_ids):
                tag = f"{i:04d}"
                ap  = attn_dir   / f"{tag}.npy"
                cp  = coords_dir / f"{tag}.npy"
                if ap.exists():
                    attn_weights[sid] = np.load(ap)
                if cp.exists():
                    coords_map[sid] = np.load(cp)

            inference[run_name] = {
                "probs":   probs,
                "labels":  labels,
                "weights": attn_weights,
                "coords":  coords_map,
            }

    return {
        "runs":         runs,
        "trajectories": trajectories,
        "p5_baseline":  p5_baseline,
        "inference":    inference,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT
        ).strip()
    except Exception:
        return "unknown"


def fmt(value, decimals: int = 4) -> str:
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


def is_singletask(run_name: str) -> bool:
    return run_name == SINGLETASK_RUN


# ── Statistical functions ─────────────────────────────────────────────────────

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
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
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


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    if len(labels) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def compute_attn_entropy(weights: np.ndarray) -> float:
    """H = −Σ wᵢ log wᵢ  averaged over tasks if weights is (N, T)."""
    eps = 1e-12
    if weights.ndim == 2:
        return float(np.mean([
            -np.sum(np.clip(weights[:, t], eps, 1.0) * np.log(np.clip(weights[:, t], eps, 1.0)))
            for t in range(weights.shape[1])
        ]))
    w = np.clip(weights.flatten(), eps, 1.0)
    return float(-np.sum(w * np.log(w)))


def weights_to_grid(weights_1d: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    grid = np.full((int(rows.max()) + 1, int(cols.max()) + 1), np.nan)
    for i in range(len(weights_1d)):
        grid[rows[i], cols[i]] = float(weights_1d[i])
    return grid


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_val_auroc_per_task(trajectories: dict, runs: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(runs):
            key  = f"val_auroc_{task}"
            vals = trajectories.get(run_name, {}).get(key, [])
            if not vals:
                continue
            label = run_name.replace("multitask-", "mt-").replace("singletask-", "st-")
            ax.plot(range(1, len(vals) + 1), vals, linewidth=1.4,
                    color=COLORS[i % len(COLORS)], label=label)
        ax.set_title(f"Val AUROC — {task.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val AUROC")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 6 — Val AUROC per Task", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "phase6_val_auroc_per_task.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def plot_val_auroc_mean(trajectories: dict, runs: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, run_name in enumerate(runs):
        key  = "val_auroc" if is_singletask(run_name) else "val_auroc_mean"
        vals = trajectories.get(run_name, {}).get(key, [])
        if not vals:
            continue
        label = run_name.replace("multitask-", "mt-").replace("singletask-", "st-")
        ax.plot(range(1, len(vals) + 1), vals, linewidth=1.5,
                color=COLORS[i % len(COLORS)], label=label)
    ax.axhline(PHASE5_MEAN_VAL, color="red", linewidth=1.0, linestyle="--",
               label=f"Phase 5 gate ({PHASE5_MEAN_VAL})")
    ax.set_title("Phase 6 — Val AUROC Mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "phase6_val_auroc_mean.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def plot_sensitivity(trajectories: dict, runs: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(runs):
            vals = trajectories.get(run_name, {}).get(f"val_sensitivity_{task}", [])
            if not vals:
                continue
            label = run_name.replace("multitask-", "mt-").replace("singletask-", "st-")
            ax.plot(range(1, len(vals) + 1), vals, linewidth=1.4,
                    color=COLORS[i % len(COLORS)], label=label)
        ax.axhline(0.5, color="red", linewidth=0.8, linestyle=":", label="gate=0.50")
        ax.set_title(f"Val Sensitivity — {task.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Sensitivity")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 6 — Val Sensitivity per Task", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "phase6_sensitivity.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def plot_grad_diagnostics(trajectories: dict, runs: dict) -> Optional[Path]:
    PAIRS = [
        ("mmr/ras",  "grad_cos_mmr_ras"),
        ("mmr/braf", "grad_cos_mmr_braf"),
        ("ras/braf", "grad_cos_ras_braf"),
    ]
    multi_runs = [n for n in runs if not is_singletask(n)]
    if not any(trajectories.get(n, {}).get("grad_cos_mmr_ras") for n in multi_runs):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax_idx, (pair_label, key) in enumerate(PAIRS):
        ax = axes[ax_idx]
        for i, run_name in enumerate(multi_runs):
            vals = trajectories.get(run_name, {}).get(key, [])
            if not vals:
                continue
            label = run_name.replace("multitask-", "mt-")
            ax.plot(range(1, len(vals) + 1), vals, linewidth=1.4,
                    color=COLORS[i % len(COLORS)], label=label)
        ax.axhline(0,    color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(-0.5, color="red",   linewidth=0.8, linestyle=":",  alpha=0.7,
                   label="conflict threshold")
        ax.set_title(f"Cos Sim — {pair_label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 6 — Task Gradient Cosine Similarity", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "phase6_grad_cos.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def plot_attn_entropy(entropy_data: dict) -> Optional[Path]:
    if not entropy_data:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(entropy_data) * 1.5), 5))
    names  = list(entropy_data.keys())
    values = [entropy_data[n] for n in names]
    bp = ax.boxplot(values, patch_artist=True,
                    labels=[n.replace("multitask-", "mt-").replace("singletask-", "st-")
                            for n in names])
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Phase 6 — Per-Slide Attention Entropy (ABMIL runs)")
    ax.set_ylabel("H = −Σ wᵢ log wᵢ")
    ax.set_xlabel("Run")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGURES_DIR / "phase6_attn_entropy.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def plot_attention_heatmaps(
    singletask_inf: dict,
    best_multi_inf: dict,
    best_multi_name: str,
    n_slides: int,
) -> list:
    st_weights = singletask_inf.get("weights", {})
    mt_weights = best_multi_inf.get("weights", {})
    st_coords  = singletask_inf.get("coords",  {})
    mt_coords  = best_multi_inf.get("coords",  {})

    common = [
        sid for sid in st_weights
        if sid in mt_weights and sid in st_coords and sid in mt_coords
    ]
    if not common:
        print("  [SKIP] Heatmaps: no common slides with coords.")
        return []

    # Sort by single-task attention entropy (most informative first)
    common_sorted = sorted(common,
                           key=lambda s: compute_attn_entropy(st_weights[s]),
                           reverse=True)[:n_slides]

    n_mt_tasks = mt_weights[common_sorted[0]].shape[-1] if common_sorted else 3
    saved: list = []

    for slide_id in common_sorted:
        st_w = st_weights[slide_id]
        mt_w = mt_weights[slide_id]
        st_c = st_coords[slide_id]
        mt_c = mt_coords[slide_id]

        st_w1d = st_w[:, 0] if st_w.ndim == 2 else st_w.flatten()
        n_panels = 1 + n_mt_tasks

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        grid_st = weights_to_grid(st_w1d, st_c)
        im = axes[0].imshow(grid_st, cmap="hot", aspect="equal",
                            vmin=0, vmax=np.nanmax(grid_st) or 1)
        axes[0].set_title(f"Single-task MMR\n{slide_id}", fontsize=8)
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        for ti, task in enumerate(TASKS[:n_mt_tasks]):
            w_t  = mt_w[:, ti] if mt_w.ndim == 2 else mt_w.flatten()
            grid = weights_to_grid(w_t, mt_c)
            im   = axes[1 + ti].imshow(grid, cmap="hot", aspect="equal",
                                       vmin=0, vmax=np.nanmax(grid) or 1)
            axes[1 + ti].set_title(f"Multitask — {task.upper()}\n{best_multi_name}", fontsize=8)
            axes[1 + ti].axis("off")
            plt.colorbar(im, ax=axes[1 + ti], fraction=0.046, pad=0.04)

        safe_id = slide_id.replace("/", "_").replace(" ", "_")
        fig.suptitle(f"Attention Heatmap — {slide_id}", fontsize=10)
        fig.tight_layout()
        out = FIGURES_DIR / f"phase6_attn_{safe_id}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")
        saved.append(out)

    return saved


# ── Report sections ───────────────────────────────────────────────────────────

def section_header(git_sha: str, runs: dict) -> str:
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_names = ", ".join(f"`{n}`" for n in runs)
    return (
        f"# Phase 6 Results Report\n\n"
        f"**Generated**: {ts}  **Git**: `{git_sha}`\n"
        f"**Runs**: {run_names}\n\n"
        f"---\n\n"
    )


def section_performance_table(runs: dict, p5_baseline) -> str:
    lines = ["## Section 1 — Performance Table\n\n"]
    lines.append(
        "_Phase 5 baseline = `multitask-cosine-accum16` "
        f"(mean val AUROC {PHASE5_MEAN_VAL:.3f}). "
        "Δ = run mean test AUROC − Phase 5 baseline._\n\n"
    )
    lines.append(
        "| Config | Task | Best Val AUROC | Test AUROC | Test AUPRC"
        " | AUPRC baseline | Best Epoch |\n"
        "|--------|------|---------------|------------|------------"
        "|----------------|------------|\n"
    )

    for run_name, run_info in runs.items():
        sc = run_info["scalars"]
        st = is_singletask(run_name)
        for task in TASKS:
            if st and task != "mmr":
                lines.append(f"| {run_name} | {task} | — | — | — | — | — |\n")
                continue
            val_a = sc.get("best_val_auroc") if st else sc.get(f"best_val_auroc_{task}")
            tst_a = sc.get("test_auroc")     if st else sc.get(f"test_auroc_{task}")
            tst_p = sc.get(f"test_auprc_{task}")
            ep    = sc.get("best_epoch")
            lines.append(
                f"| {run_name} | {task} "
                f"| {fmt(val_a)} | {fmt(tst_a)} | {fmt(tst_p)} "
                f"| {fmt(PREVALENCES.get(task), 2)} | {fmt_int(ep)} |\n"
            )

        if not st:
            mean_val   = sc.get("best_val_auroc_mean")
            test_vals  = [sc.get(f"test_auroc_{t}") for t in TASKS]
            mean_test  = (
                float(np.mean([v for v in test_vals if v is not None]))
                if any(v is not None for v in test_vals) else None
            )
            delta = "—"
            if p5_baseline and mean_test is not None:
                p5v = [p5_baseline.get(f"test_auroc_{t}") for t in TASKS]
                p5m = float(np.mean([v for v in p5v if v is not None])) if any(
                    v is not None for v in p5v
                ) else None
                if p5m is not None:
                    delta = f"{mean_test - p5m:+.4f}"
            lines.append(
                f"| **{run_name}** | **mean** "
                f"| {fmt(mean_val)} | {fmt(mean_test)} ({delta}) | — | — "
                f"| {fmt_int(sc.get('best_epoch'))} |\n"
            )
        lines.append("| | | | | | | |\n")

    lines.append(
        "\n_AUPRC baseline = prevalence (train). "
        "Δ requires Phase 5 baseline data from the same MLflow DB._\n\n"
    )
    return "".join(lines)


def section_bootstrap_ci(runs: dict, inference: dict, n_bootstrap: int) -> str:
    lines = ["## Section 2 — Bootstrap 95% CIs (AUROC & AUPRC)\n\n"]
    lines.append(
        f"_Bootstrap N={n_bootstrap} resamples. "
        "Only runs with downloaded inference data shown._\n\n"
        "| Run | Task | AUROC | 95% CI | AUPRC | 95% CI |\n"
        "|-----|------|-------|--------|-------|--------|\n"
    )
    rng = np.random.default_rng(42)
    for run_name, inf in inference.items():
        task_list = ["mmr"] if is_singletask(run_name) else TASKS
        for task in task_list:
            labels = inf["labels"].get(task, np.array([]))
            probs  = inf["probs"].get(task, np.array([]))
            if len(labels) < 10:
                lines.append(f"| {run_name} | {task} | — | — | — | — |\n")
                continue
            auroc = _auroc(labels, probs)
            auprc = _auprc(labels, probs)
            a_lo, a_hi = bootstrap_ci(labels, probs, _auroc, n_bootstrap, rng)
            p_lo, p_hi = bootstrap_ci(labels, probs, _auprc, n_bootstrap, rng)
            lines.append(
                f"| {run_name} | {task} "
                f"| {fmt(auroc)} | [{fmt(a_lo)}, {fmt(a_hi)}] "
                f"| {fmt(auprc)} | [{fmt(p_lo)}, {fmt(p_hi)}] |\n"
            )
        lines.append("| | | | | | |\n")
    lines.append("\n")
    return "".join(lines)


def section_braf_verdict(runs: dict, trajectories: dict) -> str:
    lines = ["## Section 3 — BRAF Sensitivity Recovery Verdict\n\n"]
    lines.append(
        "_Gate: val sensitivity > 0.50 at best epoch (Phase 5 range was 0.30–0.35)._\n\n"
        "| Config | BRAF Sens @ best epoch | Pass/Fail |\n"
        "|--------|------------------------|----------|\n"
    )
    any_pass = False
    for run_name, run_info in runs.items():
        if is_singletask(run_name):
            continue
        sc   = run_info["scalars"]
        traj = trajectories.get(run_name, {})
        be   = sc.get("best_epoch")
        sens = None
        if be is not None:
            series = traj.get("val_sensitivity_braf", [])
            if series:
                idx  = min(max(int(be) - 1, 0), len(series) - 1)
                sens = series[idx]
        verdict = "—"
        if sens is not None:
            verdict = "PASS ✓" if sens > 0.50 else "FAIL ✗"
            if sens > 0.50:
                any_pass = True
        lines.append(f"| {run_name} | {fmt(sens)} | {verdict} |\n")

    lines.append("\n")
    lines.append(
        "**Verdict: PASS.** Per-head class weighting recovers BRAF sensitivity above 0.50.\n\n"
        if any_pass else
        "**Verdict: FAIL.** BRAF sensitivity did not exceed 0.50 in any run.\n\n"
    )
    return "".join(lines)


def section_mmr_regularization(runs: dict, trajectories: dict) -> str:
    lines = ["## Section 4 — MMR Regularization Hypothesis\n\n"]
    lines.append(
        f"_Hypothesis: multitask co-labels reduce MMR overfitting "
        f"(vl_rise < Phase 4 single-task = {PHASE4_MMR_VL_RISE})._\n\n"
        "| Config | MMR vl_rise | Δ vs Phase 4 | Verdict |\n"
        "|--------|------------|--------------|--------|\n"
    )
    lower = 0
    for run_name, run_info in runs.items():
        traj  = trajectories.get(run_name, {})
        stab  = compute_stability_metrics(
            traj.get("val_auroc_mmr", []), traj.get("val_loss_mmr", [])
        )
        vl    = stab.get("val_loss_rise")
        delta = verdict = "—"
        if vl is not None:
            d       = vl - PHASE4_MMR_VL_RISE
            delta   = f"{d:+.3f}"
            verdict = "lower ✓" if vl < PHASE4_MMR_VL_RISE else "higher ✗"
            if vl < PHASE4_MMR_VL_RISE:
                lower += 1
        lines.append(f"| {run_name} | {fmt(vl, 3)} | {delta} | {verdict} |\n")

    lines.append("\n")
    total = len(runs)
    if lower > total // 2:
        lines.append(
            f"**Conclusion: hypothesis supported.** {lower}/{total} runs show lower MMR "
            f"vl_rise than Phase 4 ({PHASE4_MMR_VL_RISE}). "
            "Multitask training reduces MMR overfitting.\n\n"
        )
    elif lower > 0:
        lines.append(
            f"**Conclusion: mixed.** {lower}/{total} runs beat Phase 4 vl_rise. "
            "Regularisation effect is partial.\n\n"
        )
    else:
        lines.append(
            "**Conclusion: hypothesis not supported.** No run beats Phase 4 MMR vl_rise.\n\n"
        )
    return "".join(lines)


def section_ablation_delta(runs: dict) -> tuple:
    lines = ["## Section 5 — Best Config & Ablation Delta\n\n"]

    best_name  = None
    best_score = -1.0
    for rn, ri in runs.items():
        if is_singletask(rn):
            continue
        v = ri["scalars"].get("best_val_auroc_mean")
        if v is not None and float(v) > best_score:
            best_score = float(v)
            best_name  = rn

    if best_name:
        sc       = runs[best_name]["scalars"]
        tv       = [sc.get(f"test_auroc_{t}") for t in TASKS]
        mt       = float(np.mean([v for v in tv if v is not None])) if any(
            v is not None for v in tv
        ) else None
        lines.append(
            f"**Best multitask config**: `{best_name}`  \n"
            f"Mean val AUROC = {fmt(best_score)} | Mean test AUROC = {fmt(mt)}\n\n"
        )
    else:
        lines.append("_Insufficient data to identify best config._\n\n")

    lines.append(
        "### Ablation Delta (marginal mean val AUROC contribution)\n\n"
        "| Axis | With feature | Without feature | Δ mean val AUROC |\n"
        "|------|-------------|-----------------|------------------|\n"
    )
    for axis_label, with_names, without_names in ABLATION_AXES:
        ws  = [runs[n]["scalars"].get("best_val_auroc_mean") for n in with_names    if n in runs]
        wos = [runs[n]["scalars"].get("best_val_auroc_mean") for n in without_names if n in runs]
        wm  = float(np.mean([v for v in ws  if v is not None])) if any(v is not None for v in ws)  else None
        wom = float(np.mean([v for v in wos if v is not None])) if any(v is not None for v in wos) else None
        d   = (f"{wm - wom:+.4f}" if (wm is not None and wom is not None) else "—")
        lines.append(f"| {axis_label} | {fmt(wm)} | {fmt(wom)} | {d} |\n")

    lines.append("\n")
    return "".join(lines), best_name


def section_ece(inference: dict) -> str:
    lines = ["## Section 6 — Expected Calibration Error\n\n"]
    lines.append(
        "_Lower ECE = better calibrated. "
        "Only ABMIL runs with inference data shown._\n\n"
        "| Run | Task | ECE |\n"
        "|-----|------|-----|\n"
    )
    for run_name, inf in inference.items():
        task_list = ["mmr"] if is_singletask(run_name) else TASKS
        for task in task_list:
            labels = inf["labels"].get(task, np.array([]))
            probs  = inf["probs"].get(task, np.array([]))
            ece    = compute_ece(labels, probs) if len(labels) >= 5 else float("nan")
            lines.append(f"| {run_name} | {task} | {fmt(ece)} |\n")
        lines.append("| | | |\n")
    lines.append("\n")
    return "".join(lines)


def section_attention_analysis(heatmap_paths: list) -> str:
    lines = ["## Section 7 — Attention Analysis\n\n"]
    lines.append(
        "Per-patch attention weights from `singletask-mmr-abmil` and the best multitask "
        "ABMIL run, projected onto each slide's patch grid for a matched set of slides.\n\n"
    )
    if not heatmap_paths:
        lines.append(
            "_Heatmaps not generated — inference data was not downloaded "
            "or no slides with coordinates were common to both runs._\n\n"
        )
    else:
        for path in heatmap_paths:
            rel = path.relative_to(OUT_DIR)
            lines.append(f"![{path.stem}]({rel})\n\n")
    lines.append(
        "**Interpretation**: Shifts in attention between single-task and multitask runs "
        "indicate whether co-predicting RAS/BRAF forces the backbone to attend to "
        "different, potentially more biologically meaningful, tissue regions.\n\n"
    )
    return "".join(lines)


def section_attn_entropy(inference: dict, entropy_fig: Optional[Path]) -> tuple:
    lines = ["## Section 8 — Attention Weight Entropy\n\n"]
    lines.append(
        "_H = −Σ wᵢ log wᵢ per slide. "
        "Low entropy = sparse focal attention; high entropy = diffuse weighting._\n\n"
        "| Run | Mean H | Std H | N slides |\n"
        "|-----|--------|-------|----------|\n"
    )
    entropy_data: dict = {}
    for run_name, inf in inference.items():
        weights = inf.get("weights", {})
        if not weights:
            continue
        es = [compute_attn_entropy(w) for w in weights.values()]
        entropy_data[run_name] = es
        lines.append(
            f"| {run_name} | {fmt(np.mean(es))} | {fmt(np.std(es))} | {len(es)} |\n"
        )
    lines.append("\n")
    if entropy_fig is not None:
        lines.append(f"![Attention entropy]({entropy_fig.relative_to(OUT_DIR)})\n\n")
    return "".join(lines), entropy_data


def section_gradient_conflict(runs: dict, trajectories: dict) -> str:
    lines = ["## Section 9 — Task Gradient Conflict\n\n"]
    lines.append(
        "_Epoch-level cosine similarity between task gradient vectors "
        "at the final shared transformer layer. Values near −1 = persistent conflict._\n\n"
        "| Config | Pair | Mean cos | Min cos | Flag |\n"
        "|--------|------|----------|---------|------|\n"
    )
    PAIRS = [
        ("mmr/ras",  "grad_cos_mmr_ras"),
        ("mmr/braf", "grad_cos_mmr_braf"),
        ("ras/braf", "grad_cos_ras_braf"),
    ]
    any_conflict = False
    for run_name, run_info in runs.items():
        if is_singletask(run_name):
            continue
        traj = trajectories.get(run_name, {})
        for pair_label, key in PAIRS:
            series = traj.get(key, [])
            if not series:
                lines.append(f"| {run_name} | {pair_label} | — | — | — |\n")
                continue
            mean_c = float(np.mean(series))
            min_c  = float(np.min(series))
            flag   = "⚠️ conflict" if min_c < -0.5 else ""
            if min_c < -0.5:
                any_conflict = True
            lines.append(
                f"| {run_name} | {pair_label} "
                f"| {fmt(mean_c)} | {fmt(min_c)} | {flag} |\n"
            )

    lines.append("\n")
    lines.append(
        "**Conclusion**: Persistent conflict detected — consider GradNorm in Phase 7.\n\n"
        if any_conflict else
        "**Conclusion**: No persistent conflict. Gradients are orthogonal or cooperative.\n\n"
    )
    return "".join(lines)


def section_gate_checklist(runs: dict, trajectories: dict, best_name: Optional[str]) -> str:
    lines = ["## Section 10 — Phase 6 Gate Checklist\n\n"]
    lines.append(
        "| Gate | Threshold | Actual | Status |\n"
        "|------|-----------|--------|--------|\n"
    )

    # Gate 1: mean val AUROC >= 0.821
    best_mean_val = (
        runs[best_name]["scalars"].get("best_val_auroc_mean")
        if best_name and best_name in runs else None
    )
    g1 = "PASS ✓" if (best_mean_val is not None and best_mean_val >= PHASE5_MEAN_VAL) else "FAIL ✗"
    lines.append(
        f"| ABMIL mean val AUROC ≥ {PHASE5_MEAN_VAL} "
        f"| ≥ {PHASE5_MEAN_VAL} | {fmt(best_mean_val)} | {g1} |\n"
    )

    # Gate 2: BRAF sensitivity > 0.50
    braf_sens = None
    if best_name and best_name in runs and not is_singletask(best_name):
        sc   = runs[best_name]["scalars"]
        traj = trajectories.get(best_name, {})
        be   = sc.get("best_epoch")
        if be is not None:
            series = traj.get("val_sensitivity_braf", [])
            if series:
                idx       = min(max(int(be) - 1, 0), len(series) - 1)
                braf_sens = series[idx]
    g2 = "PASS ✓" if (braf_sens is not None and braf_sens > 0.50) else "FAIL ✗"
    lines.append(f"| BRAF sensitivity > 0.50 | > 0.50 | {fmt(braf_sens)} | {g2} |\n")

    # Gate 3: MMR vl_rise documented
    mmr_vl = None
    if best_name and best_name in runs:
        traj  = trajectories.get(best_name, {})
        stab  = compute_stability_metrics(
            traj.get("val_auroc_mmr", []), traj.get("val_loss_mmr", [])
        )
        mmr_vl = stab.get("val_loss_rise")
    g3 = "PASS ✓" if mmr_vl is not None else "FAIL ✗"
    lines.append(f"| MMR vl_rise documented | documented | {fmt(mmr_vl, 3)} | {g3} |\n")

    # Gate 4: Attention figure generated
    has_attn = len(glob.glob(str(FIGURES_DIR / "phase6_attn_*.png"))) > 0
    g4 = "PASS ✓" if has_attn else "FAIL ✗"
    lines.append(
        f"| Attention figure generated | present | "
        f"{'present' if has_attn else 'absent'} | {g4} |\n"
    )

    lines.append("\n")
    gates = [g1, g2, g3, g4]
    if all(g.startswith("PASS") for g in gates):
        lines.append("**All gates passed — Phase 7 may proceed.**\n\n")
    else:
        n_fail = sum(1 for g in gates if g.startswith("FAIL"))
        lines.append(f"**{n_fail} gate(s) failed — resolve before moving to Phase 7.**\n\n")

    return "".join(lines)


def section_figures(figures: dict) -> str:
    lines = ["## Figures\n\n"]
    for label, path in figures.items():
        if path is None:
            continue
        rel = path.relative_to(OUT_DIR)
        lines.append(f"### {label}\n\n![{label}]({rel})\n\n")
    return "".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 local report generator")
    p.add_argument("--data-dir",    required=True,
                   help="Path to directory downloaded from the GCP server")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-slides",    type=int, default=5,
                   help="Max matched slides for attention heatmaps")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    print(f"Loading data from {data_dir} …")
    data         = load_data(data_dir)
    runs         = data["runs"]
    trajectories = data["trajectories"]
    p5_baseline  = data["p5_baseline"]
    inference    = data["inference"]
    git_sha      = get_git_commit()

    if not runs:
        raise SystemExit("No run data found — check that phase6_extract.py completed successfully.")
    print(f"  {len(runs)} run(s) | {len(inference)} inference result(s)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Identify best multitask ABMIL run ──────────────────────────────────────
    best_multi_name = None
    best_multi_val  = -1.0
    for rn, ri in runs.items():
        if is_singletask(rn) or ABMIL_PATTERN not in rn:
            continue
        v = ri["scalars"].get("best_val_auroc_mean")
        if v is not None and float(v) > best_multi_val:
            best_multi_val  = float(v)
            best_multi_name = rn
    print(f"  Best multitask ABMIL: {best_multi_name} (mean val AUROC {best_multi_val:.4f})")

    # ── Figures ────────────────────────────────────────────────────────────────
    figures: dict = {}
    figures["Figure 1 — Val AUROC per Task"]      = plot_val_auroc_per_task(trajectories, runs)
    figures["Figure 2 — Val AUROC Mean"]           = plot_val_auroc_mean(trajectories, runs)
    figures["Figure 3 — Val Sensitivity per Task"] = plot_sensitivity(trajectories, runs)
    grad_fig = plot_grad_diagnostics(trajectories, runs)
    if grad_fig:
        figures["Figure 4 — Gradient Cosine Similarity"] = grad_fig

    # ── Attention heatmaps ─────────────────────────────────────────────────────
    heatmap_paths: list = []
    st_inf = inference.get(SINGLETASK_RUN)
    mt_inf = inference.get(best_multi_name) if best_multi_name else None
    if st_inf and mt_inf:
        heatmap_paths = plot_attention_heatmaps(
            st_inf, mt_inf, best_multi_name, args.n_slides
        )

    # ── Attention entropy figure ───────────────────────────────────────────────
    entropy_data: dict = {
        rn: [compute_attn_entropy(w) for w in inf["weights"].values()]
        for rn, inf in inference.items()
        if inf.get("weights")
    }
    entropy_fig = plot_attn_entropy(entropy_data)
    if entropy_fig:
        figures["Figure 5 — Attention Entropy"] = entropy_fig

    # ── Report sections ────────────────────────────────────────────────────────
    perf_sec        = section_performance_table(runs, p5_baseline)
    ci_sec          = section_bootstrap_ci(runs, inference, args.n_bootstrap)
    braf_sec        = section_braf_verdict(runs, trajectories)
    mmr_sec         = section_mmr_regularization(runs, trajectories)
    ablation_text, best_multi_name = section_ablation_delta(runs)
    ece_sec         = section_ece(inference)
    attn_sec        = section_attention_analysis(heatmap_paths)
    entropy_text, _ = section_attn_entropy(inference, entropy_fig)
    grad_sec        = section_gradient_conflict(runs, trajectories)
    gate_sec        = section_gate_checklist(runs, trajectories, best_multi_name)
    figs_sec        = section_figures(figures)

    report = (
        section_header(git_sha, runs) +
        perf_sec      +
        ci_sec        +
        braf_sec      +
        mmr_sec       +
        ablation_text +
        ece_sec       +
        attn_sec      +
        entropy_text  +
        grad_sec      +
        gate_sec      +
        figs_sec      +
        "\n---\n*Generated by `scripts/studies/phase6_report.py`*\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{REPORT_DATE}-phase6-results.md"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    main()
