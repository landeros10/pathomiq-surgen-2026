#!/usr/bin/env python3
"""Phase 7 Step 6 — Attention Entropy Analysis.

Fully local. Loads Phase 6 inference attention for all 5 ABMIL runs × ~165 test
slides. Computes per-slide Shannon entropy, plots distributions, runs Mann-Whitney
U test.

Tests hypothesis H6: Multi-task training sharpens attention (lower entropy vs ST).

Usage:
    python scripts/studies/phase7_entropy.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from phase7_utils import (          # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    ALL_ABMIL_RUNS,
    FIGURES_DIR,
    load_study_set,
    load_run_inference,
    load_slide_attn,
    compute_entropy,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ENTROPY_DIR = FIGURES_DIR / "entropy"

RUN_SHORT_LABELS = {
    "singletask-mmr-abmil-cosine-accum16":       "ST",
    "multitask-abmil-nope-cosine-accum16":       "MT-nope",
    "multitask-abmil-cosine-accum16":            "MT-base",
    "multitask-abmil-joined-cosine-accum16":     "MT-joined",
    "multitask-abmil-joined-pe-cosine-accum16":  "MT-joined-pe",
}

RUN_COLORS = {
    "singletask-mmr-abmil-cosine-accum16":       "steelblue",
    "multitask-abmil-nope-cosine-accum16":       "coral",
    "multitask-abmil-cosine-accum16":            "darkorange",
    "multitask-abmil-joined-cosine-accum16":     "seagreen",
    "multitask-abmil-joined-pe-cosine-accum16":  "mediumpurple",
}


# ── Data collection ────────────────────────────────────────────────────────────

def collect_entropy_data(run_name: str) -> dict | None:
    """Load attention for each slide in run_name and compute Shannon entropy.

    Returns dict with keys: run_name, slide_ids, entropies, labels, probs, correct.
    Returns None if the inference directory is missing.
    """
    try:
        inf = load_run_inference(run_name)
    except FileNotFoundError as e:
        print(f"  [WARN] {run_name}: {e}")
        return None

    slide_ids  = inf["slide_ids"]
    n          = len(slide_ids)
    probs_arr  = inf.get("probs_mmr",  np.full(n, float("nan")))
    labels_arr = inf.get("labels_mmr", np.full(n, float("nan")))

    entropies: list[float] = []
    for idx in range(n):
        try:
            attn = load_slide_attn(run_name, idx)
            w1d  = attn[:, 0] if attn.ndim == 2 else attn.flatten()
            entropies.append(compute_entropy(w1d))
        except FileNotFoundError:
            print(f"  [WARN idx={idx}] attn missing: {run_name}")
            entropies.append(float("nan"))

    probs   = np.asarray(probs_arr,  dtype=np.float64)
    labels  = np.asarray(labels_arr, dtype=np.float64)
    valid_p = ~np.isnan(probs) & ~np.isnan(labels)
    correct = np.where(valid_p, (probs > 0.5) == (labels == 1), False).tolist()

    return {
        "run_name":  run_name,
        "slide_ids": slide_ids,
        "entropies": entropies,
        "labels":    labels.tolist(),
        "probs":     probs.tolist(),
        "correct":   correct,
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_mann_whitney(st_entropies: list, mt_entropies: list) -> dict:
    """Mann-Whitney U (two-sided) with effect size r = 1 - 2U/(n1*n2)."""
    from scipy.stats import mannwhitneyu

    st = np.array([e for e in st_entropies if not np.isnan(e)])
    mt = np.array([e for e in mt_entropies if not np.isnan(e)])

    stat, pval = mannwhitneyu(st, mt, alternative="two-sided")
    n1, n2 = len(st), len(mt)
    effect_r = float(1 - 2 * stat / (n1 * n2)) if n1 > 0 and n2 > 0 else float("nan")

    mean_st = float(np.mean(st)) if len(st) else float("nan")
    mean_mt = float(np.mean(mt)) if len(mt) else float("nan")

    supported = (mean_mt < mean_st) and (pval < 0.05)
    verdict   = "SUPPORTED (lower entropy in MT)" if supported else "NOT SUPPORTED"

    return {
        "U":        float(stat),
        "p_value":  float(pval),
        "effect_r": effect_r,
        "n_st":     n1,
        "n_mt":     n2,
        "mean_st":  mean_st,
        "mean_mt":  mean_mt,
        "verdict":  verdict,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_entropy_violin(run_data_all: list, out_dir: Path) -> None:
    """Violin + jitter strip for all 5 runs. No seaborn dependency."""
    valid = [rd for rd in run_data_all if rd is not None]
    if not valid:
        print("  [WARN] No run data for violin plot — skipping")
        return

    violin_data  = []
    labels_list  = []
    colors_list  = []
    positions    = list(range(1, len(valid) + 1))

    for rd in valid:
        ents = np.array([e for e in rd["entropies"] if not np.isnan(e)])
        violin_data.append(ents)
        labels_list.append(RUN_SHORT_LABELS.get(rd["run_name"], rd["run_name"]))
        colors_list.append(RUN_COLORS.get(rd["run_name"], "grey"))

    fig, ax = plt.subplots(figsize=(10, 6))

    parts = ax.violinplot(violin_data, positions=positions,
                          showmedians=True, showextrema=True)

    for pc, col in zip(parts["bodies"], colors_list):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.2)

    # Jitter strip
    rng = np.random.default_rng(42)
    for ents, pos, col in zip(violin_data, positions, colors_list):
        jitter = rng.uniform(-0.08, 0.08, size=len(ents))
        ax.scatter(pos + jitter, ents, s=8, alpha=0.35, color=col, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, fontsize=10)
    ax.set_ylabel("Shannon Entropy H", fontsize=11)
    ax.set_title("Phase 7 Step 6 — Attention Entropy by Run", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    ax.text(0.98, 0.97,
            "H6: MT training → sharper attention (↓ entropy)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec="goldenrod", alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / "entropy_violin.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_entropy_vs_correctness(run_data_all: list, out_dir: Path) -> None:
    """Side-by-side box plots: ST vs MT-joined, correct vs incorrect."""
    rd_st = next((rd for rd in run_data_all if rd and rd["run_name"] == SINGLETASK_RUN), None)
    rd_mt = next((rd for rd in run_data_all if rd and rd["run_name"] == MULTITASK_RUN), None)

    if rd_st is None or rd_mt is None:
        print("  [WARN] ST or MT-joined data missing — skipping entropy_vs_correctness")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, rd, title in zip(axes, [rd_st, rd_mt], ["ST", "MT-joined"]):
        ents    = np.array(rd["entropies"])
        correct = np.array(rd["correct"], dtype=bool)
        valid   = ~np.isnan(ents)

        correct_ents = ents[valid & correct]
        wrong_ents   = ents[valid & ~correct]

        bp = ax.boxplot([correct_ents, wrong_ents],
                        tick_labels=["Correct", "Incorrect"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("salmon")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Shannon Entropy H", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        mean_c = float(np.mean(correct_ents)) if len(correct_ents) else float("nan")
        mean_w = float(np.mean(wrong_ents))   if len(wrong_ents)   else float("nan")
        ax.text(0.98, 0.97,
                f"Correct: {mean_c:.2f}\nWrong: {mean_w:.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    fig.suptitle("Phase 7 Step 6 — Entropy by Prediction Correctness", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "entropy_vs_correctness.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_entropy_by_category(study_set: list, rd_st: dict | None,
                              rd_mt: dict | None, out_dir: Path) -> None:
    """Grouped bars: 4 categories × ST vs MT (study set 19 slides only)."""
    if rd_st is None or rd_mt is None:
        print("  [WARN] ST or MT data missing — skipping entropy_by_category")
        return

    st_idx_map = {sid: i for i, sid in enumerate(rd_st["slide_ids"])}
    mt_idx_map = {sid: i for i, sid in enumerate(rd_mt["slide_ids"])}

    categories = ["correct_msi", "correct_mss", "wrong_msi", "wrong_mss"]
    cat_labels  = ["Correct MSI", "Correct MSS", "Wrong MSI", "Wrong MSS"]
    st_by_cat   = {c: [] for c in categories}
    mt_by_cat   = {c: [] for c in categories}

    for slide in study_set:
        sid = slide["slide_id"]
        cat = slide.get("category", "unknown")
        if cat not in categories:
            continue

        if sid in st_idx_map:
            e = rd_st["entropies"][st_idx_map[sid]]
            if not np.isnan(e):
                st_by_cat[cat].append(e)
        if sid in mt_idx_map:
            e = rd_mt["entropies"][mt_idx_map[sid]]
            if not np.isnan(e):
                mt_by_cat[cat].append(e)

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(categories))
    width = 0.35

    st_means = [float(np.mean(st_by_cat[c])) if st_by_cat[c] else float("nan") for c in categories]
    mt_means = [float(np.mean(mt_by_cat[c])) if mt_by_cat[c] else float("nan") for c in categories]
    st_stds  = [float(np.std(st_by_cat[c]))  if st_by_cat[c] else 0.0          for c in categories]
    mt_stds  = [float(np.std(mt_by_cat[c]))  if mt_by_cat[c] else 0.0          for c in categories]

    ax.bar(x - width / 2, st_means, width, yerr=st_stds, label="ST",
           color="steelblue", alpha=0.8, capsize=4)
    ax.bar(x + width / 2, mt_means, width, yerr=mt_stds, label="MT-joined",
           color="seagreen", alpha=0.8, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Mean Shannon Entropy H", fontsize=11)
    ax.set_title("Phase 7 Step 6 — Entropy by Category (Study Set)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "entropy_by_category.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Summary JSON ──────────────────────────────────────────────────────────────

def build_summary_json(run_data_all: list, mw_result: dict,
                        rd_st: dict | None, rd_mt: dict | None) -> dict:
    """Per-run stats + Mann-Whitney result + entropy-vs-correctness means."""
    per_run: dict = {}
    for rd in run_data_all:
        if rd is None:
            continue
        ents = np.array([e for e in rd["entropies"] if not np.isnan(e)])
        per_run[rd["run_name"]] = {
            "short_label": RUN_SHORT_LABELS.get(rd["run_name"], rd["run_name"]),
            "n":           int(len(ents)),
            "mean":        float(np.mean(ents))   if len(ents) else float("nan"),
            "std":         float(np.std(ents))    if len(ents) else float("nan"),
            "median":      float(np.median(ents)) if len(ents) else float("nan"),
        }

    ent_vs_correct: dict = {}
    for rd, key in [(rd_st, "ST"), (rd_mt, "MT-joined")]:
        if rd is None:
            continue
        ents    = np.array(rd["entropies"])
        correct = np.array(rd["correct"], dtype=bool)
        valid   = ~np.isnan(ents)
        ent_vs_correct[key] = {
            "correct_mean":   float(np.mean(ents[valid & correct]))  if (valid & correct).any()  else float("nan"),
            "incorrect_mean": float(np.mean(ents[valid & ~correct])) if (valid & ~correct).any() else float("nan"),
        }

    return {
        "per_run":            per_run,
        "mann_whitney":       mw_result,
        "entropy_vs_correct": ent_vs_correct,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ENTROPY_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Load study set
    print("Step 1 — loading study set")
    try:
        study_set = load_study_set()
        print(f"  {len(study_set)} slides loaded")
    except FileNotFoundError as e:
        print(f"  [WARN] {e} — category plot will be skipped")
        study_set = []

    # [2] Collect entropy data for all runs
    print("\nStep 2 — collecting entropy data")
    run_data_all: list = []
    for run_name in ALL_ABMIL_RUNS:
        print(f"  {run_name} ...")
        rd = collect_entropy_data(run_name)
        run_data_all.append(rd)
        if rd is not None:
            ents = np.array([e for e in rd["entropies"] if not np.isnan(e)])
            print(f"    n={len(ents)}, mean_H={np.mean(ents):.4f}, std={np.std(ents):.4f}")

    # [3] Violin plot
    print("\nStep 3 — entropy violin plot")
    plot_entropy_violin(run_data_all, ENTROPY_DIR)

    # [4] Mann-Whitney ST vs MT-joined
    print("\nStep 4 — Mann-Whitney U test (ST vs MT-joined)")
    rd_st = next((rd for rd in run_data_all if rd and rd["run_name"] == SINGLETASK_RUN), None)
    rd_mt = next((rd for rd in run_data_all if rd and rd["run_name"] == MULTITASK_RUN), None)

    mw_result: dict = {}
    if rd_st is not None and rd_mt is not None:
        try:
            mw_result = compute_mann_whitney(rd_st["entropies"], rd_mt["entropies"])
            print(f"  U={mw_result['U']:.1f}, p={mw_result['p_value']:.4f}, "
                  f"r={mw_result['effect_r']:.3f}")
            print(f"  mean_ST={mw_result['mean_st']:.4f}, mean_MT={mw_result['mean_mt']:.4f}")
            print(f"  Verdict: {mw_result['verdict']}")
        except Exception as e:
            print(f"  [WARN] Mann-Whitney failed: {e}")
    else:
        print("  [WARN] ST or MT-joined data missing — skipping Mann-Whitney")

    # [5] Entropy vs correctness
    print("\nStep 5 — entropy vs correctness plot")
    plot_entropy_vs_correctness(run_data_all, ENTROPY_DIR)

    # [6] Per-category plot (study set only)
    print("\nStep 6 — entropy by category plot")
    if study_set:
        plot_entropy_by_category(study_set, rd_st, rd_mt, ENTROPY_DIR)
    else:
        print("  [WARN] No study set — skipping")

    # [7] Save summary JSON
    print("\nStep 7 — saving summary JSON")
    summary  = build_summary_json(run_data_all, mw_result, rd_st, rd_mt)
    out_path = ENTROPY_DIR / "entropy_stats.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved {out_path.name}")

    print(f"\nDone. Results in {ENTROPY_DIR}")
    print(f"  entropy_violin.png")
    print(f"  entropy_vs_correctness.png")
    print(f"  entropy_by_category.png")
    print(f"  entropy_stats.json")


if __name__ == "__main__":
    main()
