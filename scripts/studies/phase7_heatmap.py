#!/usr/bin/env python3
"""Phase 7 Step 1 — Attention Heatmap Pipeline.

Back-projects saved per-slide attention weights (from Phase 6 inference)
into 2-D spatial heatmaps and generates paired singletask vs. multitask
overlay figures for a canonical 20-slide study set.

No model re-inference required — all data already exists in
  tmp/phase6-report-data/inference/

Outputs:
  reports/figures/phase7/study_set.json
  reports/figures/phase7/heatmaps/<slide_id>/singletask_mmr.png
  reports/figures/phase7/heatmaps/<slide_id>/multitask_mmr.png
  reports/figures/phase7/heatmaps/<slide_id>/paired.png
  reports/figures/phase7/heatmaps/<slide_id>/singletask_mmr_attn.npy
  reports/figures/phase7/heatmaps/<slide_id>/multitask_mmr_attn.npy
  reports/figures/phase7/heatmaps/gallery.png

Usage:
    python scripts/studies/phase7_heatmap.py
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
    FIGURES_DIR,
    build_attn_grid,
    compute_entropy,
    load_run_inference,
    load_slide_attn,
    load_slide_coords,
)

# ── Category definitions ───────────────────────────────────────────────────────
CATEGORIES   = ["correct_msi", "correct_mss", "wrong_msi", "wrong_mss"]
SAMPLES_PER  = 5          # max slides per category
TARGET_TOTAL = 20


# ── Study-set selection ────────────────────────────────────────────────────────

def select_study_set(
    st_info: dict,
    mt_probs: np.ndarray,
    mt_slide_ids: list,
) -> list:
    """Return up to TARGET_TOTAL slide records, balanced across categories.

    Args:
        st_info:      Result of load_run_inference(SINGLETASK_RUN).
        mt_probs:     (M,) float32 — multitask MMR probs (may differ in length/order).
        mt_slide_ids: list[str] slide IDs for mt_probs.

    Returns:
        List of dicts (one per selected slide), sorted by category then prob.
    """
    st_ids    = st_info["slide_ids"]
    st_probs  = st_info["probs_mmr"]
    st_labels = st_info["labels_mmr"]

    # Build multitask prob lookup by slide_id
    mt_prob_map = {sid: float(p) for sid, p in zip(mt_slide_ids, mt_probs)}

    buckets: dict = {c: [] for c in CATEGORIES}

    for idx, (sid, prob, label) in enumerate(zip(st_ids, st_probs, st_labels)):
        pred_pos = float(prob) > 0.5
        lbl_pos  = int(label) == 1

        if lbl_pos and pred_pos:
            cat = "correct_msi"
        elif not lbl_pos and not pred_pos:
            cat = "correct_mss"
        elif lbl_pos and not pred_pos:
            cat = "wrong_msi"
        else:
            cat = "wrong_mss"

        buckets[cat].append({
            "slide_id":            sid,
            "idx":                 idx,
            "category":            cat,
            "mmr_label":           int(label),
            "mmr_prob_singletask": float(prob),
            "mmr_prob_multitask":  mt_prob_map.get(sid, None),
        })

    # Sort each bucket by descending singletask prob (MSI-first) or ascending (MSS-first)
    for cat in CATEGORIES:
        rev = cat in ("correct_msi", "wrong_msi")
        buckets[cat].sort(key=lambda r: r["mmr_prob_singletask"], reverse=rev)

    selected = []
    for cat in CATEGORIES:
        selected.extend(buckets[cat][:SAMPLES_PER])

    print("  Category counts:")
    for cat in CATEGORIES:
        n_avail  = len(buckets[cat])
        n_select = min(n_avail, SAMPLES_PER)
        print(f"    {cat:<20s}: {n_select}/{n_avail}")

    return selected


# ── Heatmap rendering ──────────────────────────────────────────────────────────

def _render_heatmap(
    grid: np.ndarray,
    title: str,
    ax: plt.Axes,
    entropy: float,
    label: int,
    category: str,
) -> None:
    """Draw a single normalised viridis heatmap panel onto *ax*."""
    g_norm = grid / (grid.max() + 1e-12)
    ax.imshow(g_norm, cmap="viridis", origin="upper", vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=9, fontweight="bold")
    info = f"label={'MSI' if label else 'MSS'}  cat={category}\nH={entropy:.3f}"
    ax.set_xlabel(info, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_slide_heatmaps(record: dict) -> dict:
    """Generate and save heatmap artefacts for one slide.

    Returns a dict with keys:
        st_grid, mt_grid, st_entropy, mt_entropy
    """
    sid      = record["slide_id"]
    idx      = record["idx"]
    label    = record["mmr_label"]
    cat      = record["category"]

    out_dir = FIGURES_DIR / "heatmaps" / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load attention + coords ──────────────────────────────────────────────
    st_attn   = load_slide_attn(SINGLETASK_RUN, idx)   # (N, 1)
    st_coords = load_slide_coords(SINGLETASK_RUN, idx) # (N, 2)

    # Multitask: find matching slide index in MT run
    mt_info = _MT_INFO  # loaded once in main()
    mt_idx  = mt_info["slide_ids"].index(sid)
    mt_attn   = load_slide_attn(MULTITASK_RUN, mt_idx)   # (N, 1)
    mt_coords = load_slide_coords(MULTITASK_RUN, mt_idx) # (N, 2)

    # 1-D attention weights
    st_w1d = st_attn[:, 0] if st_attn.ndim == 2 else st_attn.flatten()
    mt_w1d = mt_attn[:, 0] if mt_attn.ndim == 2 else mt_attn.flatten()

    # ── Build grids ──────────────────────────────────────────────────────────
    st_grid = build_attn_grid(st_attn, st_coords, task_idx=0)
    mt_grid = build_attn_grid(mt_attn, mt_coords, task_idx=0)

    st_entropy = compute_entropy(st_w1d)
    mt_entropy = compute_entropy(mt_w1d)

    # ── Save .npy ────────────────────────────────────────────────────────────
    np.save(out_dir / "singletask_mmr_attn.npy", st_grid)
    np.save(out_dir / "multitask_mmr_attn.npy",  mt_grid)

    # ── Individual PNGs ──────────────────────────────────────────────────────
    for grid, name, title, ent in [
        (st_grid, "singletask_mmr", "Singletask MMR", st_entropy),
        (mt_grid, "multitask_mmr",  "Multitask MMR",  mt_entropy),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        _render_heatmap(grid, title, ax, ent, label, cat)
        ax.set_title(f"{name}\n{sid[:30]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Paired figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _render_heatmap(st_grid, f"Singletask  |  H={st_entropy:.3f}",
                    axes[0], st_entropy, label, cat)
    _render_heatmap(mt_grid, f"Multitask   |  H={mt_entropy:.3f}",
                    axes[1], mt_entropy, label, cat)
    fig.suptitle(
        f"{sid}  |  {'MSI' if label else 'MSS'}  |  {cat}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "paired.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"  {sid:<40s}  H_st={st_entropy:.3f}  H_mt={mt_entropy:.3f}")

    return {
        "st_grid":    st_grid,
        "mt_grid":    mt_grid,
        "st_entropy": st_entropy,
        "mt_entropy": mt_entropy,
    }


# ── Gallery ────────────────────────────────────────────────────────────────────

def build_gallery(study_set: list) -> None:
    """4×5 grid of paired.png thumbnails, one row per category."""
    n = len(study_set)
    n_cols = SAMPLES_PER       # up to 5 columns
    n_rows = len(CATEGORIES)   # 4 rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.5),
    )
    # Index slides by category
    by_cat: dict = {c: [] for c in CATEGORIES}
    for rec in study_set:
        by_cat[rec["category"]].append(rec)

    for r, cat in enumerate(CATEGORIES):
        slides = by_cat[cat]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(slides):
                sid    = slides[c]["slide_id"]
                paired = FIGURES_DIR / "heatmaps" / sid / "paired.png"
                if paired.exists():
                    img = plt.imread(str(paired))
                    ax.imshow(img, aspect="auto")
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                prob_st = slides[c]["mmr_prob_singletask"]
                ax.set_title(f"{sid[-20:]}\np={prob_st:.2f}", fontsize=6)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        # Row label
        axes[r, 0].set_ylabel(cat, fontsize=8, rotation=90, labelpad=4)

    fig.suptitle("Phase 7 Study Set — Attention Heatmap Gallery", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "heatmaps" / "gallery.png"
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGallery saved → {out_path}")


# ── Output directory structure ─────────────────────────────────────────────────

def _make_output_dirs() -> None:
    for sub in ("heatmaps", "attributions", "uncertainty", "deletion", "clusters"):
        (FIGURES_DIR / sub).mkdir(parents=True, exist_ok=True)


# ── Module-level MT info cache (set in main) ───────────────────────────────────
_MT_INFO: dict = {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global _MT_INFO

    print("=" * 60)
    print("Phase 7 Step 1 — Attention Heatmap Pipeline")
    print("=" * 60)

    _make_output_dirs()

    # ── 1. Load inference data ───────────────────────────────────────────────
    print("\n[1] Loading inference data …")
    st_info = load_run_inference(SINGLETASK_RUN)
    mt_info = load_run_inference(MULTITASK_RUN)
    _MT_INFO = mt_info

    print(f"  ST slides : {len(st_info['slide_ids'])}")
    print(f"  MT slides : {len(mt_info['slide_ids'])}")

    # Only keep slides present in BOTH runs (with attn files)
    st_ids_set = set(st_info["slide_ids"])
    mt_ids_set = set(mt_info["slide_ids"])
    common     = st_ids_set & mt_ids_set
    print(f"  Common    : {len(common)}")

    # Restrict ST arrays to common slides (rebuild filtered view)
    st_mask   = np.array([sid in common for sid in st_info["slide_ids"]])
    st_ids_f  = [s for s, m in zip(st_info["slide_ids"], st_mask) if m]
    st_probs_f  = st_info["probs_mmr"][st_mask]
    st_labels_f = st_info["labels_mmr"][st_mask]

    st_filtered = {
        "slide_ids": st_ids_f,
        "probs_mmr": st_probs_f,
        "labels_mmr": st_labels_f,
    }

    # For MT indices we use the original mt_info (index needed for file lookup)
    mt_probs_full = mt_info["probs_mmr"]

    # ── 2. Select study set ──────────────────────────────────────────────────
    print("\n[2] Selecting study set …")
    study_set = select_study_set(st_filtered, mt_probs_full, mt_info["slide_ids"])
    print(f"  Total selected: {len(study_set)}")

    # ── 3. Save study_set.json ───────────────────────────────────────────────
    ss_path = FIGURES_DIR / "study_set.json"
    ss_path.write_text(json.dumps(study_set, indent=2))
    print(f"\n[3] study_set.json → {ss_path}  ({len(study_set)} slides)")

    # ── 4. Generate per-slide heatmaps ───────────────────────────────────────
    print("\n[4] Generating heatmaps …")
    results = {}
    for rec in study_set:
        try:
            r = generate_slide_heatmaps(rec)
            results[rec["slide_id"]] = r
        except Exception as exc:
            print(f"  [WARN] {rec['slide_id']}: {exc}")

    # ── 5. Gallery ───────────────────────────────────────────────────────────
    print("\n[5] Building gallery …")
    build_gallery(study_set)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Done.  {len(results)}/{len(study_set)} slides processed.")
    print(f"Output root: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
