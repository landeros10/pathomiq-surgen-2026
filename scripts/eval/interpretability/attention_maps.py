"""Attention heatmap generation functions — Phase 7 Step 1.

Back-projects saved per-slide attention weights into 2-D spatial heatmaps
and generates paired singletask vs. multitask overlay figures.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import (          # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    DEFAULT_FIGURES_DIR,
    build_attn_grid,
    compute_entropy,
    load_inference,
    load_attn,
    load_coords,
)

# ── Constants ──────────────────────────────────────────────────────────────────
FIGURES_DIR  = DEFAULT_FIGURES_DIR
CATEGORIES   = ["correct_msi", "correct_mss", "wrong_msi", "wrong_mss"]
SAMPLES_PER  = 5
TARGET_TOTAL = 20


# ── Study-set selection ────────────────────────────────────────────────────────

def build_study_set(
    st_info: dict,
    mt_probs: np.ndarray,
    mt_slide_ids: list,
) -> list:
    """Return up to TARGET_TOTAL slide records, balanced across categories.

    Args:
        st_info:      Result of load_inference(SINGLETASK_RUN).
        mt_probs:     (M,) float32 — multitask MMR probs.
        mt_slide_ids: list[str] slide IDs for mt_probs.

    Returns:
        List of dicts (one per selected slide), sorted by category then prob.
    """
    st_ids    = st_info["slide_ids"]
    st_probs  = st_info["probs_mmr"]
    st_labels = st_info["labels_mmr"]

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


# Backward-compatible alias
select_study_set = build_study_set


# ── Heatmap rendering ──────────────────────────────────────────────────────────

def render_heatmap(
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


# Backward-compatible alias
_render_heatmap = render_heatmap


def render_paired_heatmap(
    record: dict,
    mt_info: dict,
    out_dir: "Path | None" = None,
) -> dict:
    """Generate and save heatmap artefacts for one slide.

    Args:
        record:   Slide record dict from build_study_set().
        mt_info:  Result of load_inference(MULTITASK_RUN).
        out_dir:  Output directory; defaults to FIGURES_DIR/heatmaps/{slide_id}.

    Returns:
        dict with keys: st_grid, mt_grid, st_entropy, mt_entropy
    """
    sid   = record["slide_id"]
    idx   = record["idx"]
    label = record["mmr_label"]
    cat   = record["category"]

    if out_dir is None:
        out_dir = FIGURES_DIR / "heatmaps" / sid
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    st_attn   = load_attn(SINGLETASK_RUN, idx)
    st_coords = load_coords(SINGLETASK_RUN, idx)

    mt_idx    = mt_info["slide_ids"].index(sid)
    mt_attn   = load_attn(MULTITASK_RUN, mt_idx)
    mt_coords = load_coords(MULTITASK_RUN, mt_idx)

    st_w1d = st_attn[:, 0] if st_attn.ndim == 2 else st_attn.flatten()
    mt_w1d = mt_attn[:, 0] if mt_attn.ndim == 2 else mt_attn.flatten()

    st_grid = build_attn_grid(st_attn, st_coords, task_idx=0)
    mt_grid = build_attn_grid(mt_attn, mt_coords, task_idx=0)

    st_entropy = compute_entropy(st_w1d)
    mt_entropy = compute_entropy(mt_w1d)

    np.save(out_dir / "singletask_mmr_attn.npy", st_grid)
    np.save(out_dir / "multitask_mmr_attn.npy",  mt_grid)

    for grid, name, title, ent in [
        (st_grid, "singletask_mmr", "Singletask MMR", st_entropy),
        (mt_grid, "multitask_mmr",  "Multitask MMR",  mt_entropy),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        render_heatmap(grid, title, ax, ent, label, cat)
        ax.set_title(f"{name}\n{sid[:30]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    render_heatmap(st_grid, f"Singletask  |  H={st_entropy:.3f}",
                   axes[0], st_entropy, label, cat)
    render_heatmap(mt_grid, f"Multitask   |  H={mt_entropy:.3f}",
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


# Backward-compatible alias used by phase7_heatmap.py
generate_slide_heatmaps = render_paired_heatmap


# ── Gallery ────────────────────────────────────────────────────────────────────

def build_gallery(study_set: list, figures_dir: Path = FIGURES_DIR) -> None:
    """4×5 grid of paired.png thumbnails, one row per category."""
    n_cols = SAMPLES_PER
    n_rows = len(CATEGORIES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 2.5),
    )
    by_cat: dict = {c: [] for c in CATEGORIES}
    for rec in study_set:
        by_cat[rec["category"]].append(rec)

    for r, cat in enumerate(CATEGORIES):
        slides = by_cat[cat]
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(slides):
                sid    = slides[c]["slide_id"]
                paired = figures_dir / "heatmaps" / sid / "paired.png"
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
        axes[r, 0].set_ylabel(cat, fontsize=8, rotation=90, labelpad=4)

    fig.suptitle("Phase 7 Study Set — Attention Heatmap Gallery", fontsize=12)
    fig.tight_layout()
    out_path = figures_dir / "heatmaps" / "gallery.png"
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGallery saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run Phase 7 Step 1 — Attention Heatmap Pipeline."""
    print("=" * 60)
    print("Phase 7 Step 1 — Attention Heatmap Pipeline")
    print("=" * 60)

    for sub in ("heatmaps", "attributions", "uncertainty", "deletion", "clusters"):
        (FIGURES_DIR / sub).mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading inference data …")
    st_info = load_inference(SINGLETASK_RUN)
    mt_info = load_inference(MULTITASK_RUN)

    print(f"  ST slides : {len(st_info['slide_ids'])}")
    print(f"  MT slides : {len(mt_info['slide_ids'])}")

    st_ids_set = set(st_info["slide_ids"])
    mt_ids_set = set(mt_info["slide_ids"])
    common     = st_ids_set & mt_ids_set
    print(f"  Common    : {len(common)}")

    st_mask    = np.array([sid in common for sid in st_info["slide_ids"]])
    st_ids_f   = [s for s, m in zip(st_info["slide_ids"], st_mask) if m]
    st_probs_f = st_info["probs_mmr"][st_mask]
    st_labels_f = st_info["labels_mmr"][st_mask]

    st_filtered = {
        "slide_ids": st_ids_f,
        "probs_mmr": st_probs_f,
        "labels_mmr": st_labels_f,
    }

    print("\n[2] Selecting study set …")
    study_set = build_study_set(st_filtered, mt_info["probs_mmr"], mt_info["slide_ids"])
    print(f"  Total selected: {len(study_set)}")

    ss_path = FIGURES_DIR / "study_set.json"
    ss_path.write_text(json.dumps(study_set, indent=2))
    print(f"\n[3] study_set.json → {ss_path}  ({len(study_set)} slides)")

    print("\n[4] Generating heatmaps …")
    results = {}
    for rec in study_set:
        try:
            r = render_paired_heatmap(rec, mt_info)
            results[rec["slide_id"]] = r
        except Exception as exc:
            print(f"  [WARN] {rec['slide_id']}: {exc}")

    print("\n[5] Building gallery …")
    build_gallery(study_set)

    print("\n" + "=" * 60)
    print(f"Done.  {len(results)}/{len(study_set)} slides processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
