#!/usr/bin/env python3
"""Phase 7 Step 1 — Attention Heatmap Pipeline. (thin wrapper)

All analysis logic lives in scripts/eval/interpretability/attention_maps.py.

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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval.interpretability.attention_maps import (  # noqa: E402
    main,
    build_study_set,
    select_study_set,
    render_heatmap,
    render_paired_heatmap,
    generate_slide_heatmaps,
    build_gallery,
    FIGURES_DIR,
    CATEGORIES,
    SAMPLES_PER,
    TARGET_TOTAL,
)
from utils.eval_utils import (  # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    build_attn_grid,
    compute_entropy,
    load_inference as load_run_inference,
    load_attn as load_slide_attn,
    load_coords as load_slide_coords,
)

if __name__ == "__main__":
    main()
