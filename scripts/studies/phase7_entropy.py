#!/usr/bin/env python3
"""Phase 7 Step 6 — Attention Entropy Analysis. (thin wrapper)

All analysis logic lives in scripts/eval/interpretability/entropy.py.

Tests hypothesis H6: Multi-task training sharpens attention (lower entropy vs ST).

Usage:
    python scripts/studies/phase7_entropy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval.interpretability.entropy import (  # noqa: E402
    main,
    compute_run_entropies,
    mann_whitney_entropy,
    plot_entropy_violin,
    plot_entropy_vs_correctness,
    plot_entropy_by_category,
    build_summary_json,
    ENTROPY_DIR,
    RUN_SHORT_LABELS,
    RUN_COLORS,
)
from utils.eval_utils import (  # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    ALL_ABMIL_RUNS,
    DEFAULT_FIGURES_DIR as FIGURES_DIR,
    load_inference as load_run_inference,
    load_attn as load_slide_attn,
    compute_entropy,
    load_study_set,
)

if __name__ == "__main__":
    main()
