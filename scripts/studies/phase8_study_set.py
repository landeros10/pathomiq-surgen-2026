#!/usr/bin/env python3
"""Phase 8 Interpretability Step 1 — Study Set Curation.

Selects slides for the Phase 8 interpretability analysis from inference outputs,
stratifying by biological hypothesis:

  MMR+/BRAF+  : overlap hypothesis — expect high MMR↔BRAF attention correlation
  MMR+/BRAF-  : MMR-driven without BRAF (control)
  MMR-/RAS+   : RAS-driven — expect low MMR↔RAS correlation
  MMR-/RAS-   : negative baseline

Selects N slides per category, prioritizing high-confidence correct predictions
for clean signal.

Output: reports/figures/phase8/study_set_phase8.json

Schema per slide:
    slide_id, idx, category, label_mmr, label_ras, label_braf,
    prob_mmr, prob_ras, prob_braf, confidence

Usage:
    python scripts/studies/phase8_study_set.py \\
        --data-dir phase8-data/ \\
        --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        [--n-per-category 5]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import load_inference  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
FIGURES_DIR    = ROOT / "reports" / "figures" / "phase8"
STUDY_SET_PATH = FIGURES_DIR / "study_set_phase8.json"
REPORT_PATH    = ROOT / "reports" / "phase8-results.md"

# Biological categories: task → required label (None = don't care)
CATEGORIES = {
    "mmr_pos_braf_pos": {"mmr": 1, "braf": 1, "ras": None},
    "mmr_pos_braf_neg": {"mmr": 1, "braf": 0, "ras": None},
    "mmr_neg_ras_pos":  {"mmr": 0, "braf": None, "ras": 1},
    "mmr_neg_ras_neg":  {"mmr": 0, "braf": None, "ras": 0},
}


# ── Study set selection ────────────────────────────────────────────────────────

def _confidence(rec: dict, cat_def: dict) -> float:
    """Mean predicted probability in the correct direction for specified tasks."""
    scores = []
    for task, req_label in cat_def.items():
        if req_label is None:
            continue
        prob = rec.get(f"prob_{task}")
        if prob is None:
            continue
        scores.append(prob if req_label == 1 else (1.0 - prob))
    return float(np.mean(scores)) if scores else 0.0


def select_slides(inference: dict, n_per_category: int) -> list:
    """Select study set slides from inference outputs."""
    slide_ids   = inference["slide_ids"]
    probs_mmr   = inference.get("probs_mmr",   np.full(len(slide_ids), np.nan))
    labels_mmr  = inference.get("labels_mmr",  np.full(len(slide_ids), np.nan))
    probs_ras   = inference.get("probs_ras",   np.full(len(slide_ids), np.nan))
    labels_ras  = inference.get("labels_ras",  np.full(len(slide_ids), np.nan))
    probs_braf  = inference.get("probs_braf",  np.full(len(slide_ids), np.nan))
    labels_braf = inference.get("labels_braf", np.full(len(slide_ids), np.nan))

    # Build per-slide records
    records = []
    for i, sid in enumerate(slide_ids):
        def _label(arr, idx):
            v = arr[idx]
            return None if (np.isnan(v) if isinstance(v, float) else False) else int(v)

        def _prob(arr, idx):
            v = float(arr[idx])
            return None if np.isnan(v) else round(v, 4)

        records.append({
            "slide_id":   sid,
            "idx":        i,
            "label_mmr":  _label(labels_mmr,  i),
            "label_ras":  _label(labels_ras,   i),
            "label_braf": _label(labels_braf,  i),
            "prob_mmr":   _prob(probs_mmr,   i),
            "prob_ras":   _prob(probs_ras,   i),
            "prob_braf":  _prob(probs_braf,  i),
        })

    study_set = []
    for cat_name, cat_def in CATEGORIES.items():
        candidates = []
        for rec in records:
            # Check label match for all specified tasks
            match = True
            for task, req_label in cat_def.items():
                if req_label is None:
                    continue
                actual = rec[f"label_{task}"]
                if actual is None or actual != req_label:
                    match = False
                    break
            if not match:
                continue
            conf = _confidence(rec, cat_def)
            candidates.append((conf, rec))

        # Sort by confidence descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        for conf, rec in candidates[:n_per_category]:
            entry = dict(rec)
            entry["category"]   = cat_name
            entry["confidence"] = round(conf, 4)
            study_set.append(entry)

    return study_set


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_summary(study_set: list) -> None:
    header = f"  {'Category':<22} {'N':>3}  {'conf':>6}  {'p_mmr':>6}  {'p_ras':>6}  {'p_braf':>7}"
    print(header)
    print("  " + "-" * 60)
    cats = list(CATEGORIES.keys())
    for cat in cats:
        slides = [s for s in study_set if s["category"] == cat]
        if not slides:
            print(f"  {cat:<22} {'0':>3}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}")
            continue
        n         = len(slides)
        mean_conf = float(np.mean([s["confidence"] for s in slides]))
        def _mprob(key):
            vals = [s[key] for s in slides if s.get(key) is not None]
            return f"{np.mean(vals):.3f}" if vals else "—"
        print(f"  {cat:<22} {n:>3}  {mean_conf:>6.3f}  "
              f"{_mprob('prob_mmr'):>6}  {_mprob('prob_ras'):>6}  {_mprob('prob_braf'):>7}")
    print(f"\n  Total: {len(study_set)} slides")


def _append_report(study_set: list, best_run: str, n_per_category: int) -> None:
    """Append Study Set section to phase8-results.md."""
    lines = [
        "",
        "---",
        "",
        "## Interpretability Analysis",
        "",
        "### Study Set",
        "",
        f"Slides were curated from the winning Phase 8 configuration (`{best_run}`) "
        "to test the primary biological hypothesis: BRAF mutation is a frequent driver "
        "of MMR deficiency, so the MMR and BRAF prediction heads should attend to "
        f"overlapping tissue regions. {n_per_category} slides per category were selected, "
        "prioritizing high-confidence correct predictions.",
        "",
        f"| slide_id | category | label_mmr | label_ras | label_braf | prob_mmr | prob_ras | prob_braf |",
        f"|----------|----------|-----------|-----------|------------|----------|----------|-----------|",
    ]
    for s in study_set:
        lines.append(
            f"| {s['slide_id']} | {s['category']} | {s['label_mmr']} | "
            f"{s['label_ras']} | {s['label_braf']} | "
            f"{s['prob_mmr']} | {s['prob_ras']} | {s['prob_braf']} |"
        )
    lines.append("")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Appended Study Set section to {REPORT_PATH.relative_to(ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",        default="reports/data/phase8/",
                   help="Path to extracted phase8 data (default: reports/data/phase8/)")
    p.add_argument("--best-run",        required=True,
                   help="Name of the winning phase8 run")
    p.add_argument("--n-per-category",  type=int, default=5,
                   help="Slides per category (default: 5)")
    p.add_argument("--force",           action="store_true",
                   help="Overwrite existing study_set_phase8.json")
    p.add_argument("--no-report",       action="store_true",
                   help="Skip appending to phase8-results.md")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir      = Path(args.data_dir)
    inference_dir = data_dir / "inference"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if STUDY_SET_PATH.exists() and not args.force:
        print(f"Study set already exists: {STUDY_SET_PATH}")
        print("  Use --force to regenerate.")
        study_set = json.loads(STUDY_SET_PATH.read_text())
        print_summary(study_set)
        return

    print(f"Phase 1 — loading inference for '{args.best_run}'")
    inference  = load_inference(args.best_run, inference_dir)
    n_slides   = len(inference["slide_ids"])
    tasks_have = [t for t in ("mmr", "ras", "braf") if f"probs_{t}" in inference]
    print(f"  {n_slides} slides loaded, tasks: {tasks_have}")

    print(f"\nPhase 2 — selecting {args.n_per_category} slides per category")
    study_set = select_slides(inference, args.n_per_category)
    print(f"  Selected {len(study_set)} total slides")

    print("\nPhase 3 — saving study set")
    STUDY_SET_PATH.write_text(json.dumps(study_set, indent=2))
    print(f"  Saved {STUDY_SET_PATH.relative_to(ROOT)}")

    print_summary(study_set)

    if not args.no_report:
        _append_report(study_set, args.best_run, args.n_per_category)

    print("\nDone.")
    print(f"  Study set: {STUDY_SET_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
