"""
Build unified multitask SurGen splits for MMR, RAS, and BRAF tasks.

Outputs 3 CSVs to data/splits/:
  SurGen_multitask_{train,validate,test}.csv

Each has columns: case_id, slide_id, label_mmr, label_ras, label_braf
Labels are binary (0/1) or NaN when no supervision is available for that task.

Also writes reports/2026-03-05-phase5-multitask-dataset.md.

Usage:
    cd /path/to/surgen
    python scripts/etl/build_multitask_splits.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
SPLITS = ["train", "validate", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recode_sr386(df: pd.DataFrame) -> pd.DataFrame:
    """Drop FAIL rows and recode M→1, WT→0. label column becomes int."""
    df = df[df["label"] != "FAIL"].copy()
    df["label"] = df["label"].map({"M": 1, "WT": 0})
    if df["label"].isna().any():
        bad = df[df["label"].isna()]["label"].unique()
        raise ValueError(f"Unexpected label values after recode: {bad}")
    df["label"] = df["label"].astype(int)
    return df


def load_sr386(gene: str, split: str) -> pd.DataFrame:
    """Load SR386_{gene}_{split}.csv and recode labels."""
    path = SPLITS_DIR / f"SR386_{gene}_{split}.csv"
    df = pd.read_csv(path)
    return recode_sr386(df)


def load_sr1482(gene: str, split: str) -> pd.DataFrame:
    """Load SR1482_{split}_{gene}.csv; labels are already 0/1."""
    path = SPLITS_DIR / f"SR1482_{split}_{gene}.csv"
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    return df


def dist_str(series: pd.Series) -> tuple[int, int, int, float]:
    """Return (n_valid, n_pos, n_neg, pct_pos) ignoring NaN."""
    valid = series.dropna()
    n = len(valid)
    n_pos = int((valid == 1).sum())
    n_neg = int((valid == 0).sum())
    pct = 100.0 * n_pos / n if n > 0 else 0.0
    return n, n_pos, n_neg, pct


# ---------------------------------------------------------------------------
# BRAF
# ---------------------------------------------------------------------------

def build_braf_split(split: str) -> pd.DataFrame:
    sr386 = load_sr386("braf", split)
    sr1482 = load_sr1482("braf", split)
    combined = pd.concat([sr386, sr1482], ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    return combined[["case_id", "slide_id", "label"]]


# ---------------------------------------------------------------------------
# RAS (KRAS | NRAS union)
# ---------------------------------------------------------------------------

def build_ras_for_cohort(
    kras_df: pd.DataFrame,
    nras_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive RAS = 1 if KRAS==1 OR NRAS==1, per case_id."""
    kras_map = kras_df.set_index("case_id")["label"].to_dict()
    nras_map = nras_df.set_index("case_id")["label"].to_dict()
    kras_slide = kras_df.set_index("case_id")["slide_id"].to_dict()
    nras_slide = nras_df.set_index("case_id")["slide_id"].to_dict()

    all_cases = sorted(set(kras_map) | set(nras_map))

    rows = []
    for case_id in all_cases:
        k = kras_map.get(case_id)
        n = nras_map.get(case_id)
        ras = 1 if (k == 1 or n == 1) else 0
        slide_id = kras_slide.get(case_id) or nras_slide.get(case_id)
        rows.append({"case_id": case_id, "slide_id": slide_id, "label": ras})

    return pd.DataFrame(rows, columns=["case_id", "slide_id", "label"])


def build_ras_split(split: str) -> pd.DataFrame:
    sr386_kras = load_sr386("kras", split)
    sr386_nras = load_sr386("nras", split)
    sr386_ras = build_ras_for_cohort(sr386_kras, sr386_nras)

    sr1482_kras = load_sr1482("kras", split)
    sr1482_nras = load_sr1482("nras", split)
    sr1482_ras = build_ras_for_cohort(sr1482_kras, sr1482_nras)

    combined = pd.concat([sr386_ras, sr1482_ras], ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    return combined[["case_id", "slide_id", "label"]]


# ---------------------------------------------------------------------------
# Multitask merge
# ---------------------------------------------------------------------------

def build_multitask_split(split: str) -> tuple[pd.DataFrame, dict]:
    """Build one multitask split DataFrame and return cohort counts."""
    # MMR: already unified; columns case_id, slide_id, label
    mmr = pd.read_csv(SPLITS_DIR / f"SurGen_msi_{split}.csv")
    mmr = mmr[["case_id", "slide_id", "label"]].rename(columns={"label": "label_mmr"})

    # BRAF
    braf = build_braf_split(split)[["case_id", "slide_id", "label"]].rename(
        columns={"label": "label_braf", "slide_id": "slide_id_braf"}
    )

    # RAS
    ras = build_ras_split(split)[["case_id", "slide_id", "label"]].rename(
        columns={"label": "label_ras", "slide_id": "slide_id_ras"}
    )

    # Outer join: mmr ← braf ← ras on case_id
    merged = mmr.merge(braf, on="case_id", how="outer")
    merged = merged.merge(ras, on="case_id", how="outer")

    # Resolve slide_id: mmr first, then braf, then ras
    merged["slide_id"] = (
        merged["slide_id"]
        .combine_first(merged["slide_id_braf"])
        .combine_first(merged["slide_id_ras"])
    )
    merged = merged.drop(columns=["slide_id_braf", "slide_id_ras"])

    # Drop rows where all three labels are NaN
    all_nan_mask = (
        merged["label_mmr"].isna()
        & merged["label_braf"].isna()
        & merged["label_ras"].isna()
    )
    merged = merged[~all_nan_mask].reset_index(drop=True)

    # Reorder columns
    merged = merged[["case_id", "slide_id", "label_mmr", "label_ras", "label_braf"]]

    cohort_counts = {
        "mmr": len(mmr),
        "braf": len(braf),
        "ras": len(ras),
        "multitask": len(merged),
    }
    return merged, cohort_counts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_multitask_outputs(dfs: dict[str, pd.DataFrame]) -> None:
    for split, df in dfs.items():
        assert list(df.columns) == ["case_id", "slide_id", "label_mmr", "label_ras", "label_braf"], \
            f"{split}: wrong columns"
        assert not df["case_id"].duplicated().any(), \
            f"{split}: duplicate case_ids"
        for col in ["label_mmr", "label_ras", "label_braf"]:
            valid = df[col].dropna()
            assert valid.isin([0, 1]).all(), \
                f"{split} {col}: contains non-binary values"
        all_nan = (
            df["label_mmr"].isna()
            & df["label_ras"].isna()
            & df["label_braf"].isna()
        )
        assert not all_nan.any(), f"{split}: rows with all labels NaN"

    sets = {s: set(df["case_id"]) for s, df in dfs.items()}
    pairs = [("train", "validate"), ("train", "test"), ("validate", "test")]
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        assert not overlap, f"case_id overlap between {a} and {b}: {overlap}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    dfs: dict[str, pd.DataFrame],
    cohort_counts: dict[str, dict],
) -> str:
    lines = [
        "# Phase 5 Multitask Dataset — MMR, RAS, BRAF",
        "",
        "*Generated: 2026-03-05*",
        "",
        "---",
        "",
        "## Overview",
        "",
        "Unified multitask splits merging MMR, RAS, and BRAF labels per case.",
        "Each row represents one case with up to three binary labels.",
        "Labels are NaN when no supervision is available for that task/case.",
        "",
        "---",
        "",
        "## Class Distribution",
        "",
        "| Task | Split | N (valid) | Positive (1) | Negative (0) | Pos% |",
        "|------|-------|-----------|--------------|--------------|------|",
    ]

    for task_col, task_name in [
        ("label_mmr", "MMR"),
        ("label_ras", "RAS"),
        ("label_braf", "BRAF"),
    ]:
        for split in SPLITS:
            df = dfs[split]
            n, n_pos, n_neg, pct = dist_str(df[task_col])
            lines.append(
                f"| {task_name} | {split} | {n} | {n_pos} | {n_neg} | {pct:.1f}% |"
            )

    lines += [
        "",
        "---",
        "",
        "## Coverage Matrix",
        "",
        "How many cases have 1, 2, or all 3 labels per split.",
        "",
        "| Split | Total cases | 1 label | 2 labels | 3 labels |",
        "|-------|------------|---------|----------|----------|",
    ]

    for split in SPLITS:
        df = dfs[split]
        label_cols = ["label_mmr", "label_ras", "label_braf"]
        n_labels = df[label_cols].notna().sum(axis=1)
        total = len(df)
        c1 = int((n_labels == 1).sum())
        c2 = int((n_labels == 2).sum())
        c3 = int((n_labels == 3).sum())
        lines.append(f"| {split} | {total} | {c1} | {c2} | {c3} |")

    lines += [
        "",
        "---",
        "",
        "## Cohort Contributions",
        "",
        "| Split | MMR cases | BRAF cases | RAS cases | Multitask total |",
        "|-------|-----------|------------|-----------|-----------------|",
    ]

    for split in SPLITS:
        cc = cohort_counts[split]
        lines.append(
            f"| {split} | {cc['mmr']} | {cc['braf']} | {cc['ras']} | {cc['multitask']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Label Sources",
        "",
        "- **MMR**: `SurGen_msi_{split}.csv` (pre-built unified split)",
        "- **BRAF**: SR386 (`M`→1, `WT`→0, `FAIL` dropped) + SR1482 (already binary)",
        "- **RAS**: KRAS|NRAS union applied independently per cohort, then concatenated",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building multitask SurGen splits (MMR + RAS + BRAF) ...\n")

    dfs: dict[str, pd.DataFrame] = {}
    all_cohort_counts: dict[str, dict] = {}

    for split in SPLITS:
        df, cohort_counts = build_multitask_split(split)
        dfs[split] = df
        all_cohort_counts[split] = cohort_counts

        for task_col, task_name in [
            ("label_mmr", "MMR "),
            ("label_ras", "RAS "),
            ("label_braf", "BRAF"),
        ]:
            n, n_pos, n_neg, pct = dist_str(df[task_col])
            print(f"  {task_name} {split:8s}: N={n:4d}  pos={n_pos:3d} ({pct:.1f}%)  neg={n_neg:3d}")
        print(f"  Total cases in split: {len(df)}\n")

    print("Validating outputs ...")
    validate_multitask_outputs(dfs)
    print("All assertions passed.\n")

    for split in SPLITS:
        out = SPLITS_DIR / f"SurGen_multitask_{split}.csv"
        dfs[split].to_csv(out, index=False)
        print(f"Wrote {out.name}")

    print()

    report = build_report(dfs, all_cohort_counts)
    report_path = REPORTS_DIR / "2026-03-05-phase5-multitask-dataset.md"
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path.write_text(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
