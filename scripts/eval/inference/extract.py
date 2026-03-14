"""Inference extraction functions for Phase 6 and Phase 8.

Provides a unified run_extraction() entry point. The actual extraction
runs on the GCP server via SSH; results are scp'd locally.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_extraction(
    experiment: str,
    config_stems: list,
    seeds: list,
    out_dir: "str | Path",
) -> None:
    """Run model inference and serialize predictions to *out_dir*.

    Args:
        experiment:    MLflow experiment name (e.g. "multitask-surgen-phase8").
        config_stems:  List of config stem strings (e.g. ["config_phase8_baseline"]).
        seeds:         List of integer seeds (e.g. [0, 1, 2]).
        out_dir:       Local output directory for serialized inference data.

    Note:
        For Phase 6 extraction use scripts/studies/phase6_extract.py directly.
        For Phase 8 extraction use scripts/studies/phase8_extract.py directly.
        This function serves as a shared interface for both.
    """
    raise NotImplementedError(
        "run_extraction() is a placeholder. Use phase6_extract.py or "
        "phase8_extract.py on the GCP server for actual extraction."
    )


def extract_phase6(out_dir: "str | Path") -> None:
    """Run Phase 6 inference extraction (delegates to phase6_extract.py)."""
    sys.path.insert(0, str(ROOT / "scripts" / "studies"))
    from phase6_extract import main as _main  # noqa: F401
    _main()


def extract_phase8(out_dir: "str | Path") -> None:
    """Run Phase 8 inference extraction (delegates to phase8_extract.py)."""
    sys.path.insert(0, str(ROOT / "scripts" / "studies"))
    from phase8_extract import main as _main  # noqa: F401
    _main()
