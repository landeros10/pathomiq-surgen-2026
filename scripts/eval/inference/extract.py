"""Inference extraction interface stub.

Actual extraction runs on the GCP server via the orchestrator scripts:
    scripts/orchestrators/phase6_extract.py
    scripts/orchestrators/phase8_extract.py
"""

from pathlib import Path


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
        Run the appropriate orchestrator script on the GCP server for actual
        extraction (scripts/orchestrators/phase6_extract.py or phase8_extract.py),
        then scp the output locally.
    """
    raise NotImplementedError(
        "run_extraction() is a placeholder. Use scripts/orchestrators/phase6_extract.py "
        "or phase8_extract.py on the GCP server for actual extraction."
    )
