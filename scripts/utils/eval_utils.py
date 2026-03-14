"""General-purpose eval utilities — generalized from phase7_utils.py.

Provides data loading helpers, attention-grid construction, entropy, and
MLflow run-ID lookup. Parameterized paths allow reuse across phases.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

# ── Root & default paths ───────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INFERENCE_DIR = ROOT / "tmp" / "phase6-report-data" / "inference"
DEFAULT_FIGURES_DIR   = ROOT / "reports" / "figures" / "phase7"
DEFAULT_MLFLOW_DB     = ROOT / "mlflow.db"

# ── Phase 6 run constants ──────────────────────────────────────────────────────

SINGLETASK_RUN = "singletask-mmr-abmil-cosine-accum16"
MULTITASK_RUN  = "multitask-abmil-joined-cosine-accum16"   # best Phase 6 run

ALL_ABMIL_RUNS = [
    "singletask-mmr-abmil-cosine-accum16",
    "multitask-abmil-nope-cosine-accum16",
    "multitask-abmil-cosine-accum16",
    "multitask-abmil-joined-cosine-accum16",
    "multitask-abmil-joined-pe-cosine-accum16",
]

TASKS = ["mmr", "ras", "braf"]


# ── Inference data loading ─────────────────────────────────────────────────────

def load_inference(
    run_name: str,
    inference_dir: Path = DEFAULT_INFERENCE_DIR,
) -> dict:
    """Load slide_ids, probs_{task}, labels_{task} for a run.

    Returns a dict with keys:
        slide_ids  : list[str]  (length N)
        probs_mmr  : np.ndarray (N,) float32  — always present
        labels_mmr : np.ndarray (N,) float32  — always present
        probs_ras, labels_ras, probs_braf, labels_braf  — if present
    """
    run_dir = Path(inference_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Inference dir not found: {run_dir}")

    result: dict = {}
    result["slide_ids"] = json.loads((run_dir / "slide_ids.json").read_text())

    for task in TASKS:
        for kind in ("probs", "labels"):
            p = run_dir / f"{kind}_{task}.npy"
            if p.exists():
                result[f"{kind}_{task}"] = np.load(p)

    return result


# Backward-compatible alias for phase7 scripts
load_run_inference = load_inference


def load_attn(
    run_name: str,
    idx: int,
    inference_dir: Path = DEFAULT_INFERENCE_DIR,
) -> np.ndarray:
    """Load (N, T) or (N, 1) attention array for slide index *idx*."""
    path = Path(inference_dir) / run_name / "attn" / f"{idx:04d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Attn file not found: {path}")
    return np.load(path)


# Backward-compatible alias
load_slide_attn = load_attn


def load_coords(
    run_name: str,
    idx: int,
    inference_dir: Path = DEFAULT_INFERENCE_DIR,
) -> np.ndarray:
    """Load (N, 2) int64 pixel coords for slide index *idx*.

    coords[:, 0] = X (horizontal / column direction)
    coords[:, 1] = Y (vertical   / row    direction)
    """
    path = Path(inference_dir) / run_name / "coords" / f"{idx:04d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Coords file not found: {path}")
    return np.load(path)


# Backward-compatible alias
load_slide_coords = load_coords


# ── Study-set I/O ──────────────────────────────────────────────────────────────

def load_study_set(figures_dir: Path = DEFAULT_FIGURES_DIR) -> list:
    """Load canonical study_set.json → list of slide record dicts."""
    path = Path(figures_dir) / "study_set.json"
    if not path.exists():
        raise FileNotFoundError(
            f"study_set.json not found at {path}. Run phase7_heatmap.py first."
        )
    return json.loads(path.read_text())


# ── MLflow run-ID lookup ───────────────────────────────────────────────────────

def mlflow_run_id(
    run_name: str,
    mlflow_db: Path = DEFAULT_MLFLOW_DB,
) -> str:
    """Return the MLflow run_uuid for *run_name*.

    Queries the local mlflow.db SQLite database.

    Raises:
        FileNotFoundError  if mlflow.db does not exist.
        KeyError           if no run matches *run_name*.
    """
    mlflow_db = Path(mlflow_db)
    if not mlflow_db.exists():
        raise FileNotFoundError(f"MLflow DB not found: {mlflow_db}")

    conn = sqlite3.connect(str(mlflow_db))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.run_uuid
            FROM   runs r
            JOIN   tags t ON r.run_uuid = t.run_uuid
                          AND t.key = 'mlflow.runName'
            WHERE  t.value = ?
              AND  r.lifecycle_stage = 'active'
            ORDER  BY r.start_time DESC
            LIMIT  1
            """,
            (run_name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise KeyError(f"No active MLflow run found with name '{run_name}'")
    return row[0]


# ── Attention grid construction ────────────────────────────────────────────────

def _estimate_stride(vals: np.ndarray) -> int:
    """Estimate patch stride (pixels) from a 1-D array of pixel positions."""
    u = np.unique(vals)
    if len(u) < 2:
        return 1
    diffs = np.diff(u)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    counts = np.bincount(diffs.astype(np.int64))
    return int(np.argmax(counts))


def build_attn_grid(
    attn_weights: np.ndarray,
    coords: np.ndarray,
    task_idx: int = 0,
) -> np.ndarray:
    """Scatter per-patch attention weights into a 2-D spatial grid.

    Args:
        attn_weights: (N, T) or (N,) float32 attention values.
        coords:       (N, 2) int64 pixel coords — col X = [:,0], row Y = [:,1].
        task_idx:     Which task column to use when attn is 2-D.

    Returns:
        (H, W) float32 grid; unoccupied cells are zero.
    """
    if attn_weights.ndim == 2:
        w = attn_weights[:, task_idx].astype(np.float32)
    else:
        w = attn_weights.astype(np.float32)

    x_px = coords[:, 0].astype(np.int64)
    y_px = coords[:, 1].astype(np.int64)

    stride_x = _estimate_stride(x_px)
    stride_y = _estimate_stride(y_px)
    stride   = max(stride_x, stride_y, 1)

    x_min, y_min = int(x_px.min()), int(y_px.min())
    col_idx = np.round((x_px - x_min) / stride).astype(np.int64)
    row_idx = np.round((y_px - y_min) / stride).astype(np.int64)

    H = int(row_idx.max()) + 1
    W = int(col_idx.max()) + 1

    grid = np.zeros((H, W), dtype=np.float32)
    grid[row_idx, col_idx] = w
    return grid


# ── Entropy ────────────────────────────────────────────────────────────────────

def compute_entropy(attn_1d: np.ndarray) -> float:
    """Shannon entropy H = -Σ w·log(w+ε) for a 1-D attention distribution."""
    eps = 1e-12
    w = np.clip(attn_1d.flatten(), eps, 1.0)
    return float(-np.sum(w * np.log(w)))
