"""Inference extraction — queries MLflow, loads models, serializes predictions.

Output layout:
  <out_dir>/
    runs.json                     # {run_name: {run_id, params, scalars}}
    trajectories.json             # {run_name: {metric: [...]}}
    inference/
      <run_name>/
        slide_ids.json
        probs_{task}.npy   labels_{task}.npy
        ...
"""

import json
import math
import sqlite3
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

_DEFAULT_TASKS    = ["mmr", "ras", "braf"]
_DEFAULT_TEST_CSV = ROOT / "data" / "splits" / "SurGen_multitask_test.csv"


# ── JSON helpers ──────────────────────────────────────────────────────────────

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if math.isnan(float(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, cls=_Encoder))


# ── MLflow / SQLite helpers ───────────────────────────────────────────────────

def _get_exp_id(conn: sqlite3.Connection, name: str):
    row = pd.read_sql(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn, params=(name,)
    )
    return None if row.empty else str(row.iloc[0]["experiment_id"])


def _load_runs(conn: sqlite3.Connection, exp_id: str, tasks: list) -> dict:
    """Return {run_name: {run_id, params, scalars}} for FINISHED non-preflight runs."""
    runs_df = pd.read_sql(
        """
        SELECT run_uuid, name
        FROM runs
        WHERE experiment_id = ? AND status = 'FINISHED'
        ORDER BY start_time ASC
        """,
        conn, params=(exp_id,)
    )
    runs_df = runs_df[~runs_df["name"].str.contains("preflight", case=False, na=False)]

    result: dict = {}
    for _, row in runs_df.iterrows():
        result[row["name"]] = {"run_id": row["run_uuid"], "params": {}, "scalars": {}}

    if not result:
        return result

    run_uuids   = [v["run_id"] for v in result.values()]
    uid_to_name = {v["run_id"]: k for k, v in result.items()}
    ph          = ",".join("?" * len(run_uuids))

    params_df = pd.read_sql(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({ph})",
        conn, params=run_uuids
    )
    for _, row in params_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["params"][row["key"]] = row["value"]

    scalar_keys = (
        ["best_val_auroc_mean", "best_val_auprc_mean", "best_epoch",
         "test_auroc_mean", "test_auprc_mean"] +
        [f"best_val_auroc_{t}" for t in tasks] +
        [f"best_val_auprc_{t}" for t in tasks] +
        [f"test_auroc_{t}"     for t in tasks] +
        [f"test_auprc_{t}"     for t in tasks]
    )
    ph_k = ",".join("?" * len(scalar_keys))
    scalars_df = pd.read_sql(
        f"""
        SELECT run_uuid, key, MAX(step) as step, value
        FROM metrics
        WHERE run_uuid IN ({ph}) AND key IN ({ph_k})
        GROUP BY run_uuid, key
        """,
        conn, params=run_uuids + scalar_keys
    )
    for _, row in scalars_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["scalars"][row["key"]] = float(row["value"])

    return result


def _load_trajectories(conn: sqlite3.Connection, runs: dict) -> dict:
    """Return {run_name: {metric: [val, ...]}} per epoch."""
    if not runs:
        return {}

    run_uuids   = [v["run_id"] for v in runs.values()]
    uid_to_name = {v["run_id"]: k for k, v in runs.items()}
    ph          = ",".join("?" * len(run_uuids))
    traj_keys   = ["val_auroc_mean", "train_loss", "val_loss"]
    ph_k        = ",".join("?" * len(traj_keys))

    df = pd.read_sql(
        f"""
        SELECT run_uuid, key, step, value
        FROM metrics
        WHERE run_uuid IN ({ph}) AND key IN ({ph_k})
        ORDER BY run_uuid, key, step
        """,
        conn, params=run_uuids + traj_keys
    )

    result: dict = {}
    for name, run_info in runs.items():
        uid = run_info["run_id"]
        sub = df[df["run_uuid"] == uid]
        result[name] = {}
        for key in traj_keys:
            series = sub[sub["key"] == key].sort_values("step")["value"].tolist()
            result[name][key] = series
    return result


# ── Model inference ───────────────────────────────────────────────────────────

def _load_model(run_id: str, run_name: str, mlflow_db: Path):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
        model.eval()
        return model
    except Exception as exc:
        print(f"  [WARN] Cannot load model for {run_name}: {exc}")
        return None


def _run_inference(model, embeddings_dir: Path, tasks: list, split_csv: Path, device: str) -> dict:
    """Run inference on full test split. Returns probs, labels, slide_ids."""
    import torch
    from etl.dataset import MultitaskMILDataset, mil_collate_fn
    from torch.utils.data import DataLoader

    ds = MultitaskMILDataset(
        split_csv=str(split_csv),
        embeddings_dir=str(embeddings_dir),
        tasks=tasks,
        slide_id_col="case_id",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    all_probs:  dict = {t: [] for t in tasks}
    all_labels: dict = {t: [] for t in tasks}
    slide_ids:  list = []

    model.to(device)
    model.eval()

    amp_enabled = (device == "cuda")
    with torch.no_grad():
        for batch in loader:
            emb, labels_t, mask_t, batch_slide_ids, coords = batch
            emb = emb.to(device)
            coords_gpu = (
                coords.to(device)
                if (coords is not None and not isinstance(coords, list))
                else None
            )
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(emb, coords=coords_gpu)
            probs = torch.sigmoid(logits.float()).cpu().numpy()

            slide_id = batch_slide_ids[0] if isinstance(batch_slide_ids, list) else batch_slide_ids
            slide_ids.append(slide_id)

            for ti, task in enumerate(tasks):
                lab  = float(labels_t[0, ti].item())
                msk  = float(mask_t[0, ti].item())
                prob = float(probs[0, ti]) if probs.ndim > 1 else float(probs.flat[0])
                if msk > 0.5:
                    all_probs[task].append(prob)
                    all_labels[task].append(lab)

    return {
        "probs":     {t: np.array(all_probs[t],  dtype=np.float32) for t in tasks},
        "labels":    {t: np.array(all_labels[t], dtype=np.float32) for t in tasks},
        "slide_ids": slide_ids,
    }


def _save_inference(result: dict, out_dir: Path, tasks: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(result["slide_ids"], out_dir / "slide_ids.json")
    for task in tasks:
        np.save(out_dir / f"probs_{task}.npy",  result["probs"][task])
        np.save(out_dir / f"labels_{task}.npy", result["labels"][task])


# ── Public interface ──────────────────────────────────────────────────────────

def run_extraction(
    experiment: str,
    config_stems: list,
    seeds: list,
    out_dir: "str | Path",
    mlflow_db: str = "mlflow.db",
    embeddings_dir=None,
    tasks: list = None,
) -> None:
    """Run model inference and serialize predictions to *out_dir*.

    Queries *experiment* in the MLflow SQLite database, filters to runs matching
    ``f"{cfg}-s{seed}"`` for each (cfg, seed) pair, loads each model checkpoint,
    runs inference on the test split, and writes per-run prediction arrays.

    Args:
        experiment:    MLflow experiment name (e.g. "multitask-surgen-phase8").
        config_stems:  List of config stem strings (e.g. ["config_phase8_baseline"]).
        seeds:         List of integer seeds (e.g. [0, 1, 2]).
        out_dir:       Local output directory for serialized inference data.
        mlflow_db:     Path to MLflow SQLite DB (default: "mlflow.db").
        embeddings_dir: Path to embeddings directory (zarr store). Required for inference.
        tasks:         Task names to extract (default: ["mmr", "ras", "braf"]).
    """
    import torch

    if tasks is None:
        tasks = _DEFAULT_TASKS

    out_dir   = Path(out_dir)
    mlflow_db = Path(mlflow_db)
    split_csv = _DEFAULT_TEST_CSV

    if not mlflow_db.exists():
        raise FileNotFoundError(f"MLflow DB not found: {mlflow_db}")
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {split_csv}\n"
            f"Expected at data/splits/SurGen_multitask_test.csv relative to project root."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    conn   = sqlite3.connect(str(mlflow_db))
    exp_id = _get_exp_id(conn, experiment)
    if exp_id is None:
        conn.close()
        raise ValueError(f"Experiment '{experiment}' not found in {mlflow_db}")

    print(f"Loading MLflow run metadata from '{experiment}' ...")
    runs = _load_runs(conn, exp_id, tasks)
    if not runs:
        conn.close()
        raise RuntimeError("No FINISHED non-preflight runs found")
    print(f"  {len(runs)} run(s): {sorted(runs.keys())}")

    print("Loading epoch trajectories ...")
    trajectories = _load_trajectories(conn, runs)
    conn.close()

    _save_json(runs,         out_dir / "runs.json")
    _save_json(trajectories, out_dir / "trajectories.json")
    print(f"Metadata saved -> {out_dir}")

    if embeddings_dir is None:
        print("embeddings_dir not provided; skipping model inference.")
        print(f"\nExtraction complete -> {out_dir}")
        return

    emb_dir = Path(embeddings_dir)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    target_names = {f"{cfg}-s{seed}" for cfg in config_stems for seed in seeds}
    infer_dir    = out_dir / "inference"

    for run_name in sorted(target_names):
        if run_name not in runs:
            print(f"\n[skip] {run_name} — not in DB")
            continue

        print(f"\n[inference] {run_name}")
        model = _load_model(runs[run_name]["run_id"], run_name, mlflow_db)
        if model is None:
            continue

        try:
            result = _run_inference(model, emb_dir, tasks, split_csv, device)
            _save_inference(result, infer_dir / run_name, tasks)
            print(
                f"  Saved. slides={len(result['slide_ids'])} | " +
                " | ".join(f"{t}: n={len(result['labels'][t])}" for t in tasks)
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()

    print(f"\nExtraction complete -> {out_dir}")
