#!/usr/bin/env python3
"""Phase 6 server-side data extraction.

Queries the multitask-surgen-phase6 MLflow experiment, loads ABMIL model
checkpoints, runs inference on the test set, and serializes everything to
an output directory for secure download.

Nothing is written to reports/. Run this on GCP, scp the output dir locally,
then delete it from the server before running phase6_report.py locally.

Output layout:
  <out-dir>/
    runs.json                     run metadata: scalars, params
    trajectories.json             epoch-level metric series per run
    phase5_baseline.json          Phase 5 baseline scalars (or null)
    inference/
      <run_name>/
        slide_ids.json            ordered list of slide IDs with attn weights
        probs_<task>.npy          (N_valid,) float32
        labels_<task>.npy         (N_valid,) float32
        attn/
          <NNNN>.npy              (N_patches, T) float32 — indexed by slide_ids.json
        coords/
          <NNNN>.npy              (N_patches, 2) int64  — indexed by slide_ids.json

Usage:
    python scripts/studies/phase6_extract.py --out-dir /tmp/phase6-data
    python scripts/studies/phase6_extract.py --out-dir /tmp/phase6-data --skip-inference
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "models"))
sys.path.insert(0, str(ROOT / "scripts" / "etl"))

# ── Constants ─────────────────────────────────────────────────────────────────
TASKS           = ["mmr", "ras", "braf"]
SINGLETASK_RUN  = "singletask-mmr-abmil-cosine-accum16"
ABMIL_RUNS      = {
    "multitask-abmil-nope-cosine-accum16",
    "multitask-abmil-cosine-accum16",
    "multitask-abmil-joined-cosine-accum16",
    "multitask-abmil-joined-pe-cosine-accum16",
    SINGLETASK_RUN,
}
EXPERIMENT_NAME = "multitask-surgen-phase6"
PHASE5_EXP_NAME = "multitask-surgen"
PHASE5_BASELINE = "multitask-cosine-accum16"
MLFLOW_DB_PATH  = ROOT / "mlflow.db"
DEFAULT_EMB_DIR = Path("/mnt/data-surgen/embeddings")
DEFAULT_DATA_DIR = ROOT / "data" / "splits"


# ── JSON serialisation helpers ────────────────────────────────────────────────

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


def _load_runs(conn: sqlite3.Connection, exp_id: str) -> dict:
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

    # params
    params_df = pd.read_sql(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({ph})",
        conn, params=run_uuids
    )
    for _, row in params_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["params"][row["key"]] = row["value"]

    # final scalars
    scalar_keys = (
        ["best_val_auroc_mean", "best_val_auprc_mean", "best_epoch",
         "best_val_auroc", "test_auroc"] +
        [f"best_val_auroc_{t}" for t in TASKS] +
        [f"best_val_auprc_{t}" for t in TASKS] +
        [f"test_auroc_{t}"     for t in TASKS] +
        [f"test_auprc_{t}"     for t in TASKS]
    )
    ph_k = ",".join("?" * len(scalar_keys))
    scalars_df = pd.read_sql(
        f"""
        SELECT run_uuid, key, value, step
        FROM metrics
        WHERE run_uuid IN ({ph}) AND key IN ({ph_k})
        """,
        conn, params=run_uuids + scalar_keys
    )
    scalars_df = (
        scalars_df.sort_values("step")
        .groupby(["run_uuid", "key"])
        .last()
        .reset_index()
    )
    for _, row in scalars_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["scalars"][row["key"]] = float(row["value"])

    return result


def _load_trajectories(conn: sqlite3.Connection, runs: dict) -> dict:
    if not runs:
        return {}

    run_uuids   = [v["run_id"] for v in runs.values()]
    uid_to_name = {v["run_id"]: k for k, v in runs.items()}
    ph          = ",".join("?" * len(run_uuids))

    traj_keys = (
        ["train_loss", "val_loss", "val_auroc", "val_auroc_mean"] +
        [f"val_auroc_{t}"       for t in TASKS] +
        [f"train_auroc_{t}"     for t in TASKS] +
        [f"val_loss_{t}"        for t in TASKS] +
        [f"train_loss_{t}"      for t in TASKS] +
        [f"val_sensitivity_{t}" for t in TASKS] +
        [f"val_specificity_{t}" for t in TASKS] +
        [f"task_grad_norm_{t}"  for t in TASKS] +
        ["grad_cos_mmr_ras", "grad_cos_mmr_braf", "grad_cos_ras_braf"]
    )
    ph_k = ",".join("?" * len(traj_keys))

    rows = pd.read_sql(
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
        uid  = run_info["run_id"]
        sub  = rows[rows["run_uuid"] == uid]
        result[name] = {}
        for key in traj_keys:
            series = sub[sub["key"] == key].sort_values("step")["value"].tolist()
            result[name][key] = series
        # alias for single-task runs so local script has uniform key names
        if name == SINGLETASK_RUN:
            for src, dst in [
                ("val_auroc",  "val_auroc_mmr"),
                ("val_loss",   "val_loss_mmr"),
                ("train_loss", "train_loss_mmr"),
            ]:
                if dst not in result[name] and result[name].get(src):
                    result[name][dst] = result[name][src]

    return result


# ── Model inference ───────────────────────────────────────────────────────────

def _load_mlflow_model(run_id: str, run_name: str):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
        model.eval()
        return model
    except Exception as exc:
        print(f"  [WARN] Cannot load model for {run_name}: {exc}")
        return None


def _run_inference(model, test_csv: Path, embeddings_dir: Path,
                   tasks: list, device: str) -> dict:
    """Return {probs, labels, attn_weights, coords} for the full test split."""
    import torch
    from etl.dataset import MultitaskMILDataset, MILDataset, mil_collate_fn
    from torch.utils.data import DataLoader

    is_multi = len(tasks) > 1

    if is_multi:
        ds = MultitaskMILDataset(
            split_csv=str(test_csv),
            embeddings_dir=str(embeddings_dir),
            tasks=tasks,
            slide_id_col="case_id",
        )
    else:
        ds = MILDataset(
            split_csv=str(test_csv),
            embeddings_dir=str(embeddings_dir),
            label_col="label",
            slide_id_col="case_id",
        )

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    all_probs:   dict = {t: [] for t in tasks}
    all_labels:  dict = {t: [] for t in tasks}
    attn_weights: dict = {}   # slide_id → np.ndarray (N, T)
    coords_map:  dict = {}   # slide_id → np.ndarray (N, 2)

    has_attn = (
        hasattr(model, "aggregation") and
        getattr(model, "aggregation", None) == "attention"
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if is_multi:
                emb, labels_t, mask_t, slide_ids, coords = batch
            else:
                emb, label_t, slide_ids, coords = batch
                labels_t = label_t.unsqueeze(-1)
                mask_t   = torch.ones(1, 1)

            emb = emb.to(device)
            coords_gpu = (
                coords.to(device)
                if (coords is not None and not isinstance(coords, list))
                else None
            )

            if has_attn:
                out = model(emb, coords=coords_gpu, return_weights=True)
                logits, weights = out if isinstance(out, tuple) else (out, None)
            else:
                logits  = model(emb, coords=coords_gpu)
                weights = None

            probs = torch.sigmoid(logits).cpu().numpy()
            slide_id = slide_ids[0] if isinstance(slide_ids, list) else slide_ids

            for ti, task in enumerate(tasks):
                if is_multi:
                    lab = float(labels_t[0, ti].item())
                    msk = float(mask_t[0, ti].item())
                    prob = float(probs[0, ti]) if probs.ndim > 1 else float(probs.flat[0])
                else:
                    lab  = float(labels_t[0, 0].item())
                    msk  = 1.0
                    prob = float(probs.flat[0])
                if msk > 0.5:
                    all_probs[task].append(prob)
                    all_labels[task].append(lab)

            if weights is not None:
                attn_weights[slide_id] = weights[0].cpu().numpy()  # (N, T)
            if coords is not None and not isinstance(coords, list):
                coords_map[slide_id] = coords[0].cpu().numpy()     # (N, 2)

    return {
        "probs":   {t: np.array(all_probs[t],  dtype=np.float32) for t in tasks},
        "labels":  {t: np.array(all_labels[t], dtype=np.float32) for t in tasks},
        "weights": attn_weights,
        "coords":  coords_map,
    }


# ── Serialisation ─────────────────────────────────────────────────────────────

def _save_inference(result: dict, tasks: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        np.save(out_dir / f"probs_{task}.npy",  result["probs"][task])
        np.save(out_dir / f"labels_{task}.npy", result["labels"][task])

    weights = result["weights"]
    coords  = result["coords"]
    slide_ids = list(weights.keys())
    _save_json(slide_ids, out_dir / "slide_ids.json")

    attn_dir   = out_dir / "attn"
    coords_dir = out_dir / "coords"
    attn_dir.mkdir(exist_ok=True)
    coords_dir.mkdir(exist_ok=True)

    for i, sid in enumerate(slide_ids):
        tag = f"{i:04d}"
        np.save(attn_dir   / f"{tag}.npy", weights[sid].astype(np.float32))
        if sid in coords:
            np.save(coords_dir / f"{tag}.npy", coords[sid].astype(np.int64))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 server-side data extraction")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write extracted data (e.g. /tmp/phase6-data)")
    p.add_argument("--skip-inference", action="store_true",
                   help="Only extract MLflow metrics; skip model loading and inference")
    p.add_argument("--embeddings-dir", default=str(DEFAULT_EMB_DIR))
    p.add_argument("--data-dir",       default=str(DEFAULT_DATA_DIR))
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    out_dir     = Path(args.out_dir)
    emb_dir     = Path(args.embeddings_dir)
    data_dir    = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not MLFLOW_DB_PATH.exists():
        raise SystemExit(f"MLflow DB not found: {MLFLOW_DB_PATH}")

    conn   = sqlite3.connect(str(MLFLOW_DB_PATH))
    exp_id = _get_exp_id(conn, EXPERIMENT_NAME)
    if exp_id is None:
        raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found")

    print("Loading MLflow run metadata …")
    runs = _load_runs(conn, exp_id)
    if not runs:
        raise SystemExit("No FINISHED non-preflight runs found")
    print(f"  {len(runs)} run(s): {list(runs.keys())}")

    print("Loading epoch trajectories …")
    trajectories = _load_trajectories(conn, runs)

    # Phase 5 baseline
    phase5_baseline = None
    p5_exp_id = _get_exp_id(conn, PHASE5_EXP_NAME)
    if p5_exp_id:
        p5_runs = _load_runs(conn, p5_exp_id)
        if PHASE5_BASELINE in p5_runs:
            phase5_baseline = p5_runs[PHASE5_BASELINE]["scalars"]
            print(f"  Phase 5 baseline loaded: {PHASE5_BASELINE}")

    conn.close()

    _save_json(runs,         out_dir / "runs.json")
    _save_json(trajectories, out_dir / "trajectories.json")
    _save_json(phase5_baseline, out_dir / "phase5_baseline.json")
    print(f"Metadata saved → {out_dir}")

    if args.skip_inference:
        print("--skip-inference set; skipping model loading.")
        print(f"\nExtraction complete → {out_dir}")
        return

    import torch
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    infer_dir = out_dir / "inference"
    abmil_runs = {n: v for n, v in runs.items() if n in ABMIL_RUNS}

    for run_name, run_info in abmil_runs.items():
        print(f"\n[inference] {run_name}")
        model = _load_mlflow_model(run_info["run_id"], run_name)
        if model is None:
            continue

        is_st     = run_name == SINGLETASK_RUN
        task_list = ["mmr"] if is_st else TASKS
        test_csv  = data_dir / ("SurGen_msi_test.csv" if is_st else "SurGen_multitask_test.csv")

        if not test_csv.exists():
            print(f"  [WARN] Test CSV not found: {test_csv} — skipping")
            continue

        try:
            result = _run_inference(model, test_csv, emb_dir, task_list, device)
            _save_inference(result, task_list, infer_dir / run_name)
            n_attn = len(result["weights"])
            print(
                f"  Saved. attn slides={n_attn} | " +
                " | ".join(f"{t}: n={len(result['labels'][t])}" for t in task_list)
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print(f"\nExtraction complete → {out_dir}")


if __name__ == "__main__":
    main()
