#!/usr/bin/env python3
"""Phase 8 server-side data extraction.

Queries the multitask-surgen-phase8 MLflow experiment, loads model checkpoints,
runs inference on the test set, and serializes per-slide predictions to disk.

Run this on GCP. scp the output dir locally, then run:
    python scripts/studies/phase8_ablation.py --data-dir <out-dir>

Output defaults to reports/data/ locally.

Output layout:
  <out-dir>/
    runs.json                     # {run_name: {run_id, params, scalars}}
    trajectories.json             # {run_name: {metric: [...]}}
    inference/
      <run_name>/
        slide_ids.json
        probs_mmr.npy   labels_mmr.npy
        probs_ras.npy   labels_ras.npy
        probs_braf.npy  labels_braf.npy

Usage:
    python scripts/orchestrators/phase8_extract.py \
        --mlflow-db mlflow.db \
        --embeddings-dir /mnt/data-surgen/embeddings \
        --out-dir phase8-data/
"""

import argparse
import json
import math
import sqlite3
import sys
import traceback
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
EXPERIMENT_NAME = "multitask-surgen-phase8"
TEST_CSV        = ROOT / "data" / "splits" / "SurGen_multitask_test.csv"
DEFAULT_EMB_DIR = Path("/mnt/data-surgen/embeddings")

CONFIGS = [
    "config_phase8_baseline",
    "config_phase8_mlp_rpb",
    "config_phase8_dropout10",
    "config_phase8_dropout25",
    "config_phase8_mlp_rpb_dropout10",
    "config_phase8_mlp_rpb_dropout25",
]
SEEDS = [0, 1, 2]
ALL_RUN_NAMES = [f"{cfg}-s{seed}" for cfg in CONFIGS for seed in SEEDS]


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


# ── MLflow / SQLite ───────────────────────────────────────────────────────────

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

    params_df = pd.read_sql(
        f"SELECT run_uuid, key, value FROM params WHERE run_uuid IN ({ph})",
        conn, params=run_uuids
    )
    for _, row in params_df.iterrows():
        result[uid_to_name[row["run_uuid"]]]["params"][row["key"]] = row["value"]

    scalar_keys = (
        ["best_val_auroc_mean", "best_val_auprc_mean", "best_epoch",
         "test_auroc_mean", "test_auprc_mean"] +
        [f"best_val_auroc_{t}" for t in TASKS] +
        [f"best_val_auprc_{t}" for t in TASKS] +
        [f"test_auroc_{t}"     for t in TASKS] +
        [f"test_auprc_{t}"     for t in TASKS]
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


def _run_inference(model, embeddings_dir: Path, device: str) -> dict:
    """Run inference on full test split. Returns probs, labels, slide_ids."""
    import torch
    from etl.dataset import MultitaskMILDataset, mil_collate_fn
    from torch.utils.data import DataLoader

    ds = MultitaskMILDataset(
        split_csv=str(TEST_CSV),
        embeddings_dir=str(embeddings_dir),
        tasks=TASKS,
        slide_id_col="case_id",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    all_probs:  dict = {t: [] for t in TASKS}
    all_labels: dict = {t: [] for t in TASKS}
    slide_ids:  list = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            emb, labels_t, mask_t, batch_slide_ids, coords = batch
            emb = emb.to(device)
            coords_gpu = (
                coords.to(device)
                if (coords is not None and not isinstance(coords, list))
                else None
            )
            logits = model(emb, coords=coords_gpu)
            probs  = torch.sigmoid(logits).cpu().numpy()

            slide_id = batch_slide_ids[0] if isinstance(batch_slide_ids, list) else batch_slide_ids
            slide_ids.append(slide_id)

            for ti, task in enumerate(TASKS):
                lab  = float(labels_t[0, ti].item())
                msk  = float(mask_t[0, ti].item())
                prob = float(probs[0, ti]) if probs.ndim > 1 else float(probs.flat[0])
                if msk > 0.5:
                    all_probs[task].append(prob)
                    all_labels[task].append(lab)

    return {
        "probs":     {t: np.array(all_probs[t],  dtype=np.float32) for t in TASKS},
        "labels":    {t: np.array(all_labels[t], dtype=np.float32) for t in TASKS},
        "slide_ids": slide_ids,
    }


def _save_inference(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(result["slide_ids"], out_dir / "slide_ids.json")
    for task in TASKS:
        np.save(out_dir / f"probs_{task}.npy",  result["probs"][task])
        np.save(out_dir / f"labels_{task}.npy", result["labels"][task])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8 server-side data extraction")
    p.add_argument("--mlflow-db",      default="mlflow.db",
                   help="Path to MLflow SQLite DB (default: mlflow.db)")
    p.add_argument("--embeddings-dir", default=str(DEFAULT_EMB_DIR),
                   help="Path to embeddings directory")
    p.add_argument("--out-dir",        required=True,
                   help="Directory to write extracted data")
    p.add_argument("--skip-inference", action="store_true",
                   help="Only extract MLflow metrics; skip model loading and inference")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    mlflow_db = Path(args.mlflow_db)
    emb_dir   = Path(args.embeddings_dir)
    out_dir   = Path(args.out_dir)

    if not mlflow_db.exists():
        raise SystemExit(f"ERROR: MLflow DB not found: {mlflow_db}")
    if not TEST_CSV.exists():
        raise SystemExit(
            f"ERROR: Test CSV not found: {TEST_CSV}\n"
            f"Expected at data/splits/SurGen_multitask_test.csv relative to project root."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    conn   = sqlite3.connect(str(mlflow_db))
    exp_id = _get_exp_id(conn, EXPERIMENT_NAME)
    if exp_id is None:
        raise SystemExit(f"ERROR: Experiment '{EXPERIMENT_NAME}' not found in {mlflow_db}")

    print(f"Loading MLflow run metadata from '{EXPERIMENT_NAME}' …")
    runs = _load_runs(conn, exp_id)
    if not runs:
        raise SystemExit("ERROR: No FINISHED non-preflight runs found")
    print(f"  {len(runs)} run(s): {sorted(runs.keys())}")

    missing_expected = [n for n in ALL_RUN_NAMES if n not in runs]
    if missing_expected:
        print(f"  [WARN] {len(missing_expected)} expected run(s) not yet complete:")
        for m in missing_expected:
            print(f"    {m}")

    print("Loading epoch trajectories …")
    trajectories = _load_trajectories(conn, runs)
    conn.close()

    _save_json(runs,         out_dir / "runs.json")
    _save_json(trajectories, out_dir / "trajectories.json")
    print(f"Metadata saved → {out_dir}")

    if args.skip_inference:
        print("--skip-inference set; skipping model loading.")
        print(f"\nExtraction complete → {out_dir}")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    infer_dir = out_dir / "inference"

    for run_name in ALL_RUN_NAMES:
        if run_name not in runs:
            print(f"\n[skip] {run_name} — not in DB")
            continue

        print(f"\n[inference] {run_name}")
        model = _load_model(runs[run_name]["run_id"], run_name, mlflow_db)
        if model is None:
            continue

        try:
            result = _run_inference(model, emb_dir, device)
            _save_inference(result, infer_dir / run_name)
            print(
                f"  Saved. slides={len(result['slide_ids'])} | " +
                " | ".join(f"{t}: n={len(result['labels'][t])}" for t in TASKS)
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()

    print(f"\nExtraction complete → {out_dir}")


if __name__ == "__main__":
    main()
