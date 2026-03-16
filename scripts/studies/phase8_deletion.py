#!/usr/bin/env python3
"""Phase 8 Interpretability — Deletion / Insertion Analysis (Ensemble + Per-Task).

Two co-equal goals:
  1. Within ABMIL: compare attn vs ixg attribution faithfulness on ensemble_baseline.
  2. Cross-model: compare ixg faithfulness for ensemble_baseline (ABMIL) vs mean_agg
     (mean pool) — tests whether ABMIL's attention focuses on more causally relevant
     regions than a non-attentive baseline.

Groups:
  ensemble_baseline  — config_phase8_baseline-s{0,1,2}     (attn, ixg)
  mean_agg           — config_phase5_baseline-s{0,1,2}      (ixg only)

Attribution notes:
  gradnorm excluded: constant 1/N·||W||₂ for mean_agg (degenerate); not needed for ABMIL.
  ixg valid for both: ABMIL — gradient×input modulated by attention routing;
                      mean_agg — each patch's additive logit contribution (h_i ⊙ (1/N)·Wᵀ).
  Cross-model comparison: ixg on shared balanced slide set; same global mean insertion baseline.
  Insertion baseline: N copies of global mean embedding (precomputed from test-set Zarr).
  Primary metric: SRG = AUPC_ins − AUPC_del. Supplementary: del-AUC.

Three-phase pattern:
  Phase A0  (local/remote) : Extract mean_agg inference if not already local
  Phase A0b (local/remote) : Compute global mean embedding from test-set Zarr
  Phase A   (local)        : Build shared balanced per-task slide lists (seed=42)
  Phase B   (remote)       : Compute deletion/insertion curves on GCP
  Phase C   (local)        : Download results, generate figures

Outputs (reports/data/phase8/deletion/):
  slide_lists/{group}_{task}.json
  curves/{sid}_{group}_{task}_{method}_{del|ins}.npy
  auc_table.csv
  canonical_method.txt
  summary.json

Figures (reports/figures/phase8/):
  deletion_curves_{group}.png               (3-row × 2-col per group)
  deletion_curves_cross_model_ixg.png       (3-row × 2-col cross-model ixg)
  deletion_auc_table.png

Usage:
    # Dry-run (no GCP):
    python scripts/studies/phase8_deletion.py --data-dir reports/data/phase8/ --no-report

    # Full run:
    python scripts/studies/phase8_deletion.py \\
        --data-dir reports/data/phase8/ \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'

    # Re-visualise only:
    python scripts/studies/phase8_deletion.py \\
        --data-dir reports/data/phase8/ --visualise-only
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import load_inference  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p8_del"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"

GCP_INFER_SENTINEL      = f"{GCP_REMOTE_TMP}/infer_done.sentinel"
GCP_GLOBAL_MEAN_SENTINEL = f"{GCP_REMOTE_TMP}/global_mean_done.sentinel"
GCP_DEL_SENTINEL        = f"{GCP_REMOTE_TMP}/del_done.sentinel"

FIGURES_DIR     = ROOT / "reports" / "figures" / "phase8"
REPORT_PATH     = ROOT / "reports" / "phase8-results.md"

K_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
TASKS    = ["mmr", "ras", "braf"]
TASK_IDX = {"mmr": 0, "ras": 1, "braf": 2}

GROUPS = {
    "ensemble_baseline": {
        "run_names": [
            "config_phase8_baseline-s0",
            "config_phase8_baseline-s1",
            "config_phase8_baseline-s2",
        ],
        "methods": ["attn", "ixg"],
    },
    "mean_agg": {
        "run_names": [
            "config_phase5_baseline-s0",
            "config_phase5_baseline-s1",
            "config_phase5_baseline-s2",
        ],
        "methods": ["ixg"],
    },
}

MEAN_AGG_GROUP = "mean_agg"


# ── Remote: mean_agg inference extraction ─────────────────────────────────────

REMOTE_INFER_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side inference extraction for config_phase5_baseline seeds.\"\"\"
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path("{root_dir}")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "models"))
sys.path.insert(0, str(ROOT_DIR / "scripts" / "etl"))

import mlflow
from torch.utils.data import DataLoader
from etl.dataset import MultitaskMILDataset, mil_collate_fn

MLFLOW_DB = "{mlflow_db}"
ZARR_DIR  = Path("{zarr_dir}")
OUT_DIR   = Path("{out_dir}")
RUN_NAMES = {run_names}
TEST_CSV  = str(ROOT_DIR / "data" / "splits" / "SurGen_multitask_test.csv")
TASKS     = ["mmr", "ras", "braf"]

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_run_id(run_name):
    conn = sqlite3.connect(MLFLOW_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT r.run_uuid FROM runs r "
            "JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.runName' "
            "WHERE t.value = ? AND r.lifecycle_stage = 'active' "
            "ORDER BY r.start_time DESC LIMIT 1",
            (run_name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"No active MLflow run: {{run_name}}")
    return row[0]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

for run_name in RUN_NAMES:
    print(f"\\nExtracting inference for {{run_name}} ...")
    run_out = OUT_DIR / run_name
    if run_out.exists() and (run_out / "slide_ids.json").exists():
        print(f"  [SKIP] Already exists: {{run_out}}")
        continue
    run_out.mkdir(parents=True, exist_ok=True)

    try:
        run_id = _resolve_run_id(run_name)
        mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
        model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
        model.eval().to(device)
    except Exception as e:
        print(f"  [ERROR] Cannot load model: {{e}}")
        continue

    ds = MultitaskMILDataset(
        split_csv=TEST_CSV,
        embeddings_dir=str(ZARR_DIR),
        tasks=TASKS,
        slide_id_col="case_id",
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=mil_collate_fn)

    all_probs  = {{t: [] for t in TASKS}}
    all_labels = {{t: [] for t in TASKS}}
    slide_ids  = []

    with torch.no_grad():
        for batch in loader:
            emb, labels_t, mask_t, batch_sids, coords = batch
            emb = emb.to(device)
            logits = model(emb)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            sid = batch_sids[0] if isinstance(batch_sids, list) else batch_sids
            slide_ids.append(sid)
            for ti, task in enumerate(TASKS):
                prob = float(probs[0, ti]) if probs.ndim > 1 else float(probs.flat[ti])
                lab  = float(labels_t[0, ti].item())
                all_probs[task].append(prob)
                all_labels[task].append(lab)

    (run_out / "slide_ids.json").write_text(json.dumps(slide_ids))
    for task in TASKS:
        np.save(str(run_out / f"probs_{{task}}.npy"),  np.array(all_probs[task],  dtype=np.float32))
        np.save(str(run_out / f"labels_{{task}}.npy"), np.array(all_labels[task], dtype=np.float32))
    print(f"  Saved {{len(slide_ids)}} slides → {{run_out}}")

Path("{sentinel}").touch()
print("\\nInference extraction done. Sentinel written.")
""")


# ── Remote: global mean embedding computation ─────────────────────────────────

REMOTE_GLOBAL_MEAN_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side global mean embedding computation from test-set Zarr.\"\"\"
import re, sys
from pathlib import Path

import numpy as np
import zarr

ROOT_DIR = Path("{root_dir}")
ZARR_DIR = Path("{zarr_dir}")
TEST_CSV = str(ROOT_DIR / "data" / "splits" / "SurGen_multitask_test.csv")
OUT_PATH = Path("{out_path}")

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import pandas as pd


def _pad_id(slide_id):
    m = re.match(r'^(.*_T)(\\d{{1,3}})$', slide_id)
    if m:
        return f"{{m.group(1)}}{{int(m.group(2)):03d}}_01"
    return slide_id


def _find_zarr(slide_id):
    for sid in (slide_id, _pad_id(slide_id)):
        p = ZARR_DIR / f"{{sid}}.zarr"
        if p.exists():
            return str(p)
    return None


df = pd.read_csv(TEST_CSV)
slide_ids = df["case_id"].astype(str).tolist()
print(f"Computing global mean from {{len(slide_ids)}} test slides ...")

running_sum = None
count = 0
skipped = 0

for sid in slide_ids:
    path = _find_zarr(sid)
    if path is None:
        skipped += 1
        continue
    store = zarr.open(path, mode="r")
    feats = store["features"][:]  # (N, 1024)
    if running_sum is None:
        running_sum = feats.sum(axis=0).astype(np.float64)
    else:
        running_sum += feats.sum(axis=0)
    count += feats.shape[0]

print(f"Skipped {{skipped}} slides. Total patches: {{count}}")
global_mean = (running_sum / count).astype(np.float32)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(str(OUT_PATH), global_mean)
print(f"Saved global mean (shape={{global_mean.shape}}) to {{OUT_PATH}}")

Path("{sentinel}").touch()
print("Global mean done. Sentinel written.")
""")


# ── Remote: deletion/insertion computation ────────────────────────────────────

REMOTE_DEL_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side deletion/insertion computation for Phase 8 (ensemble + per-task).\"\"\"
import json, re, sqlite3, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "{root_dir}")
sys.path.insert(0, "{scripts_dir}")
sys.path.insert(0, "{scripts_dir}/models")
sys.path.insert(0, "{scripts_dir}/etl")

import mlflow
import zarr

MLFLOW_DB = "{mlflow_db}"
ZARR_DIR  = Path("{zarr_dir}")
OUT_DIR   = Path("{out_dir}")
MANIFEST  = json.loads(Path("{manifest_path}").read_text())

K_LEVELS    = MANIFEST["k_levels"]
GLOBAL_MEAN = np.load(MANIFEST["global_mean_emb_path"])  # (D,)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_run_id(run_name):
    conn = sqlite3.connect(MLFLOW_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT r.run_uuid FROM runs r "
            "JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.runName' "
            "WHERE t.value = ? AND r.lifecycle_stage = 'active' "
            "ORDER BY r.start_time DESC LIMIT 1",
            (run_name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"No active MLflow run: {{run_name}}")
    return row[0]


def _pad_id(slide_id):
    m = re.match(r'^(.*_T)(\\d{{1,3}})$', slide_id)
    if m:
        return f"{{m.group(1)}}{{int(m.group(2)):03d}}_01"
    return slide_id


def _find_zarr(slide_id):
    for sid in (slide_id, _pad_id(slide_id)):
        p = ZARR_DIR / f"{{sid}}.zarr"
        if p.exists():
            return str(p)
    return None


def _load_zarr(slide_id):
    path = _find_zarr(slide_id)
    if path is None:
        raise FileNotFoundError(f"No zarr for {{slide_id}} in {{ZARR_DIR}}")
    store = zarr.open(path, mode="r")
    return store["features"][:]  # (N, 1024)


def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    model.eval()
    return model


def _get_task_prob(models, emb_np, device, task_idx):
    \"\"\"Average sigmoid(logit[task_idx]) across ensemble members.\"\"\"
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
    probs = []
    with torch.no_grad():
        for model in models:
            out = model(emb)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim <= 1:
                logit = out.flatten()[0]
            else:
                logit = out[0, task_idx]
            probs.append(float(torch.sigmoid(logit).item()))
    return float(np.mean(probs))


def _get_attn_task(models, emb_np, device, task_idx):
    \"\"\"Average per-task attention scores across ensemble members.\"\"\"
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
    valid = []
    for model in models:
        with torch.no_grad():
            out = model(emb, return_weights=True)
        if not isinstance(out, tuple):
            continue
        _, weights = out
        if weights is None:
            continue
        w = weights.detach().cpu().numpy()
        if w.ndim == 3:
            w = w[0]   # (B, N, T) → (N, T)
        elif w.ndim == 2 and w.shape[0] == 1:
            w = w[0]   # (1, N) → (N,)
        if w.ndim == 2:
            col = min(task_idx, w.shape[1] - 1)
            valid.append(w[:, col].astype(np.float32))
        else:
            valid.append(w.astype(np.float32))
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def _get_ixg_task(models, emb_np, device, task_idx):
    \"\"\"Average InputXGrad across ensemble members for a given task.\"\"\"
    all_ixg = []
    for model in models:
        emb_leaf = torch.tensor(emb_np, dtype=torch.float32, device=device,
                                requires_grad=True)
        emb = emb_leaf.unsqueeze(0)
        out = model(emb)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim <= 1:
            logit = out.flatten()[0]
        else:
            logit = out[0, task_idx]
        logit.backward()
        grad    = emb_leaf.grad.detach().cpu().numpy()
        emb_val = emb_leaf.detach().cpu().numpy()
        all_ixg.append(np.linalg.norm(grad * emb_val, axis=1).astype(np.float32))
    return np.mean(np.stack(all_ixg, axis=0), axis=0)


def run_deletion_task(features_np, scores, k_levels, models, device, task_idx):
    N = features_np.shape[0]
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_remove = int(k * N)
        keep = rank_desc[n_remove:]
        bag  = features_np[keep] if len(keep) > 0 else features_np[:1]
        probs.append(_get_task_prob(models, bag, device, task_idx))
    return np.array(probs, dtype=np.float32)


def run_insertion_task(features_np, scores, k_levels, models, device, task_idx):
    \"\"\"Fixed-bag insertion: N copies of global mean; reveal top-k real patches.\"\"\"
    N = features_np.shape[0]
    baseline = np.tile(GLOBAL_MEAN, (N, 1))  # (N, D) — all global mean
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_reveal = int(k * N)
        bag = baseline.copy()
        if n_reveal > 0:
            bag[rank_desc[:n_reveal]] = features_np[rank_desc[:n_reveal]]
        probs.append(_get_task_prob(models, bag, device, task_idx))
    return np.array(probs, dtype=np.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

for group_name, group_cfg in MANIFEST["groups"].items():
    run_names  = group_cfg["run_names"]
    methods    = group_cfg["methods"]
    slides_per_task = group_cfg["slides_per_task"]

    print(f"\\n{{'='*60}}")
    print(f"Group: {{group_name}}  runs: {{run_names}}")

    print(f"  Loading {{len(run_names)}} models ...")
    models = []
    for rn in run_names:
        try:
            rid = _resolve_run_id(rn)
            m   = _load_model(rid)
            m.to(device)
            models.append(m)
            print(f"    Loaded {{rn}}")
        except Exception as e:
            print(f"    [WARN] Cannot load {{rn}}: {{e}}")

    if not models:
        print(f"  [ERROR] No models for {{group_name}} — skipping")
        continue

    for task_name, slide_list in slides_per_task.items():
        if not slide_list:
            print(f"  [SKIP] No slides for {{group_name}}/{{task_name}}")
            continue
        task_idx = slide_list[0]["task_idx"]
        print(f"\\n  Task: {{task_name}} (idx={{task_idx}})  slides={{len(slide_list)}}")

        for slide in slide_list:
            sid = slide["slide_id"]
            try:
                features_np = _load_zarr(sid)
            except FileNotFoundError as e:
                print(f"    [WARN] {{e}}")
                continue

            N = features_np.shape[0]
            scores_dict = {{}}

            if "attn" in methods:
                try:
                    attn = _get_attn_task(models, features_np, device, task_idx)
                    if attn is not None and attn.shape[0] == N:
                        scores_dict["attn"] = attn
                    elif attn is None:
                        print(f"    [WARN] attn=None for {{sid}}/{{group_name}} — skipping attn")
                    else:
                        print(f"    [WARN] attn shape mismatch {{sid}}: {{attn.shape[0]}} vs N={{N}}")
                except Exception as e:
                    print(f"    [WARN] attn failed for {{sid}}: {{e}}")

            if "ixg" in methods:
                try:
                    ixg = _get_ixg_task(models, features_np, device, task_idx)
                    scores_dict["ixg"] = ixg
                except Exception as e:
                    print(f"    [WARN] ixg failed for {{sid}}: {{e}}")

            for method, scores in scores_dict.items():
                try:
                    fname_base = f"{{sid}}_{{group_name}}_{{task_name}}_{{method}}"
                    del_path = OUT_DIR / f"{{fname_base}}_del.npy"
                    ins_path = OUT_DIR / f"{{fname_base}}_ins.npy"
                    del_probs = run_deletion_task(features_np, scores, K_LEVELS, models, device, task_idx)
                    ins_probs = run_insertion_task(features_np, scores, K_LEVELS, models, device, task_idx)
                    np.save(str(del_path), del_probs)
                    np.save(str(ins_path), ins_probs)
                    k20_idx = K_LEVELS.index(0.20) if 0.20 in K_LEVELS else 3
                    drop = (del_probs[0] - del_probs[k20_idx]) * 100
                    print(f"    {{method}}: del@0%={{del_probs[0]:.3f}} "
                          f"del@20%={{del_probs[k20_idx]:.3f}} (drop={{drop:.1f}}pp) "
                          f"ins@50%={{ins_probs[-1]:.3f}}")
                except Exception as exc:
                    print(f"    [ERROR] {{method}} {{sid}}: {{exc}}")

Path("{sentinel}").touch()
print("\\nDone. Sentinel written.")
""")


# ── SSH/SCP helpers ────────────────────────────────────────────────────────────

def _ssh_cmd(host, user, password, remote_cmd):
    return ["sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}", remote_cmd]


def _scp_to(host, user, password, local, remote):
    return ["sshpass", "-p", password, "scp", "-o", "StrictHostKeyChecking=no",
            local, f"{user}@{host}:{remote}"]


def _scp_from(host, user, password, remote, local):
    return ["sshpass", "-p", password, "scp", "-r", "-o", "StrictHostKeyChecking=no",
            f"{user}@{host}:{remote}", local]


def _run(cmd, desc=""):
    label = desc or " ".join(cmd[:4])
    print(f"  >> {label}", flush=True)
    is_ssh = len(cmd) > 3 and cmd[3] == "ssh"
    if is_ssh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out_lines: list = []
        err_lines: list = []

        def _drain(stream, lines, tag):
            for line in stream:
                l = line.rstrip()
                lines.append(l)
                print(f"    {tag}{l}", flush=True)

        t1 = threading.Thread(target=_drain, args=(proc.stdout, out_lines, ""))
        t2 = threading.Thread(target=_drain, args=(proc.stderr, err_lines, "[err] "))
        t1.start(); t2.start()
        t1.join();  t2.join()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {label}")
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout}")
            print(f"  STDERR:\n{result.stderr}")
            raise RuntimeError(f"Command failed ({result.returncode}): {label}")
        if result.stdout.strip():
            print(result.stdout.strip())


def _launch_nohup_and_poll(host, user, password, script_remote, log_remote,
                            sentinel_remote, poll_interval=30) -> None:
    """Launch script with nohup on GCP; poll sentinel file until done."""
    launch_cmd = (
        f"nohup python3 {script_remote} > {log_remote} 2>&1 & echo $!"
    )
    proc = subprocess.run(
        _ssh_cmd(host, user, password, launch_cmd),
        capture_output=True, text=True
    )
    pid = proc.stdout.strip()
    print(f"  >> launched nohup python3 on GCP (PID {pid})", flush=True)

    while True:
        time.sleep(poll_interval)
        check = (
            f"if [ -f {sentinel_remote} ]; then echo done; "
            f"elif kill -0 {pid} 2>/dev/null; then echo running; "
            f"else echo dead; fi"
        )
        r = subprocess.run(
            _ssh_cmd(host, user, password, check),
            capture_output=True, text=True
        )
        status = r.stdout.strip()
        if status == "done":
            print("  >> GCP computation finished", flush=True)
            break
        elif status == "dead":
            raise RuntimeError(
                f"GCP process {pid} died without sentinel — check {log_remote}"
            )
        else:
            tail = subprocess.run(
                _ssh_cmd(host, user, password, f"tail -3 {log_remote}"),
                capture_output=True, text=True
            )
            if tail.stdout.strip():
                print(f"  [GCP] {tail.stdout.strip()}", flush=True)


# ── Phase A0: extract mean_agg inference (if missing) ─────────────────────────

def _mean_agg_inference_exists(inference_dir: Path) -> bool:
    """Return True if all 3 config_phase5_baseline seeds have inference data."""
    for seed in range(3):
        run_dir = inference_dir / f"config_phase5_baseline-s{seed}"
        if not (run_dir / "slide_ids.json").exists():
            return False
    return True


def phase_a0_extract_mean_agg(inference_dir: Path, host: str, user: str,
                               password: str) -> None:
    """SSH to GCP, run inference for config_phase5_baseline seeds, SCP back."""
    print("\nPhase A0 — extracting mean_agg inference from GCP")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        run_names_json = json.dumps(GROUPS[MEAN_AGG_GROUP]["run_names"])
        script_str = REMOTE_INFER_PY.format(
            root_dir    = GCP_ROOT_DIR,
            mlflow_db   = GCP_MLFLOW_DB,
            zarr_dir    = GCP_ZARR_DIR,
            out_dir     = GCP_REMOTE_TMP + "/infer_out",
            run_names   = run_names_json,
            sentinel    = GCP_INFER_SENTINEL,
        )
        script_local  = tmp / "infer_extract.py"
        script_remote = f"{GCP_REMOTE_TMP}/infer_extract.py"
        script_local.write_text(script_str)

        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/infer_out"),
             "mkdir remote infer_out")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp infer_extract.py")
        log_remote = f"{GCP_REMOTE_TMP}/infer_extract.log"
        _launch_nohup_and_poll(host, user, password, script_remote, log_remote,
                                GCP_INFER_SENTINEL)

        # tar and download
        _run(_ssh_cmd(host, user, password,
                      f"cd {GCP_REMOTE_TMP} && tar czf infer_out.tar.gz infer_out/ && echo 'tar done'"),
             "tar infer results")
        tar_local = tmp / "infer_out.tar.gz"
        _run(_scp_from(host, user, password,
                       f"{GCP_REMOTE_TMP}/infer_out.tar.gz", str(tar_local)),
             "scp infer_out.tar.gz")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}/infer_out*"),
             "rm remote infer_out")

        with tarfile.open(tar_local) as tf:
            tf.extractall(tmp)
        infer_src = tmp / "infer_out"
        inference_dir.mkdir(parents=True, exist_ok=True)
        for item in infer_src.iterdir():
            dest = inference_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(inference_dir))
        print(f"  Extracted mean_agg inference to {inference_dir}")


# ── Phase A0b: compute global mean embedding (if missing) ─────────────────────

def _global_mean_exists(inference_dir: Path) -> bool:
    return (inference_dir / "global_mean_emb.npy").exists()


def phase_a0b_compute_global_mean(inference_dir: Path, host: str, user: str,
                                   password: str) -> None:
    """SSH to GCP, compute global mean of all test-set patch embeddings, SCP back."""
    print("\nPhase A0b — computing global mean embedding on GCP")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        remote_out_path = f"{GCP_REMOTE_TMP}/global_mean_emb.npy"
        script_str = REMOTE_GLOBAL_MEAN_PY.format(
            root_dir  = GCP_ROOT_DIR,
            zarr_dir  = GCP_ZARR_DIR,
            out_path  = remote_out_path,
            sentinel  = GCP_GLOBAL_MEAN_SENTINEL,
        )
        script_local  = tmp / "global_mean.py"
        script_remote = f"{GCP_REMOTE_TMP}/global_mean.py"
        script_local.write_text(script_str)

        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}"),
             "mkdir remote tmp")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp global_mean.py")

        log_remote = f"{GCP_REMOTE_TMP}/global_mean.log"
        _launch_nohup_and_poll(host, user, password, script_remote, log_remote,
                                GCP_GLOBAL_MEAN_SENTINEL)

        local_npy = tmp / "global_mean_emb.npy"
        _run(_scp_from(host, user, password, remote_out_path, str(local_npy)),
             "scp global_mean_emb.npy")

        inference_dir.mkdir(parents=True, exist_ok=True)
        dest = inference_dir / "global_mean_emb.npy"
        if dest.exists():
            dest.unlink()
        shutil.move(str(local_npy), str(dest))
        mean_val = np.load(str(dest))
        print(f"  Saved global mean embedding: shape={mean_val.shape}, "
              f"mean={mean_val.mean():.4f} → {dest}")


# ── Phase A: build shared balanced per-task slide lists ───────────────────────

def _load_test_labels(test_csv: Path) -> dict:
    """Load per-slide task labels/masks from the test split CSV.

    Returns {case_id: {task: label_or_None}} where None means masked (no label).
    """
    df = pd.read_csv(test_csv)
    result: dict = {}
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        result[case_id] = {}
        for task in TASKS:
            col = f"label_{task}"
            val = row.get(col, float("nan"))
            try:
                fval = float(val)
                result[case_id][task] = None if pd.isna(fval) else int(round(fval))
            except (TypeError, ValueError):
                result[case_id][task] = None
    return result


def collect_candidates(group_name: str, run_names: list,
                       inference_dir: Path, tasks: list) -> dict:
    """Load ensemble probs for a group; return all labeled slides per task (no TP filter).

    Returns {task_name: [{"slide_id", "label", "ensemble_prob"}, ...]}
    """
    inferences = []
    for run_name in run_names:
        inf = load_inference(run_name, inference_dir)
        inferences.append(inf)

    slide_ids = inferences[0]["slide_ids"]

    test_csv   = ROOT / "data" / "splits" / "SurGen_multitask_test.csv"
    csv_labels = _load_test_labels(test_csv) if test_csv.exists() else {}

    candidates: dict = {}

    for task in tasks:
        probs_key = f"probs_{task}"

        if csv_labels:
            has_label = [bool(csv_labels.get(sid, {}).get(task) is not None)
                         for sid in slide_ids]
        else:
            first_probs = inferences[0].get(probs_key, [])
            n_labeled   = len(first_probs)
            has_label   = [i < n_labeled for i in range(len(slide_ids))]

        per_seed_probs: list = []
        for inf in inferences:
            arr = inf.get(probs_key)
            if arr is None:
                continue
            seed_probs: dict = {}
            arr_idx = 0
            for sid, hlab in zip(slide_ids, has_label):
                if hlab:
                    if arr_idx < len(arr):
                        seed_probs[sid] = float(arr[arr_idx])
                    arr_idx += 1
            per_seed_probs.append(seed_probs)

        if not per_seed_probs:
            print(f"  [WARN] No probs for {group_name}/{task} — skipping")
            candidates[task] = []
            continue

        slide_list = []
        for sid, hlab in zip(slide_ids, has_label):
            if not hlab:
                continue

            label = None
            if csv_labels:
                label = csv_labels.get(sid, {}).get(task)
            if label is None:
                labels_arr = inferences[0].get(f"labels_{task}")
                if labels_arr is not None:
                    idx = sum(has_label[:slide_ids.index(sid)])
                    if idx < len(labels_arr):
                        label = int(round(float(labels_arr[idx])))
            if label is None:
                continue

            probs_for_slide = [s[sid] for s in per_seed_probs if sid in s]
            if not probs_for_slide:
                continue
            ensemble_prob = float(np.mean(probs_for_slide))
            slide_list.append({
                "slide_id":      sid,
                "label":         label,
                "ensemble_prob": ensemble_prob,
            })

        candidates[task] = slide_list

    return candidates


def build_shared_balanced_lists(groups: dict, inference_dir: Path,
                                 lists_dir: Path, seed: int = 42) -> dict:
    """Build shared balanced slide lists across all groups.

    For each task:
      1. Collect candidates for each group (all labeled slides, no TP filter)
      2. Intersect slide IDs across all groups
      3. Balance: downsample majority to min(n_pos, n_neg) (seed=42)
      4. Write per-group JSONs with the same slide IDs but group-specific probs

    Returns {group_name: {task: [slide_dict, ...]}} with task_idx added.
    """
    lists_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect candidates for all groups
    all_candidates: dict = {}
    for group_name, group_cfg in groups.items():
        print(f"    collecting candidates: {group_name}")
        try:
            all_candidates[group_name] = collect_candidates(
                group_name, group_cfg["run_names"], inference_dir, TASKS
            )
        except FileNotFoundError as e:
            print(f"    [ERROR] {e}")
            all_candidates[group_name] = {t: [] for t in TASKS}

    group_names = list(groups.keys())
    rng = np.random.default_rng(seed)
    result: dict = {g: {} for g in group_names}

    for task in TASKS:
        task_idx = TASK_IDX[task]

        # Step 2: intersect slide IDs across all groups
        id_sets = []
        for g in group_names:
            cands = all_candidates[g].get(task, [])
            id_sets.append({s["slide_id"] for s in cands})
        shared_ids = set.intersection(*id_sets) if id_sets else set()

        if not shared_ids:
            print(f"    {task}: no shared slides across groups — skipping")
            for g in group_names:
                result[g][task] = []
            continue

        # Step 3: balance using reference group labels (same test CSV → same labels)
        ref_group = group_names[0]
        ref_dict  = {s["slide_id"]: s["label"]
                     for s in all_candidates[ref_group].get(task, [])
                     if s["slide_id"] in shared_ids}
        pos_ids = sorted([sid for sid, lbl in ref_dict.items() if lbl == 1])
        neg_ids = sorted([sid for sid, lbl in ref_dict.items() if lbl == 0])
        n = min(len(pos_ids), len(neg_ids))
        if n == 0:
            print(f"    {task}: no positive or negative slides after intersection — skipping")
            for g in group_names:
                result[g][task] = []
            continue
        chosen_pos = rng.choice(pos_ids, n, replace=False).tolist()
        chosen_neg = rng.choice(neg_ids, n, replace=False).tolist()
        balanced_ids = set(chosen_pos + chosen_neg)
        print(f"    {task}: {len(balanced_ids)} slides ({n} pos + {n} neg)")

        # Step 4: write per-group JSONs (same slide IDs, group-specific probs)
        for g in group_names:
            cand_dict = {s["slide_id"]: s
                         for s in all_candidates[g].get(task, [])}
            slide_list = []
            for sid in balanced_ids:
                entry = cand_dict.get(sid)
                if entry is None:
                    continue
                slide_list.append({
                    "slide_id":      entry["slide_id"],
                    "task_idx":      task_idx,
                    "label":         entry["label"],
                    "ensemble_prob": entry["ensemble_prob"],
                })
            result[g][task] = slide_list
            out_path = lists_dir / f"{g}_{task}.json"
            out_path.write_text(json.dumps(slide_list, indent=2))
            print(f"      {g}/{task}: {len(slide_list)} slides → {out_path.name}")

    return result


def phase_a_build_slide_lists(groups: dict, inference_dir: Path,
                               deletion_dir: Path, force: bool = False) -> dict:
    """Build and save shared balanced per-task slide lists for all groups."""
    lists_dir = deletion_dir / "slide_lists"

    # Check if all lists already cached
    all_cached = all(
        (lists_dir / f"{g}_{t}.json").exists()
        for g in groups for t in TASKS
    )

    if all_cached and not force:
        print("  All slide lists cached — loading from disk")
        result: dict = {}
        for group_name in groups:
            result[group_name] = {}
            for task in TASKS:
                path = lists_dir / f"{group_name}_{task}.json"
                slide_list = json.loads(path.read_text())
                result[group_name][task] = slide_list
                print(f"    {group_name}/{task}: {len(slide_list)} slides (cached)")
        return result

    print("  Building shared balanced slide lists (seed=42)")
    return build_shared_balanced_lists(groups, inference_dir, lists_dir, seed=42)


def load_slide_lists(groups: dict, deletion_dir: Path) -> dict:
    """Load slide lists from disk. Returns group→task→list."""
    lists_dir = deletion_dir / "slide_lists"
    result: dict = {}
    for group_name in groups:
        result[group_name] = {}
        for task in TASKS:
            path = lists_dir / f"{group_name}_{task}.json"
            if path.exists():
                result[group_name][task] = json.loads(path.read_text())
            else:
                result[group_name][task] = []
    return result


# ── Phase B: remote compute ────────────────────────────────────────────────────

def build_manifest(groups: dict, all_slide_lists: dict, global_mean_emb_path: str) -> dict:
    manifest_groups = {}
    for group_name, group_cfg in groups.items():
        manifest_groups[group_name] = {
            "run_names":       group_cfg["run_names"],
            "methods":         group_cfg["methods"],
            "slides_per_task": all_slide_lists.get(group_name, {t: [] for t in TASKS}),
        }
    return {
        "k_levels":            K_LEVELS,
        "groups":              manifest_groups,
        "global_mean_emb_path": global_mean_emb_path,
    }


def phase_b_remote_compute(manifest: dict, curves_dir: Path,
                            inference_dir: Path,
                            host: str, user: str, password: str) -> None:
    curves_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local     = tmp / "manifest.json"
        script_local       = tmp / "del_compute.py"
        manifest_remote    = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote      = f"{GCP_REMOTE_TMP}/del_compute.py"
        global_mean_remote = f"{GCP_REMOTE_TMP}/global_mean_emb.npy"

        manifest_local.write_text(json.dumps(manifest, indent=2))
        script_local.write_text(REMOTE_DEL_PY.format(
            root_dir      = GCP_ROOT_DIR,
            scripts_dir   = GCP_SCRIPTS_DIR,
            mlflow_db     = GCP_MLFLOW_DB,
            zarr_dir      = GCP_ZARR_DIR,
            out_dir       = GCP_REMOTE_TMP + "/out",
            manifest_path = manifest_remote,
            sentinel      = GCP_DEL_SENTINEL,
        ))

        print("\nPhase B1 — prepare remote directory")
        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/out"),
             "mkdir remote tmp")

        print("Phase B2 — upload manifest, script, and global mean embedding")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp del_compute.py")
        local_global_mean = str(inference_dir / "global_mean_emb.npy")
        _run(_scp_to(host, user, password, local_global_mean, global_mean_remote),
             "scp global_mean_emb.npy")

        print("Phase B3 — run deletion/insertion on GCP")
        log_remote = f"{GCP_REMOTE_TMP}/del_compute.log"
        _launch_nohup_and_poll(host, user, password, script_remote, log_remote,
                                GCP_DEL_SENTINEL)

        print("Phase B4 — download results")
        _run(_ssh_cmd(host, user, password,
                      f"cd {GCP_REMOTE_TMP} && tar czf out.tar.gz out/ && echo 'tar done'"),
             "tar results on GCP")
        out_tar_local = tmp / "out.tar.gz"
        _run(_scp_from(host, user, password,
                       f"{GCP_REMOTE_TMP}/out.tar.gz", str(out_tar_local)),
             "scp out.tar.gz → local")
        _run(_ssh_cmd(host, user, password, f"rm {GCP_REMOTE_TMP}/out.tar.gz"),
             "rm remote out.tar.gz")

        with tarfile.open(out_tar_local) as tf:
            tf.extractall(tmp)
        out_src = tmp / "out"
        for item in out_src.iterdir():
            dest = curves_dir / item.name
            if dest.exists():
                dest.unlink() if dest.is_file() else shutil.rmtree(dest)
            shutil.move(str(item), str(curves_dir))
        print(f"  Extracted results to {curves_dir}")

        print("Phase B5 — clean up remote")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
             "rm remote tmp")


# ── Phase C: local visualisation ──────────────────────────────────────────────

def compute_auc(probs: np.ndarray, k_levels: list) -> float:
    x = np.array(k_levels)
    x_norm = x / x.max() if x.max() > 0 else x
    return float(np.trapezoid(probs, x_norm))


def _load_curves(slide_list: list, methods: list, group: str,
                 task: str, curves_dir: Path) -> dict:
    """Load saved npy curves for one (group, task). Returns {sid: {method: {del, ins}}}."""
    results: dict = {}
    for entry in slide_list:
        sid = entry["slide_id"]
        results[sid] = {}
        for method in methods:
            base = curves_dir / f"{sid}_{group}_{task}_{method}"
            d, i = base.parent / (base.name + "_del.npy"), base.parent / (base.name + "_ins.npy")
            if d.exists() and i.exists():
                results[sid][method] = {"del": np.load(d), "ins": np.load(i)}
            else:
                if not d.exists():
                    print(f"    [WARN] Missing {d.name}")
    return results


def aggregate_curves(results: dict, slide_list: list, curve_type: str) -> dict:
    """Aggregate per-slide curves → {method: {mean, std}}."""
    all_methods: set = set()
    for per_method in results.values():
        all_methods |= set(per_method.keys())

    agg = {}
    for method in sorted(all_methods):
        arrays = []
        for entry in slide_list:
            entry_data = results.get(entry["slide_id"], {}).get(method)
            if entry_data and curve_type in entry_data:
                arrays.append(entry_data[curve_type])
        if arrays:
            stacked = np.stack(arrays, axis=0)
            agg[method] = {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}
    return agg


def plot_group_curves(group_name: str, tasks_data: dict,
                      k_levels: list, out_dir: Path) -> None:
    """Plot 3-row × 2-col figure (tasks × del/ins) for one group."""
    method_colors = {"attn": "steelblue", "ixg": "seagreen"}
    k_pct = [k * 100 for k in k_levels]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Phase 8 — Deletion / Insertion Curves ({group_name})", fontsize=13)

    for row_i, task in enumerate(TASKS):
        del_agg = tasks_data.get(task, {}).get("del", {})
        ins_agg = tasks_data.get(task, {}).get("ins", {})

        for col_i, (agg, curve_label) in enumerate([(del_agg, "Deletion"), (ins_agg, "Insertion")]):
            ax = axes[row_i, col_i]
            ax.set_title(f"{task.upper()} — {curve_label}", fontsize=10)
            ax.set_ylabel("Task probability", fontsize=8)
            ax.grid(True, alpha=0.3)

            for method, data in agg.items():
                color = method_colors.get(method, "gray")
                mean, std = data["mean"], data["std"]
                ax.plot(k_pct, mean, marker="o", markersize=4, label=method, color=color)
                ax.fill_between(k_pct, mean - std, mean + std, alpha=0.15, color=color)

            if agg:
                ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Patches removed / inserted (%)", fontsize=9)

    plt.tight_layout()
    fname = f"deletion_curves_{group_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def build_auc_records(all_slide_lists: dict, groups: dict,
                      curves_dir: Path) -> list:
    """Compute AUC and SRG stats for all (group, task, method) combinations."""
    records = []
    for group_name, group_cfg in groups.items():
        methods = group_cfg["methods"]
        for task in TASKS:
            slide_list = all_slide_lists.get(group_name, {}).get(task, [])
            if not slide_list:
                continue
            results = _load_curves(slide_list, methods, group_name, task, curves_dir)
            for method in methods:
                del_aucs, ins_aucs = [], []
                for entry in slide_list:
                    entry_data = results.get(entry["slide_id"], {}).get(method)
                    if entry_data:
                        del_aucs.append(compute_auc(entry_data["del"], K_LEVELS))
                        ins_aucs.append(compute_auc(entry_data["ins"], K_LEVELS))
                if del_aucs:
                    srg_vals = [ins - d for ins, d in zip(ins_aucs, del_aucs)]
                    records.append({
                        "group":          group_name,
                        "task":           task,
                        "method":         method,
                        "srg_mean":       round(float(np.mean(srg_vals)), 4),
                        "srg_std":        round(float(np.std(srg_vals)),  4),
                        "del_auc_mean":   round(float(np.mean(del_aucs)), 4),
                        "del_auc_std":    round(float(np.std(del_aucs)),  4),
                        "ins_auc_mean":   round(float(np.mean(ins_aucs)), 4),
                        "ins_auc_std":    round(float(np.std(ins_aucs)),  4),
                        "n_slides":       len(del_aucs),
                    })
    return records


def plot_auc_table(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("  [WARN] AUC table empty — skipping plot")
        return

    cols = ["group", "task", "method", "srg_mean", "srg_std",
            "del_auc_mean", "del_auc_std", "ins_auc_mean", "ins_auc_std", "n_slides"]
    col_labels = ["Group", "Task", "Method",
                  "SRG (↑)", "SRG std",
                  "Del AUC (↓)", "Del AUC std", "Ins AUC (↑)", "Ins AUC std", "N"]

    cell_data = []
    for _, row in df[cols].iterrows():
        cell_data.append([
            row["group"], row["task"], row["method"],
            f"{row['srg_mean']:.4f}",     f"{row['srg_std']:.4f}",
            f"{row['del_auc_mean']:.4f}", f"{row['del_auc_std']:.4f}",
            f"{row['ins_auc_mean']:.4f}", f"{row['ins_auc_std']:.4f}",
            str(int(row["n_slides"])),
        ])

    # Highlight row per (group, task) with highest SRG
    best_rows = set()
    for (g, t), sub in df.groupby(["group", "task"]):
        idx = sub["srg_mean"].idxmax()
        best_rows.add(df.index.get_loc(idx))

    cell_colors = []
    for ri in range(len(df)):
        cell_colors.append(["#c6efce"] * len(cols) if ri in best_rows else ["white"] * len(cols))

    fig_h = max(4, len(df) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellColours=cell_colors, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)
    ax.set_title("Phase 8 — Deletion/Insertion AUC Summary (all groups × tasks × methods)",
                 fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "deletion_auc_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved deletion_auc_table.png")


def select_canonical(auc_df: pd.DataFrame) -> str:
    """Select canonical ABMIL method by SRG rank within ensemble_baseline.

    Rank by srg_mean descending (higher = more faithful), del_auc_mean ascending as tiebreaker.
    Returns method with lowest mean rank across all tasks.
    """
    abmil_df = auc_df[auc_df["group"] == "ensemble_baseline"] if not auc_df.empty else auc_df
    if abmil_df.empty:
        return "attn"

    method_ranks: dict = {}
    for (_, _), sub in abmil_df.groupby(["group", "task"]):
        ranked = sub.sort_values(
            ["srg_mean", "del_auc_mean"],
            ascending=[False, True]
        ).reset_index(drop=True)
        for rank_i, row in ranked.iterrows():
            method_ranks.setdefault(row["method"], []).append(rank_i + 1)

    if not method_ranks:
        return "attn"

    mean_ranks = {m: np.mean(v) for m, v in method_ranks.items()}
    return min(mean_ranks, key=mean_ranks.get)


def plot_cross_model_ixg(all_slide_lists: dict, groups: dict,
                         auc_df: pd.DataFrame, curves_dir: Path,
                         k_levels: list, out_dir: Path) -> None:
    """3-row × 2-col cross-model ixg comparison: ensemble_baseline vs mean_agg."""
    group_style = {
        "ensemble_baseline": {"color": "steelblue",  "linestyle": "-",  "label": "ABMIL (ensemble_baseline)"},
        "mean_agg":          {"color": "darkorange", "linestyle": "--", "label": "Mean pool (mean_agg)"},
    }
    k_pct = [k * 100 for k in k_levels]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.suptitle("Phase 8 — Cross-Model IxG Faithfulness: ABMIL vs Mean Pool", fontsize=13)

    for row_i, task in enumerate(TASKS):
        for col_i, curve_type in enumerate(["del", "ins"]):
            ax = axes[row_i, col_i]
            curve_label = "Deletion" if curve_type == "del" else "Insertion"
            ax.set_title(f"{task.upper()} — {curve_label}", fontsize=10)
            ax.set_ylabel("Task probability", fontsize=8)
            ax.grid(True, alpha=0.3)

            for group_name, style in group_style.items():
                if group_name not in groups:
                    continue
                if "ixg" not in groups[group_name]["methods"]:
                    continue
                slide_list = all_slide_lists.get(group_name, {}).get(task, [])
                if not slide_list:
                    continue

                results = _load_curves(slide_list, ["ixg"], group_name, task, curves_dir)
                agg = aggregate_curves(results, slide_list, curve_type)
                data = agg.get("ixg")
                if data is None:
                    continue

                # Build legend label with SRG if available
                srg_label = ""
                if not auc_df.empty:
                    row = auc_df[
                        (auc_df["group"] == group_name) &
                        (auc_df["task"]  == task) &
                        (auc_df["method"] == "ixg")
                    ]
                    if not row.empty:
                        srg_val = row.iloc[0]["srg_mean"]
                        srg_label = f" (SRG={srg_val:.3f})"

                mean, std = data["mean"], data["std"]
                label = style["label"] + srg_label
                ax.plot(k_pct, mean, marker="o", markersize=4,
                        color=style["color"], linestyle=style["linestyle"], label=label)
                ax.fill_between(k_pct, mean - std, mean + std,
                                alpha=0.15, color=style["color"])

            ax.legend(fontsize=7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Patches removed / inserted (%)", fontsize=9)

    plt.tight_layout()
    fname = "deletion_curves_cross_model_ixg.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def phase_c_visualise(groups: dict, all_slide_lists: dict,
                      deletion_dir: Path, figures_dir: Path) -> str:
    """Generate figures and return canonical method name."""
    curves_dir = deletion_dir / "curves"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nPhase C1 — building AUC records")
    records = build_auc_records(all_slide_lists, groups, curves_dir)
    auc_df = pd.DataFrame(records)
    if not auc_df.empty:
        auc_csv = deletion_dir / "auc_table.csv"
        auc_df.to_csv(auc_csv, index=False)
        print(f"  Saved auc_table.csv ({len(auc_df)} rows)")
        print(auc_df.to_string(index=False))

    print("\nPhase C2 — plotting per-group curves")
    for group_name, group_cfg in groups.items():
        methods = group_cfg["methods"]
        tasks_data: dict = {}
        for task in TASKS:
            slide_list = all_slide_lists.get(group_name, {}).get(task, [])
            if not slide_list:
                tasks_data[task] = {"del": {}, "ins": {}}
                continue
            results = _load_curves(slide_list, methods, group_name, task, curves_dir)
            tasks_data[task] = {
                "del": aggregate_curves(results, slide_list, "del"),
                "ins": aggregate_curves(results, slide_list, "ins"),
            }
        plot_group_curves(group_name, tasks_data, K_LEVELS, figures_dir)

    print("\nPhase C3 — AUC table figure")
    plot_auc_table(auc_df, figures_dir)

    print("\nPhase C4 — cross-model IxG comparison figure")
    plot_cross_model_ixg(all_slide_lists, groups, auc_df, curves_dir, K_LEVELS, figures_dir)

    canonical = select_canonical(auc_df)
    print(f"\n  Canonical method: {canonical}")

    (deletion_dir / "canonical_method.txt").write_text(canonical + "\n")

    summary = {
        "groups":           {g: cfg["run_names"] for g, cfg in groups.items()},
        "canonical_method": canonical,
        "auc_table":        records,
    }
    (deletion_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("  Saved canonical_method.txt and summary.json")

    return canonical


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",       default="reports/data/phase8/",
                   help="Phase 8 data root (default: reports/data/phase8/)")
    p.add_argument("--groups",         default=None,
                   help="Comma-separated subset of groups (default: all)")
    p.add_argument("--gcp-host",       default=None,
                   help="GCP server IP (omit for dry-run / visualise-only)")
    p.add_argument("--gcp-user",       default="chris")
    p.add_argument("--gcp-pass",       default=None)
    p.add_argument("--visualise-only", action="store_true",
                   help="Skip GCP; visualise from existing local curve files")
    p.add_argument("--no-report",      action="store_true",
                   help="Skip appending to phase8-results.md")
    p.add_argument("--force",          action="store_true",
                   help="Regenerate slide lists even if they already exist")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir      = ROOT / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    inference_dir = data_dir / "inference"
    deletion_dir  = data_dir / "deletion"
    curves_dir    = deletion_dir / "curves"

    # Filter groups if requested
    active_groups = dict(GROUPS)
    if args.groups:
        requested = [g.strip() for g in args.groups.split(",")]
        active_groups = {k: v for k, v in GROUPS.items() if k in requested}
        unknown = set(requested) - set(GROUPS)
        if unknown:
            print(f"[WARN] Unknown groups: {unknown}")

    # ── visualise-only shortcut ──────────────────────────────────────────────
    if args.visualise_only:
        print("--visualise-only: loading slide lists from disk")
        all_slide_lists = load_slide_lists(active_groups, deletion_dir)
        for g, per_task in all_slide_lists.items():
            for t, sl in per_task.items():
                print(f"  {g}/{t}: {len(sl)} slides")
        canonical = phase_c_visualise(active_groups, all_slide_lists,
                                      deletion_dir, FIGURES_DIR)
        print(f"\nDone. Canonical method: {canonical}")
        return

    # ── Phase A0 — mean_agg inference (if needed and GCP available) ──────────
    if MEAN_AGG_GROUP in active_groups:
        if not _mean_agg_inference_exists(inference_dir):
            if args.gcp_host is None:
                print(f"[WARN] mean_agg inference missing at {inference_dir} "
                      "and --gcp-host not provided. "
                      "mean_agg group will be skipped.")
                active_groups = {k: v for k, v in active_groups.items()
                                 if k != MEAN_AGG_GROUP}
            else:
                if args.gcp_pass is None:
                    print("ERROR: --gcp-pass required when --gcp-host is set",
                          file=sys.stderr)
                    sys.exit(1)
                phase_a0_extract_mean_agg(inference_dir, args.gcp_host,
                                          args.gcp_user, args.gcp_pass)
        else:
            print(f"  mean_agg inference already present — skipping Phase A0")

    # ── Phase A0b — global mean embedding (if needed and GCP available) ──────
    if not _global_mean_exists(inference_dir):
        if args.gcp_host is None:
            print(f"[WARN] global_mean_emb.npy missing at {inference_dir} "
                  "and --gcp-host not provided. Cannot proceed without global mean.")
            sys.exit(1)
        else:
            if args.gcp_pass is None:
                print("ERROR: --gcp-pass required when --gcp-host is set",
                      file=sys.stderr)
                sys.exit(1)
            phase_a0b_compute_global_mean(inference_dir, args.gcp_host,
                                          args.gcp_user, args.gcp_pass)
    else:
        print(f"  global_mean_emb.npy already present — skipping Phase A0b")

    # ── Phase A — build shared balanced slide lists ───────────────────────────
    print("\nPhase A — building shared balanced per-task slide lists")
    all_slide_lists = phase_a_build_slide_lists(
        active_groups, inference_dir, deletion_dir, force=args.force
    )

    # Summary
    print("\nSlide counts:")
    for g, per_task in all_slide_lists.items():
        for t, sl in per_task.items():
            print(f"  {g}/{t}: {len(sl)} slides")

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        if curves_dir.exists() and list(curves_dir.glob("*_del.npy")):
            print("Local curve data found — running visualisation")
            canonical = phase_c_visualise(active_groups, all_slide_lists,
                                          deletion_dir, FIGURES_DIR)
            print(f"  Canonical method: {canonical}")
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase B — remote compute ──────────────────────────────────────────────
    print("\nPhase B — remote deletion/insertion compute")
    global_mean_remote = f"{GCP_REMOTE_TMP}/global_mean_emb.npy"
    manifest = build_manifest(active_groups, all_slide_lists, global_mean_remote)
    phase_b_remote_compute(manifest, curves_dir, inference_dir,
                           args.gcp_host, args.gcp_user, args.gcp_pass)

    print(f"\nDone.")
    print(f"  Curves dir : {curves_dir}")
    print(f"  Run --visualise-only to generate figures and AUC table.")


if __name__ == "__main__":
    main()
