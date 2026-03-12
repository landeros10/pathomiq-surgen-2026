#!/usr/bin/env python3
"""Phase 7 Step 4 — Patch Deletion / Insertion Study.

Ranks patches by three attribution methods (attn, grad, ixg) and measures:
  - Deletion curve : remove top-k% patches → prediction should drop
  - Insertion curve: add top-k% patches to mean-baseline bag → prediction recovers

AUC comparison selects the canonical attribution method for the Phase 7 report.

Steps 2 and 3 may not have been run on GCP yet — if local attr data is missing,
only the 'attn' method runs.

Usage:
    # Full run (GCP required):
    python scripts/studies/phase7_deletion.py \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase7_deletion.py

    # Skip GCP if data already downloaded:
    python scripts/studies/phase7_deletion.py --visualise-only
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from phase7_utils import (          # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    INFERENCE_DIR,
    FIGURES_DIR,
    load_study_set,
    mlflow_run_id,
)

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p7_del"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"
GCP_INFERENCE   = "/home/chris/surgen/tmp/phase6-report-data/inference"
GCP_ATTR_DIR    = "/home/chris/surgen/tmp/phase7-attr-data"

LOCAL_DATA_DIR  = ROOT / "tmp" / "phase7-del-data"
LOCAL_ATTR_DIR  = ROOT / "tmp" / "phase7-attr-data"
DEL_OUT_DIR     = FIGURES_DIR / "deletion"

K_LEVELS        = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]

ST_RUN_ID_FALLBACK = "e60812c9ab0e462080a6bde3bc0141fe"
MT_RUN_ID_FALLBACK = "1013f94132dd4ea092266b334abc3c2e"


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_DEL_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side deletion/insertion computation for Phase 7 Step 4.\"\"\"
import json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "{root_dir}")
sys.path.insert(0, "{scripts_dir}")
sys.path.insert(0, "{scripts_dir}/models")
sys.path.insert(0, "{scripts_dir}/etl")

import mlflow
import zarr

MLFLOW_DB     = "{mlflow_db}"
ZARR_DIR      = Path("{zarr_dir}")
INFERENCE_DIR = Path("{inference_dir}")
ATTR_DIR      = Path("{attr_dir}")
OUT_DIR       = Path("{out_dir}")
MANIFEST      = json.loads(Path("{manifest_path}").read_text())

ST_RUN_ID   = MANIFEST["st_run_id"]
MT_RUN_ID   = MANIFEST["mt_run_id"]
ST_RUN_NAME = MANIFEST["st_run_name"]
MT_RUN_NAME = MANIFEST["mt_run_name"]
METHODS     = MANIFEST["methods"]
K_LEVELS    = MANIFEST["k_levels"]
SLIDES      = MANIFEST["slides"]

OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return store["features"][:]  # (N, 1024) numpy float32


def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    model.eval()
    return model


def _get_prob(model, emb_np, device, is_multitask):
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(emb)
    if out.ndim == 0:
        logit = out
    elif out.ndim == 1:
        logit = out[0]      # ST: (B,) → scalar
    else:
        logit = out[0, 0]   # MT: (B, T) → MMR logit
    return float(torch.sigmoid(logit).item())


def run_deletion(features_np, scores, k_levels, model, device, is_mt):
    N = features_np.shape[0]
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_remove = int(k * N)
        bag = features_np.copy()
        if n_remove > 0:
            bag[rank_desc[:n_remove]] = 0.0
        probs.append(_get_prob(model, bag, device, is_mt))
    return np.array(probs, dtype=np.float32)


def run_insertion(features_np, scores, k_levels, model, device, is_mt):
    N = features_np.shape[0]
    mean_emb = features_np.mean(axis=0)  # (1024,)
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_insert = int(k * N)
        bag = np.tile(mean_emb, (N, 1))  # (N, 1024) mean baseline
        if n_insert > 0:
            bag[rank_desc[:n_insert]] = features_np[rank_desc[:n_insert]]
        probs.append(_get_prob(model, bag, device, is_mt))
    return np.array(probs, dtype=np.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

print("Loading ST model ...")
st_model = _load_model(ST_RUN_ID)
st_model.to(device)
print("Loading MT model ...")
mt_model = _load_model(MT_RUN_ID)
mt_model.to(device)

for slide in SLIDES:
    sid    = slide["slide_id"]
    st_idx = slide["st_idx"]
    mt_idx = slide["mt_idx"]
    print(f"\\nProcessing {{sid}} ...")

    try:
        features_np = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    N = features_np.shape[0]

    # Collect per-method attribution scores: {{method: {{"st": arr, "mt": arr}}}}
    scores_dict = {{}}

    if "attn" in METHODS:
        try:
            st_attn_path = INFERENCE_DIR / ST_RUN_NAME / "attn" / f"{{st_idx:04d}}.npy"
            mt_attn_path = INFERENCE_DIR / MT_RUN_NAME / "attn" / f"{{mt_idx:04d}}.npy"
            st_attn = np.load(str(st_attn_path))
            mt_attn = np.load(str(mt_attn_path))
            if st_attn.ndim == 2:
                st_attn = st_attn[:, 0]
            if mt_attn.ndim == 2:
                mt_attn = mt_attn[:, 0]
            if st_attn.shape[0] != N:
                print(f"  [WARN] attn N={{st_attn.shape[0]}} != zarr N={{N}} for ST {{sid}}, skipping attn")
            elif mt_attn.shape[0] != N:
                print(f"  [WARN] attn N={{mt_attn.shape[0]}} != zarr N={{N}} for MT {{sid}}, skipping attn")
            else:
                scores_dict["attn"] = {{"st": st_attn, "mt": mt_attn}}
        except Exception as e:
            print(f"  [WARN] attn load failed for {{sid}}: {{e}}")

    for method in ("grad", "ixg"):
        if method not in METHODS:
            continue
        try:
            st_arr = np.load(str(ATTR_DIR / f"{{sid}}_st_{{method}}.npy"))
            mt_arr = np.load(str(ATTR_DIR / f"{{sid}}_mt_{{method}}.npy"))
            if st_arr.shape[0] != N:
                print(f"  [WARN] {{method}} N mismatch for ST {{sid}}, skipping")
                continue
            if mt_arr.shape[0] != N:
                print(f"  [WARN] {{method}} N mismatch for MT {{sid}}, skipping")
                continue
            scores_dict[method] = {{"st": st_arr, "mt": mt_arr}}
        except Exception as e:
            print(f"  [WARN] {{method}} load failed for {{sid}}: {{e}}")

    for method, model_scores in scores_dict.items():
        for prefix, model, is_mt in (("st", st_model, False), ("mt", mt_model, True)):
            scores = model_scores[prefix]
            try:
                del_probs = run_deletion(features_np, scores, K_LEVELS, model, device, is_mt)
                ins_probs = run_insertion(features_np, scores, K_LEVELS, model, device, is_mt)
                np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_{{method}}_del.npy"), del_probs)
                np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_{{method}}_ins.npy"), ins_probs)
                k20_drop = (del_probs[0] - del_probs[3]) * 100
                print(f"  {{prefix}}/{{method}}: del@0%={{del_probs[0]:.3f}} "
                      f"del@20%={{del_probs[3]:.3f}} (drop={{k20_drop:.1f}}pp) "
                      f"ins@0%={{ins_probs[0]:.3f}} ins@50%={{ins_probs[-1]:.3f}}")
            except Exception as exc:
                print(f"  [ERROR] {{prefix}}/{{method}} {{sid}}: {{exc}}")

print("\\nDone.")
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
        # Stream stdout/stderr live for long-running SSH commands
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


# ── Phase 1: local prep ────────────────────────────────────────────────────────

def detect_available_methods(study_set: list) -> list:
    """Always returns ['attn']; adds 'grad'/'ixg' if all 19 slides have both files."""
    methods = ["attn"]
    slide_ids = [s["slide_id"] for s in study_set]
    for method in ("grad", "ixg"):
        all_present = all(
            (LOCAL_ATTR_DIR / f"{sid}_st_{method}.npy").exists()
            and (LOCAL_ATTR_DIR / f"{sid}_mt_{method}.npy").exists()
            for sid in slide_ids
        )
        if all_present:
            methods.append(method)
        else:
            print(f"  [INFO] Method '{method}' not fully available in {LOCAL_ATTR_DIR} — skipping")
    return methods


def _resolve_mt_idx(study_set: list) -> dict:
    """Return {slide_id: mt_idx} by looking up each slide in MT slide_ids.json."""
    mt_ids_path = INFERENCE_DIR / MULTITASK_RUN / "slide_ids.json"
    if not mt_ids_path.exists():
        raise FileNotFoundError(f"MT slide_ids.json not found: {mt_ids_path}")
    mt_ids: list = json.loads(mt_ids_path.read_text())
    idx_map = {sid: i for i, sid in enumerate(mt_ids)}
    result = {}
    for slide in study_set:
        sid = slide["slide_id"]
        if sid not in idx_map:
            raise KeyError(f"Slide '{sid}' not found in MT slide_ids.json")
        result[sid] = idx_map[sid]
    return result


def build_manifest(study_set: list, st_run_id: str, mt_run_id: str,
                   methods: list) -> dict:
    mt_idx_map = _resolve_mt_idx(study_set)
    slides = []
    for s in study_set:
        sid = s["slide_id"]
        slides.append({
            "slide_id":  sid,
            "category":  s["category"],
            "mmr_label": s["mmr_label"],
            "st_idx":    s["idx"],
            "mt_idx":    mt_idx_map[sid],
        })
    return {
        "st_run_id":   st_run_id,
        "mt_run_id":   mt_run_id,
        "st_run_name": SINGLETASK_RUN,
        "mt_run_name": MULTITASK_RUN,
        "methods":     methods,
        "k_levels":    K_LEVELS,
        "slides":      slides,
    }


# ── Phase 2: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local  = tmp / "manifest.json"
        script_local    = tmp / "del_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/del_compute.py"

        manifest_local.write_text(json.dumps(manifest, indent=2))

        script_content = REMOTE_DEL_PY.format(
            root_dir      = GCP_ROOT_DIR,
            scripts_dir   = GCP_SCRIPTS_DIR,
            mlflow_db     = GCP_MLFLOW_DB,
            zarr_dir      = GCP_ZARR_DIR,
            inference_dir = GCP_INFERENCE,
            attr_dir      = GCP_ATTR_DIR,
            out_dir       = GCP_REMOTE_TMP + "/out",
            manifest_path = manifest_remote,
        )
        script_local.write_text(script_content)

        print("\nPhase 2a — prepare remote directory")
        _run(_ssh_cmd(host, user, password,
                      f"mkdir -p {GCP_REMOTE_TMP}/out "
                      f"{GCP_INFERENCE}/{SINGLETASK_RUN}/attn "
                      f"{GCP_INFERENCE}/{MULTITASK_RUN}/attn "
                      f"{GCP_ATTR_DIR}"),
             "mkdir remote tmp")

        print("Phase 2b — upload manifest + script")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp del_compute.py")

        # Phase 2b2 — zip + upload phase-6 attn files
        local_st_attn = INFERENCE_DIR / SINGLETASK_RUN / "attn"
        local_mt_attn = INFERENCE_DIR / MULTITASK_RUN / "attn"
        zip_local = tmp / "attn.zip"
        with zipfile.ZipFile(zip_local, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(local_st_attn.glob("*.npy")):
                zf.write(f, f"{SINGLETASK_RUN}/attn/{f.name}")
            for f in sorted(local_mt_attn.glob("*.npy")):
                zf.write(f, f"{MULTITASK_RUN}/attn/{f.name}")
        zip_remote = f"{GCP_REMOTE_TMP}/attn.zip"
        print("Phase 2b2 — upload + unzip phase-6 attn files")
        _run(_scp_to(host, user, password, str(zip_local), zip_remote), "scp attn.zip")
        _run(_ssh_cmd(host, user, password,
                      f"mkdir -p {GCP_INFERENCE} && "
                      f"cd {GCP_INFERENCE} && unzip -o {zip_remote} && rm {zip_remote}"),
             "unzip attn on GCP")

        print("Phase 2c — run deletion/insertion on GCP  (~5 min)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 del_compute.py")

        print("Phase 2d — download results (tar-transfer)")
        _run(_ssh_cmd(host, user, password,
                      f"cd {GCP_REMOTE_TMP} && tar czf out.tar.gz out/ && echo 'tar done'"),
             "tar results on GCP")
        out_tar_local = tmp / "out.tar.gz"
        _run(_scp_from(host, user, password,
                       f"{GCP_REMOTE_TMP}/out.tar.gz",
                       str(out_tar_local)),
             "scp out.tar.gz → local")
        _run(_ssh_cmd(host, user, password,
                      f"rm {GCP_REMOTE_TMP}/out.tar.gz"),
             "rm remote out.tar.gz")
        import tarfile
        with tarfile.open(out_tar_local) as tf:
            tf.extractall(tmp)
        out_src = tmp / "out"
        for item in out_src.iterdir():
            dest = LOCAL_DATA_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(LOCAL_DATA_DIR))
        print(f"  Extracted results to {LOCAL_DATA_DIR}")

        print("Phase 2e — clean up remote")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
             "rm remote tmp")


# ── Phase 3: local visualisation ──────────────────────────────────────────────

def compute_auc(probs: np.ndarray, k_levels: list) -> float:
    """Trapezoidal AUC; x-axis normalised to [0, 1]."""
    x = np.array(k_levels)
    x_norm = x / x.max()
    return float(np.trapezoid(probs, x_norm))


def load_del_results(study_set: list, methods: list):
    """Load saved npy files → (del_results, ins_results).

    Returns nested dicts:
        {slide_id: {"st": {method: arr(6,)}, "mt": {method: arr(6,)}}}
    """
    del_results: dict = {}
    ins_results: dict = {}
    for slide in study_set:
        sid = slide["slide_id"]
        del_results[sid] = {"st": {}, "mt": {}}
        ins_results[sid] = {"st": {}, "mt": {}}
        for prefix in ("st", "mt"):
            for method in methods:
                del_path = LOCAL_DATA_DIR / f"{sid}_{prefix}_{method}_del.npy"
                ins_path = LOCAL_DATA_DIR / f"{sid}_{prefix}_{method}_ins.npy"
                if del_path.exists():
                    del_results[sid][prefix][method] = np.load(del_path)
                else:
                    print(f"  [WARN] Missing {del_path.name}")
                if ins_path.exists():
                    ins_results[sid][prefix][method] = np.load(ins_path)
                else:
                    print(f"  [WARN] Missing {ins_path.name}")
    return del_results, ins_results


def aggregate_curves(results_per_model: dict, study_set: list) -> dict:
    """Aggregate per-slide curve arrays for one model.

    Args:
        results_per_model: {slide_id: {method: arr(6,)}}
        study_set: list of slide dicts (for ordering)

    Returns:
        {method: {"mean": arr(6,), "std": arr(6,)}}
    """
    all_methods = set()
    for per_method in results_per_model.values():
        all_methods |= set(per_method.keys())

    agg = {}
    for method in sorted(all_methods):
        arrays = []
        for slide in study_set:
            sid = slide["slide_id"]
            arr = results_per_model.get(sid, {}).get(method)
            if arr is not None:
                arrays.append(arr)
        if arrays:
            stacked = np.stack(arrays, axis=0)  # (n_slides, 6)
            agg[method] = {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}
    return agg


def plot_curves(del_agg: dict, ins_agg: dict, model_label: str,
                k_levels: list, out_dir: Path) -> None:
    """Plot deletion and insertion curves for one model; saves 2 PNGs."""
    k_pct = [k * 100 for k in k_levels]
    method_colors = {"attn": "steelblue", "grad": "darkorange", "ixg": "seagreen"}

    for curve_type, agg, ylabel, fname in (
        ("Deletion",  del_agg, "MMR probability", f"deletion_curves_{model_label}.png"),
        ("Insertion", ins_agg, "MMR probability", f"insertion_curves_{model_label}.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, data in agg.items():
            color = method_colors.get(method, None)
            mean  = data["mean"]
            std   = data["std"]
            ax.plot(k_pct, mean, marker="o", label=method, color=color)
            ax.fill_between(k_pct, mean - std, mean + std, alpha=0.15, color=color)
            if curve_type == "Deletion":
                # Annotate k=20% drop
                k20_idx = k_levels.index(0.20)
                drop = (mean[0] - mean[k20_idx]) * 100
                ax.annotate(
                    f"−{drop:.1f}pp",
                    xy=(20, mean[k20_idx]),
                    xytext=(22, mean[k20_idx] + 0.03),
                    fontsize=7, color=color,
                )

        ax.set_xlabel("Patches removed / inserted (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Phase 7 Step 4 — {curve_type} Curves ({model_label.upper()})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def build_auc_table(del_results: dict, ins_results: dict, methods: list,
                    study_set: list, k_levels: list) -> pd.DataFrame:
    """Build AUC summary DataFrame.

    Rows: (method, model) combos.
    Columns: del_auc_mean, del_auc_std, ins_auc_mean, ins_auc_std.
    """
    records = []
    for method in methods:
        for model_prefix in ("st", "mt"):
            del_aucs, ins_aucs = [], []
            for slide in study_set:
                sid = slide["slide_id"]
                d = del_results.get(sid, {}).get(model_prefix, {}).get(method)
                i = ins_results.get(sid, {}).get(model_prefix, {}).get(method)
                if d is not None:
                    del_aucs.append(compute_auc(d, k_levels))
                if i is not None:
                    ins_aucs.append(compute_auc(i, k_levels))
            if del_aucs or ins_aucs:
                records.append({
                    "method":       method,
                    "model":        model_prefix,
                    "del_auc_mean": float(np.mean(del_aucs)) if del_aucs else float("nan"),
                    "del_auc_std":  float(np.std(del_aucs))  if del_aucs else float("nan"),
                    "ins_auc_mean": float(np.mean(ins_aucs)) if ins_aucs else float("nan"),
                    "ins_auc_std":  float(np.std(ins_aucs))  if ins_aucs else float("nan"),
                })
    return pd.DataFrame(records)


def plot_auc_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Render AUC table as PNG; highlight winning row in green."""
    if df.empty:
        print("  [WARN] AUC table is empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.5 + 1)))
    ax.axis("off")

    cols = ["method", "model", "del_auc_mean", "del_auc_std",
            "ins_auc_mean", "ins_auc_std"]
    col_labels = ["Method", "Model", "Del AUC (↓)", "Del AUC std",
                  "Ins AUC (↑)", "Ins AUC std"]
    cell_data = []
    for _, row in df[cols].iterrows():
        cell_data.append([
            row["method"],
            row["model"],
            f"{row['del_auc_mean']:.4f}",
            f"{row['del_auc_std']:.4f}",
            f"{row['ins_auc_mean']:.4f}",
            f"{row['ins_auc_std']:.4f}",
        ])

    # Find winning row (min del_auc_mean for ST)
    st_mask = df["model"] == "st"
    win_row = None
    if st_mask.any():
        win_row = df.loc[st_mask, "del_auc_mean"].idxmin()
        win_row = df.index.get_loc(win_row)

    cell_colors = [["white"] * len(cols) for _ in range(len(df))]
    if win_row is not None:
        cell_colors[win_row] = ["#c6efce"] * len(cols)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)

    ax.set_title("Phase 7 Step 4 — Deletion/Insertion AUC Summary",
                 fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "auc_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved auc_table.png")


def select_canonical_method(auc_df: pd.DataFrame) -> str:
    """Return method with lowest del_auc_mean for ST; tie-break by max ins_auc_mean."""
    if auc_df.empty:
        return "attn"
    st_df = auc_df[auc_df["model"] == "st"].copy()
    if st_df.empty:
        return "attn"
    st_df = st_df.sort_values(
        ["del_auc_mean", "ins_auc_mean"],
        ascending=[True, False],
    )
    return str(st_df.iloc[0]["method"])


def gate_check(study_set: list, del_results: dict, k_levels: list,
               canonical_method: str = "attn") -> dict:
    """PASS if > 50% of correct_msi slides show >= 5pp drop at k=20%."""
    k20_idx = k_levels.index(0.20)
    eligible = [s for s in study_set if s["category"] == "correct_msi"]
    details = []
    n_pass = 0
    for slide in eligible:
        sid = slide["slide_id"]
        probs = del_results.get(sid, {}).get("st", {}).get(canonical_method)
        if probs is None:
            print(f"  [WARN] gate_check: no ST/{canonical_method} data for {sid}")
            continue
        drop_pp = float((probs[0] - probs[k20_idx]) * 100)
        passed = drop_pp >= 5.0
        if passed:
            n_pass += 1
        details.append({"slide_id": sid, "drop_pp": drop_pp, "passed": passed})

    n_eligible = len(details)
    gate_passed = n_eligible > 0 and n_pass > n_eligible / 2
    return {
        "passed":     gate_passed,
        "n_pass":     n_pass,
        "n_eligible": n_eligible,
        "details":    details,
    }


def visualise(study_set: list, methods: list) -> None:
    DEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPhase 3a — loading saved del/ins npy files")
    del_results, ins_results = load_del_results(study_set, methods)

    # Separate by model
    del_st = {sid: del_results[sid]["st"] for sid in del_results}
    del_mt = {sid: del_results[sid]["mt"] for sid in del_results}
    ins_st = {sid: ins_results[sid]["st"] for sid in ins_results}
    ins_mt = {sid: ins_results[sid]["mt"] for sid in ins_results}

    print("Phase 3b — aggregating curves")
    del_agg_st = aggregate_curves(del_st, study_set)
    del_agg_mt = aggregate_curves(del_mt, study_set)
    ins_agg_st = aggregate_curves(ins_st, study_set)
    ins_agg_mt = aggregate_curves(ins_mt, study_set)

    print("Phase 3c — plotting curves")
    plot_curves(del_agg_st, ins_agg_st, "st", K_LEVELS, DEL_OUT_DIR)
    plot_curves(del_agg_mt, ins_agg_mt, "mt", K_LEVELS, DEL_OUT_DIR)

    print("Phase 3d — building AUC table")
    auc_df = build_auc_table(del_results, ins_results, methods, study_set, K_LEVELS)
    print(auc_df.to_string(index=False))
    plot_auc_table(auc_df, DEL_OUT_DIR)

    canonical = select_canonical_method(auc_df)
    print(f"  Canonical method: {canonical}")
    (DEL_OUT_DIR / "canonical_method.txt").write_text(canonical + "\n")

    print("Phase 3e — gate check")
    gate = gate_check(study_set, del_results, K_LEVELS, canonical)
    status = "PASS" if gate["passed"] else "FAIL"
    print(f"  Gate: {status}  ({gate['n_pass']}/{gate['n_eligible']} slides ≥ 5pp drop)")

    # Build per_slide summary
    per_slide = {}
    for slide in study_set:
        sid = slide["slide_id"]
        entry = {}
        for prefix in ("st", "mt"):
            for method in methods:
                d = del_results.get(sid, {}).get(prefix, {}).get(method)
                i = ins_results.get(sid, {}).get(prefix, {}).get(method)
                if d is not None:
                    entry[f"{prefix}_{method}_del"] = d.tolist()
                if i is not None:
                    entry[f"{prefix}_{method}_ins"] = i.tolist()
        per_slide[sid] = entry

    summary = {
        "methods_available": methods,
        "canonical_method":  canonical,
        "gate":              gate,
        "auc_table":         auc_df.to_dict(orient="records"),
        "per_slide":         per_slide,
    }
    (DEL_OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary.json")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcp-host",       default=None,
                   help="GCP server IP (omit for dry-run)")
    p.add_argument("--gcp-user",       default="chris")
    p.add_argument("--gcp-pass",       default=None)
    p.add_argument("--visualise-only", action="store_true",
                   help="Skip GCP; visualise from existing LOCAL_DATA_DIR files")
    return p.parse_args()


def main():
    args = parse_args()

    if args.visualise_only:
        print("--visualise-only: skipping Phases 1 & 2")
        study_set = load_study_set()
        print(f"  {len(study_set)} slides loaded")
        # Detect methods from already-downloaded local data
        npy_files = list(LOCAL_DATA_DIR.glob("*_del.npy"))
        methods = ["attn"]
        for m in ("grad", "ixg"):
            if any(f"_{m}_del" in f.name for f in npy_files):
                methods.append(m)
        print(f"  Methods detected: {methods}")
        visualise(study_set, methods)
        return

    # ── Phase 1 — local prep ──────────────────────────────────────────────────
    print("Phase 1 — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} slides: "
          + ", ".join(s["slide_id"] for s in study_set))

    print("\nPhase 1b — detecting available attribution methods")
    methods = detect_available_methods(study_set)
    print(f"  Methods: {methods}")

    print("\nPhase 1c — resolving MLflow run IDs")
    try:
        st_run_id = mlflow_run_id(SINGLETASK_RUN)
        mt_run_id = mlflow_run_id(MULTITASK_RUN)
        print(f"  ST run ID: {st_run_id}")
        print(f"  MT run ID: {mt_run_id}")
    except (FileNotFoundError, KeyError) as e:
        print(f"  [WARN] Cannot resolve run IDs locally: {e}")
        st_run_id = ST_RUN_ID_FALLBACK
        mt_run_id = MT_RUN_ID_FALLBACK
        print(f"  Using hard-coded IDs: ST={st_run_id}, MT={mt_run_id}")

    print("\nPhase 1d — building manifest")
    manifest = build_manifest(study_set, st_run_id, mt_run_id, methods)
    print(f"  {len(manifest['slides'])} slides, methods={manifest['methods']}")
    for s in manifest["slides"]:
        print(f"    {s['slide_id']:40s} st_idx={s['st_idx']:4d}  mt_idx={s['mt_idx']:4d}"
              f"  [{s['category']}]")

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase 2 — remote compute ──────────────────────────────────────────────
    remote_compute(manifest, args.gcp_host, args.gcp_user, args.gcp_pass)

    # ── Phase 3 — local visualisation ─────────────────────────────────────────
    print("\nPhase 3 — local visualisation")
    visualise(study_set, methods)

    print(f"\nDone.")
    print(f"  Deletion curves : {DEL_OUT_DIR}/deletion_curves_st.png")
    print(f"  Insertion curves: {DEL_OUT_DIR}/insertion_curves_st.png")
    print(f"  AUC table       : {DEL_OUT_DIR}/auc_table.png")
    print(f"  Summary         : {DEL_OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
