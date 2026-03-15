#!/usr/bin/env python3
"""Phase 8 Interpretability Step 2 — Deletion / Insertion Analysis.

Ranks patches by three attribution methods (attn, gradnorm, ixg) and measures:
  - Deletion curve : remove top-k% patches → MMR prediction should drop
  - Insertion curve: add top-k% patches to mean-baseline bag → prediction recovers

Compares winner config vs baseline to identify the canonical attribution method.
AUC comparison selects the canonical method used in subsequent interpretability
sections (cross-head heatmaps, MC Dropout).

Three-phase pattern:
  Phase A (local) : Load study set, build manifest, upload to GCP
  Phase B (remote): Compute deletion/insertion curves for 3 methods × 2 runs
  Phase C (local) : Download results, generate figures, append to report

Outputs:
  reports/figures/phase8/deletion_curves_winner.png
  reports/figures/phase8/deletion_curves_baseline.png
  reports/figures/phase8/deletion_auc_table.png
  reports/figures/phase8/deletion/canonical_method.txt
  Appends '### Deletion Analysis' to reports/phase8-results.md

Usage:
    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase8_deletion.py \\
        --data-dir phase8-data/ --best-run config_phase8_baseline_seed0

    # Full run (GCP required):
    python scripts/studies/phase8_deletion.py \\
        --data-dir phase8-data/ --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'

    # Re-visualise from already-downloaded results:
    python scripts/studies/phase8_deletion.py \\
        --data-dir phase8-data/ --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --visualise-only
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import build_attn_grid  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p8_del"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"

FIGURES_DIR     = ROOT / "reports" / "figures" / "phase8"
LOCAL_DATA_DIR  = ROOT / "results" / "phase8" / "deletion"
REPORT_PATH     = ROOT / "reports" / "phase8-results.md"
STUDY_SET_PATH  = FIGURES_DIR / "study_set_phase8.json"

K_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
TASKS    = ["mmr", "ras", "braf"]


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_DEL_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side deletion/insertion computation for Phase 8 Step 2.\"\"\"
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

MLFLOW_DB  = "{mlflow_db}"
ZARR_DIR   = Path("{zarr_dir}")
OUT_DIR    = Path("{out_dir}")
MANIFEST   = json.loads(Path("{manifest_path}").read_text())

WINNER_RUN    = MANIFEST["winner_run"]
BASELINE_RUN  = MANIFEST["baseline_run"]
METHODS       = MANIFEST["methods"]
K_LEVELS      = MANIFEST["k_levels"]
SLIDES        = MANIFEST["slides"]

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
    return store["features"][:]  # (N, 1024) numpy float32


def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    model.eval()
    return model


def _get_mmr_prob(model, emb_np, device):
    \"\"\"Get MMR prediction probability from multitask model.\"\"\"
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(emb)
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 0:
        logit = out
    elif out.ndim == 1:
        logit = out[0]
    else:
        logit = out[0, 0]  # MMR head (index 0)
    return float(torch.sigmoid(logit).item())


def _get_attn(model, emb_np, device):
    \"\"\"Get attention weights (N, T) or (N,) from forward pass.\"\"\"
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(emb, return_weights=True)
    if not isinstance(out, tuple):
        return None
    _, weights = out
    if weights is None:
        return None
    w = weights.detach().cpu().numpy()
    if w.ndim == 3:
        w = w[0]   # (B, N, T) → (N, T)
    elif w.ndim == 2 and w.shape[0] == 1:
        w = w[0]   # (1, N) → (N,)
    return w  # (N, T) or (N,)


def _get_grad_ixg(model, emb_np, device):
    \"\"\"Compute GradNorm and InputXGrad for MMR head.\"\"\"
    emb_leaf = torch.tensor(emb_np, dtype=torch.float32, device=device,
                            requires_grad=True)
    emb = emb_leaf.unsqueeze(0)
    out = model(emb)
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 0:
        logit = out
    elif out.ndim == 1:
        logit = out[0]
    else:
        logit = out[0, 0]  # MMR head
    logit.backward()
    grad = emb_leaf.grad.detach().cpu().numpy()  # (N, 1024)
    emb_val = emb_leaf.detach().cpu().numpy()    # (N, 1024)
    gradnorm = np.linalg.norm(grad, axis=1).astype(np.float32)
    ixg      = np.linalg.norm(grad * emb_val, axis=1).astype(np.float32)
    return gradnorm, ixg


def run_deletion(features_np, scores, k_levels, model, device):
    N = features_np.shape[0]
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_remove = int(k * N)
        bag = features_np.copy()
        if n_remove > 0:
            bag[rank_desc[:n_remove]] = 0.0
        probs.append(_get_mmr_prob(model, bag, device))
    return np.array(probs, dtype=np.float32)


def run_insertion(features_np, scores, k_levels, model, device):
    N = features_np.shape[0]
    mean_emb  = features_np.mean(axis=0)
    rank_desc = np.argsort(scores)[::-1]
    probs = []
    for k in k_levels:
        n_insert = int(k * N)
        bag = np.tile(mean_emb, (N, 1))
        if n_insert > 0:
            bag[rank_desc[:n_insert]] = features_np[rank_desc[:n_insert]]
        probs.append(_get_mmr_prob(model, bag, device))
    return np.array(probs, dtype=np.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

print(f"Resolving winner run ID: {{WINNER_RUN}}")
winner_run_id = _resolve_run_id(WINNER_RUN)
print(f"  winner run_id: {{winner_run_id}}")

baseline_run_id = None
if BASELINE_RUN:
    try:
        baseline_run_id = _resolve_run_id(BASELINE_RUN)
        print(f"  baseline run_id: {{baseline_run_id}}")
    except KeyError as e:
        print(f"  [WARN] {{e}} — skipping baseline")

print("Loading winner model ...")
winner_model = _load_model(winner_run_id)
winner_model.to(device)

baseline_model = None
if baseline_run_id:
    print("Loading baseline model ...")
    baseline_model = _load_model(baseline_run_id)
    baseline_model.to(device)

for slide in SLIDES:
    sid = slide["slide_id"]
    print(f"\\nProcessing {{sid}} ...")

    try:
        features_np = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    N = features_np.shape[0]

    for model, prefix in [(winner_model, "winner")] + (
        [(baseline_model, "baseline")] if baseline_model else []
    ):
        scores_dict = {{}}

        if "attn" in METHODS:
            try:
                attn = _get_attn(model, features_np, device)
                if attn is not None:
                    if attn.ndim == 2:
                        attn_mmr = attn[:, 0]  # MMR head
                    else:
                        attn_mmr = attn
                    if attn_mmr.shape[0] == N:
                        scores_dict["attn"] = attn_mmr
                    else:
                        print(f"  [WARN] attn shape mismatch for {{sid}}/{{prefix}}")
            except Exception as e:
                print(f"  [WARN] attn failed for {{sid}}/{{prefix}}: {{e}}")

        if "gradnorm" in METHODS or "ixg" in METHODS:
            try:
                gradnorm, ixg = _get_grad_ixg(model, features_np, device)
                if "gradnorm" in METHODS:
                    scores_dict["gradnorm"] = gradnorm
                if "ixg" in METHODS:
                    scores_dict["ixg"] = ixg
            except Exception as e:
                print(f"  [WARN] grad computation failed for {{sid}}/{{prefix}}: {{e}}")

        for method, scores in scores_dict.items():
            try:
                del_probs = run_deletion(features_np, scores, K_LEVELS, model, device)
                ins_probs = run_insertion(features_np, scores, K_LEVELS, model, device)
                np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_{{method}}_del.npy"), del_probs)
                np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_{{method}}_ins.npy"), ins_probs)
                k20_drop = (del_probs[0] - del_probs[3]) * 100
                print(f"  {{prefix}}/{{method}}: del@0%={{del_probs[0]:.3f}} "
                      f"del@20%={{del_probs[3]:.3f}} (drop={{k20_drop:.1f}}pp) "
                      f"ins@50%={{ins_probs[-1]:.3f}}")
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


# ── Phase A: local prep ────────────────────────────────────────────────────────

def load_study_set() -> list:
    if not STUDY_SET_PATH.exists():
        raise FileNotFoundError(
            f"Study set not found: {STUDY_SET_PATH}\n"
            "Run phase8_study_set.py first."
        )
    return json.loads(STUDY_SET_PATH.read_text())


def build_manifest(study_set: list, winner_run: str, baseline_run: str,
                   methods: list) -> dict:
    return {
        "winner_run":   winner_run,
        "baseline_run": baseline_run,
        "methods":      methods,
        "k_levels":     K_LEVELS,
        "slides":       [{"slide_id": s["slide_id"], "category": s["category"]}
                         for s in study_set],
    }


# ── Phase B: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local  = tmp / "manifest.json"
        script_local    = tmp / "del_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/del_compute.py"

        manifest_local.write_text(json.dumps(manifest, indent=2))
        script_local.write_text(REMOTE_DEL_PY.format(
            root_dir      = GCP_ROOT_DIR,
            scripts_dir   = GCP_SCRIPTS_DIR,
            mlflow_db     = GCP_MLFLOW_DB,
            zarr_dir      = GCP_ZARR_DIR,
            out_dir       = GCP_REMOTE_TMP + "/out",
            manifest_path = manifest_remote,
        ))

        print("\nPhase B1 — prepare remote directory")
        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/out"),
             "mkdir remote tmp")

        print("Phase B2 — upload manifest + script")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp del_compute.py")

        print("Phase B3 — run deletion/insertion on GCP  (~15 min)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 del_compute.py")

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
            dest = LOCAL_DATA_DIR / item.name
            if dest.exists():
                dest.unlink() if dest.is_file() else shutil.rmtree(dest)
            shutil.move(str(item), str(LOCAL_DATA_DIR))
        print(f"  Extracted results to {LOCAL_DATA_DIR}")

        print("Phase B5 — clean up remote")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
             "rm remote tmp")


# ── Phase C: local visualisation ──────────────────────────────────────────────

def compute_auc(probs: np.ndarray, k_levels: list) -> float:
    x = np.array(k_levels)
    x_norm = x / x.max()
    return float(np.trapezoid(probs, x_norm))


def _load_curves(study_set: list, methods: list, prefix: str):
    """Load saved npy files for one run prefix (winner or baseline).

    Returns {slide_id: {method: {"del": arr, "ins": arr}}}
    """
    results: dict = {}
    for slide in study_set:
        sid = slide["slide_id"]
        results[sid] = {}
        for method in methods:
            d = LOCAL_DATA_DIR / f"{sid}_{prefix}_{method}_del.npy"
            i = LOCAL_DATA_DIR / f"{sid}_{prefix}_{method}_ins.npy"
            if d.exists() and i.exists():
                results[sid][method] = {
                    "del": np.load(d),
                    "ins": np.load(i),
                }
            else:
                if not d.exists():
                    print(f"  [WARN] Missing {d.name}")
    return results


def aggregate_curves(results: dict, study_set: list, curve_type: str):
    """Aggregate per-slide curves → {method: {"mean": arr, "std": arr}}."""
    all_methods: set = set()
    for per_method in results.values():
        all_methods |= set(per_method.keys())

    agg = {}
    for method in sorted(all_methods):
        arrays = []
        for slide in study_set:
            entry = results.get(slide["slide_id"], {}).get(method)
            if entry and curve_type in entry:
                arrays.append(entry[curve_type])
        if arrays:
            stacked = np.stack(arrays, axis=0)
            agg[method] = {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}
    return agg


def plot_curves(del_agg: dict, ins_agg: dict, run_label: str,
                k_levels: list, out_dir: Path) -> None:
    k_pct = [k * 100 for k in k_levels]
    method_colors = {"attn": "steelblue", "gradnorm": "darkorange", "ixg": "seagreen"}

    for curve_type, agg, ylabel, fname in (
        ("Deletion",  del_agg, "MMR probability",
         f"deletion_curves_{run_label}.png"),
        ("Insertion", ins_agg, "MMR probability",
         f"insertion_curves_{run_label}.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, data in agg.items():
            color = method_colors.get(method)
            mean, std = data["mean"], data["std"]
            ax.plot(k_pct, mean, marker="o", label=method, color=color)
            ax.fill_between(k_pct, mean - std, mean + std, alpha=0.15, color=color)
            if curve_type == "Deletion":
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
        ax.set_title(f"Phase 8 — {curve_type} Curves ({run_label})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def build_auc_table(winner_res: dict, baseline_res: dict,
                    methods: list, study_set: list, k_levels: list) -> pd.DataFrame:
    records = []
    for method in methods:
        for prefix, res in [("winner", winner_res), ("baseline", baseline_res)]:
            if res is None:
                continue
            del_aucs, ins_aucs = [], []
            for slide in study_set:
                sid   = slide["slide_id"]
                entry = res.get(sid, {}).get(method)
                if entry:
                    del_aucs.append(compute_auc(entry["del"], k_levels))
                    ins_aucs.append(compute_auc(entry["ins"], k_levels))
            if del_aucs:
                records.append({
                    "method":       method,
                    "run":          prefix,
                    "del_auc_mean": round(float(np.mean(del_aucs)), 4),
                    "del_auc_std":  round(float(np.std(del_aucs)),  4),
                    "ins_auc_mean": round(float(np.mean(ins_aucs)), 4),
                    "ins_auc_std":  round(float(np.std(ins_aucs)),  4),
                })
    return pd.DataFrame(records)


def plot_auc_table(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("  [WARN] AUC table is empty — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.5 + 1)))
    ax.axis("off")
    cols = ["method", "run", "del_auc_mean", "del_auc_std", "ins_auc_mean", "ins_auc_std"]
    col_labels = ["Method", "Run", "Del AUC (↓)", "Del AUC std", "Ins AUC (↑)", "Ins AUC std"]

    cell_data = []
    for _, row in df[cols].iterrows():
        cell_data.append([
            row["method"], row["run"],
            f"{row['del_auc_mean']:.4f}", f"{row['del_auc_std']:.4f}",
            f"{row['ins_auc_mean']:.4f}", f"{row['ins_auc_std']:.4f}",
        ])

    # Highlight row with lowest del_auc_mean for winner
    win_row = None
    winner_mask = df["run"] == "winner"
    if winner_mask.any():
        win_idx = df.loc[winner_mask, "del_auc_mean"].idxmin()
        win_row = df.index.get_loc(win_idx)

    cell_colors = [["white"] * len(cols) for _ in range(len(df))]
    if win_row is not None:
        cell_colors[win_row] = ["#c6efce"] * len(cols)

    tbl = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellColours=cell_colors, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    ax.set_title("Phase 8 Step 2 — Deletion/Insertion AUC Summary", fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(out_dir / "deletion_auc_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved deletion_auc_table.png")


def select_canonical(auc_df: pd.DataFrame) -> str:
    if auc_df.empty:
        return "attn"
    winner_df = auc_df[auc_df["run"] == "winner"].copy()
    if winner_df.empty:
        return "attn"
    winner_df = winner_df.sort_values(
        ["del_auc_mean", "ins_auc_mean"], ascending=[True, False]
    )
    return str(winner_df.iloc[0]["method"])


def _detect_methods(study_set: list) -> list:
    """Detect which methods have data in LOCAL_DATA_DIR."""
    methods = []
    for method in ("attn", "gradnorm", "ixg"):
        any_found = any(
            (LOCAL_DATA_DIR / f"{s['slide_id']}_winner_{method}_del.npy").exists()
            for s in study_set
        )
        if any_found:
            methods.append(method)
    return methods or ["attn"]


def visualise(study_set: list, winner_run: str, baseline_run: str,
              methods: list) -> str:
    """Generate figures and return canonical method name."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    del_dir = FIGURES_DIR / "deletion"
    del_dir.mkdir(parents=True, exist_ok=True)

    print("Phase C1 — loading saved del/ins npy files")
    winner_res   = _load_curves(study_set, methods, "winner")
    baseline_res = _load_curves(study_set, methods, "baseline") if baseline_run else None

    print("Phase C2 — aggregating curves")
    del_winner = aggregate_curves(winner_res, study_set, "del")
    ins_winner = aggregate_curves(winner_res, study_set, "ins")

    print("Phase C3 — plotting curves")
    plot_curves(del_winner, ins_winner, "winner", K_LEVELS, FIGURES_DIR)
    if baseline_res:
        del_base = aggregate_curves(baseline_res, study_set, "del")
        ins_base = aggregate_curves(baseline_res, study_set, "ins")
        plot_curves(del_base, ins_base, "baseline", K_LEVELS, FIGURES_DIR)

    print("Phase C4 — building AUC table")
    auc_df = build_auc_table(winner_res, baseline_res, methods, study_set, K_LEVELS)
    if not auc_df.empty:
        print(auc_df.to_string(index=False))
    plot_auc_table(auc_df, FIGURES_DIR)

    canonical = select_canonical(auc_df)
    print(f"  Canonical method: {canonical}")
    (del_dir / "canonical_method.txt").write_text(canonical + "\n")

    # Save summary
    summary = {
        "winner_run":       winner_run,
        "baseline_run":     baseline_run,
        "methods":          methods,
        "canonical_method": canonical,
        "auc_table":        auc_df.to_dict(orient="records") if not auc_df.empty else [],
    }
    (del_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary.json")

    return canonical


def _append_report(canonical: str, winner_run: str, baseline_run: str,
                   auc_df: pd.DataFrame) -> None:
    """Append Deletion Analysis section to phase8-results.md."""
    figures_rel = Path("reports/figures/phase8")
    lines = [
        "",
        "### Deletion Analysis",
        "",
        "Deletion and insertion curves were computed for three attribution methods "
        "(attention weights, GradNorm, and InputXGrad) on the winning and baseline "
        f"configurations (`{winner_run}` vs `{baseline_run}`). For each method, the "
        "top-k% most salient patches were removed (deletion) or revealed from a "
        "mean-embedding baseline (insertion) at k ∈ {0, 5, 10, 20, 30, 50}%. "
        "AUC of the deletion curve (lower = more faithful) selects the canonical "
        "attribution method for subsequent interpretability sections.",
        "",
        f"![Deletion curves — winner]({figures_rel}/deletion_curves_winner.png)",
        "",
    ]
    if baseline_run:
        lines += [
            f"![Deletion curves — baseline]({figures_rel}/deletion_curves_baseline.png)",
            "",
        ]
    lines += [
        f"![Deletion AUC table]({figures_rel}/deletion_auc_table.png)",
        "",
        f"**Canonical metric selected: `{canonical}`** — "
        f"lowest deletion AUC (most faithful) for the winner configuration across "
        "the study set.",
        "",
    ]

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Appended Deletion Analysis section to {REPORT_PATH.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",       default="reports/data/phase8/",
                   help="Path to extracted phase8 data (default: reports/data/phase8/)")
    p.add_argument("--best-run",       required=True,
                   help="Name of the winning phase8 run")
    p.add_argument("--baseline-run",   default="config_phase8_baseline_seed0",
                   help="Baseline run for comparison (default: config_phase8_baseline_seed0)")
    p.add_argument("--methods",        default="attn,gradnorm,ixg",
                   help="Comma-separated attribution methods (default: attn,gradnorm,ixg)")
    p.add_argument("--gcp-host",       default=None,
                   help="GCP server IP (omit for dry-run)")
    p.add_argument("--gcp-user",       default="chris")
    p.add_argument("--gcp-pass",       default=None)
    p.add_argument("--visualise-only", action="store_true",
                   help="Skip GCP; visualise from existing LOCAL_DATA_DIR files")
    p.add_argument("--no-report",      action="store_true",
                   help="Skip appending to phase8-results.md")
    p.add_argument("--force",          action="store_true",
                   help="Re-run even if results already exist")
    return p.parse_args()


def main():
    args    = parse_args()
    methods = [m.strip() for m in args.methods.split(",")]

    if args.visualise_only:
        print("--visualise-only: skipping remote compute")
        study_set = load_study_set()
        detected  = _detect_methods(study_set)
        print(f"  Methods detected from local data: {detected}")
        canonical = visualise(study_set, args.best_run, args.baseline_run, detected)
        if not args.no_report:
            auc_path = FIGURES_DIR / "deletion" / "summary.json"
            auc_df   = pd.DataFrame()
            if auc_path.exists():
                auc_df = pd.DataFrame(json.loads(auc_path.read_text()).get("auc_table", []))
            _append_report(canonical, args.best_run, args.baseline_run, auc_df)
        return

    # ── Phase A — local prep ──────────────────────────────────────────────────
    print("Phase A — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} slides loaded")

    print("\nPhase A2 — building manifest")
    manifest = build_manifest(study_set, args.best_run, args.baseline_run, methods)
    print(f"  Methods: {manifest['methods']}")

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        print(f"  Winner run     : {args.best_run}")
        print(f"  Baseline run   : {args.baseline_run}")
        print(f"  Methods        : {methods}")
        print(f"  Study slides   : {len(study_set)}")
        if LOCAL_DATA_DIR.exists() and list(LOCAL_DATA_DIR.glob("*_del.npy")):
            print("\nLocal data found — running visualisation")
            detected  = _detect_methods(study_set)
            canonical = visualise(study_set, args.best_run, args.baseline_run, detected)
            if not args.no_report:
                auc_df = pd.DataFrame()
                _append_report(canonical, args.best_run, args.baseline_run, auc_df)
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase B — remote compute ──────────────────────────────────────────────
    remote_compute(manifest, args.gcp_host, args.gcp_user, args.gcp_pass)

    # ── Phase C — local visualisation ─────────────────────────────────────────
    print("\nPhase C — local visualisation")
    detected  = _detect_methods(study_set)
    canonical = visualise(study_set, args.best_run, args.baseline_run, detected)

    if not args.no_report:
        auc_path = FIGURES_DIR / "deletion" / "summary.json"
        auc_df   = pd.DataFrame()
        if auc_path.exists():
            auc_df = pd.DataFrame(json.loads(auc_path.read_text()).get("auc_table", []))
        _append_report(canonical, args.best_run, args.baseline_run, auc_df)

    print("\nDone.")
    print(f"  Deletion curves  : {FIGURES_DIR}/deletion_curves_winner.png")
    print(f"  AUC table        : {FIGURES_DIR}/deletion_auc_table.png")
    print(f"  Canonical method : {canonical}")


if __name__ == "__main__":
    main()
