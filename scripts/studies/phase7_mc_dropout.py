#!/usr/bin/env python3
"""Phase 7 Step 3 — MC Dropout Uncertainty.

Runs T=50 stochastic forward passes (model kept in .train() mode to activate
transformer dropout p=0.15) over the full 165-slide test split for both ST and
MT models. Produces predictive uncertainty estimates and spatial attention
variance maps for the 19-slide study set.

Computation runs on GCP (models live there). Results are SCP'd back locally
and visualised:
  - Reliability diagram (calibration)
  - Violin plot of per-slide uncertainty by category
  - Spatial attention-std maps per study slide
  - Summary JSON

Usage:
    # Full run (GCP required):
    python scripts/studies/phase7_mc_dropout.py \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase7_mc_dropout.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from utils.eval_utils import (      # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    DEFAULT_FIGURES_DIR as FIGURES_DIR,
    build_attn_grid,
    load_slide_coords,
    load_study_set,
    mlflow_run_id,
)

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP   = "/tmp/surgen_p7_mc"
GCP_ZARR_DIR     = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB    = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR  = "/home/chris/surgen/scripts"
GCP_ROOT_DIR     = "/home/chris/surgen"
GCP_DATA_DIR     = "/home/chris/surgen/data/splits"

LOCAL_DATA_DIR   = ROOT / "results" / "phase7" / "mc"
UNC_OUT_DIR      = FIGURES_DIR / "uncertainty"

MC_PASSES        = 50
COLORMAP         = "viridis"


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_MC_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side MC Dropout computation for Phase 7 Step 3.\"\"\"
import csv, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "{root_dir}")
sys.path.insert(0, "{scripts_dir}")
sys.path.insert(0, "{scripts_dir}/models")
sys.path.insert(0, "{scripts_dir}/etl")

import mlflow
import zarr

MLFLOW_DB    = "{mlflow_db}"
ZARR_DIR     = Path("{zarr_dir}")
OUT_DIR      = Path("{out_dir}")
DATA_DIR     = Path("{data_dir}")
ST_RUN_ID    = "{st_run_id}"
MT_RUN_ID    = "{mt_run_id}"
MC_PASSES    = {mc_passes}
STUDY_SLIDES = set({study_slides_repr})   # set of slide_ids for attn collection

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
    features = torch.from_numpy(store["features"][:]).float()
    coords   = torch.from_numpy(store["coords"][:]).long() if "coords" in store else None
    return features, coords

def _load_csv_slides(csv_path, id_col="case_id", label_col="label"):
    slides, labels = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            slides.append(row[id_col])
            try:
                labels.append(float(row[label_col]))
            except (KeyError, ValueError):
                labels.append(float("nan"))
    return slides, labels

def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

# Load test slides from ST CSV (MMR labels)
test_csv = DATA_DIR / "SurGen_msi_test.csv"
slide_ids, true_labels = _load_csv_slides(str(test_csv))
print(f"Test slides: {{len(slide_ids)}}")

# Save slide_ids once (shared)
(OUT_DIR / "slide_ids.json").write_text(json.dumps(slide_ids))

# Load models
print("Loading ST model ...")
st_model = _load_model(ST_RUN_ID)
st_model.to(device)
print("Loading MT model ...")
mt_model = _load_model(MT_RUN_ID)
mt_model.to(device)

# MC Dropout: keep models in .train() mode to activate dropout stochasticity
st_model.train()
mt_model.train()

st_probs_all = np.full((len(slide_ids), MC_PASSES), np.nan, dtype=np.float32)
mt_probs_all = np.full((len(slide_ids), MC_PASSES), np.nan, dtype=np.float32)

for i, sid in enumerate(slide_ids):
    if i % 20 == 0:
        print(f"  Slide {{i+1}}/{{len(slide_ids)}} ...")
    try:
        features, coords = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    emb = features.unsqueeze(0).to(device)   # (1, N, 1024)
    is_study = sid in STUDY_SLIDES
    st_attn_passes = [] if is_study else None
    mt_attn_passes = [] if is_study else None

    with torch.no_grad():
        for t in range(MC_PASSES):
            # ST model
            if is_study:
                out = st_model(emb, return_weights=True)
                logit, w = out if isinstance(out, tuple) else (out, None)
                if w is not None:
                    st_attn_passes.append(w[0, :, 0].cpu().numpy())
            else:
                logit = st_model(emb)
                if isinstance(logit, tuple):
                    logit = logit[0]
            prob = float(torch.sigmoid(
                logit[0] if logit.ndim == 1 else logit.squeeze()
            ).item())
            st_probs_all[i, t] = prob

            # MT model
            if is_study:
                out = mt_model(emb, return_weights=True)
                logit_mt, w_mt = out if isinstance(out, tuple) else (out, None)
                if w_mt is not None:
                    mt_attn_passes.append(w_mt[0, :, 0].cpu().numpy())
            else:
                logit_mt = mt_model(emb)
                if isinstance(logit_mt, tuple):
                    logit_mt = logit_mt[0]
            # Extract MMR logit (index 0)
            if logit_mt.ndim == 2:
                logit_mt = logit_mt[0, 0]
            elif logit_mt.ndim == 1:
                logit_mt = logit_mt[0]
            mt_probs_all[i, t] = float(torch.sigmoid(logit_mt).item())

    # Save per-study-slide attention arrays
    if is_study and st_attn_passes:
        np.save(str(OUT_DIR / f"{{sid}}_st_mc_attn.npy"),
                np.stack(st_attn_passes).astype(np.float32))  # (T, N)
    if is_study and mt_attn_passes:
        np.save(str(OUT_DIR / f"{{sid}}_mt_mc_attn.npy"),
                np.stack(mt_attn_passes).astype(np.float32))  # (T, N)

np.save(str(OUT_DIR / "st_mc_probs.npy"), st_probs_all)
np.save(str(OUT_DIR / "mt_mc_probs.npy"), mt_probs_all)
np.save(str(OUT_DIR / "st_labels.npy"),   np.array(true_labels, dtype=np.float32))
np.save(str(OUT_DIR / "mt_labels.npy"),   np.array(true_labels, dtype=np.float32))

print(f"Saved: st_mc_probs {{st_probs_all.shape}}, mt_mc_probs {{mt_probs_all.shape}}")
print("Done.")
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


# ── Phase 2: remote compute ────────────────────────────────────────────────────

def remote_compute(
    study_set: list,
    st_run_id: str,
    mt_run_id: str,
    host: str,
    user: str,
    password: str,
) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    study_slide_ids = [s["slide_id"] for s in study_set]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script_local  = tmp / "mc_compute.py"
        script_remote = f"{GCP_REMOTE_TMP}/mc_compute.py"

        script_content = REMOTE_MC_PY.format(
            root_dir          = GCP_ROOT_DIR,
            scripts_dir       = GCP_SCRIPTS_DIR,
            mlflow_db         = GCP_MLFLOW_DB,
            zarr_dir          = GCP_ZARR_DIR,
            out_dir           = GCP_REMOTE_TMP + "/out",
            data_dir          = GCP_DATA_DIR,
            st_run_id         = st_run_id,
            mt_run_id         = mt_run_id,
            mc_passes         = MC_PASSES,
            study_slides_repr = repr(study_slide_ids),
        )
        script_local.write_text(script_content)

        print("\nPhase 2a — prepare remote directory")
        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/out"),
             "mkdir remote tmp")

        print("Phase 2b — upload script")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp mc_compute.py")

        print(f"Phase 2c — run MC Dropout on GCP ({MC_PASSES} passes × 165 slides × 2 models)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 mc_compute.py")

        print("Phase 2d — download results")
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _run(_scp_from(host, user, password,
                       f"{GCP_REMOTE_TMP}/out/",
                       str(LOCAL_DATA_DIR) + "/"),
             "scp MC Dropout results")

        # scp -r may copy the 'out' dir itself rather than its contents;
        # flatten if needed so files land directly in LOCAL_DATA_DIR.
        nested_out = LOCAL_DATA_DIR / "out"
        if nested_out.is_dir():
            import shutil
            for item in nested_out.iterdir():
                dest = LOCAL_DATA_DIR / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(LOCAL_DATA_DIR))
            nested_out.rmdir()
            print("  [INFO] Flattened nested out/ dir into LOCAL_DATA_DIR")

        print("Phase 2e — clean up remote")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
             "rm remote tmp")


# ── Phase 3: local visualisation ──────────────────────────────────────────────

def _reliability_diagram(
    ax,
    probs: np.ndarray,    # (N, T) — MC probs
    labels: np.ndarray,   # (N,)   — true binary labels
    label_str: str,
    n_bins: int = 10,
    color: str = "steelblue",
) -> float:
    """Draw reliability diagram; return ECE."""
    mc_mean = probs.mean(axis=1)    # (N,) — mean predicted prob
    valid   = ~np.isnan(labels) & ~np.isnan(mc_mean)
    mc_mean = mc_mean[valid]
    labels  = labels[valid]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc   = []
    bin_conf  = []
    bin_n     = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (mc_mean >= lo) & (mc_mean < hi)
        if mask.sum() > 0:
            bin_acc.append(labels[mask].mean())
            bin_conf.append(mc_mean[mask].mean())
            bin_n.append(mask.sum())

    if not bin_acc:
        return 0.0

    bin_acc  = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_n    = np.array(bin_n)
    ece      = float(np.average(np.abs(bin_acc - bin_conf), weights=bin_n))

    ax.bar(bin_conf, bin_acc, width=0.07, alpha=0.6, color=color, label=f"{label_str} ECE={ece:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    return ece


def visualise(study_set: list) -> None:
    UNC_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load raw MC data ───────────────────────────────────────────────────────
    st_probs_path = LOCAL_DATA_DIR / "st_mc_probs.npy"
    mt_probs_path = LOCAL_DATA_DIR / "mt_mc_probs.npy"
    st_labels_path = LOCAL_DATA_DIR / "st_labels.npy"
    slide_ids_path = LOCAL_DATA_DIR / "slide_ids.json"

    missing = [p for p in [st_probs_path, mt_probs_path, st_labels_path, slide_ids_path]
               if not p.exists()]
    if missing:
        print(f"  [WARN] Missing data files: {[str(p) for p in missing]}")
        print("  Skipping visualisation (run with --gcp-host first)")
        return

    st_probs  = np.load(st_probs_path)   # (N, T)
    mt_probs  = np.load(mt_probs_path)   # (N, T)
    st_labels = np.load(st_labels_path)  # (N,)
    slide_ids = json.loads(slide_ids_path.read_text())

    # Build study-slide category lookup
    study_map = {s["slide_id"]: s["category"] for s in study_set}

    # ── 1. Calibration ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    _reliability_diagram(ax, st_probs, st_labels, "ST", color="steelblue")
    _reliability_diagram(ax, mt_probs, st_labels, "MT", color="tomato")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive (true)")
    ax.set_title("Reliability Diagram — ST vs MT (MC Dropout)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(UNC_OUT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved calibration.png")

    # ── 2. Uncertainty by category (full 165-slide test split) ───────────────
    # Derive category from predictions vs labels for every slide, not just the
    # 19-slide study set, so the violin reflects the full distribution.
    CATEGORIES = ["correct_msi", "correct_mss", "wrong_msi", "wrong_mss"]

    st_std  = st_probs.std(axis=1)    # (N,) epistemic uncertainty
    mt_std  = mt_probs.std(axis=1)
    st_mean = st_probs.mean(axis=1)   # (N,) calibrated prediction
    mt_mean = mt_probs.mean(axis=1)

    cat_st: dict = {c: [] for c in CATEGORIES}
    cat_mt: dict = {c: [] for c in CATEGORIES}
    for i, sid in enumerate(slide_ids):
        label = int(st_labels[i])
        suffix = "msi" if label == 1 else "mss"

        st_cat = ("correct" if (st_mean[i] > 0.5) == bool(label) else "wrong") + "_" + suffix
        mt_cat = ("correct" if (mt_mean[i] > 0.5) == bool(label) else "wrong") + "_" + suffix

        if not np.isnan(st_std[i]):
            cat_st[st_cat].append(float(st_std[i]))
        if not np.isnan(mt_std[i]):
            cat_mt[mt_cat].append(float(mt_std[i]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cat_labels = [c.replace("_", "\n") for c in CATEGORIES]

    for ax, data_by_cat, title in zip(
        axes, [cat_st, cat_mt], ["Single-Task", "Multi-Task"]
    ):
        parts = ax.violinplot(
            [data_by_cat[c] for c in CATEGORIES],
            positions=range(len(CATEGORIES)),
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.set_ylabel("Epistemic uncertainty (std of MC probs)")
        ax.set_title(f"{title} — Uncertainty by Category")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"MC Dropout Uncertainty by Category — full test split (n=165, T={MC_PASSES} passes)", fontsize=12)
    plt.tight_layout()
    fig.savefig(UNC_OUT_DIR / "by_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved by_category.png")

    # ── 3. Spatial uncertainty maps per study slide ───────────────────────────
    heatmap_dir = FIGURES_DIR / "heatmaps"

    for slide in study_set:
        sid      = slide["slide_id"]
        idx      = slide["idx"]
        category = slide["category"]

        # Load coords for 2D grid building
        try:
            coords = load_slide_coords(SINGLETASK_RUN, idx)
        except FileNotFoundError:
            print(f"  [WARN] No coords for {sid} idx={idx}")
            continue

        slide_out = UNC_OUT_DIR / sid
        slide_out.mkdir(parents=True, exist_ok=True)

        panels = []
        for prefix in ("st", "mt"):
            attn_path = LOCAL_DATA_DIR / f"{sid}_{prefix}_mc_attn.npy"
            if not attn_path.exists():
                continue
            mc_attn = np.load(attn_path)   # (T, N)
            attn_std_1d = mc_attn.std(axis=0)  # (N,)
            std_grid    = build_attn_grid(attn_std_1d, coords)  # (H, W)
            panels.append((prefix.upper(), std_grid))

        if not panels:
            continue

        # attn_std.png for first available model
        _, std_grid_primary = panels[0]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(std_grid_primary, cmap=COLORMAP, aspect="equal",
                       interpolation="nearest")
        ax.set_title(f"{sid} [{category}]\nAttention Std (MC Dropout, T={MC_PASSES})",
                     fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(slide_out / "attn_std.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # paired_uncertainty.png
        if len(panels) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, (name, grid) in zip(axes, panels[:2]):
                im = ax.imshow(grid, cmap=COLORMAP, aspect="equal",
                               interpolation="nearest")
                ax.set_title(f"{name} Attention Std", fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{sid} [{category}]", fontsize=10)
            plt.tight_layout()
            fig.savefig(slide_out / "paired_uncertainty.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    print("  Saved per-slide spatial uncertainty maps")

    # ── 4. Summary JSON ───────────────────────────────────────────────────────
    summary = []
    st_mc_mean = st_probs.mean(axis=1)
    mt_mc_mean = mt_probs.mean(axis=1)

    slide_idx_map = {sid: i for i, sid in enumerate(slide_ids)}
    for slide in study_set:
        sid = slide["slide_id"]
        if sid not in slide_idx_map:
            continue
        i = slide_idx_map[sid]
        summary.append({
            "slide_id":   sid,
            "category":   slide["category"],
            "st_mc_mean": round(float(st_mc_mean[i]), 4) if not np.isnan(st_mc_mean[i]) else None,
            "st_mc_std":  round(float(st_std[i]),     4) if not np.isnan(st_std[i])     else None,
            "mt_mc_mean": round(float(mt_mc_mean[i]), 4) if not np.isnan(mt_mc_mean[i]) else None,
            "mt_mc_std":  round(float(mt_std[i]),     4) if not np.isnan(mt_std[i])     else None,
        })

    (UNC_OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary.json ({len(summary)} study slides)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcp-host", default=None,
                   help="GCP server IP (omit for dry-run)")
    p.add_argument("--gcp-user", default="chris")
    p.add_argument("--gcp-pass", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Phase 1 — local prep ──────────────────────────────────────────────────
    print("Phase 1 — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} study slides")
    for s in study_set:
        print(f"    {s['slide_id']}  [{s['category']}]")

    print("\nPhase 1b — resolving MLflow run IDs")
    try:
        st_run_id = mlflow_run_id(SINGLETASK_RUN)
        mt_run_id = mlflow_run_id(MULTITASK_RUN)
        print(f"  ST run ID: {st_run_id}")
        print(f"  MT run ID: {mt_run_id}")
    except (FileNotFoundError, KeyError) as e:
        print(f"  [WARN] Cannot resolve run IDs locally: {e}")
        st_run_id = "e60812c9ab0e462080a6bde3bc0141fe"
        mt_run_id = "1013f94132dd4ea092266b334abc3c2e"
        print(f"  Using hard-coded IDs: ST={st_run_id}, MT={mt_run_id}")

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print(f"\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        print(f"  MC passes planned : {MC_PASSES}")
        print(f"  Study slides       : {len(study_set)}")
        print(f"  ST run ID          : {st_run_id}")
        print(f"  MT run ID          : {mt_run_id}")

        # If data already downloaded, run visualisation
        if (LOCAL_DATA_DIR / "st_mc_probs.npy").exists():
            print("\nLocal data found — running visualisation")
            visualise(study_set)
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase 2 — remote compute ──────────────────────────────────────────────
    remote_compute(study_set, st_run_id, mt_run_id,
                   args.gcp_host, args.gcp_user, args.gcp_pass)

    # ── Phase 3 — local visualisation ─────────────────────────────────────────
    print("\nPhase 3 — local visualisation")
    visualise(study_set)

    print(f"\nDone.")
    print(f"  Calibration plot   : {UNC_OUT_DIR}/calibration.png")
    print(f"  By-category violin : {UNC_OUT_DIR}/by_category.png")
    print(f"  Summary JSON       : {UNC_OUT_DIR}/summary.json")
    print(f"  Slide dirs         : {UNC_OUT_DIR}/<slide_id>/")


if __name__ == "__main__":
    main()
