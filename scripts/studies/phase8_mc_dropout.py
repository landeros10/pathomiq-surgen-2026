#!/usr/bin/env python3
"""Phase 8 Interpretability Step 4 — MC Dropout Uncertainty.

Runs T=50 stochastic forward passes (model kept in .train() mode to activate
transformer dropout) over the study set slides for the winning Phase 8
configuration. Produces:
  - Per-slide predictive uncertainty (std of MC probs) for all three task heads
  - Spatial attention variance maps (std of MC attention weights) per slide

Three-phase pattern:
  Phase A (local) : Load study set, build manifest
  Phase B (remote): T=50 stochastic forward passes on study slides;
                    save per-slide (T,) prediction arrays and (N, T_mc) attn arrays
  Phase C (local) : Calibration diagram, per-category violin, spatial uncertainty maps

Outputs:
  reports/figures/phase8/mc_calibration.png
  reports/figures/phase8/mc_violin_by_category.png
  reports/figures/phase8/mc_uncertainty_maps/<slide_id>.png
  Appends '### MC Dropout Uncertainty' to reports/phase8-results.md

Usage:
    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase8_mc_dropout.py \\
        --data-dir phase8-data/ --best-run config_phase8_baseline_seed0

    # Full run (GCP required):
    python scripts/studies/phase8_mc_dropout.py \\
        --data-dir phase8-data/ --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'

    # Re-visualise from already-downloaded results:
    python scripts/studies/phase8_mc_dropout.py \\
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

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import build_attn_grid  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p8_mc"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"

FIGURES_DIR    = ROOT / "reports" / "figures" / "phase8"
LOCAL_DATA_DIR = ROOT / "results" / "phase8" / "mc"
REPORT_PATH    = ROOT / "reports" / "phase8-results.md"
STUDY_SET_PATH = FIGURES_DIR / "study_set_phase8.json"

MC_PASSES    = 50
COLORMAP     = "viridis"
TASKS        = ["mmr", "ras", "braf"]
CATEGORIES   = ["mmr_pos_braf_pos", "mmr_pos_braf_neg", "mmr_neg_ras_pos", "mmr_neg_ras_neg"]


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_MC_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side MC Dropout computation for Phase 8 Step 4.\"\"\"
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

WINNER_RUN  = MANIFEST["winner_run"]
SLIDES      = MANIFEST["slides"]
MC_PASSES   = {mc_passes}

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
    store  = zarr.open(path, mode="r")
    feats  = torch.from_numpy(store["features"][:]).float()
    coords = torch.from_numpy(store["coords"][:]).long() if "coords" in store else None
    return feats, coords


def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

print(f"Resolving winner run ID: {{WINNER_RUN}}")
winner_run_id = _resolve_run_id(WINNER_RUN)
print(f"  winner run_id: {{winner_run_id}}")

print("Loading winner model ...")
model = _load_model(winner_run_id)
model.to(device)
# Keep in .train() mode to activate dropout stochasticity
model.train()

n_slides = len(SLIDES)
# Per-slide probs array: (n_slides, MC_PASSES, n_tasks)  — up to 3 tasks
probs_all = np.full((n_slides, MC_PASSES, 3), np.nan, dtype=np.float32)

for i, slide in enumerate(SLIDES):
    sid = slide["slide_id"]
    print(f"  Slide {{i+1}}/{{n_slides}}: {{sid}}")

    try:
        features, coords = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    emb = features.unsqueeze(0).to(device)  # (1, N, 1024)
    attn_passes = []

    with torch.no_grad():
        for t in range(MC_PASSES):
            out = model(emb, return_weights=True)
            if isinstance(out, tuple):
                logits, weights = out
                if weights is not None:
                    w = weights.detach().cpu()
                    if w.ndim == 3:
                        w = w[0]   # (B, N, T) → (N, T)
                    elif w.ndim == 2 and w.shape[0] == 1:
                        w = w[0]
                    attn_passes.append(w.numpy())
            else:
                logits = out

            # Extract per-task probabilities
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.ndim == 2:
                task_logits = logits[0]   # (T,)
            elif logits.ndim == 1:
                task_logits = logits      # could be (T,) or scalar (1-task)
            else:
                task_logits = logits.unsqueeze(0)

            n_tasks = min(3, task_logits.shape[0])
            for task_idx in range(n_tasks):
                probs_all[i, t, task_idx] = float(
                    torch.sigmoid(task_logits[task_idx]).item()
                )

    # Save per-slide MC attention if available
    if attn_passes:
        mc_attn = np.stack(attn_passes, axis=0).astype(np.float32)  # (T, N, n_tasks) or (T, N)
        np.save(str(OUT_DIR / f"{{sid}}_mc_attn.npy"), mc_attn)

    # Save coords if available
    if coords is not None:
        np.save(str(OUT_DIR / f"{{sid}}_coords.npy"), coords.numpy().astype(np.int64))

np.save(str(OUT_DIR / "probs_all.npy"), probs_all)
slide_ids = [s["slide_id"] for s in SLIDES]
(OUT_DIR / "slide_ids.json").write_text(json.dumps(slide_ids))

print(f"Saved probs_all {{probs_all.shape}}")
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


def build_manifest(study_set: list, winner_run: str, mc_passes: int) -> dict:
    return {
        "winner_run": winner_run,
        "mc_passes":  mc_passes,
        "slides":     [{"slide_id": s["slide_id"], "category": s["category"]}
                       for s in study_set],
    }


# ── Phase B: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local  = tmp / "manifest.json"
        script_local    = tmp / "mc_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/mc_compute.py"

        manifest_local.write_text(json.dumps(manifest, indent=2))
        script_local.write_text(REMOTE_MC_PY.format(
            root_dir      = GCP_ROOT_DIR,
            scripts_dir   = GCP_SCRIPTS_DIR,
            mlflow_db     = GCP_MLFLOW_DB,
            zarr_dir      = GCP_ZARR_DIR,
            out_dir       = GCP_REMOTE_TMP + "/out",
            manifest_path = manifest_remote,
            mc_passes     = manifest["mc_passes"],
        ))

        print("\nPhase B1 — prepare remote directory")
        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/out"),
             "mkdir remote tmp")

        print("Phase B2 — upload manifest + script")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp mc_compute.py")

        n_passes  = manifest["mc_passes"]
        n_slides  = len(manifest["slides"])
        print(f"Phase B3 — run MC Dropout on GCP  ({n_passes} passes × {n_slides} slides, ~20 min)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 mc_compute.py")

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

def _reliability_diagram(ax, probs_mc: np.ndarray, labels: np.ndarray,
                          label_str: str, n_bins: int = 10,
                          color: str = "steelblue") -> float:
    """Draw reliability diagram, return ECE."""
    mc_mean = probs_mc.mean(axis=1)
    valid   = ~np.isnan(labels) & ~np.isnan(mc_mean)
    mc_mean = mc_mean[valid]
    labs    = labels[valid]
    if len(mc_mean) == 0:
        return 0.0

    edges = np.linspace(0, 1, n_bins + 1)
    b_acc, b_conf, b_n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (mc_mean >= lo) & (mc_mean < hi)
        if mask.sum() > 0:
            b_acc.append(labs[mask].mean())
            b_conf.append(mc_mean[mask].mean())
            b_n.append(mask.sum())
    if not b_acc:
        return 0.0
    b_acc  = np.array(b_acc)
    b_conf = np.array(b_conf)
    b_n    = np.array(b_n)
    ece    = float(np.average(np.abs(b_acc - b_conf), weights=b_n))
    ax.bar(b_conf, b_acc, width=0.07, alpha=0.6, color=color,
           label=f"{label_str} ECE={ece:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    return ece


def visualise(study_set: list, mc_passes: int) -> None:
    unc_dir = FIGURES_DIR / "mc_uncertainty_maps"
    unc_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    probs_path    = LOCAL_DATA_DIR / "probs_all.npy"
    slide_ids_path = LOCAL_DATA_DIR / "slide_ids.json"

    missing = [p for p in [probs_path, slide_ids_path] if not p.exists()]
    if missing:
        print(f"  [WARN] Missing data files: {[str(p) for p in missing]}")
        print("  Skipping visualisation (run with --gcp-host first)")
        return

    probs_all = np.load(probs_path)  # (n_slides, T, 3)
    slide_ids = json.loads(slide_ids_path.read_text())
    n_slides  = len(slide_ids)
    study_map = {s["slide_id"]: s for s in study_set}

    # Per-task mean and std across MC passes
    probs_mean = np.nanmean(probs_all, axis=1)   # (n_slides, 3)
    probs_std  = np.nanstd(probs_all, axis=1)    # (n_slides, 3)

    # ── 1. Calibration (MMR head) ─────────────────────────────────────────────
    # Build MMR labels from study set
    mmr_labels = np.full(n_slides, np.nan)
    for i, sid in enumerate(slide_ids):
        if sid in study_map and study_map[sid].get("label_mmr") is not None:
            mmr_labels[i] = float(study_map[sid]["label_mmr"])

    fig, ax = plt.subplots(figsize=(6, 6))
    mmr_probs_mc = probs_all[:, :, 0]  # (n_slides, T)
    _reliability_diagram(ax, mmr_probs_mc, mmr_labels, "MMR", color="steelblue")

    if probs_all.shape[2] > 1:
        braf_labels = np.full(n_slides, np.nan)
        for i, sid in enumerate(slide_ids):
            if sid in study_map and study_map[sid].get("label_braf") is not None:
                braf_labels[i] = float(study_map[sid]["label_braf"])
        braf_probs_mc = probs_all[:, :, 2]  # BRAF index 2
        _reliability_diagram(ax, braf_probs_mc, braf_labels, "BRAF", color="tomato")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive (true)")
    ax.set_title(f"Reliability Diagram — Phase 8 Winner (T={mc_passes} MC passes)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "mc_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mc_calibration.png")

    # ── 2. Uncertainty violin by category ─────────────────────────────────────
    mmr_std = probs_std[:, 0]  # (n_slides,) epistemic uncertainty for MMR head
    cat_std: dict = {c: [] for c in CATEGORIES}
    for i, sid in enumerate(slide_ids):
        if sid in study_map and not np.isnan(mmr_std[i]):
            cat = study_map[sid]["category"]
            if cat in cat_std:
                cat_std[cat].append(float(mmr_std[i]))

    cat_labels = [c.replace("_", "\n") for c in CATEGORIES]
    valid_cats = [c for c in CATEGORIES if cat_std[c]]
    data_vals  = [cat_std[c] for c in valid_cats]
    labels_use = [c.replace("_", "\n") for c in valid_cats]

    if data_vals:
        fig, ax = plt.subplots(figsize=(8, 5))
        parts = ax.violinplot(data_vals, positions=range(len(valid_cats)), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(valid_cats)))
        ax.set_xticklabels(labels_use, fontsize=9)
        ax.set_ylabel("Epistemic uncertainty (std of MC probs — MMR head)")
        ax.set_title(
            f"MC Dropout Uncertainty by Category (T={mc_passes}, n={n_slides} slides)",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "mc_violin_by_category.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved mc_violin_by_category.png")

    # ── 3. Spatial uncertainty maps per slide ─────────────────────────────────
    for i, sid in enumerate(slide_ids):
        if sid not in study_map:
            continue

        mc_attn_path = LOCAL_DATA_DIR / f"{sid}_mc_attn.npy"
        coords_path  = LOCAL_DATA_DIR / f"{sid}_coords.npy"

        if not mc_attn_path.exists() or not coords_path.exists():
            continue

        mc_attn = np.load(mc_attn_path)  # (T, N, n_tasks) or (T, N)
        coords  = np.load(coords_path)   # (N, 2)

        category = study_map[sid]["category"]

        # Compute attention std across MC passes
        if mc_attn.ndim == 3:
            attn_std_mmr = mc_attn[:, :, 0].std(axis=0)   # (N,)
        elif mc_attn.ndim == 2:
            attn_std_mmr = mc_attn.std(axis=0)            # (N,)
        else:
            continue

        std_grid = build_attn_grid(attn_std_mmr, coords)  # (H, W)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(std_grid, cmap=COLORMAP, aspect="equal",
                       interpolation="nearest")
        ax.set_title(
            f"{sid}  [{category}]\nMMR Attention Std (T={mc_passes} MC passes)",
            fontsize=9,
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(unc_dir / f"{sid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved per-slide spatial uncertainty maps to {unc_dir.name}/")

    # ── 4. Summary JSON ───────────────────────────────────────────────────────
    summary = []
    for i, sid in enumerate(slide_ids):
        if sid not in study_map:
            continue
        entry = {
            "slide_id": sid,
            "category": study_map[sid]["category"],
        }
        for t_idx, task in enumerate(TASKS):
            if t_idx < probs_all.shape[2]:
                m = float(np.nanmean(probs_all[i, :, t_idx]))
                s = float(np.nanstd(probs_all[i, :, t_idx]))
                entry[f"mc_mean_{task}"] = round(m, 4) if not np.isnan(m) else None
                entry[f"mc_std_{task}"]  = round(s, 4) if not np.isnan(s) else None
        summary.append(entry)

    (LOCAL_DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary.json ({len(summary)} study slides)")


def _append_report(winner_run: str, mc_passes: int) -> None:
    figures_rel = Path("reports/figures/phase8")
    unc_maps = sorted((FIGURES_DIR / "mc_uncertainty_maps").glob("*.png")) if (
        FIGURES_DIR / "mc_uncertainty_maps"
    ).exists() else []

    lines = [
        "",
        "### MC Dropout Uncertainty",
        "",
        f"MC Dropout uncertainty was quantified using T={mc_passes} stochastic forward "
        f"passes on the winning Phase 8 configuration (`{winner_run}`). The model was "
        "kept in `.train()` mode to activate transformer dropout, producing per-slide "
        "predictive uncertainty (std of MC probabilities) and spatial uncertainty maps "
        "(std of MC attention weights). All three task heads (MMR, BRAF, RAS) were "
        "monitored. Calibration was assessed via reliability diagrams for the MMR and "
        "BRAF heads.",
        "",
        f"![Calibration]({figures_rel}/mc_calibration.png)",
        "",
        f"![Uncertainty by category]({figures_rel}/mc_violin_by_category.png)",
        "",
    ]

    if unc_maps:
        lines += [
            f"**Spatial uncertainty maps** (per-slide attention std, "
            f"{len(unc_maps)} slides):",
            "",
        ]
        for p in unc_maps[:6]:  # show first 6 in report
            sid = p.stem
            lines.append(f"![{sid}]({figures_rel}/mc_uncertainty_maps/{p.name})")
        lines.append("")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Appended MC Dropout Uncertainty section to {REPORT_PATH.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",       default="reports/data/phase8/",
                   help="Path to extracted phase8 data (default: reports/data/phase8/)")
    p.add_argument("--best-run",       required=True,
                   help="Name of the winning phase8 run")
    p.add_argument("--n-passes",       type=int, default=MC_PASSES,
                   help=f"MC Dropout passes (default: {MC_PASSES})")
    p.add_argument("--gcp-host",       default=None)
    p.add_argument("--gcp-user",       default="chris")
    p.add_argument("--gcp-pass",       default=None)
    p.add_argument("--visualise-only", action="store_true",
                   help="Skip GCP; visualise from existing local data")
    p.add_argument("--no-report",      action="store_true",
                   help="Skip appending to phase8-results.md")
    p.add_argument("--force",          action="store_true",
                   help="Re-run even if results already exist")
    return p.parse_args()


def main():
    args = parse_args()

    if args.visualise_only:
        print("--visualise-only: skipping remote compute")
        study_set = load_study_set()
        visualise(study_set, args.n_passes)
        if not args.no_report:
            _append_report(args.best_run, args.n_passes)
        return

    # ── Phase A — local prep ──────────────────────────────────────────────────
    print("Phase A — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} study slides")

    print("\nPhase A2 — building manifest")
    manifest = build_manifest(study_set, args.best_run, args.n_passes)

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        print(f"  Winner run   : {args.best_run}")
        print(f"  MC passes    : {args.n_passes}")
        print(f"  Study slides : {len(study_set)}")
        if (LOCAL_DATA_DIR / "probs_all.npy").exists():
            print("\nLocal data found — running visualisation")
            visualise(study_set, args.n_passes)
            if not args.no_report:
                _append_report(args.best_run, args.n_passes)
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase B — remote compute ──────────────────────────────────────────────
    remote_compute(manifest, args.gcp_host, args.gcp_user, args.gcp_pass)

    # ── Phase C — local visualisation ─────────────────────────────────────────
    print("\nPhase C — local visualisation")
    visualise(study_set, args.n_passes)

    if not args.no_report:
        _append_report(args.best_run, args.n_passes)

    print("\nDone.")
    print(f"  Calibration    : {FIGURES_DIR}/mc_calibration.png")
    print(f"  Violin         : {FIGURES_DIR}/mc_violin_by_category.png")
    print(f"  Uncertainty maps: {FIGURES_DIR}/mc_uncertainty_maps/")
    print(f"  Summary JSON   : {LOCAL_DATA_DIR}/summary.json")


if __name__ == "__main__":
    main()
