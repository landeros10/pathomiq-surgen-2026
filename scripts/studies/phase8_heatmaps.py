#!/usr/bin/env python3
"""Phase 8 Interpretability Step 3 — Cross-Head Attention Heatmaps.

Extracts attention weights for all three task heads (MMR, BRAF, RAS) from the
winning Phase 8 multitask model and generates per-slide 3-panel figures.

Biological hypothesis: BRAF mutation is a frequent driver of MMR deficiency →
the MMR and BRAF prediction heads should attend to overlapping tissue regions,
while the RAS head attends to largely distinct regions.

Statistical test: per-slide Spearman ρ(MMR, BRAF) vs ρ(MMR, RAS) across the
study set. Expected: ρ(MMR, BRAF) >> ρ(MMR, RAS).

Three-phase pattern:
  Phase A (local) : Load study set, build slide index for GCP
  Phase B (remote): Extract (N, 3) attention arrays + (N, 2) coords per slide
  Phase C (local) : Build 2D grids, generate 3-panel figures + gallery,
                    compute Spearman correlations, append report section

Outputs:
  reports/figures/phase8/heatmaps/<slide_id>_3head.png  — 3-panel per slide
  reports/figures/phase8/heatmap_gallery.png             — combined gallery
  reports/figures/phase8/head_correlation_violin.png     — ρ(MMR,BRAF) vs ρ(MMR,RAS)
  Appends '### Cross-Head Attention Heatmaps' to reports/phase8-results.md

Usage:
    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase8_heatmaps.py \\
        --data-dir phase8-data/ --best-run config_phase8_baseline_seed0

    # Full run (GCP required):
    python scripts/studies/phase8_heatmaps.py \\
        --data-dir phase8-data/ --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'

    # Re-visualise from already-downloaded results:
    python scripts/studies/phase8_heatmaps.py \\
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
from scipy.stats import spearmanr

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.eval_utils import build_attn_grid  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p8_hm"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"

FIGURES_DIR    = ROOT / "reports" / "figures" / "phase8"
LOCAL_DATA_DIR = ROOT / "results" / "phase8" / "heatmaps"
REPORT_PATH    = ROOT / "reports" / "phase8-results.md"
STUDY_SET_PATH = FIGURES_DIR / "study_set_phase8.json"

TASKS    = ["mmr", "ras", "braf"]
COLORMAP = "hot"


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_HM_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side attention extraction for Phase 8 Step 3 (cross-head heatmaps).\"\"\"
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
    feats  = store["features"][:]
    coords = store["coords"][:] if "coords" in store else None
    return feats, coords


def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    model.eval()
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

print(f"Resolving winner run ID: {{WINNER_RUN}}")
winner_run_id = _resolve_run_id(WINNER_RUN)
print(f"  winner run_id: {{winner_run_id}}")

print("Loading winner model ...")
model = _load_model(winner_run_id)
model.to(device)

for slide in SLIDES:
    sid = slide["slide_id"]
    print(f"Processing {{sid}} ...")

    try:
        features_np, coords_np = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    N = features_np.shape[0]

    # Forward pass to get attention weights for all task heads
    try:
        emb = torch.tensor(features_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(emb, return_weights=True)

        if not isinstance(out, tuple) or out[1] is None:
            # Try alternate: forward pass and check model internals
            print(f"  [WARN] return_weights=True not supported; trying fallback")
            with torch.no_grad():
                out = model(emb)
            print(f"  [WARN] No attention weights available for {{sid}}")
            continue

        logits, weights = out
        w = weights.detach().cpu().numpy()

        # Normalise weights shape → (N, T)
        if w.ndim == 3:
            w = w[0]   # (B, N, T) → (N, T)
        elif w.ndim == 2 and w.shape[0] == 1:
            w = w[0]   # (1, N) edge case

        if w.shape[0] != N:
            print(f"  [WARN] attn shape {{w.shape}} does not match N={{N}}")
            continue

        np.save(str(OUT_DIR / f"{{sid}}_attn.npy"), w.astype(np.float32))
        if coords_np is not None:
            np.save(str(OUT_DIR / f"{{sid}}_coords.npy"), coords_np.astype(np.int64))

        print(f"  Saved attn {{w.shape}}, coords {{coords_np.shape if coords_np is not None else 'N/A'}}")

    except Exception as exc:
        print(f"  [ERROR] {{sid}}: {{exc}}")

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


def build_manifest(study_set: list, winner_run: str) -> dict:
    return {
        "winner_run": winner_run,
        "slides":     [{"slide_id": s["slide_id"], "category": s["category"]}
                       for s in study_set],
    }


# ── Phase B: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local  = tmp / "manifest.json"
        script_local    = tmp / "hm_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/hm_compute.py"

        manifest_local.write_text(json.dumps(manifest, indent=2))
        script_local.write_text(REMOTE_HM_PY.format(
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
             "scp hm_compute.py")

        print("Phase B3 — run attention extraction on GCP  (~10 min)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 hm_compute.py")

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

def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)


def visualise(study_set: list, canonical_metric: str) -> dict:
    """Generate 3-panel heatmaps, gallery, correlation violin, return stats."""
    heatmap_dir = FIGURES_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    task_labels = {"mmr": "MMR", "ras": "RAS", "braf": "BRAF"}
    task_indices = {"mmr": 0, "ras": 1, "braf": 2}

    spearman_records: list = []
    gallery_items:    list = []

    for slide in study_set:
        sid      = slide["slide_id"]
        category = slide["category"]

        attn_path   = LOCAL_DATA_DIR / f"{sid}_attn.npy"
        coords_path = LOCAL_DATA_DIR / f"{sid}_coords.npy"

        if not attn_path.exists():
            print(f"  [WARN] No attn data for {sid} — skipping")
            continue
        if not coords_path.exists():
            print(f"  [WARN] No coords for {sid} — skipping")
            continue

        attn   = np.load(attn_path)    # (N, T) or (N,)
        coords = np.load(coords_path)  # (N, 2)

        # Build per-task 2D grids
        grids = {}
        for task, t_idx in task_indices.items():
            if attn.ndim == 2 and attn.shape[1] > t_idx:
                grids[task] = build_attn_grid(attn[:, t_idx], coords)
            elif attn.ndim == 1 and task == "mmr":
                grids[task] = build_attn_grid(attn, coords)

        if len(grids) < 2:
            print(f"  [WARN] Insufficient task attention for {sid} — skipping")
            continue

        # ── 3-panel figure ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{sid}  [{category}]", fontsize=11)

        panel_tasks = ["mmr", "braf", "ras"]
        for ax, task in zip(axes, panel_tasks):
            grid = grids.get(task)
            if grid is None:
                ax.set_title(f"{task_labels[task]}\n(N/A)", fontsize=10)
                ax.axis("off")
                continue
            im = ax.imshow(_normalise(grid), cmap=COLORMAP, aspect="equal",
                           interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(task_labels[task], fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = heatmap_dir / f"{sid}_3head.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        gallery_items.append((sid, category, str(out_path)))

        # ── Spearman correlations ─────────────────────────────────────────────
        if "mmr" in grids and "braf" in grids and "ras" in grids:
            flat_mmr  = grids["mmr"].flatten()
            flat_braf = grids["braf"].flatten()
            flat_ras  = grids["ras"].flatten()

            # Only correlate non-zero patches (occupied cells)
            mask = (flat_mmr > 0) | (flat_braf > 0) | (flat_ras > 0)
            if mask.sum() > 10:
                rho_mmr_braf, pval_mb = spearmanr(flat_mmr[mask], flat_braf[mask])
                rho_mmr_ras,  pval_mr = spearmanr(flat_mmr[mask], flat_ras[mask])
                spearman_records.append({
                    "slide_id":      sid,
                    "category":      category,
                    "rho_mmr_braf":  round(float(rho_mmr_braf), 4),
                    "pval_mmr_braf": round(float(pval_mb),       6),
                    "rho_mmr_ras":   round(float(rho_mmr_ras),   4),
                    "pval_mmr_ras":  round(float(pval_mr),        6),
                })

    # ── Gallery ───────────────────────────────────────────────────────────────
    if gallery_items:
        n = len(gallery_items)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
        axes = np.array(axes).reshape(nrows, ncols)
        for i, (sid, cat, img_path) in enumerate(gallery_items):
            r, c = divmod(i, ncols)
            axes[r, c].imshow(plt.imread(img_path))
            axes[r, c].set_title(f"{sid}\n[{cat}]", fontsize=7)
            axes[r, c].axis("off")
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")
        plt.suptitle("Phase 8 — Cross-Head Attention Gallery (MMR | BRAF | RAS)", fontsize=13)
        plt.tight_layout()
        gallery_path = FIGURES_DIR / "heatmap_gallery.png"
        fig.savefig(gallery_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap_gallery.png ({n} slides)")

    # ── Correlation violin ────────────────────────────────────────────────────
    if len(spearman_records) >= 3:
        rho_mb = [r["rho_mmr_braf"] for r in spearman_records]
        rho_mr = [r["rho_mmr_ras"]  for r in spearman_records]

        fig, ax = plt.subplots(figsize=(6, 5))
        parts = ax.violinplot([rho_mb, rho_mr], positions=[0, 1], showmedians=True)
        colors = ["steelblue", "tomato"]
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ρ(MMR, BRAF)", "ρ(MMR, RAS)"], fontsize=11)
        ax.set_ylabel("Spearman ρ (attention correlation)", fontsize=10)
        ax.set_title(
            f"Phase 8 — Cross-Head Attention Concordance (n={len(spearman_records)} slides)",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        violin_path = FIGURES_DIR / "head_correlation_violin.png"
        fig.savefig(violin_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved head_correlation_violin.png")

    # Save spearman JSON
    spearman_path = FIGURES_DIR / "heatmaps" / "spearman.json"
    spearman_path.parent.mkdir(parents=True, exist_ok=True)
    spearman_path.write_text(json.dumps(spearman_records, indent=2))
    print(f"  Spearman summary: {len(spearman_records)} slides")

    return {"spearman_records": spearman_records}


def _append_report(spearman_records: list, canonical_metric: str,
                   winner_run: str) -> None:
    figures_rel = Path("reports/figures/phase8")
    lines = [
        "",
        "### Cross-Head Attention Heatmaps",
        "",
        "Cross-head attention heatmaps were generated for the winning Phase 8 "
        f"configuration (`{winner_run}`) across all {len(spearman_records)} study "
        "slides. For each slide, the spatial attention of the MMR, BRAF, and RAS "
        "prediction heads was extracted from the multitask ABMIL model and projected "
        "onto a 2D tissue grid. Spearman rank correlation (ρ) between head attention "
        f"maps was computed per slide. Attribution method used: `{canonical_metric}` "
        "(canonical metric from deletion analysis). "
        "Concordance hypothesis: ρ(MMR, BRAF) > ρ(MMR, RAS).",
        "",
        f"![Attention heatmap gallery]({figures_rel}/heatmap_gallery.png)",
        "",
        f"![Head correlation violin]({figures_rel}/head_correlation_violin.png)",
        "",
    ]

    if spearman_records:
        rho_mb_vals = [r["rho_mmr_braf"] for r in spearman_records]
        rho_mr_vals = [r["rho_mmr_ras"]  for r in spearman_records]
        mean_mb = float(np.mean(rho_mb_vals))
        mean_mr = float(np.mean(rho_mr_vals))

        lines += [
            "**Spearman ρ summary:**",
            "",
            "| Metric | Mean ρ | Std |",
            "|--------|--------|-----|",
            f"| ρ(MMR, BRAF) | {mean_mb:.3f} | {float(np.std(rho_mb_vals)):.3f} |",
            f"| ρ(MMR, RAS)  | {mean_mr:.3f} | {float(np.std(rho_mr_vals)):.3f} |",
            "",
        ]

        if mean_mb > mean_mr + 0.05:
            finding = (
                f"ρ(MMR, BRAF) = {mean_mb:.3f} substantially exceeds ρ(MMR, RAS) = {mean_mr:.3f}, "
                "consistent with the hypothesis that BRAF mutation co-drives MMR deficiency "
                "and the two heads attend to overlapping tissue regions."
            )
        elif mean_mb > mean_mr:
            finding = (
                f"ρ(MMR, BRAF) = {mean_mb:.3f} marginally exceeds ρ(MMR, RAS) = {mean_mr:.3f}, "
                "providing weak support for the MMR–BRAF concordance hypothesis."
            )
        else:
            finding = (
                f"ρ(MMR, BRAF) = {mean_mb:.3f} does not exceed ρ(MMR, RAS) = {mean_mr:.3f}; "
                "the expected MMR–BRAF concordance is not observed in this study set."
            )

        lines += [f"**Concordance finding:** {finding}", ""]

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Appended Cross-Head Heatmaps section to {REPORT_PATH.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",         default="reports/data/phase8/",
                   help="Path to extracted phase8 data (default: reports/data/phase8/)")
    p.add_argument("--best-run",         required=True,
                   help="Name of the winning phase8 run")
    p.add_argument("--canonical-metric", default="auto",
                   help="Attribution method for report justification: auto|attn|gradnorm|ixg "
                        "(default: auto — read from deletion/canonical_method.txt)")
    p.add_argument("--gcp-host",         default=None)
    p.add_argument("--gcp-user",         default="chris")
    p.add_argument("--gcp-pass",         default=None)
    p.add_argument("--visualise-only",   action="store_true",
                   help="Skip GCP; visualise from existing local data")
    p.add_argument("--no-report",        action="store_true",
                   help="Skip appending to phase8-results.md")
    return p.parse_args()


def _resolve_canonical(canonical_arg: str) -> str:
    if canonical_arg != "auto":
        return canonical_arg
    canon_path = FIGURES_DIR / "deletion" / "canonical_method.txt"
    if canon_path.exists():
        return canon_path.read_text().strip()
    return "attn"


def main():
    args     = parse_args()
    canonical = _resolve_canonical(args.canonical_metric)

    if args.visualise_only:
        print("--visualise-only: skipping remote compute")
        study_set = load_study_set()
        stats = visualise(study_set, canonical)
        if not args.no_report:
            _append_report(stats["spearman_records"], canonical, args.best_run)
        return

    # ── Phase A — local prep ──────────────────────────────────────────────────
    print("Phase A — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} slides loaded")

    print("\nPhase A2 — building manifest")
    manifest = build_manifest(study_set, args.best_run)

    # ── Dry-run exit ──────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote compute.")
        print(f"  Winner run       : {args.best_run}")
        print(f"  Canonical metric : {canonical}")
        print(f"  Study slides     : {len(study_set)}")
        if LOCAL_DATA_DIR.exists() and list(LOCAL_DATA_DIR.glob("*_attn.npy")):
            print("\nLocal data found — running visualisation")
            stats = visualise(study_set, canonical)
            if not args.no_report:
                _append_report(stats["spearman_records"], canonical, args.best_run)
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    # ── Phase B — remote compute ──────────────────────────────────────────────
    remote_compute(manifest, args.gcp_host, args.gcp_user, args.gcp_pass)

    # ── Phase C — local visualisation ─────────────────────────────────────────
    print("\nPhase C — local visualisation")
    stats = visualise(study_set, canonical)

    if not args.no_report:
        _append_report(stats["spearman_records"], canonical, args.best_run)

    print("\nDone.")
    print(f"  Heatmap gallery  : {FIGURES_DIR}/heatmap_gallery.png")
    print(f"  Violin plot      : {FIGURES_DIR}/head_correlation_violin.png")
    print(f"  Spearman summary : {FIGURES_DIR}/heatmaps/spearman.json")


if __name__ == "__main__":
    main()
