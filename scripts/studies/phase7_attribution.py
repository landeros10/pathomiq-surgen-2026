#!/usr/bin/env python3
"""Phase 7 Step 2 — Gradient Attribution.

Computes per-patch gradient-based attribution scores for the 19-slide study set
using two methods:
  - GradNorm   : ||∇_{emb} logit||₂  per patch
  - InputXGrad : ||(∇ ⊙ emb)||₂      per patch

Computation runs on GCP (models live there). Results are SCP'd back locally
and visualised as 3-panel PNGs (Attention | GradNorm | InputXGrad).

Usage:
    # Full run (GCP required):
    python scripts/studies/phase7_attribution.py \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Dry-run (validate local state only, no GCP):
    python scripts/studies/phase7_attribution.py
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
from scipy.stats import spearmanr

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
GCP_REMOTE_TMP  = "/tmp/surgen_p7_attr"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_MLFLOW_DB   = "/home/chris/surgen/mlflow.db"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_ROOT_DIR    = "/home/chris/surgen"
GCP_ATTR_DIR    = "/home/chris/surgen/tmp/phase7-attr-data"

LOCAL_DATA_DIR  = ROOT / "results" / "phase7" / "attr"
ATTR_OUT_DIR    = FIGURES_DIR / "attributions"

COLORMAP = "hot"


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_ATTR_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side attribution computation for Phase 7 Step 2.\"\"\"
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

MLFLOW_DB   = "{mlflow_db}"
ZARR_DIR    = Path("{zarr_dir}")
OUT_DIR     = Path("{out_dir}")
MANIFEST    = json.loads(Path("{manifest_path}").read_text())
ST_RUN_ID   = MANIFEST["st_run_id"]
MT_RUN_ID   = MANIFEST["mt_run_id"]
STUDY_SLIDES = MANIFEST["slides"]   # list of {{"slide_id": ..., "category": ...}}

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
    features = torch.from_numpy(store["features"][:]).float()  # (N, 1024)
    coords   = torch.from_numpy(store["coords"][:]).long() if "coords" in store else None
    return features, coords

def _load_model(run_id):
    mlflow.set_tracking_uri(f"sqlite:///{{MLFLOW_DB}}")
    model = mlflow.pytorch.load_model(f"runs:/{{run_id}}/best_model")
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

# Load models once
print("Loading ST model ...")
st_model = _load_model(ST_RUN_ID)
st_model.to(device)
print("Loading MT model ...")
mt_model = _load_model(MT_RUN_ID)
mt_model.to(device)

for slide in STUDY_SLIDES:
    sid = slide["slide_id"]
    print(f"Processing {{sid}} ...")
    try:
        features, coords = _load_zarr(sid)
    except FileNotFoundError as e:
        print(f"  [WARN] {{e}}")
        continue

    emb_np = features.numpy()  # (N, 1024)
    N = emb_np.shape[0]

    for model, prefix in ((st_model, "st"), (mt_model, "mt")):
        try:
            emb_leaf = torch.tensor(emb_np, dtype=torch.float32, device=device,
                                    requires_grad=True)          # leaf: (N, 1024)
            emb = emb_leaf.unsqueeze(0)                          # (1, N, 1024)
            out = model(emb)
            # Extract MMR scalar logit
            if out.ndim == 0:
                logit = out
            elif out.ndim == 1:
                logit = out[0]          # ST: (B,) → scalar
            else:
                logit = out[0, 0]       # MT: (B, T) → MMR logit
            logit.backward()
            grad = emb_leaf.grad.detach().cpu().numpy()  # (N, 1024)
            emb_val = emb_leaf.detach().cpu().numpy()    # (N, 1024)

            grad_norm = np.linalg.norm(grad, axis=1).astype(np.float32)          # (N,)
            ixg       = np.linalg.norm(grad * emb_val, axis=1).astype(np.float32) # (N,)

            np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_grad.npy"), grad_norm)
            np.save(str(OUT_DIR / f"{{sid}}_{{prefix}}_ixg.npy"),  ixg)
            print(f"  {{prefix}} done. grad_norm max={{grad_norm.max():.4f}}, ixg max={{ixg.max():.4f}}")
        except Exception as exc:
            print(f"  [ERROR] {{prefix}} {{sid}}: {{exc}}")

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


# ── Phase 1: local prep ────────────────────────────────────────────────────────

def build_manifest(study_set: list, st_run_id: str, mt_run_id: str) -> dict:
    return {
        "st_run_id": st_run_id,
        "mt_run_id": mt_run_id,
        "slides": [{"slide_id": s["slide_id"], "category": s["category"]}
                   for s in study_set],
    }


# ── Phase 2: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local = tmp / "manifest.json"
        manifest_local.write_text(json.dumps(manifest, indent=2))

        script_local = tmp / "attr_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/attr_compute.py"

        script_content = REMOTE_ATTR_PY.format(
            root_dir      = GCP_ROOT_DIR,
            scripts_dir   = GCP_SCRIPTS_DIR,
            mlflow_db     = GCP_MLFLOW_DB,
            zarr_dir      = GCP_ZARR_DIR,
            out_dir       = GCP_ATTR_DIR,
            manifest_path = manifest_remote,
        )
        script_local.write_text(script_content)

        print("\nPhase 2a — prepare remote directory")
        _run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP} {GCP_ATTR_DIR}"),
             "mkdir remote tmp")

        print("Phase 2b — upload manifest + script")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp attr_compute.py")

        print("Phase 2c — run attribution on GCP")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 attr_compute.py")

        print("Phase 2d — download results")
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _run(_scp_from(host, user, password,
                       f"{GCP_ATTR_DIR}/",
                       str(LOCAL_DATA_DIR) + "/"),
             "scp attribution npy files")

        print("Phase 2e — clean up remote")
        _run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
             "rm remote tmp")


# ── Phase 3: local visualisation ──────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


def visualise(study_set: list) -> None:
    ATTR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_dir = FIGURES_DIR / "heatmaps"

    spearman_records: list = []
    gallery_panels: list   = []   # (slide_id, category, img_array)

    for slide in study_set:
        sid      = slide["slide_id"]
        idx      = slide["idx"]
        category = slide["category"]

        # Load existing ST attention grid (from Step 1)
        attn_grid_path = heatmap_dir / sid / "singletask_mmr_attn.npy"
        if not attn_grid_path.exists():
            print(f"  [WARN] No attention grid for {sid} — skipping")
            continue

        attn_grid = np.load(attn_grid_path)   # (H, W)

        # Load coords for this slide (for grid building)
        try:
            coords = load_slide_coords(SINGLETASK_RUN, idx)
        except FileNotFoundError:
            print(f"  [WARN] No coords for {sid} idx={idx} — skipping")
            continue

        slide_out = ATTR_OUT_DIR / sid
        slide_out.mkdir(parents=True, exist_ok=True)

        for prefix in ("st", "mt"):
            grad_path = LOCAL_DATA_DIR / f"{sid}_{prefix}_grad.npy"
            ixg_path  = LOCAL_DATA_DIR / f"{sid}_{prefix}_ixg.npy"

            if not grad_path.exists() or not ixg_path.exists():
                print(f"  [WARN] Missing {prefix} attribution files for {sid}")
                continue

            grad_arr = np.load(grad_path)   # (N,)
            ixg_arr  = np.load(ixg_path)    # (N,)

            # Build 2D grids
            grad_grid = build_attn_grid(grad_arr, coords)
            ixg_grid  = build_attn_grid(ixg_arr,  coords)

            # Save IxG grid for Step 4
            np.save(slide_out / f"{prefix}_ixg_grid.npy", ixg_grid)

            # ── 3-panel PNG ────────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"{sid} [{category}] — {prefix.upper()} Attribution",
                         fontsize=11, y=1.02)

            for ax, grid, title in zip(
                axes,
                [attn_grid if prefix == "st" else np.load(heatmap_dir / sid / "multitask_mmr_attn.npy"),
                 grad_grid, ixg_grid],
                ["Attention", "GradNorm", "InputXGrad"],
            ):
                im = ax.imshow(_normalise(grid), cmap=COLORMAP, aspect="equal",
                               interpolation="nearest", vmin=0, vmax=1)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            fig.savefig(slide_out / f"{prefix}_attribution_panel.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Spearman correlation (ST attention vs ST IxG)
            if prefix == "st":
                attn_flat = attn_grid.flatten()
                ixg_flat  = ixg_grid.flatten()
                mask = (attn_flat > 0) | (ixg_flat > 0)
                if mask.sum() > 10:
                    rho, pval = spearmanr(attn_flat[mask], ixg_flat[mask])
                    spearman_records.append({
                        "slide_id":  sid,
                        "category":  category,
                        "spearman_rho": float(rho),
                        "pvalue":       float(pval),
                    })

        # Collect for gallery (use ST panels)
        st_panel = slide_out / "st_attribution_panel.png"
        if st_panel.exists():
            img = plt.imread(str(st_panel))
            gallery_panels.append((sid, category, img))

    # ── Spearman summary ───────────────────────────────────────────────────────
    (ATTR_OUT_DIR / "spearman.json").write_text(
        json.dumps(spearman_records, indent=2)
    )
    print(f"  Spearman ρ saved ({len(spearman_records)} slides)")

    # ── Gallery ────────────────────────────────────────────────────────────────
    if gallery_panels:
        n = len(gallery_panels)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
        axes = np.array(axes).reshape(nrows, ncols)

        for i, (sid, cat, img) in enumerate(gallery_panels):
            r, c = divmod(i, ncols)
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{sid}\n[{cat}]", fontsize=7)
            axes[r, c].axis("off")

        # Hide unused axes
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")

        plt.suptitle("Phase 7 Step 2 — Attribution Gallery (ST)", fontsize=13)
        plt.tight_layout()
        fig.savefig(ATTR_OUT_DIR / "gallery.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Gallery saved ({n} panels)")


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
    print(f"  {len(study_set)} slides: "
          + ", ".join(s["slide_id"] for s in study_set))

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

    manifest = build_manifest(study_set, st_run_id, mt_run_id)
    print(f"\n  Manifest: {len(manifest['slides'])} slides to process")
    for s in manifest["slides"]:
        print(f"    {s['slide_id']}  [{s['category']}]")

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
    visualise(study_set)

    print(f"\nDone.")
    print(f"  Attribution panels : {ATTR_OUT_DIR}/")
    print(f"  Gallery            : {ATTR_OUT_DIR}/gallery.png")
    print(f"  Spearman summary   : {ATTR_OUT_DIR}/spearman.json")


if __name__ == "__main__":
    main()
