#!/usr/bin/env python3
"""Phase 7 Step 5 — Top-Patch Contact Sheets & Morphologic Clustering.

Three phases:
  1. Local  — build manifest of top/bottom patch coords for Part A (19 study slides)
              and Part B (all correctly-classified MSI slides, full test split).
  2. Remote — send manifest + extraction script to GCP; run CZI extraction +
              UMAP/HDBSCAN clustering; download results.
  3. Local  — assemble gallery PNGs, UMAP scatter plots, cluster comparison charts.

Part A: 4×5 contact sheets (top-20 and bottom-20 patches) per slide per model.
Part B: UMAP + HDBSCAN of top-20 zarr embeddings from correctly-classified MSI slides.

Usage:
    # Full run (GCP required):
    python scripts/studies/phase7_top_patches.py \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Dry-run (validate local state, no GCP):
    python scripts/studies/phase7_top_patches.py

    # Skip GCP if data already downloaded:
    python scripts/studies/phase7_top_patches.py --visualise-only
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

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from utils.eval_utils import (      # noqa: E402
    SINGLETASK_RUN,
    MULTITASK_RUN,
    DEFAULT_INFERENCE_DIR as INFERENCE_DIR,
    DEFAULT_FIGURES_DIR as FIGURES_DIR,
    load_study_set,
    load_run_inference,
    load_slide_attn,
    load_slide_coords,
)

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_p7_top"
GCP_ZARR_DIR    = "/mnt/data-surgen/embeddings"
GCP_CZI_DIR     = "/mnt/data-surgen"
GCP_SCRIPTS_DIR = "/home/chris/surgen/scripts"
GCP_INFERENCE   = "/home/chris/surgen/tmp/phase6-report-data/inference"

LOCAL_DATA_DIR  = ROOT / "results" / "phase7" / "top"
LOCAL_ATTR_DIR  = ROOT / "results" / "phase7" / "attr"
TOP_OUT_DIR     = FIGURES_DIR / "top_patches"

TARGET_MPP = 1.0
PATCH_PX   = 224
TOP_K      = 20
GRID_COLS  = 5
GRID_ROWS  = 4   # 4×5 = 20 patches

ST_RUN_ID_FALLBACK = "e60812c9ab0e462080a6bde3bc0141fe"
MT_RUN_ID_FALLBACK = "1013f94132dd4ea092266b334abc3c2e"


# ── Remote computation script ──────────────────────────────────────────────────

REMOTE_TOP_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"GCP-side top-patch extraction + UMAP clustering for Phase 7 Step 5.\"\"\"
import json, re, sys
from pathlib import Path

import numpy as np
import aicspylibczi
from PIL import Image
import zarr
import umap
import hdbscan

ZARR_DIR      = Path("{zarr_dir}")
CZI_DIR       = Path("{czi_dir}")
OUT_DIR       = Path("{out_dir}")
MANIFEST_PATH = Path("{manifest_path}")
TARGET_MPP    = {target_mpp}
PATCH_PX      = {patch_px}
TOP_K         = {top_k}

MANIFEST = json.loads(MANIFEST_PATH.read_text())
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "contact_sheets").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "cluster_patches").mkdir(parents=True, exist_ok=True)


# ── CZI helpers ───────────────────────────────────────────────────────────────

def _pad_id(slide_id):
    m = re.match(r'^(.*_T)(\\d{{1,3}})$', slide_id)
    if m:
        return f"{{m.group(1)}}{{int(m.group(2)):03d}}_01"
    return slide_id


def _find_czi(slide_id):
    for sid in (slide_id, _pad_id(slide_id)):
        p = CZI_DIR / f"{{sid}}.czi"
        if p.exists():
            return str(p)
    return None


def _find_zarr(slide_id):
    for sid in (slide_id, _pad_id(slide_id)):
        p = ZARR_DIR / f"{{sid}}.zarr"
        if p.exists():
            return str(p)
    return None


def _get_mpp(czi):
    import xml.etree.ElementTree as ET
    try:
        meta_xml = czi.meta
        root = ET.fromstring(meta_xml if isinstance(meta_xml, str)
                             else meta_xml.decode())
        val_el = root.find('./Metadata/Scaling/Items/Distance[@Id="X"]/Value')
        return float(val_el.text) * 1e6 if val_el is not None else 0.1112
    except Exception:
        return 0.1112  # fallback 40× CZI ≈ 0.11 µm/px


def _grey_patch(px):
    \"\"\"Return a grey PIL Image as a placeholder for missing patches.\"\"\"
    arr = np.full((px, px, 3), 180, dtype=np.uint8)
    return Image.fromarray(arr)


def extract_patch(czi, bb, mpp, x_px, y_px):
    \"\"\"Extract a single patch from CZI at 1.0 MPP.\"\"\"
    downsample = TARGET_MPP / mpp
    px_at_l0   = int(PATCH_PX * downsample)
    czi_x      = x_px + bb.x
    czi_y      = y_px + bb.y
    mosaic = czi.read_mosaic(
        region=(czi_x, czi_y, px_at_l0, px_at_l0),
        scale_factor=1.0 / downsample,
        C=0,
    )
    arr = np.squeeze(mosaic)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr / (arr.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((PATCH_PX, PATCH_PX))


def assemble_grid(patches, rows, cols):
    \"\"\"Assemble list of PIL images into rows×cols contact sheet.\"\"\"
    w, h = patches[0].size
    canvas = Image.new("RGB", (cols * w, rows * h), (200, 200, 200))
    for i, p in enumerate(patches):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        canvas.paste(p, (c * w, r * h))
    return canvas


# ── Part A: contact sheets ─────────────────────────────────────────────────────

print("\\n=== Part A: Contact Sheets ===")
for entry in MANIFEST.get("part_a", []):
    sid       = entry["slide_id"]
    czi_path  = entry.get("czi_path") or _find_czi(sid)
    safe_id   = sid.replace("/", "_")

    if not czi_path or not Path(czi_path).exists():
        print(f"[WARN] CZI not found for {{sid}} — writing grey placeholders")
        for model_key in ("st", "mt"):
            for rank_type in ("top", "bot"):
                grey_patches = [_grey_patch(PATCH_PX)] * TOP_K
                sheet = assemble_grid(grey_patches, {grid_rows}, {grid_cols})
                out_p = OUT_DIR / "contact_sheets" / f"{{safe_id}}_{{model_key}}_{{rank_type}}.png"
                sheet.save(str(out_p))
        continue

    try:
        czi = aicspylibczi.CziFile(czi_path)
        mpp = _get_mpp(czi)
        bb  = czi.get_mosaic_bounding_box()
    except Exception as e:
        print(f"[WARN] Cannot open CZI for {{sid}}: {{e}}")
        continue

    for model_key in ("st", "mt"):
        for rank_type in ("top", "bot"):
            coords_list = entry.get(f"{{model_key}}_{{rank_type}}", [])
            patches = []
            for x_px, y_px in coords_list[:TOP_K]:
                try:
                    p = extract_patch(czi, bb, mpp, x_px, y_px)
                except Exception as e:
                    print(f"  [WARN] {{sid}} {{model_key}} {{rank_type}} patch: {{e}}")
                    p = _grey_patch(PATCH_PX)
                patches.append(p)
            # Pad with grey if fewer than TOP_K
            while len(patches) < TOP_K:
                patches.append(_grey_patch(PATCH_PX))
            sheet = assemble_grid(patches, {grid_rows}, {grid_cols})
            out_p = OUT_DIR / "contact_sheets" / f"{{safe_id}}_{{model_key}}_{{rank_type}}.png"
            sheet.save(str(out_p))
            print(f"  Saved {{out_p.name}}")

print("Part A done.")


# ── Part B: UMAP clustering ────────────────────────────────────────────────────

print("\\n=== Part B: UMAP Clustering ===")
part_b      = MANIFEST.get("part_b", {{}})
part_b_slides = part_b.get("slides", {{}})

for model_key in ("st", "mt"):
    eligible_sids = part_b.get(f"{{model_key}}_correct_msi", [])
    print(f"\\n-- {{model_key.upper()}} correct-MSI slides: {{len(eligible_sids)}} --")

    embs_list  = []   # (20, 1024) per slide
    meta_list  = []   # {{slide_id, patch_rank, zarr_idx, coords}}

    for sid in eligible_sids:
        slide_info = part_b_slides.get(sid, {{}})
        zarr_hint  = slide_info.get("zarr_path_hint") or _find_zarr(sid)
        zarr_idxs  = slide_info.get(f"{{model_key}}_top20_zarr_indices", [])
        coords_l   = slide_info.get(f"{{model_key}}_top20_coords", [])

        if not zarr_hint or not Path(zarr_hint).exists():
            zarr_path = _find_zarr(sid)
            if zarr_path is None:
                print(f"  [WARN] zarr not found for {{sid}}")
                continue
        else:
            zarr_path = zarr_hint

        try:
            store    = zarr.open(zarr_path, mode="r")
            features = store["features"][:]   # (N, 1024)
        except Exception as e:
            print(f"  [WARN] zarr load failed for {{sid}}: {{e}}")
            continue

        for rank, (zi, coord) in enumerate(zip(zarr_idxs, coords_l)):
            if zi < 0 or zi >= features.shape[0]:
                continue
            embs_list.append(features[zi])
            meta_list.append({{
                "slide_id":  sid,
                "patch_rank": rank,
                "zarr_idx":   int(zi),
                "coords":     coord,
            }})

    M = len(embs_list)
    print(f"  Pooled {{M}} embeddings")

    if M < 20:
        print(f"  [WARN] Only {{M}} embeddings — skipping UMAP for {{model_key}}")
        continue

    X = np.stack(embs_list, axis=0).astype(np.float32)

    # UMAP
    print("  Running UMAP ...")
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0, random_state=42, verbose=False)
    xy      = reducer.fit_transform(X)   # (M, 2)
    np.save(str(OUT_DIR / f"{{model_key}}_umap.npy"), xy)

    # HDBSCAN
    print("  Running HDBSCAN ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=2)
    labels    = clusterer.fit_predict(xy)
    np.save(str(OUT_DIR / f"{{model_key}}_cluster_labels.npy"), labels)

    unique_clusters = [int(k) for k in np.unique(labels) if k >= 0]
    print(f"  Clusters: {{len(unique_clusters)}} (noise: {{(labels == -1).sum()}})")

    # Save meta
    meta_json = {{
        "model_key":        model_key,
        "n_embeddings":     M,
        "n_clusters":       len(unique_clusters),
        "cluster_ids":      unique_clusters,
        "patches":          meta_list,
    }}
    (OUT_DIR / f"{{model_key}}_part_b_meta.json").write_text(json.dumps(meta_json, indent=2))

    if len(unique_clusters) < 3:
        print(f"  [WARN] Fewer than 3 clusters for {{model_key}} — skipping cluster patches")
        continue

    # Cluster representative patches
    czi_cache = {{}}
    for k in unique_clusters:
        mask      = labels == k
        centroid  = xy[mask].mean(axis=0)
        dists     = np.linalg.norm(xy[mask] - centroid, axis=1)
        local_top = np.argsort(dists)[:9]
        global_idxs = np.where(mask)[0][local_top]

        patch_imgs = []
        for gi in global_idxs:
            m   = meta_list[gi]
            sid = m["slide_id"]
            x_px, y_px = m["coords"]

            if sid not in czi_cache:
                czi_path = _find_czi(sid)
                if czi_path:
                    try:
                        c   = aicspylibczi.CziFile(czi_path)
                        mpp = _get_mpp(c)
                        bb  = c.get_mosaic_bounding_box()
                        czi_cache[sid] = (c, mpp, bb)
                    except Exception as e:
                        print(f"  [WARN] CZI open failed {{sid}}: {{e}}")
                        czi_cache[sid] = None
                else:
                    czi_cache[sid] = None

            if czi_cache[sid] is None:
                patch_imgs.append(_grey_patch(PATCH_PX))
                continue

            c, mpp, bb = czi_cache[sid]
            try:
                patch_imgs.append(extract_patch(c, bb, mpp, x_px, y_px))
            except Exception as e:
                print(f"  [WARN] patch extract {{sid}}: {{e}}")
                patch_imgs.append(_grey_patch(PATCH_PX))

        while len(patch_imgs) < 9:
            patch_imgs.append(_grey_patch(PATCH_PX))

        sheet    = assemble_grid(patch_imgs, 3, 3)
        out_name = f"cluster_{{k:02d}}_{{model_key}}_sheet.png"
        sheet.save(str(OUT_DIR / "cluster_patches" / out_name))
        print(f"  Saved {{out_name}}")

print("\\nPart B done.")
print("Remote script complete.")
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


# ── Phase 1 helpers ────────────────────────────────────────────────────────────

def load_canonical_method() -> str:
    """Read deletion/canonical_method.txt; fall back to 'attn' with [WARN]."""
    p = FIGURES_DIR / "deletion" / "canonical_method.txt"
    if not p.exists():
        print(f"  [WARN] canonical_method.txt not found at {p} — defaulting to 'attn'")
        return "attn"
    method = p.read_text().strip()
    if method not in {"attn", "grad", "ixg"}:
        print(f"  [WARN] Unknown canonical method '{method}' — defaulting to 'attn'")
        return "attn"
    return method


def _pad_slide_id(slide_id: str) -> str:
    """SR386_40X_HE_T27 → SR386_40X_HE_T027_01."""
    import re
    m = re.match(r'^(.*_T)(\d{1,3})$', slide_id)
    if m:
        return f"{m.group(1)}{int(m.group(2)):03d}_01"
    return slide_id


def _resolve_mt_idx(run_name: str = MULTITASK_RUN) -> dict:
    """Return {sid: int} full reverse-map from run's slide_ids.json."""
    p = INFERENCE_DIR / run_name / "slide_ids.json"
    if not p.exists():
        raise FileNotFoundError(f"slide_ids.json not found: {p}")
    ids: list = json.loads(p.read_text())
    return {sid: i for i, sid in enumerate(ids)}


def load_attribution_scores(sid: str, model_prefix: str, method: str, idx: int) -> np.ndarray:
    """Return (N,) attribution scores for one slide.

    method='attn': loads from inference attn/ dir.
    method='grad'|'ixg': loads from LOCAL_ATTR_DIR.
    """
    run_name = SINGLETASK_RUN if model_prefix == "st" else MULTITASK_RUN
    if method == "attn":
        attn = load_slide_attn(run_name, idx)
        return attn[:, 0].astype(np.float32) if attn.ndim == 2 else attn.flatten().astype(np.float32)
    else:
        p = LOCAL_ATTR_DIR / f"{sid}_{model_prefix}_{method}.npy"
        return np.load(p).astype(np.float32)


def build_manifest(study_set: list, st_run_inf: dict, mt_run_inf: dict,
                   mt_idx_map: dict, method: str) -> dict:
    """Build the two-part manifest for GCP remote script.

    Part A: top/bottom coords for 19 study slides.
    Part B: zarr indices + coords for all correctly-classified MSI slides.
    """
    # ── Part A ──────────────────────────────────────────────────────────────
    part_a = []
    for slide in study_set:
        sid    = slide["slide_id"]
        st_idx = slide["idx"]                      # index in ST inference
        mt_idx = mt_idx_map.get(sid)
        if mt_idx is None:
            print(f"  [WARN] {sid} not in MT inference — skipping Part A entry")
            continue

        czi_path = f"{GCP_CZI_DIR}/{_pad_slide_id(sid)}.czi"
        entry: dict = {
            "slide_id":  sid,
            "czi_path":  czi_path,
            "category":  slide.get("category", ""),
            "mmr_label": slide.get("mmr_label", -1),
        }

        for model_prefix, idx in (("st", st_idx), ("mt", mt_idx)):
            try:
                run_name = SINGLETASK_RUN if model_prefix == "st" else MULTITASK_RUN
                coords   = load_slide_coords(run_name, idx)
                scores   = load_attribution_scores(sid, model_prefix, method, idx)
                sorted_i = np.argsort(scores)
                top_i    = sorted_i[-TOP_K:][::-1]
                bot_i    = sorted_i[:TOP_K]
                entry[f"{model_prefix}_top"] = [[int(coords[i, 0]), int(coords[i, 1])] for i in top_i]
                entry[f"{model_prefix}_bot"] = [[int(coords[i, 0]), int(coords[i, 1])] for i in bot_i]
            except Exception as e:
                print(f"  [WARN] {sid}/{model_prefix}: {e}")
                entry[f"{model_prefix}_top"] = []
                entry[f"{model_prefix}_bot"] = []

        part_a.append(entry)

    # ── Part B ──────────────────────────────────────────────────────────────
    # Correctly-classified MSI slides from each run
    def _correct_msi(run_inf: dict) -> list[str]:
        ids    = run_inf["slide_ids"]
        probs  = run_inf.get("probs_mmr",  np.full(len(ids), 0.0))
        labels = run_inf.get("labels_mmr", np.full(len(ids), 0.0))
        return [sid for i, sid in enumerate(ids)
                if i < len(labels) and int(labels[i]) == 1 and float(probs[i]) > 0.5]

    st_correct_msi = _correct_msi(st_run_inf)
    mt_correct_msi = _correct_msi(mt_run_inf)

    st_sid_to_idx = {sid: i for i, sid in enumerate(st_run_inf["slide_ids"])}

    all_part_b = set(st_correct_msi) | set(mt_correct_msi)
    part_b_slides: dict = {}

    for sid in sorted(all_part_b):
        czi_path  = f"{GCP_CZI_DIR}/{_pad_slide_id(sid)}.czi"
        zarr_hint = f"{GCP_ZARR_DIR}/{_pad_slide_id(sid)}.zarr"
        slide_entry: dict = {"czi_path": czi_path, "zarr_path_hint": zarr_hint}

        for model_prefix, sid_to_idx in (("st", st_sid_to_idx), ("mt", mt_idx_map)):
            if sid not in sid_to_idx:
                continue
            idx = sid_to_idx[sid]
            try:
                run_name = SINGLETASK_RUN if model_prefix == "st" else MULTITASK_RUN
                coords   = load_slide_coords(run_name, idx)
                scores   = load_attribution_scores(sid, model_prefix, method, idx)
                sorted_i = np.argsort(scores)
                top_i    = sorted_i[-TOP_K:][::-1]
                slide_entry[f"{model_prefix}_top20_zarr_indices"] = [int(i) for i in top_i]
                slide_entry[f"{model_prefix}_top20_coords"]       = [[int(coords[i, 0]), int(coords[i, 1])] for i in top_i]
            except Exception as e:
                print(f"  [WARN] Part B {sid}/{model_prefix}: {e}")

        part_b_slides[sid] = slide_entry

    return {
        "part_a": part_a,
        "part_b": {
            "st_correct_msi": st_correct_msi,
            "mt_correct_msi": mt_correct_msi,
            "slides":         part_b_slides,
        },
        "method": method,
    }


# ── Phase 2: remote compute ────────────────────────────────────────────────────

def remote_compute(manifest: dict, host: str, user: str, password: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        manifest_local  = tmp / "manifest.json"
        script_local    = tmp / "top_compute.py"
        manifest_remote = f"{GCP_REMOTE_TMP}/manifest.json"
        script_remote   = f"{GCP_REMOTE_TMP}/top_compute.py"

        manifest_local.write_text(json.dumps(manifest, indent=2))

        script_content = REMOTE_TOP_PY.format(
            zarr_dir      = GCP_ZARR_DIR,
            czi_dir       = GCP_CZI_DIR,
            out_dir       = f"{GCP_REMOTE_TMP}/out",
            manifest_path = manifest_remote,
            target_mpp    = TARGET_MPP,
            patch_px      = PATCH_PX,
            top_k         = TOP_K,
            grid_rows     = GRID_ROWS,
            grid_cols     = GRID_COLS,
        )
        script_local.write_text(script_content)

        print("\nPhase 2a — prepare remote directory + pre-install packages")
        _run(_ssh_cmd(host, user, password,
                      f"mkdir -p {GCP_REMOTE_TMP}/out"),
             "mkdir remote tmp")
        _run(_ssh_cmd(host, user, password,
                      "python3 -c 'import umap, hdbscan, aicspylibczi' 2>/dev/null || "
                      "pip3 install umap-learn hdbscan aicspylibczi pillow zarr -q"),
             "pre-install GCP packages")

        print("Phase 2b — upload manifest + script")
        _run(_scp_to(host, user, password, str(manifest_local), manifest_remote),
             "scp manifest.json")
        _run(_scp_to(host, user, password, str(script_local), script_remote),
             "scp top_compute.py")

        print("Phase 2c — run top-patch extraction on GCP  (~10 min)")
        _run(_ssh_cmd(host, user, password, f"python3 {script_remote}"),
             "python3 top_compute.py")

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

CATEGORY_ORDER = ["correct_msi", "correct_mss", "wrong_msi", "wrong_mss"]
CATEGORY_COLORS = {
    "correct_msi": "#4caf50",
    "correct_mss": "#2196f3",
    "wrong_msi":   "#ff5722",
    "wrong_mss":   "#9c27b0",
}


def build_contact_sheet_gallery(study_set: list) -> None:
    """Assemble per-category gallery of Part A contact sheets."""
    sheets_dir = LOCAL_DATA_DIR / "contact_sheets"
    if not sheets_dir.exists():
        print(f"  [WARN] contact_sheets dir not found: {sheets_dir}")
        return

    TOP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rank_type in ("top", "bot"):
        panels = []
        labels = []
        for slide in sorted(study_set, key=lambda s: (
                CATEGORY_ORDER.index(s.get("category", "wrong_mss")), s["slide_id"])):
            sid = slide["slide_id"]
            safe_id = sid.replace("/", "_")
            cat  = slide.get("category", "")

            # Use ST sheet (could also use MT — pick ST as primary)
            sheet_path = sheets_dir / f"{safe_id}_st_{rank_type}.png"
            if not sheet_path.exists():
                print(f"  [WARN] Missing {sheet_path.name}")
                continue

            try:
                from PIL import Image
                img = Image.open(sheet_path)
                panels.append(img)
                labels.append(f"{sid}\n[{cat}]")
            except Exception as e:
                print(f"  [WARN] Cannot open {sheet_path}: {e}")

        if not panels:
            print(f"  [WARN] No panels for gallery_{rank_type} — skipping")
            continue

        n_panels = len(panels)
        n_cols   = min(4, n_panels)
        n_rows   = (n_panels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 5.5, n_rows * 5.5))
        axes_flat = np.array(axes).flatten()

        for i, (img, lbl) in enumerate(zip(panels, labels)):
            ax = axes_flat[i]
            ax.imshow(np.array(img))
            ax.set_title(lbl, fontsize=6, pad=2)
            ax.axis("off")
        for i in range(n_panels, len(axes_flat)):
            axes_flat[i].axis("off")

        fig.suptitle(f"Phase 7 Step 5 — Part A {rank_type.upper()}-20 Contact Sheets (ST)",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = TOP_OUT_DIR / f"gallery_{rank_type}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def build_umap_scatter(model_key: str) -> None:
    """Scatter plot of Part B UMAP embeddings coloured by HDBSCAN cluster."""
    umap_path    = LOCAL_DATA_DIR / f"{model_key}_umap.npy"
    labels_path  = LOCAL_DATA_DIR / f"{model_key}_cluster_labels.npy"
    meta_path    = LOCAL_DATA_DIR / f"{model_key}_part_b_meta.json"

    for p in (umap_path, labels_path, meta_path):
        if not p.exists():
            print(f"  [WARN] Missing {p.name} — skipping {model_key} UMAP scatter")
            return

    xy      = np.load(umap_path)
    labels  = np.load(labels_path)
    meta    = json.loads(meta_path.read_text())

    unique_k = [k for k in np.unique(labels) if k >= 0]
    n_k      = len(unique_k)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Noise (label == -1) in grey
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
                   s=10, c="lightgrey", alpha=0.4, label="noise", zorder=1)

    for k in unique_k:
        mask  = labels == k
        color = cmap(k % 20 / 20)
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=18, color=color, alpha=0.7, label=f"cluster {k}", zorder=2)

    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    model_label = "ST" if model_key == "st" else "MT-joined"
    ax.set_title(f"Phase 7 Step 5 — Part B UMAP ({model_label})\n"
                 f"n={len(xy)}, clusters={n_k}", fontsize=11)
    if n_k <= 15:
        ax.legend(fontsize=7, markerscale=1.5, loc="best")

    TOP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    out_path = TOP_OUT_DIR / f"{model_key}_umap_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def build_cluster_comparison() -> None:
    """Stacked bar: per-cluster slide-origin fractions for ST and MT."""
    TOP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key in ("st", "mt"):
        meta_path   = LOCAL_DATA_DIR / f"{model_key}_part_b_meta.json"
        labels_path = LOCAL_DATA_DIR / f"{model_key}_cluster_labels.npy"

        if not meta_path.exists() or not labels_path.exists():
            print(f"  [WARN] Missing meta/labels for {model_key} — skipping cluster comparison")
            continue

        meta   = json.loads(meta_path.read_text())
        labels = np.load(labels_path)

        patches     = meta.get("patches", [])
        unique_k    = [k for k in np.unique(labels) if k >= 0]
        all_sids    = sorted(set(p["slide_id"] for p in patches))

        if not unique_k or not all_sids:
            print(f"  [WARN] No clusters/slides for {model_key}")
            continue

        # Build cluster × slide count matrix
        matrix = np.zeros((len(unique_k), len(all_sids)), dtype=int)
        sid_idx = {sid: i for i, sid in enumerate(all_sids)}

        for pi, (lbl, patch) in enumerate(zip(labels, patches)):
            if lbl < 0:
                continue
            ki = unique_k.index(lbl) if lbl in unique_k else -1
            if ki < 0:
                continue
            si = sid_idx.get(patch["slide_id"], -1)
            if si < 0:
                continue
            matrix[ki, si] += 1

        # Normalise rows to fractions
        row_sums = matrix.sum(axis=1, keepdims=True)
        fracs    = np.where(row_sums > 0, matrix / row_sums, 0.0)

        fig, ax = plt.subplots(figsize=(max(6, len(unique_k) * 0.8), 5))
        cmap    = plt.get_cmap("tab20")
        bottom  = np.zeros(len(unique_k))
        x       = np.arange(len(unique_k))

        for si, sid in enumerate(all_sids):
            col = cmap(si % 20 / 20)
            ax.bar(x, fracs[:, si], bottom=bottom, color=col,
                   label=sid[:18], alpha=0.85)
            bottom += fracs[:, si]

        ax.set_xticks(x)
        ax.set_xticklabels([f"C{k}" for k in unique_k], fontsize=9)
        ax.set_xlabel("Cluster", fontsize=10)
        ax.set_ylabel("Slide fraction", fontsize=10)
        model_label = "ST" if model_key == "st" else "MT-joined"
        ax.set_title(f"Phase 7 Step 5 — Cluster Slide Composition ({model_label})", fontsize=11)
        if len(all_sids) <= 15:
            ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.18, 1))
        ax.set_ylim(0, 1)

        plt.tight_layout()
        out_path = TOP_OUT_DIR / f"cluster_slide_overlap_{model_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


def visualise(study_set: list) -> None:
    TOP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPhase 3a — building contact sheet gallery")
    build_contact_sheet_gallery(study_set)

    print("\nPhase 3b — building UMAP scatter plots")
    for model_key in ("st", "mt"):
        build_umap_scatter(model_key)

    print("\nPhase 3c — building cluster comparison charts")
    build_cluster_comparison()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcp-host",       default=None,
                   help="GCP server IP (omit for dry-run)")
    p.add_argument("--gcp-user",       default="chris")
    p.add_argument("--gcp-pass",       default=None)
    p.add_argument("--visualise-only", action="store_true",
                   help="Skip Phases 1 & 2; visualise from existing LOCAL_DATA_DIR")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.visualise_only:
        print("--visualise-only: skipping Phases 1 & 2")
        study_set = load_study_set()
        print(f"  {len(study_set)} slides loaded")
        visualise(study_set)
        return

    # ── Phase 1 — local prep ──────────────────────────────────────────────────
    print("Phase 1a — loading study set")
    study_set = load_study_set()
    print(f"  {len(study_set)} slides: "
          + ", ".join(s["slide_id"] for s in study_set))

    print("\nPhase 1b — resolving canonical attribution method")
    method = load_canonical_method()
    print(f"  Method: {method}")

    print("\nPhase 1c — loading inference data")
    st_run_inf = load_run_inference(SINGLETASK_RUN)
    mt_run_inf = load_run_inference(MULTITASK_RUN)
    print(f"  ST slides: {len(st_run_inf['slide_ids'])}")
    print(f"  MT slides: {len(mt_run_inf['slide_ids'])}")

    print("\nPhase 1d — resolving MT slide indices")
    mt_idx_map = _resolve_mt_idx(MULTITASK_RUN)

    print("\nPhase 1e — building manifest")
    manifest = build_manifest(study_set, st_run_inf, mt_run_inf, mt_idx_map, method)

    n_part_a    = len(manifest["part_a"])
    n_st_msi    = len(manifest["part_b"]["st_correct_msi"])
    n_mt_msi    = len(manifest["part_b"]["mt_correct_msi"])
    n_part_b    = len(manifest["part_b"]["slides"])
    print(f"  Part A: {n_part_a} study slides")
    print(f"  Part B: {n_part_b} unique slides  "
          f"(ST correct-MSI={n_st_msi}, MT correct-MSI={n_mt_msi})")
    for s in manifest["part_a"]:
        n_st_top = len(s.get("st_top", []))
        n_mt_top = len(s.get("mt_top", []))
        print(f"    {s['slide_id']:40s} [{s['category']}]  "
              f"st_top={n_st_top}  mt_top={n_mt_top}")

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
    print(f"  Contact sheet gallery : {TOP_OUT_DIR}/gallery_top.png")
    print(f"  UMAP scatter          : {TOP_OUT_DIR}/st_umap_scatter.png")
    print(f"  Cluster comparison    : {TOP_OUT_DIR}/cluster_slide_overlap_st.png")


if __name__ == "__main__":
    main()
