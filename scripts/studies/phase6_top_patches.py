#!/usr/bin/env python3
"""Phase 6 — top-attended patch RGB extractor.

Identifies the patches with the highest attention weights for the top-N slides
(selected by single-task attention entropy), then extracts RGB images of those
patches from the raw .czi files on the GCP server.

Three phases:
  1. Local  — load attn/coords .npy files, build manifest JSON of top-patch coords.
  2. Remote — scp manifest + extraction script to GCP; run openslide extraction.
  3. Local  — scp results back; organise into <out-dir>/patches/.

Usage:
    python scripts/studies/phase6_top_patches.py \\
        --inf-dir  tmp/phase6-report-data/inference \\
        --st-run   singletask-mmr-abmil-cosine-accum16 \\
        --mt-run   multitask-abmil-joined-cosine-accum16 \\
        --out-dir  tmp/phase6-patches \\
        --gcp-host 136.109.153.16 \\
        --gcp-user chris \\
        --gcp-pass 'chris@7yz' \\
        --n-slides 5 \\
        --n-patches 3

    # Dry-run (manifest only, no GCP):
    python scripts/studies/phase6_top_patches.py \\
        --inf-dir tmp/phase6-report-data/inference \\
        --st-run  singletask-mmr-abmil-cosine-accum16 \\
        --mt-run  multitask-abmil-joined-cosine-accum16 \\
        --out-dir tmp/phase6-patches
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
GCP_REMOTE_TMP  = "/tmp/surgen_patches"
GCP_DATA_DIR    = "/mnt/data-surgen"
TARGET_MPP      = 1.0          # µm/px at extraction; ~9× downsample from 0.11 µm/px
PATCH_PX        = 224          # output patch size (pixels)

# ── Slide-ID → CZI filename mapping ───────────────────────────────────────────

def _pad_slide_id(slide_id: str) -> str:
    """Map un-padded CSV slide IDs to zero-padded server CZI filenames.

    Mirrors dataset.py::_pad_sr386_id().
    'SR386_40X_HE_T27'       → 'SR386_40X_HE_T027_01'
    'SR1482_40X_HE_T118_01'  → 'SR1482_40X_HE_T118_01'  (already padded)
    """
    m = re.match(r'^(.*_T)(\d{1,3})$', slide_id)
    if m:
        return f"{m.group(1)}{int(m.group(2)):03d}_01"
    return slide_id


def slide_id_to_czi(slide_id: str) -> str:
    padded = _pad_slide_id(slide_id)
    return f"{GCP_DATA_DIR}/{padded}.czi"


# ── Attention entropy ──────────────────────────────────────────────────────────

def compute_attn_entropy(weights: np.ndarray) -> float:
    """H = −Σ wᵢ log wᵢ, averaged over tasks if (N, T)."""
    eps = 1e-12
    if weights.ndim == 2:
        return float(np.mean([
            -np.sum(np.clip(weights[:, t], eps, 1.0) * np.log(np.clip(weights[:, t], eps, 1.0)))
            for t in range(weights.shape[1])
        ]))
    w = np.clip(weights.flatten(), eps, 1.0)
    return float(-np.sum(w * np.log(w)))


# ── Phase 1: build manifest ────────────────────────────────────────────────────

def build_manifest(
    inf_dir: Path,
    st_run: str,
    mt_run: str,
    n_slides: int,
    n_patches: int,
) -> dict:
    """Return manifest dict: {slide_id: {"st": [[row,col],...], "mt": [[row,col],...]}}."""

    def _load_run_paths(run_name: str) -> tuple[dict, dict]:
        run_dir   = inf_dir / run_name
        sid_path  = run_dir / "slide_ids.json"
        if not sid_path.exists():
            raise FileNotFoundError(f"Missing slide_ids.json in {run_dir}")
        slide_ids  = json.loads(sid_path.read_text())
        attn_paths : dict = {}
        coord_paths: dict = {}
        attn_dir   = run_dir / "attn"
        coords_dir = run_dir / "coords"
        for i, sid in enumerate(slide_ids):
            tag = f"{i:04d}"
            ap  = attn_dir   / f"{tag}.npy"
            cp  = coords_dir / f"{tag}.npy"
            if ap.exists():
                attn_paths[sid]  = ap
            if cp.exists():
                coord_paths[sid] = cp
        return attn_paths, coord_paths

    st_attn, st_coords = _load_run_paths(st_run)
    mt_attn, mt_coords = _load_run_paths(mt_run)

    common = [
        sid for sid in st_attn
        if sid in mt_attn and sid in st_coords and sid in mt_coords
    ]
    if not common:
        raise RuntimeError("No slides found in common between the two runs (with both attn + coords).")

    print(f"  Common slides: {len(common)}")

    # Sort by ST entropy descending; take top n_slides
    def _st_entropy(sid):
        return compute_attn_entropy(np.load(st_attn[sid]))

    selected = sorted(common, key=_st_entropy, reverse=True)[:n_slides]
    print(f"  Selected slides: {selected}")

    manifest: dict = {}
    for sid in selected:
        st_w  = np.load(st_attn[sid])
        mt_w  = np.load(mt_attn[sid])
        st_c  = np.load(st_coords[sid])
        mt_c  = np.load(mt_coords[sid])

        # Flatten to 1-D weights (take task-0 column for multitask joined)
        st_w1d = st_w[:, 0] if st_w.ndim == 2 else st_w.flatten()
        mt_w1d = mt_w[:, 0] if mt_w.ndim == 2 else mt_w.flatten()

        def _top_coords(w1d, coords, k):
            idxs = np.argsort(w1d)[-k:][::-1]
            # coords[:,0] = X (horizontal), coords[:,1] = Y (vertical)
            return [[int(coords[i, 0]), int(coords[i, 1])] for i in idxs]

        manifest[sid] = {
            "czi_path": slide_id_to_czi(sid),
            "st": _top_coords(st_w1d, st_c, n_patches),
            "mt": _top_coords(mt_w1d, mt_c, n_patches),
        }

    return manifest


# ── Remote extraction script ───────────────────────────────────────────────────

REMOTE_EXTRACT_PY = textwrap.dedent("""\
#!/usr/bin/env python3
\"\"\"Remote extraction script — runs on GCP server.
Uses aicspylibczi (no system libopenslide.so required) to read .czi files.
\"\"\"
import json, numpy as np
from pathlib import Path

try:
    import aicspylibczi
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install",
                           "aicspylibczi", "-q"])
    import aicspylibczi

try:
    from PIL import Image
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install",
                           "Pillow", "-q"])
    from PIL import Image

MANIFEST_PATH = Path("{remote_tmp}/manifest.json")
OUT_DIR       = Path("{remote_tmp}/out")
TARGET_MPP    = {target_mpp}
PATCH_PX      = {patch_px}

manifest = json.loads(MANIFEST_PATH.read_text())
OUT_DIR.mkdir(parents=True, exist_ok=True)

for slide_id, entry in manifest.items():
    czi_path = entry["czi_path"]
    if not Path(czi_path).exists():
        print(f"[WARN] CZI not found: {{czi_path}} — skipping {{slide_id}}")
        continue

    try:
        czi = aicspylibczi.CziFile(czi_path)
    except Exception as e:
        print(f"[WARN] Cannot open {{czi_path}}: {{e}}")
        continue

    # Read physical pixel size from CZI metadata (µm/px)
    try:
        import xml.etree.ElementTree as ET
        meta_xml = czi.meta
        root = ET.fromstring(meta_xml) if isinstance(meta_xml, str) else ET.fromstring(meta_xml.decode())
        # XY scaling is in Scaling/Items/Distance[@Id="X"]/Value (metres)
        ns = {{'x': 'http://www.zeiss.com/czi/2010/Image'}}
        val_el = root.find('./Metadata/Scaling/Items/Distance[@Id="X"]/Value')
        mpp_x = float(val_el.text) * 1e6 if val_el is not None else 0.1112
    except Exception:
        mpp_x = 0.1112  # fallback: ~0.11 µm/px for 40× CZI

    downsample  = TARGET_MPP / mpp_x          # ~9× for 0.11 µm/px native
    px_at_l0    = int(PATCH_PX * downsample)  # native px covering 224 µm

    # CZI has an absolute coordinate system; npy coords are offsets from
    # the image top-left (0,0). Translate by adding the bounding box origin.
    # coords[:,0] spans the X (horizontal/width) dimension of the CZI.
    # coords[:,1] spans the Y (vertical/height) dimension of the CZI.
    bb = czi.get_mosaic_bounding_box()

    safe_id = slide_id.replace("/", "_").replace(" ", "_")
    for model_key in ("st", "mt"):
        for rank, (x_px, y_px) in enumerate(entry[model_key]):
            out_path = OUT_DIR / f"{{safe_id}}_{{model_key}}_{{rank}}.png"
            try:
                # aicspylibczi uses (x, y, width, height) in level-0 px
                czi_x = x_px + bb.x
                czi_y = y_px + bb.y
                mosaic = czi.read_mosaic(
                    region=(czi_x, czi_y, px_at_l0, px_at_l0),
                    scale_factor=1.0 / downsample,
                    C=0,
                )
                # mosaic shape: (1, Y, X, C) or (Y, X, C) or (Y, X)
                arr = np.squeeze(mosaic)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[-1] > 3:
                    arr = arr[..., :3]
                # Normalise to uint8 if needed
                if arr.dtype != np.uint8:
                    arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
                patch = Image.fromarray(arr).resize((PATCH_PX, PATCH_PX))
                patch.save(str(out_path))
                print(f"  Saved {{out_path}}")
            except Exception as e:
                print(f"  [WARN] Failed {{slide_id}} {{model_key}}[{{rank}}]: {{e}}")

print("Done.")
""")


# ── SSH/SCP helpers ────────────────────────────────────────────────────────────

def _ssh_cmd(host: str, user: str, password: str, remote_cmd: str) -> list[str]:
    return [
        "sshpass", "-p", password,
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}", remote_cmd,
    ]


def _scp_to(host: str, user: str, password: str,
             local: str, remote: str) -> list[str]:
    return [
        "sshpass", "-p", password,
        "scp", "-o", "StrictHostKeyChecking=no",
        local, f"{user}@{host}:{remote}",
    ]


def _scp_from(host: str, user: str, password: str,
               remote: str, local: str) -> list[str]:
    return [
        "sshpass", "-p", password,
        "scp", "-r", "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}:{remote}", local,
    ]


def run(cmd: list[str], desc: str = "") -> None:
    label = desc or " ".join(cmd[:4])
    print(f"  >> {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[:500]}")
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"Command failed ({result.returncode}): {label}")
    if result.stdout.strip():
        print(result.stdout.strip())


# ── Phase 2: remote extraction ─────────────────────────────────────────────────

def remote_extract(
    manifest: dict,
    host: str,
    user: str,
    password: str,
    out_dir: Path,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write manifest
        manifest_local = tmp / "manifest.json"
        manifest_local.write_text(json.dumps(manifest, indent=2))

        # Write extraction script (fill template)
        extract_script = REMOTE_EXTRACT_PY.format(
            remote_tmp=GCP_REMOTE_TMP,
            target_mpp=TARGET_MPP,
            patch_px=PATCH_PX,
        )
        extract_local = tmp / "extract_patches.py"
        extract_local.write_text(extract_script)

        # Prepare remote tmp dir
        print("\nPhase 2a — prepare remote directory")
        run(_ssh_cmd(host, user, password, f"mkdir -p {GCP_REMOTE_TMP}/out"),
            "mkdir remote tmp")

        # Upload manifest + script
        print("Phase 2b — upload manifest and extraction script")
        run(_scp_to(host, user, password,
                    str(manifest_local), f"{GCP_REMOTE_TMP}/manifest.json"),
            "scp manifest.json")
        run(_scp_to(host, user, password,
                    str(extract_local), f"{GCP_REMOTE_TMP}/extract_patches.py"),
            "scp extract_patches.py")

        # Execute
        print("Phase 2c — run extraction on GCP")
        run(_ssh_cmd(host, user, password,
                     f"python3 {GCP_REMOTE_TMP}/extract_patches.py"),
            "python3 extract_patches.py")

        # Download results
        print("Phase 2d — download patches")
        patches_local = out_dir / "patches"
        patches_local.mkdir(parents=True, exist_ok=True)
        run(_scp_from(host, user, password,
                      f"{GCP_REMOTE_TMP}/out/*",
                      str(patches_local) + "/"),
            "scp patches back")

        # Clean up remote
        run(_ssh_cmd(host, user, password, f"rm -rf {GCP_REMOTE_TMP}"),
            "rm remote tmp")


# ── Phase 3: verify output ─────────────────────────────────────────────────────

def verify_output(out_dir: Path) -> None:
    patches_dir = out_dir / "patches"
    pngs = sorted(patches_dir.glob("*.png")) if patches_dir.exists() else []
    print(f"\nPhase 3 — output verification")
    print(f"  Patches found: {len(pngs)}")
    for p in pngs:
        print(f"    {p.name}  ({p.stat().st_size // 1024} KB)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inf-dir",   required=True,
                   help="Path to inference/ directory (contains <run>/attn etc.)")
    p.add_argument("--st-run",    required=True,
                   help="Single-task run name subdirectory")
    p.add_argument("--mt-run",    required=True,
                   help="Multitask run name subdirectory")
    p.add_argument("--out-dir",   required=True,
                   help="Local output directory for patches")
    p.add_argument("--gcp-host",  default=None,
                   help="GCP server IP (omit for dry-run / manifest-only mode)")
    p.add_argument("--gcp-user",  default="chris")
    p.add_argument("--gcp-pass",  default=None)
    p.add_argument("--n-slides",  type=int, default=5,
                   help="Number of top slides to process")
    p.add_argument("--n-patches", type=int, default=3,
                   help="Number of top patches per model per slide")
    return p.parse_args()


def main():
    args = parse_args()
    inf_dir = Path(args.inf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("Phase 1 — building top-patch manifest")
    manifest = build_manifest(
        inf_dir, args.st_run, args.mt_run, args.n_slides, args.n_patches
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved → {manifest_path}")

    for sid, entry in manifest.items():
        print(f"  {sid}")
        print(f"    ST top patches: {entry['st']}")
        print(f"    MT top patches: {entry['mt']}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if args.gcp_host is None:
        print("\n[Dry-run] --gcp-host not provided; skipping remote extraction.")
        print(f"Manifest written to {manifest_path}")
        return

    if args.gcp_pass is None:
        print("ERROR: --gcp-pass required when --gcp-host is set", file=sys.stderr)
        sys.exit(1)

    remote_extract(manifest, args.gcp_host, args.gcp_user, args.gcp_pass, out_dir)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    verify_output(out_dir)
    print(f"\nDone. Patches in {out_dir / 'patches'}")


if __name__ == "__main__":
    main()
