# SurGen Reproduction Plan

---

## Phase 0 — Close Code Gaps (Local, ~1–2 days)

Work on the M4 Mac using the synthetic data pipeline. Goal: a clean, correct, tested codebase that can be pushed to GCP without debugging model code remotely.

### Gaps

- [x] **Gap 1 — Zarr reader** *(blocking)*
  Add a Zarr branch to `MILDataset.__getitem__` using `zarr.open(path, mode='r')[:]` converted to a torch tensor. Auto-detect format from file extension. The Zenodo release (`zenodo.14047723`) stores embeddings per-slide as individual Zarr arrays — verify the exact structure by downloading one file before writing the reader.

- [x] **Gap 2 — `layer_norm_eps`** *(correctness)*
  Pass `layer_norm_eps=1e-5` to `TransformerEncoderLayer`. Specified in Table 5 of the paper. Threaded through `config.yaml` → `MILTransformer` → `TransformerEncoderLayer`.

- [x] **Gap 3 — AMP** *(CUDA only)*
  `torch.autocast(device_type='cuda', dtype=torch.float16)` in `train.py`, gated with `if device.type == 'cuda'`. AMP on MPS is a dead end — skip it. On GCP (V100) it activates automatically.

- [x] **Gap 4 — Official data splits** *(data integrity)*
  36 CSVs downloaded from [`CraigMyles/SurGen-Dataset`](https://github.com/CraigMyles/SurGen-Dataset) into `data/splits/`. SR386 MSI counts verified against Table 4: 255/84/84 slides, 20/6/6 dMMR. `config.yaml` updated to point at the correct filenames and `label_column: label`.

- [x] **Gap 5 — Commit `model_design.md`**
  Already tracked since commit `5f9fd37`. Single source of truth for architectural decisions.

### Gate: before moving to Phase 1

Run a full synthetic end-to-end train (5 epochs) with the Zarr reader, AMP disabled (MPS), and confirm MLflow logs. Do not proceed to GCP until this passes cleanly.

```bash
python scripts/etl/synthetic.py
python scripts/train.py --data-dir data/synthetic --embeddings-dir embeddings/synthetic
```

---

## Phase 1 — GCP Setup & Data Pipeline (~2–3 days)

### Storage — Google Cloud Storage

Create one GCS bucket and organize it as:

```
gs://surgen-repro/
├── data/embeddings/zarr/     # Zenodo Zarr embeddings (~large)
├── data/splits/              # CSVs (tiny — also version-control locally)
├── checkpoints/              # Model .pth files
├── mlflow/                   # MLflow artifact store
└── results/                  # Evaluation outputs
```

Upload Zenodo Zarr embeddings with parallel transfer:

```bash
gsutil -m cp -r <local_zarr_dir> gs://surgen-repro/data/embeddings/zarr/
```

### Compute — GCP VM with V100

Use `n1-standard-8 + 1× NVIDIA Tesla V100` (Compute Engine), **not** Vertex AI. Vertex AI orchestration is useful for sweeps but distracting during initial reproduction. Use a Deep Learning VM image (PyTorch, CUDA 11.8 pre-installed).

**Cost:** V100 preemptible spot ~$0.74/hr vs $2.48/hr on-demand. Preemptible is fine for ~3-hour training runs.

### Mounting GCS on the VM

Two options (prefer option B for I/O-heavy workloads):

- **Option A — `gcsfuse`:** Mount `gs://surgen-repro/` at `/mnt/surgen`. Point `embeddings_dir` in `config.yaml` to `/mnt/surgen/data/embeddings/zarr`. Minimal config change, but slower I/O.
- **Option B — local SSD copy:** Pre-copy embeddings to the VM's local SSD at startup via `gsutil cp` in a startup script. Meaningfully faster for patch embedding reads.

### MLflow Tracking Server

Run MLflow on the same VM, with artifact store pointed at GCS:

```python
mlflow.set_tracking_uri("gs://surgen-repro/mlflow/")
# or set via environment variable to keep it out of code:
# MLFLOW_TRACKING_URI=gs://surgen-repro/mlflow/
```

Backend store (run metadata): SQLite file, also synced to GCS. Optionally, deploy a persistent Cloud Run MLflow instance later — but that's not needed for Phase 1.

---

## Phase 2 — Experiment Tracking Structure

MLflow is already wired in. Build on it rather than migrating to W&B now (though W&B would be better long-term — the comparison UI, sweep integration, and artifact versioning are more mature; the free tier covers everything here).

### Run Naming Convention

```
{task}-{encoder}-{aggregator}-{seed}-{note}
```

Example: `mmr-uni-transformer-seed42-baseline`

### Required MLflow Logging (every run)

| What | How |
|---|---|
| All hyperparameters from `config.yaml` | `mlflow.log_params(flat_cfg)` — log the full flattened dict, not just a few fields |
| Git commit hash | `mlflow.set_tag("git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())` |
| Train/val AUROC per epoch | `mlflow.log_metric("val_auroc", ..., step=epoch)` |
| Val loss per epoch | `mlflow.log_metric("val_loss", ..., step=epoch)` |
| Best val AUROC + epoch | `mlflow.log_metric("best_val_auroc", ...)` |
| Test AUROC | Logged once after training, using the best checkpoint — **never touch the test set during training** |
| Confusion matrix (threshold 0.5) | Logged as an artifact |
| Best model checkpoint | `mlflow.log_artifact(best_path)` |

### Experiment Hierarchy

One MLflow experiment per task:
- `mmr-reproduction`
- `kras-prediction`
- `survival-prediction`

### Seeds & Determinism

Fix `random_seed: 42` in config and apply to all RNG sources:

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

Run the paper's exact config first. Only after confirming the baseline result should you do multi-seed runs.

### The `config.yaml` is the Experiment Contract

Every training run must be 100% determined by `config.yaml` + the git commit. No magic defaults buried in code. If anything is overridden via CLI args, log the override as an MLflow param.

---

## Phase 3 — First Real Training Run

### Pre-launch Checklist

- [ ] Config matches Table 5 exactly: `d_model=512`, `n_layers=2`, `n_heads=2`, `d_ff=2048`, `dropout=0.15`, `layer_norm_eps=1e-5`, `lr=1e-4`, `epochs=200`, `batch_size=1`, `BCEWithLogitsLoss`
- [x] Splits are the official paper CSVs (not regenerated) — `data/splits/SR386_msi_{train,validate,test}.csv`
- [ ] Embeddings are the Zenodo pre-extracted UNI Zarr files (not re-extracted locally)
- [ ] Git commit is clean and logged to MLflow
- [ ] AMP enabled (CUDA auto-activates on V100)
- [ ] No class weighting (the paper does not use it)

### Why Use the Zenodo Embeddings?

Re-extracting introduces MPP rounding differences, background subtraction threshold differences, and patch ordering differences. Using the paper's fixed embeddings eliminates all of these variables. Once you reproduce the target AUROC, verify whether re-extraction produces the same embeddings — it should, and confirming it is scientifically useful.

### Expected Results & Tolerance

| Split | Target AUROC |
|---|---|
| Val | ~0.9297 |
| Test | ~0.8273 |

With `batch_size=1` and deterministic training you should get very close. A gap of ±0.01 is acceptable given PyTorch version differences. A gap >0.03 indicates something is systematically wrong — check the split CSVs first, then the model architecture, then AMP behavior.

### Class Imbalance Note

The dataset is ~92% MSS / 8% MSI. The paper uses `BCEWithLogitsLoss` without explicit weighting. Reproduce this first. Document it as a deliberate methodological choice — it is likely a weakness to address in Phase 2 extensions (`pos_weight`, focal loss).

### Compute Estimate

The paper reports **3h 2m 36s** on a V100. At ~$0.74/hr (preemptible), that is roughly **$2.25 per full training run**. Budget accordingly for sweeps.
