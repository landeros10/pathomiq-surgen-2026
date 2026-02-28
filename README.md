# SurGen — MMR/MSI Prediction from Whole-Slide Image Embeddings

Transformer-based Multiple Instance Learning (MIL) classifier for predicting
mismatch-repair (MMR) / microsatellite instability (MSI) status from colorectal
cancer whole-slide images, replicating the baseline model from the SurGen paper.

---

## Dataset

| Cohort | Slides | Source |
|--------|--------|--------|
| SR386  | 427    | SurGen |
| SR1482 | 593    | SurGen |

**Task**: binary classification — `0` MSS/pMMR vs `1` MSI/dMMR
**Input**: pre-extracted UNI patch embeddings (1024-dim, one `.pt` file per slide)
**Splits**: predefined 60 : 20 : 20 train / val / test CSVs

> **Class imbalance**: MSI/dMMR represents ~8 % of slides. AUROC is the
> primary metric. Evaluation at four thresholds (0.0119, 0.25, 0.50, 0.75)
> is mandatory — threshold 0.0119 reflects the true class prior.

---

## Project structure

```
surgen/
├── configs/
│   └── config.yaml          # all hyperparameters and paths
├── data/                    # CSV splits (train/val/test.csv)
├── embeddings/              # .pt embedding files          [gitignored]
├── models/                  # saved checkpoints            [gitignored]
├── logs/                    # MLflow & training logs       [gitignored]
├── results/                 # plots and exported metrics
├── notebooks/               # exploration notebooks
├── requirements.txt
└── scripts/
    ├── train.py             # main training entry-point
    ├── evaluate.py          # evaluation with multi-threshold reporting
    ├── etl/
    │   ├── dataset.py       # lazy-loading MIL Dataset (local + GCS)
    │   ├── synthetic.py     # synthetic data generator for pipeline tests
    │   └── splits.py        # stratified split creation & validation
    ├── models/
    │   └── mil_transformer.py  # transformer MIL model
    └── utils/
        ├── metrics.py       # AUROC, AUPRC, per-threshold metrics
        └── mlflow_utils.py  # confusion-matrix logging helpers
```

---

## Quick start — synthetic pipeline test

Run the full pipeline without any real data in ~2 minutes:

```bash
# 1. Activate the environment
source surgen-env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate 10 synthetic slides (100 patches × 1024-dim each)
python scripts/etl/synthetic.py

# 4. Train on synthetic data
python scripts/train.py \
    --data-dir data/synthetic \
    --embeddings-dir embeddings/synthetic

# 5. Evaluate best checkpoint
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --split test \
    --data-dir data/synthetic \
    --embeddings-dir embeddings/synthetic

# 6. Inspect results in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://localhost:5000
```

---

## Training on real data

1. Place UNI `.pt` files in `embeddings/` (one file per slide, named `<slide_id>.pt`).
2. Place `train.csv`, `val.csv`, `test.csv` in `data/`
   Required columns: `slide_id`, `mmr_status` (0/1).
3. *(Optional)* verify coverage before training:
   ```bash
   python - <<'EOF'
   from scripts.etl.splits import validate_splits
   validate_splits("data/", "embeddings/")
   EOF
   ```
4. Train:
   ```bash
   python scripts/train.py
   ```
5. Evaluate:
   ```bash
   python scripts/evaluate.py --checkpoint models/best_model.pt --split test
   ```

---

## Switching to GCP paths

Edit `configs/config.yaml`:

```yaml
paths:
  embeddings_dir: "gs://your-bucket/surgen/embeddings/"
```

Install the GCS filesystem library and authenticate:

```bash
pip install gcsfs
gcloud auth application-default login
```

> Use `DataLoader(num_workers=0)` with GCP paths to avoid multiprocessing
> conflicts with the GCS client (the default in this repo).

---

## Model architecture (SurGen baseline)

```
Input patches  (N × 1024)
  → Linear(1024 → 512) + ReLU
  → TransformerEncoder  [2 layers, 2 heads, FFN=2048, dropout=0.15]
  → mean pool over N patches  →  (512,)
  → Linear(512 → 1)           →  logit
  → BCEWithLogitsLoss
```

| Hyperparameter    | Value  |
|-------------------|--------|
| Optimizer         | Adam   |
| Learning rate     | 1e-4   |
| Batch size        | 1 slide |
| Max epochs        | 200    |
| Early stopping    | 30 epochs patience |

---

## MLflow experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Experiment name: **`mmr-prediction-baseline`**

Each run logs:
- Hyperparameters
- `train_loss`, `val_loss`, `val_auroc` per epoch
- Precision / recall / F1 / specificity at thresholds 0.0119, 0.25, 0.50, 0.75
- Confusion-matrix figures at each threshold
- Best model checkpoint as an artefact

---

## Creating predefined splits from scratch

If you have a master metadata CSV:

```python
from scripts.etl.splits import create_splits

create_splits(
    metadata_csv="data/metadata.csv",
    output_dir="data/",
    label_col="mmr_status",
    slide_id_col="slide_id",
    train_frac=0.6,
    val_frac=0.2,
    seed=42,
)
```
