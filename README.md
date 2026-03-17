# SurGen — Multitask Molecular Marker Prediction from Whole-Slide Images

Multiple Instance Learning for joint prediction of MMR/MSI, RAS, and BRAF status from colorectal cancer whole-slide image embeddings (UNI foundation model), with ABMIL and MLP-RPB aggregation variants.

## Structure

```
surgen/
├── configs/                   # training configs (singletask + multitask)
├── data/splits/               # train/val/test CSV splits
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── etl/                   # dataset, splits, synthetic data
│   ├── models/                # MIL Transformer, ABMIL, MLP-RPB layers
│   ├── eval/                  # inference, interpretability, performance
│   └── utils/                 # metrics, MLflow helpers
├── tests/                     # unit + integration tests
├── report/figures/            # figures for report.md
├── report.md                  # full project write-up
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick test (no real data required)

```bash
python scripts/etl/synthetic.py
python scripts/train.py --config configs/config_singletask_mmr_baseline.yaml
```

## Report

See `report.md` for the full methodology, experiments, and results write-up.
