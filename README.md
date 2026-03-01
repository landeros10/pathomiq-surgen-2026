# SurGen — MMR/MSI Prediction from Whole-Slide Images

Transformer-based Multiple Instance Learning (MIL) for predicting mismatch-repair (MMR) / microsatellite instability (MSI) status from colorectal cancer whole-slide image embeddings.

## Structure

```
surgen/
├── configs/            # training configs
├── data/splits/        # train/val/test CSV splits
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── etl/            # dataset, splits, synthetic data
│   ├── models/         # MIL Transformer
│   └── utils/          # metrics, MLflow helpers
├── tests/              # unit + integration tests
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick test (no real data required)

```bash
python scripts/etl/synthetic.py --config configs/config_gate_test.yaml
python scripts/train.py --config configs/config_gate_test.yaml
```
