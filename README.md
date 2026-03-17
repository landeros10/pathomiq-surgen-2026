# SurGen — Multitask Molecular Marker Prediction from Whole-Slide Images

Multiple Instance Learning for joint prediction of MMR/MSI, RAS, and BRAF status from colorectal cancer whole-slide image embeddings (UNI foundation model), with ABMIL aggregation and an MLP-RPB spatial position bias extension.

## Abstract

We investigate simultaneous prediction of three clinically actionable molecular markers — MMR status, RAS mutation, and BRAF V600E mutation — from H&E whole-slide images of colorectal cancer, using the SurGen dataset (Myles et al., 2025). Patch embeddings are extracted from a frozen UNI foundation model and processed under a multiple instance learning framework in which a two-layer Transformer encoder aggregates patch-level representations into slide-level predictions. Two aggregation strategies are compared: mean pooling and attention-based MIL (ABMIL), in which task-specific gating networks learn independent patch weightings for each prediction head. A learned relative position bias (MLP-RPB) is evaluated as an extension to the ABMIL architecture to incorporate spatial relationships between patches. Training stability is first established on single-task MMR prediction, then extended to the multitask setting. Model faithfulness is assessed via deletion and insertion perturbation curves using InputXGrad attribution scores, and spatial attribution is examined across a biologically stratified set of test slides grouped by the co-occurrence of MMR, RAS, and BRAF status.

![Stratified IxG attribution heatmaps](report/figures/f4_heatmaps.png)

**Figure 4. Stratified IxG attribution heatmaps by molecular genotype.** Rows correspond to the four genotype groups; columns show three test slides per group, ranked by prediction confidence in the label-correct direction. Upper panel: slide thumbnail overlaid with IxG attribution coloured by task. Lower panel: the top-4 patches by mean IxG score across all three task heads.

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

See [`report.md`](report.md) or [`report.pdf`](report.pdf) for the full methodology, experiments, and results write-up.
