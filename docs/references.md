# Repository Summary — SurGen Project Reference

> Compiled for the surgen-26-tmb startup interview project. Covers all repositories listed in `Repositories`, recursively summarized for relevance.

---

## 1. `landeros10/surgen-26-tmb` (THIS PROJECT — Private)

**Purpose:** The primary project repository for this interview task. Reproduces the Myles et al. (GigaScience 2025) SurGen paper, targeting MMR/MSI status prediction from colorectal cancer WSIs.

**Key Architecture:**
- UNI foundation model (1024-dim patch embeddings) → Linear(1024→512) + ReLU → TransformerEncoder (2L, 2H, FFN=2048, dropout=0.15) → Mean Pool → Linear(512→1) → BCEWithLogitsLoss
- Patches: 224×224 at 1.0 MPP, background-subtracted
- Optimizer: Adam, lr=1e-4, batch size 1, 200 epochs, AMP

**Targets:** Val AUROC 0.9297 / Test AUROC 0.8273

**Known Gaps:**
- Dataset loader only reads `.pt` files — Zenodo embeddings are in `.zarr` format (reader not yet implemented)
- AMP (`torch.autocast`) not yet wired in for MPS
- `layer_norm_eps=1e-5` not yet passed to `TransformerEncoderLayer`
- `data/` is empty — CSV splits need to be downloaded from `CraigMyles/SurGen-Dataset`
- `model_design.md` exists locally but not committed

**Structure:**
```
surgen/
├── configs/config.yaml         # All hyperparameters (single source of truth)
├── scripts/
│   ├── train.py                # Training loop + MLflow logging
│   ├── evaluate.py             # Multi-threshold eval + confusion matrices
│   ├── etl/
│   │   ├── dataset.py          # MILDataset (currently .pt only)
│   │   ├── synthetic.py        # Fake slides for pipeline testing
│   │   └── splits.py          # Stratified split creation
│   ├── models/mil_transformer.py
│   └── utils/metrics.py, mlflow_utils.py
├── data/                       # CSV splits go here (currently empty)
├── embeddings/                 # .pt / Zarr (gitignored)
├── models/                     # Checkpoints (gitignored)
└── results/                    # Plots and metrics
```

---

## 2. `landeros10/brca_riskformer` (Past Work — Public)

**Purpose:** Breast cancer recurrence risk prediction from WSIs using a hierarchical transformer (RiskFormer). Highly relevant as architectural prior art.

**Clinical Problem:** Risk stratification for BRCA1/2 mutation carriers (45–87% lifetime risk). Provides cost-effective alternative to genetic tests like Oncotype DX™.

**Key Architecture:**
- Hierarchical transformer: patch → region → patient-level predictions
- Patch extraction → UNI/pretrained encoder → region-level transformer blocks with convolution → attention pooling across regions → dual prediction paths (region-level + patient-level)
- Tile dropout, region-level predictions, and attention maps used for interpretability/explainability

**Production-Grade MLOps:**
- PyTorch Lightning with distributed multi-GPU training
- Weights & Biases (W&B) experiment tracking
- Docker containerization for deployment
- AWS (S3, EC2) batch processing infrastructure
- Optuna hyperparameter sweeps via `orchestrators/run_sweep.py`
- Comprehensive unit + integration test suite (`pytest`)

**Structure:**
```
brca_riskformer/
├── configs/                     # preprocessing / training / inference
├── docker/Dockerfile
├── entrypoints/preprocess.py, train.py
├── orchestrators/run_preprocess.py, run_train.py, run_sweep.py
├── riskformer/
│   ├── data/datasets.py, data_preprocess.py
│   ├── training/model.py, layers.py, train.py
│   └── utils/aws_utils.py, data_utils.py, training_utils.py, logger_config.py
└── tests/
```

**Takeaways for surgen-26-tmb:**
- Attention pooling and hierarchical aggregation are architecturally superior to plain mean pooling for large WSIs
- W&B / Lightning / Optuna stack is worth adopting over raw PyTorch + MLflow
- The `orchestrators/` pattern (separate from `entrypoints/`) cleanly separates batch execution logic from single-run logic

---

## 3. `rationai/digital-pathology/pathology/colorectal-cancer` (GitLab — Public)

**Purpose:** RationAI group's colorectal cancer digital pathology pipeline. 12 commits, MIT license, created January 2025.

**Relevance:** Direct domain overlap (colorectal cancer WSI classification). The RationAI group's broader digital pathology framework is well-documented in the literature (see Wagner et al. 2023, *Cancer Cell*).

**Key Insight from Associated Literature (Wagner et al. 2023):**
- Transformer-based biomarker prediction (combining a pre-trained ViT encoder with a transformer aggregator) substantially outperforms CNN-based approaches for MSI prediction in CRC
- Trained across 13,000+ patients from 16 CRC cohorts; achieved 0.99 sensitivity for MSI on surgical resections
- Demonstrates that transformers improve generalizability, data efficiency, and interpretability vs. MIL-ABMIL baselines
- Architecture: ViT patch encoder → TransMIL (transformer aggregator) → slide-level classification

**Takeaways for surgen-26-tmb:** The SurGen paper's model (TransformerEncoder + mean pool) is a direct instantiation of this general paradigm, using UNI as the frozen patch encoder. RationAI's code likely includes stain normalization, patch sampling strategies, and evaluation utilities reusable here.

---

## 4. `rationai/digital-pathology/pathology/breast-cancer` (GitLab — Public)

**Purpose:** RationAI group's breast cancer digital pathology pipeline. 48 commits, 20 branches, MIT license, created August 2024.

**Relevance:** Structurally parallel to the colorectal cancer pipeline; useful as a reference for how the RationAI group organizes MIL training code, data preprocessing, and evaluation across cancer types.

**Takeaways for surgen-26-tmb:** Useful as a cross-cancer architecture reference. The higher branch count (20 vs 3) suggests more active development and potentially more mature utilities (stain normalization, augmentation, WSI preprocessing).

---

## 5. `zhangyn1415/CGPN-GAN` (GitHub — Unavailable/Private)

**Status:** Repository could not be accessed (likely private or deleted at time of fetch).

**What CGPN-GAN likely refers to:** Based on naming conventions in the literature, CGPN-GAN likely refers to a **Conditional GAN with Patch-level or Graph-based architecture** for histopathological image synthesis or augmentation in colorectal cancer. This aligns with a known body of work using GANs for:
- Data augmentation of H&E slides to address class imbalance (MSI cases represent only ~9% of SurGen)
- Stain normalization/transfer across scanners
- Synthetic patch generation for underrepresented mutation classes (e.g., dMMR, BRAF)

**Relevance to surgen-26-tmb:** GAN-based augmentation could be highly relevant given the severe 11:1 MSS:MSI class imbalance in SurGen. If a GAN were used to synthesize additional MSI patches, it could replace or complement class-weight rebalancing strategies.

---

## 6. `Project-MONAI/MONAI` (Toolkit — Public, 7.6k ⭐)

**Purpose:** Medical Open Network for AI — Apache-licensed PyTorch toolkit for healthcare imaging. The de facto standard toolkit for medical image deep learning.

**Key Relevant Modules:**
- `monai.transforms`: WSI-specific transforms (stain normalization, tissue masking, random patch sampling, intensity normalization)
- `monai.data`: `PatchWSIDataset`, `SlidingWindowSplitter`, `MaskedPatchWSIDataset` for efficient patch loading from large slides
- `monai.networks.nets`: ABMIL, TransMIL, and other MIL architectures available out of the box
- `monai.losses`: Focal loss, DiceCELoss — useful for class imbalance (MSS:MSI = 11:1)
- `monai.metrics`: AUROC, AUPRC, confusion matrices
- `monai.apps`: Pre-built pathology-specific app templates

**Takeaways for surgen-26-tmb:**
- `MONAI`'s `MaskedPatchWSIDataset` could replace the custom `etl/dataset.py` for more robust WSI reading (including CZI support via OpenSlide)
- Focal loss (`monai.losses.FocalLoss`) is a plug-in improvement for the 11:1 class imbalance — superior to vanilla BCEWithLogitsLoss
- MONAI's MIL implementations (ABMIL, TransMIL) could serve as baseline comparisons or direct replacements

---

## 7. Other Studies — NCI Data Commons (datacommons.cancer.gov/media/295)

**What it is:** The NCI Cancer Research Data Commons media page, linking to publications and datasets from TCGA, CPTAC, and GDC. Relevant datasets include TCGA-COAD (451 cases, MSI labels) and TCGA-READ (164 cases).

**Relevance to surgen-26-tmb:**
- TCGA-COAD/READ are the most commonly used external validation sets for CRC MSI prediction models
- SurGen's test AUROC of 0.8273 should ideally be contextualized against these cohorts
- Pre-extracted TCGA embeddings (UNI) are available from the Mahmood Lab (Harvard), enabling zero-additional-compute external validation
- The NCI data also includes clinical tabular data (staging, survival) that could be fused with image features for multimodal prediction

---

*Summary compiled 2026-02-28. Links verified at time of fetch. `surgen-26-tmb` is private; all other repos are public unless noted.*
