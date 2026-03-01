# SurGen MMR Classification — Reproduction Requirements

**Goal:** Reproduce the paper's proof-of-concept MMR/MSI classification result from *SurGen: 1020 H&E-stained whole-slide images with survival and genetic markers* (Myles et al., GigaScience 2025).  
**Target metrics:** Validation AUROC **0.9297**, Test AUROC **0.8273**

---

## 1. Dataset

| Property | Value |
|---|---|
| Cohort used | Full SurGen (SR386 + SR1482 combined) |
| Total WSIs | 1,020 |
| Task | Binary classification: MSS/pMMR (0) vs MSI/dMMR (1) |
| Data splits | Pre-defined CSV files from the GitHub repository (`reproducibility/dataset_csv/`) |
| Split ratio | 60 : 20 : 20 (train : validation : test) |
| Stratification | Balanced by MMR/MSI status across splits |
| WSI format | `.CZI` (Zeiss) |
| Magnification | 40× (0.1112 µm/pixel native) |

**Data sources:**
- WSIs: EBI FTP — `ftp.ebi.ac.uk/biostudies/fire/S-BIAD/285/S-BIAD1285/Files/`
- Pre-extracted UNI embeddings: Zenodo — `https://zenodo.org/records/14047723`  
  *(Using the pre-extracted embeddings is strongly recommended to skip the 110-hour extraction step)*

**Class distribution (full SurGen):**
- MSS/pMMR: 745 cases (88.4%)
- MSI/dMMR: 79 cases (9.4%)
- Unknown: 19 cases (excluded)

> **Note:** The dataset is significantly class-imbalanced (~11:1). The paper did **not** apply any class balancing.

---

## 2. Preprocessing Pipeline

### 2.1 Patch Extraction

| Parameter | Value |
|---|---|
| Patch size | 224 × 224 pixels |
| Overlap | None (non-overlapping) |
| Resolution | Extracted at **1.0 MPP** (1 micron per pixel) |
| Native MPP | 0.1112 µm/pixel → requires downsampling by ~9× |
| Background subtraction | Yes — tissue/background mask applied before embedding |

**Background subtraction approach:** Otsu thresholding or similar; tissue patches outlined in green, background/holes in red (as shown in paper Figure 11). Only tissue patches are embedded.

### 2.2 Feature Extraction

| Parameter | Value |
|---|---|
| Foundation model | **UNI** (Chen et al., Nature Medicine 2024) |
| Model type | Self-supervised Vision Transformer (ViT) |
| Pretraining data | 100,000+ H&E-stained WSIs |
| Output embedding | 1,024-dimensional vector per patch |
| Processing | All patches from a WSI processed together |
| Hardware (paper) | Single NVIDIA V100 32 GB GPU |
| Time (paper) | ~110.5 hours for full SurGen |

> **Practical note for local setup (M4 MPS):** Load the pre-extracted Zarr-format embeddings from Zenodo rather than re-running UNI extraction. This skips the 110-hour bottleneck entirely.

---

## 3. Model Architecture

The model is a **Multiple Instance Learning (MIL)** framework with a Transformer encoder for patch aggregation.

### 3.1 Full Forward Pass (per WSI)

```
WSI
 └─ Patch Extraction (224×224 @ 1.0 MPP, background removed)
     └─ UNI Feature Extractor
         └─ N patches × 1024-dim embeddings
             └─ [1] Embedding Projection Layer
                 └─ N × 512-dim
                     └─ [2] Transformer Encoder
                         └─ N × 512-dim
                             └─ [3] Mean Pooling (across N patches)
                                 └─ 1 × 512-dim slide representation
                                     └─ [4] Classification Head
                                         └─ Scalar logit → BCEWithLogitsLoss
```

### 3.2 Layer-by-Layer Specification

#### [1] Embedding Projection Layer
| Parameter | Value |
|---|---|
| Type | Fully connected (Linear) |
| Input dim | 1,024 |
| Output dim | 512 |
| Activation | ReLU |

#### [2] Transformer Encoder
| Parameter | Value |
|---|---|
| Number of layers (L) | 2 |
| Attention heads (H) | 2 |
| Model dimension (d_model) | 512 |
| Feedforward dimension (d_ff) | 2,048 |
| Activation | ReLU |
| Dropout rate | 0.15 |
| Layer norm epsilon | 1 × 10⁻⁵ |
| Positional encoding | Not explicitly mentioned — standard or none |

> Uses standard PyTorch `nn.TransformerEncoder` with `nn.TransformerEncoderLayer`.

#### [3] Aggregation
| Parameter | Value |
|---|---|
| Method | **Mean pooling** across all patch embeddings |
| Output | Single 512-dim slide-level vector |

#### [4] Classification Head
| Parameter | Value |
|---|---|
| Type | Fully connected (Linear) |
| Input dim | 512 |
| Output dim | 1 (binary classification) |
| Activation | None (raw logit; sigmoid applied by loss function) |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Loss function | `BCEWithLogitsLoss` (binary cross-entropy with sigmoid) |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 1 (one WSI per step) |
| Epochs | 200 |
| Automatic Mixed Precision (AMP) | Enabled (`torch.cuda.amp`) |
| Class balancing | None |
| Forward pass | All patches from a single WSI in one forward pass |
| Hardware (paper) | NVIDIA V100 32 GB |
| Training time (paper) | ~3 hours |

**Model selection:** Best model selected based on **highest validation AUROC** during training (not lowest loss).

---

## 5. Evaluation

| Metric | Value (to reproduce) |
|---|---|
| Primary metric | AUROC (Area Under ROC Curve) |
| Validation AUROC | **0.9297** |
| Test AUROC | **0.8273** |

**Threshold analysis** (reported in paper, not required for AUROC):
- Optimal threshold (95% sensitivity on val set): 0.0119
- Standard thresholds: 0.25, 0.50, 0.75

---

## 6. Implementation Checklist

- [ ] Obtain pre-split CSV files from `CraigMyles/SurGen-Dataset` GitHub (`reproducibility/dataset_csv/`)
- [ ] Download pre-extracted UNI embeddings from Zenodo (`zenodo.org/records/14047723`)
- [ ] Understand the Zarr embedding format (see `zarr_examined.ipynb` notebook)
- [ ] Build `Dataset` class that loads patches for a given WSI from Zarr store
- [ ] Implement the `TransformerMIL` model with exact parameters above
- [ ] Implement training loop:
  - One WSI at a time (batch_size=1 at the WSI level)
  - AMP enabled
  - Track validation AUROC each epoch
  - Save checkpoint at best validation AUROC
- [ ] Evaluate on held-out test set using saved best checkpoint
- [ ] Log metrics with MLflow (already configured locally)

---

## 7. Key Differences vs. Prior Work

The paper explicitly compares this experiment to an earlier study (Myles et al., MIUA 2024) which achieved only **0.7136 AUROC** on the smaller SR386 subset alone. The key improvements leading to 0.8273 are:

1. **Larger dataset** — full SurGen (1,020 WSIs) vs. SR386 only (427 WSIs)
2. **Greater tumour heterogeneity** — including metastatic sites from SR1482
3. **No additional hyperparameter tuning** — same architecture applied to more data

---

## 8. Local Setup Notes

Environment: `surgen-env` (Python 3.11)  
Hardware: Apple M4 with MPS GPU

**AMP compatibility:** `torch.cuda.amp` is CUDA-specific. On MPS, use `torch.autocast(device_type='mps')` instead, or disable AMP and rely on MPS's native efficiency.

**UNI model access:** Requires a Hugging Face account and accepting the model license at `hf.co/MahmoodLab/UNI`. If using pre-extracted embeddings from Zenodo, UNI itself does not need to run locally.

**Batch strategy:** With batch_size=1 (one WSI), the number of patches per forward pass varies widely (WSI sizes range from small biopsies to large resections). Ensure sufficient RAM for the largest slides.

---

## 9. Reference Architecture Summary Table

This matches Table 5 in the paper exactly:

| Parameter | Value |
|---|---|
| Task | MMR/MSI detection |
| Cohort | SurGen (SR386 + SR1482) |
| Feature extractor | UNI |
| Patch size | 224 × 224 |
| MPP | 1.0 |
| Embedding dimension (d_model) | 512 |
| Transformer encoder layers | 2 |
| Attention heads | 2 |
| Feedforward dimension | 2,048 |
| Activation | ReLU |
| Dropout rate | 0.15 |
| Layer norm epsilon | 1 × 10⁻⁵ |
| Loss function | BCEWithLogitsLoss |
| Optimiser | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 1 |
| Epochs | 200 |
| AMP | True |

---

*Document prepared for Phase 1 reproduction of the SurGen baseline result.*