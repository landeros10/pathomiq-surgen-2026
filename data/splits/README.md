# Official SurGen Dataset Splits

Source: [`CraigMyles/SurGen-Dataset`](https://github.com/CraigMyles/SurGen-Dataset), path `reproducibility/dataset_csv/`

These are the canonical train/validate/test splits used in the SurGen paper. Do not regenerate them — using these exact files is required for results to be comparable to the published numbers.

## Columns

| Column | Description |
|---|---|
| `case_id` | Slide identifier (e.g. `SR386_40X_HE_T1`) |
| `slide_id` | Same as `case_id` for SR386; identical naming convention |
| `label` | Binary label: `1` = positive (dMMR / mutant / event), `0` = negative |

## Files and verified row counts

Counts verified against Table 4 of the paper. Format: `total (positive+)`.

### SR386 — MSI/MMR status

| File | Rows | Positive (dMMR) |
|---|---|---|
| `SR386_msi_train.csv` | 255 | 20 |
| `SR386_msi_validate.csv` | 84 | 6 |
| `SR386_msi_test.csv` | 84 | 6 |

**Total: 423 slides, 32 dMMR (7.6% positive).** Matches Table 4 exactly.

### SR1482 — MSI/MMR status

| File | Rows | Positive (dMMR) |
|---|---|---|
| `SR1482_train_msi.csv` | 235 | 30 |
| `SR1482_validate_msi.csv` | 81 | 8 |
| `SR1482_test_msi.csv` | 81 | 9 |

**Total: 397 slides, 47 dMMR (11.8% positive).**

### SurGen — combined cohort MSI/MMR status

| File | Rows | Positive (dMMR) |
|---|---|---|
| `SurGen_msi_train.csv` | 490 | 50 |
| `SurGen_msi_validate.csv` | 165 | 14 |
| `SurGen_msi_test.csv` | 165 | 15 |

**Total: 820 slides, 79 dMMR (9.6% positive).**

### SR386 — other mutation tasks

| Task | Train | Val | Test |
|---|---|---|---|
| KRAS | `SR386_kras_train.csv` (243) | `SR386_kras_validate.csv` (83) | `SR386_kras_test.csv` (83) |
| BRAF | `SR386_braf_train.csv` (255) | `SR386_braf_validate.csv` (84) | `SR386_braf_test.csv` (84) |
| NRAS | `SR386_nras_train.csv` (245) | `SR386_nras_validate.csv` (84) | `SR386_nras_test.csv` (82) |
| RAS  | `SR386_ras_train.csv` | `SR386_ras_validate.csv` | `SR386_ras_test.csv` |
| 5-year survival | `SR386_5y_sur_train.csv` (255) | `SR386_5y_sur_validate.csv` (84) | `SR386_5y_sur_test.csv` (84) |

### SR1482 — other mutation tasks

| Task | Train | Val | Test |
|---|---|---|---|
| KRAS | `SR1482_train_kras.csv` | `SR1482_validate_kras.csv` | `SR1482_test_kras.csv` |
| BRAF | `SR1482_train_braf.csv` | `SR1482_validate_braf.csv` | `SR1482_test_braf.csv` |
| NRAS | `SR1482_train_nras.csv` | `SR1482_validate_nras.csv` | `SR1482_test_nras.csv` |

### Segmentation parameters

`segment_params_SR386.csv`, `segment_params_SR1482.csv`, `segment_params_SurGen.csv` — patch extraction parameters used to generate the Zenodo embeddings. Keep for reference; not used directly during training.

## Naming note

SR386 MSI files use `SR386_msi_*.csv` while SR1482 MSI files use `SR1482_*_msi.csv`. This inconsistency is in the upstream repo.
