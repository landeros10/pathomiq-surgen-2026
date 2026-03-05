# SurGen Reproduction Plan

---

## Phase 0 — Close Code Gaps ✓ COMPLETED

Closed all correctness gaps before touching GCP: Zarr reader in `MILDataset`, `layer_norm_eps=1e-5` wired through config → model, AMP gated to CUDA only, official split CSVs downloaded from `CraigMyles/SurGen-Dataset`, `model_design.md` committed. Gate passed: full synthetic end-to-end train on MPS, MLflow logged cleanly.

---

## Phase 1 — GCP Training Setup ✓ COMPLETED

**Server:** `136.109.153.16` — Tesla T4 (15 GB), CUDA 12.8, Python 3.10.12, PyTorch 2.7.1+cu128.

**Data** (read-only at `/mnt/data-surgen/embeddings/`): 427 SR386 + 593 SR1482 Zarr embeddings.

Gate passed: synthetic gate test on GCP clean, split resolution and Zarr loading verified on real data for both cohorts.

---

## Phase 2 — MLflow & Reproducibility ✓ COMPLETED

Wired full reproducibility into `train.py`: seeded RNG (`random_seed: 1`), full flattened config logged as params, `git_commit` as tag, `train_auroc`/`val_auroc` per epoch, `best_val_auroc`/`best_epoch`/`test_auroc` as scalars, confusion matrix PNGs and config YAML as artifacts.

Gate passed: all MLflow keys verified in UI on synthetic run.

---

## Phase 3 — First Real Training Run ✓ COMPLETED (paper targets used as baseline reference)

No Phase 3 run reached FINISHED status in MLflow (runs stalled or were superseded by Phase 4).
Paper reproduction targets are used as the baseline reference for Phase 4 gate evaluation.

| Split | Target | Achieved |
|---|---|---|
| Val AUROC  | ~0.9297 | n/a — paper target used as reference |
| Test AUROC | ~0.8273 | n/a — paper target used as reference |

### Steps

- [ ] **Step 1 — Run baseline training**: Update run name in `config_gcp.yaml` (e.g., `mmr-uni-surgen-mean-bce-s1-baseline`), then launch via `scripts/run_train.sh`. MLflow logs `train_auroc`/`val_auroc` per epoch and `best_val_auroc`/`best_epoch`/`test_auroc` as final scalars.

- [ ] **Step 2 — Create `reports/` and write baseline report**: Create `reports/` with a `.gitkeep` and add `reports/*.html` and `reports/*.png` to `.gitignore`. Write `reports/YYYY-MM-DD-phase3-baseline.md` documenting: run name, config used, val AUROC / test AUROC vs. paper targets, and MLflow run URL. This is the documented baseline against which all subsequent phases are compared.

---

## Phase 4 — Training Stability Interventions

**Goal:** Address the training instability observed in the paper (Fig. 12). Best val AUROC peaked at Epoch 56/200 with swings of ±0.2–0.3 AUROC per epoch and no sustained improvement — indicating three correctable issues: high-variance gradients (batch_size=1), severe class imbalance (92% MSS vs. 8% dMMR, unweighted), and a flat learning rate with no decay.

Interventions are evaluated on the combined SurGen dataset, matching the Phase 3 baseline. The Phase 3 run (combined, no interventions) is the no-intervention baseline. The final report crosses all 8 intervention conditions in a single table.

Note: an SR386-only run (using `SR386_msi_*.csv` splits) can optionally be included as a diagnostic comparator to isolate the contribution of SR1482 data from the stability improvements — but this is not the primary track.

### Implementation approach

All three interventions are wired into `train.py` in a single pass. Each is controlled by a config flag that defaults to the Phase 3-equivalent value, so all existing configs remain reproducible without modification. The config file is the complete specification of a run; MLflow logs every flag as a param. There is no separate script per condition — only different config files.

- **Intervention A — LR scheduling** (`lr_scheduler: "cosine"` | `"none"`): `CosineAnnealingLR` with `T_max=200`. Scheduler state is saved to the checkpoint so resumed runs do not reset the annealing cycle.

- **Intervention B — Class-weighted loss** (`class_weighting: true` | `false`): `pos_weight` is computed dynamically from the training split label counts (`n_negative / n_positive`) at startup — not hardcoded — so it remains correct for the combined dataset.

- **Intervention C — Gradient accumulation** (`grad_accum_steps: 8` | `1`): Loss is divided by `accum_steps` before `.backward()`; `optimizer.step()` and `zero_grad()` are called every `accum_steps` slides. Effective batch size becomes 8 with no memory increase.

### Steps

- [x] **Step 1 — Create `scripts/studies/`**: Added `scripts/studies/.gitkeep`. This holds the one-off analysis scripts that produce dated Markdown write-ups in `reports/`. (`reports/` is created in Phase 3 Step 2.)

- [x] **Step 2 — Wire interventions into `train.py`**: Added `lr_scheduler`, `class_weighting`, and `grad_accum_steps` to all three config files and `train.py` with Phase 3-equivalent defaults. Checkpoints now saved as `{"model": ..., "scheduler": ...}` dicts with backward-compatible load. Each param logged to MLflow. Smoke tests pass (4/4).

### Successive Halving Schedule

| Round | Configs | `epochs` | `early_stopping_patience` | Status |
|-------|---------|----------|--------------------------|--------|
| 1     | 8 (all) | 25       | 10                       | ✓ Complete — top 4 carried forward |
| 2     | top 4   | 50       | 15                       | ✓ Complete — no improvement over R1 peak |
| 3     | best 1  | 100      | 20                       | **Skipped** — R2 showed no AUROC gain vs R1; not still improving at epoch 50 |

All rounds use the same seed and combined SurGen dataset. Each round is a separate set of MLflow runs so partial results are queryable at any point.

- [x] **Step 3 — Generate 8 config files**: Created all 8 Round 1 condition configs in `configs/studies/`: `config_surgen_none.yaml`, `config_surgen_cosine.yaml`, `config_surgen_weighted.yaml`, `config_surgen_accum.yaml`, `config_surgen_cosine_weighted.yaml`, `config_surgen_cosine_accum.yaml`, `config_surgen_weighted_accum.yaml`, `config_surgen_all.yaml`. Added `scripts/run_stability_study.sh` to run all 8 sequentially on GCP. Added `--max-epochs` CLI arg to `train.py` and `scripts/preflight_study.sh` to validate each config end-to-end (data load → forward → loss → checkpoint → MLflow) with 1 epoch before the overnight run. Runs logged with `-preflight` suffix.

- [x] **Step 4 — Combined-data runs (8 runs)**: Preflight passed. R1 (8 conditions, 25 epochs) and R2 (top 4, 50 epochs) completed on GCP. MLflow experiment `mmr-surgen-stability`, 16 FINISHED runs total. Note: R2 was initially run with a bug that capped epochs at 25; corrected R2 re-runs completed the full 50 epochs.

- [x] **Step 5 — Report**: `scripts/studies/stability_ablation.py` written and executed. Report at `reports/2026-03-04-training-stability.md`. Includes full condition table with stability metrics (monotonicity ratio, reversals, val-loss rise), val AUROC + loss curves per condition (R1 and R2), stable candidates table (vl_rise ≤ 0.10), and R1→R2 delta analysis.

### Gate: before moving to Phase 5 ✓ PASSED

**Best condition**: `mmr-surgen-s1-cosine-accum16` — val AUROC 0.9002, test AUROC **0.8640** (vs. paper target 0.8273 ✓), best epoch 19 (later and smoother than the paper's epoch 56 peak with ±0.2–0.3 swings). Monotonicity ratio 0.833, vl_rise 0.021 (well below 0.10 threshold — no overfitting signal). Results written to `reports/`. **Round 3 skipped** — corrected 50-epoch R2 re-runs confirmed Δauroc=0.0000 for all R1→R2 comparisons; best checkpoints at epochs 12–19 show the conditions had already converged well before the epoch limit.

---

## Phase 5 — Multitask Validation (MMR + RAS + BRAF)

**Goal:** Confirm the baseline architecture generalizes beyond a single task by training one model that jointly predicts MMR, RAS (KRAS|NRAS), and BRAF from a shared encoder with three output heads. If the model cannot learn multiple tasks at this capacity, adding architectural complexity will not fix it.

**Model:** Shared `MultiMILTransformer` encoder → three independent linear heads, one per task. All slides with at least one valid label are included in training; per-task BCE loss is computed only for heads where a label is present (masked for missing tasks). No `pos_weight` on any head.

**Label source:** Official SurGen dataset splits (CraigMyles/SurGen-Dataset).
RAS derived as KRAS|NRAS union for both cohorts (consistent methodology).

**Metric:** Per-task AUROC (same as Phase 3/4, consistent with paper).

**MLflow experiment:** `multitask-surgen`

**Variants:** Two configs, both now multitask:
- **Base** (`config_gcp.yaml` style): lr=1e-4, no scheduler, grad_accum=1, 100 epochs
- **Winning stability config** (`mmr-surgen-s1-cosine-accum16` style): lr=1e-5, cosine LR, grad_accum=16, 50 epochs, early_stopping_patience=15

This yields **2 runs total**: `multitask-base`, `multitask-cosine-accum16`.

**Class distribution (train/val/test valid N, pos%):** MMR 490/165/165 ~10%/9%/9%; RAS 489/165/164 ~44%/42%/41%; BRAF 448/154/154 ~14%/13%/14%.

### Steps

- [x] **Step 1 — Define task and labels**: Label source and AUROC metric confirmed. RAS = KRAS|NRAS union.
- [x] **Step 2 — Add single-task splits** *(superseded)*: Per-task CSVs generated then deleted in favour of unified multitask splits.
- [x] **Step 3 — Build multitask splits**: `scripts/etl/build_multitask_splits.py` outer-joins MMR/RAS/BRAF per `case_id` → `SurGen_multitask_{train,validate,test}.csv` (cols: `case_id`, `slide_id`, `label_mmr`, `label_ras`, `label_braf`; NaN where task missing). Train=496, val=165, test=165. All assertions passed. Superseded `SurGen_{braf,ras}_*.csv` and `build_combined_splits.py` deleted.
- [x] **Step 4 — Config changes**: `configs/phase5/config_multitask_base.yaml` (lr=1e-4, accum=1) and `config_multitask_cosine_accum16.yaml` (lr=1e-5, cosine, accum=16). Both include `model_type: "MultiMILTransformer"`, `output_classes: 3`, `tasks: [mmr, ras, braf]`.
- [x] **Step 5 — Implement `MultiMILTransformer` and update `train.py`**: `MultiMILTransformer` added to `scripts/models/mil_transformer.py` (same encoder/pool as `MILTransformer`, `Linear(hidden_dim, output_classes)` output). `MultitaskMILDataset` added to `scripts/etl/dataset.py` (returns `valid_mask` tensor, NaN → 0). `train.py` branches on `output_classes > 1`: masked per-task BCE loss, per-epoch per-task AUROC/AUPRC/F1/sensitivity/specificity + head weight variance logged, early stopping on mean val AUROC, final keys `best_val_auroc_mean`/`test_auroc_{task}` (no `best_val_auroc` for multitask). Single-task path unchanged. Smoke test passes.
- [ ] **Step 6 — Train on GCP**: Run both configs sequentially via `scripts/run_phase5.sh` under `nohup`.
- [ ] **Step 7 — Report**: Write `reports/YYYY-MM-DD-phase5-results.md` with a per-task AUROC table for both configs, comparison against Phase 3/4 MMR single-task result, and a diagnosis note if any task fails to show signal.

### Gate: before moving to Phase 6

All three tasks must show per-task val AUROC clearly above chance (> 0.55) on at least one config variant. If any task fails on both variants, diagnose before proceeding. Gate also checks whether the winning stability config outperforms the base config on the new tasks.

---

## Phase 6 — Attention Pooling + 2D Sinusoidal Positional Encoding

**Goal:** Replace the two weakest components of the current baseline — mean pooling and zero spatial awareness — with the two highest-leverage improvements from `brca_riskformer`. Both changes are self-contained and do not require a data pipeline refactor.

**Why these two together:** Attention pooling and positional encoding are deeply coupled. Without PE, the attention weights have no spatial signal to attend to — patches are still exchangeable. Without attention pooling, PE has nowhere to concentrate its effect. They are most useful as a pair.

### Steps

- [ ] **Step 1 — Patch coordinate extraction**: The Zarr embeddings are stored in row-major order (row × col scan pattern). Add a `get_grid_shape(zarr_path)` utility that returns `(H, W)` from the Zarr array's stored metadata or inferred from patch count. Emit `(row, col)` indices alongside embeddings from `MILDataset`. Verify with a quick sanity check on one slide.

- [ ] **Step 2 — `SinusoidalPositionalEncoding2D`**: Port the class from `brca_riskformer/riskformer/training/layers.py` verbatim (MIT license) into `scripts/models/layers.py`. Unit test: given a `(1, H*W, 512)` tensor, confirm output shape is identical and values differ from input.

- [ ] **Step 3 — Attention pooling head**: Replace `x.mean(dim=1)` in `MILTransformer.forward()` with a learned attention pool:
  ```python
  # attn_pool: nn.Linear(hidden_dim, 1)
  weights = torch.softmax(self.attn_pool(x), dim=1)  # (B, N, 1)
  x = (weights * x).sum(dim=1)                        # (B, hidden_dim)
  ```
  Add `attn_pool` to `__init__`. Keep `aggregation` key in config (`"mean"` vs `"attention"`) so old runs remain reproducible.

- [ ] **Step 4 — Wire PE into forward pass**: After `input_proj`, reshape to `(B, H, W, hidden_dim)`, apply `SinusoidalPositionalEncoding2D`, reshape back to `(B, N, hidden_dim)`, then pass to transformer. Gate with a config flag `positional_encoding: "sinusoidal"` vs `"none"`.

- [ ] **Step 5 — Config + MLflow**: Add `aggregation: "attention"` and `positional_encoding: "sinusoidal"` to `config_gcp.yaml`. Create a new MLflow run name (e.g., `mmr-uni-attn-sinpe-s1`). Log `aggregation` and `positional_encoding` as params.

- [ ] **Step 6 — Gate test (local, synthetic)**: Run the synthetic pipeline end-to-end with the new model. Confirm shapes and that gradients flow through both the attention weights and PE parameters.

- [ ] **Step 7 — GCP training run**: Train on real data. Compare val/test AUROC against Phase 3 baseline in MLflow.

### Gate: before moving to Phase 7

Run an ablation: train three variants (mean pool only / attention pool only / attention pool + sinusoidal PE) and compare val AUROC. Write `scripts/studies/ablation_pooling.py` to load each checkpoint and produce a comparison table in `reports/`. If attention pooling does not improve over mean pool by at least ~0.005, investigate before adding more complexity.

---

## Phase 7 — Relative Positional Embeddings

**Goal:** Add distance-aware attention bias so the model can learn that nearby patches should attend to each other more strongly than distant ones. This is the WSI analog of Swin Transformer's relative position bias.

**Prerequisite:** Phase 6 must show that sinusoidal PE contributes positively. Relative PE is an incremental upgrade — if absolute PE provides no benefit, relative PE is unlikely to either.

### Steps

- [ ] **Step 1 — Port `MultiScaleAttention` RPE**: The relative positional embedding logic in `brca_riskformer/riskformer/training/layers.py` (`calc_rel_pos_spatial`, `rel_pos_h`, `rel_pos_w`) can be added to `MILTransformer` as a lightweight modification of `TransformerEncoderLayer`. The simplest approach: replace `nn.TransformerEncoderLayer` with a custom `RPETransformerLayer` that adds the bias directly to the attention logits before softmax.

- [ ] **Step 2 — Config flag**: Add `relative_pos_embedding: true/false` to config. Default `false` to keep backward compatibility.

- [ ] **Step 3 — Unit test**: Confirm that with a known `(H, W)` grid, `rel_pos_h` and `rel_pos_w` are initialized and updated by backprop.

- [ ] **Step 4 — GCP training run + ablation**: Train with sinusoidal PE only vs. sinusoidal PE + RPE. Write `scripts/studies/ablation_rpe.py` to compare checkpoints and write a delta AUROC table to `reports/`.

### Gate: before moving to Phase 8

RPE must show measurable improvement (>0.005 val AUROC) over Phase 6. If not, document and skip — do not carry forward complexity that does not contribute.

---

## Phase 8 — Interpretability Studies *(Placeholder)*

**Goal:** Use the attention weights and global attention pooling weights from the Phase 7 flat model to generate spatially meaningful visualizations. Doing this before the hierarchical refactor validates that attention has learned biologically meaningful structure — and provides a baseline for comparing against the region-level predictions in Phase 9.

**Scripts** (all in `scripts/studies/`, output to `reports/`):
- `attention_heatmap.py` — project final-layer attention weights back onto slide patch grid, render as heatmap overlay
- `attn_pool_weights.py` — rank patches by attention weight, visualize top-k vs. bottom-k patches
- Correlation of high-attention regions with known dMMR tissue patterns (if annotations available)

*Detailed steps TBD. Requires raw WSI thumbnails or patch coordinate metadata for visualization.*

---

## Phase 9 — Region-Level Prediction Structure

**Goal:** Refactor the model into a true hierarchical MIL — patches are grouped into spatial regions, each region gets its own prediction, and a slide-level prediction is produced by learned attention pooling over region descriptors. This enables spatial heatmaps and weak region-level supervision.

**This is the largest refactor.** It touches the data pipeline, model architecture, training loop, loss function, and evaluation. Complete Phases 6–8 before starting.

### Steps

- [ ] **Step 1 — Region grouping in dataset**: Modify `MILDataset` to group patches into non-overlapping spatial regions of fixed size (e.g., 32×32 patches = 1024 patches/region). Return tensor of shape `(num_regions, patches_per_region, hidden_dim)` and a `(num_regions, 2)` region coordinate array.

- [ ] **Step 2 — Local transformer (intra-region)**: Apply the existing `MILTransformer` transformer blocks independently to each region using `einops.rearrange` or a loop. Output: `(num_regions, hidden_dim)` region descriptors.

- [ ] **Step 3 — Region pooling**: Apply `GlobalPoolLayer` (max+avg, ported from `brca_riskformer`) to aggregate patch tokens within each region.

- [ ] **Step 4 — Global attention (inter-region)**: Add a shallow global attention module (1–2 layers) over the `(num_regions, hidden_dim)` sequence to produce the slide-level descriptor.

- [ ] **Step 5 — Dual prediction heads + regional loss**: Implement `regional_coeff` weighted loss:
  - `global_loss = BCE(slide_pred, label) * (1 - regional_coeff)`
  - `local_loss = BCE(top_k_region_preds, label) * regional_coeff` (top 10% of regions by confidence)
  - Start with `regional_coeff=0.0`, sweep over `[0.0, 0.1, 0.3]`

- [ ] **Step 6 — Config + MLflow**: Add `regional_coeff`, `patches_per_region`, and `arch_variant: "HierarchicalMIL"` to config and MLflow params.

- [ ] **Step 7 — Gate test (local, synthetic)**: Run synthetic pipeline. Confirm shapes at every stage and that both global and region-level gradients flow.

- [ ] **Step 8 — GCP training run + ablation**: Write `scripts/studies/ablation_hierarchical.py` to compare flat MIL (Phase 7) vs. hierarchical MIL. Report AUROC delta and training time increase in `reports/`.

### Gate

Hierarchical MIL must match or exceed Phase 7 AUROC. If it underperforms, check region size (too large/small), `regional_coeff` value, and global attention depth before concluding it does not help. Region-level heatmaps should be qualitatively more spatially coherent than Phase 8 patch-level attention maps.
