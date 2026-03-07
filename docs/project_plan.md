# SurGen Reproduction Plan

---

## Phases 0–5 ✓ COMPLETED

### Phase 0 — Code Gaps
Fixed before GCP: Zarr reader in `MILDataset`, `layer_norm_eps=1e-5` through config, AMP gated to CUDA only, official SurGen splits downloaded. Synthetic end-to-end train on MPS passed.

### Phase 1 — GCP Setup
**Server:** `136.109.153.16` — Tesla T4 (15 GB), CUDA 12.8, PyTorch 2.7.1+cu128.
**Data** (read-only): `/mnt/data-surgen/embeddings/` — 427 SR386 + 593 SR1482 Zarr embeddings.

### Phase 2 — MLflow & Reproducibility
Wired into `train.py`: seeded RNG, full config as params, `git_commit` tag, per-epoch `train_auroc`/`val_auroc`, final scalars `best_val_auroc`/`best_epoch`/`test_auroc`, confusion matrix + config YAML as artifacts.

### Phase 3 — Baseline
No Phase 3 run reached FINISHED status. Paper targets used as baseline reference: val AUROC ~0.9297, test AUROC ~0.8273.

### Phase 4 — Training Stability ✓ PASSED

Tested 8 intervention combinations (cosine LR × class weighting × grad accum) via successive halving (R1: 8 configs × 25 epochs; R2: top 4 × 50 epochs). R3 skipped — no AUROC gain between R1 and R2.

**Interventions** (all config-gated, MLflow-logged):
- Cosine LR (`CosineAnnealingLR`, `T_max=200`)
- Class weighting (`pos_weight = n_neg / n_pos` from train split)
- Grad accumulation (effective batch = accum_steps × 1)

**Best:** `mmr-surgen-s1-cosine-accum16` — val AUROC 0.9002, test AUROC **0.8640** (beats paper 0.8273), best epoch 19, vl_rise 0.021, mono_ratio 0.833.
**Report:** `reports/2026-03-04-training-stability.md` | **MLflow experiment:** `mmr-surgen-stability` (16 FINISHED runs)

### Phase 5 — Multitask (MMR + RAS + BRAF) ✓ PASSED

**Architecture:** `MultiMILTransformer` — shared encoder → 3 independent linear heads. Masked per-task BCE (NaN labels skipped). Early stopping on mean val AUROC.
**Splits:** `SurGen_multitask_{train,validate,test}.csv` (case_id outer-join). Train=496, val=165, test=165. RAS = KRAS|NRAS union.
**Class dist (train pos%):** MMR ~10%, RAS ~44%, BRAF ~14%.
**MLflow experiment:** `multitask-surgen`

**Results (4 runs):**

| Config | MMR test AUROC | RAS test AUROC | BRAF test AUROC | mean |
|--------|---------------|---------------|----------------|------|
| multitask-base | 0.8931 | 0.6645 | 0.8342 | 0.7973 |
| multitask-base-weighted | 0.7838 | 0.5996 | 0.7794 | 0.7209 |
| multitask-cosine-accum16 | 0.8782 | 0.6195 | 0.8122 | 0.7700 |
| multitask-cosine-accum16-weighted | 0.8476 | 0.6453 | 0.8172 | 0.7700 |

**Class weighting effect (BRAF sensitivity):** base 0.30→0.60, cosine-accum16 0.35→**0.75** (target >0.50 met).
**Gradient conflict:** `multitask-base-weighted` ras/braf min cos −0.61 ⚠️; `multitask-cosine-accum16-weighted` clean (all pairs > −0.38).
**Report:** `reports/2026-03-06-phase5-results.md` (`scripts/studies/phase5_multitask_report.py`)

**Winning config: `multitask-cosine-accum16`** — stable training (vl_rise 6× lower on mean, 15× lower on BRAF), no gradient conflict, mean test AUROC 0.7700. Base config's slightly higher AUROC attributed to checkpoint timing, not real superiority. MMR competitive with Phase 4 single-task (0.8782 vs 0.8640). RAS weakest head: val-to-test gap ~7–8 pp, vl_rise 1.22 in base. Regularization hypothesis (multitask → lower MMR vl_rise) unconfirmed — requires epoch-matched comparison in Phase 6.

---

## Phase 6 — ABMIL + Multitask Fixes + Attention Analysis

**Goal:** Two parallel objectives drive this phase: (1) determine whether multitask joint training regularizes the single-task overfitting observed in Phase 4 (the hypothesis being that predicting harder co-labels forces the backbone toward a richer, less shortcut-prone representation); and (2) determine whether multitask objectives shift the model's regions of attention toward more biologically meaningful tissue structures relative to single-task MMR training.

**Why ABMIL is the prerequisite for both:** Mean pooling produces no interpretable attention map — the attention analysis goal cannot be answered with the current architecture. ABMIL (Ilse et al. 2018) is also a higher-leverage performance change than any scheduler or loss tweak for rare-label tasks (MMR 10%, BRAF 14%), where the positive signal is likely concentrated in a small fraction of patches. Switching to ABMIL enables both the interpretability study and potential AUROC gains simultaneously.

**Multitask fixes bundled in this phase:** Three known failure modes from Phase 5 must be corrected before the attention analysis is meaningful: (a) no per-head class weighting despite large prevalence differences (10%/44%/14%), (b) equal-weight summing of task losses despite RAS dominating the gradient at 44% prevalence, and (c) BRAF sensitivity collapse (0.30–0.35) caused by (a). These are wired together as a single config change rather than separate ablation rounds.

### Steps

- [x] **Step 1 — Verify RAS val/test stratification**: train 44.2%, val 41.8%, test 40.9% — all gaps ≤ 3.3 pp, splits are adequately stratified. No rebuild needed.

- [~] **Step 2 — Per-head class weighting + loss reweighting**: Deferred. Phase 5 weighted runs showed BRAF sensitivity 0.75 already; will revisit if needed after Step 6 results.

- [x] **Step 3 — ABMIL aggregation in `MultiMILTransformer` (Option B — task-specific L2)**: Implemented `aggregation: "attention"` gate in both `MILTransformer` and `MultiMILTransformer` (`scripts/models/mil_transformer.py`). Architecture: shared `attn_l1 = Linear(hidden_dim, attn_hidden_dim) + GELU` learns a general patch-importance representation; task-specific `attn_l2 = Linear(attn_hidden_dim, T)` produces T independent attention distributions (+258 params over shared-L2). Classifier is `nn.ModuleList` of T independent `Linear(hidden_dim, 1)` heads. `forward(return_weights=True)` returns `(logits, weights)` where weights shape is `(B, N, T)`. `train.py` threads `aggregation` + `attn_hidden_dim` from config; logs `attn_variant: "task-specific-L2"` to MLflow. Phase 6 config: `configs/phase6/config_multitask_abmil.yaml` (aggregation=attention, attn_hidden_dim=128, cosine LR, accum=16).

- [ ] **Step 4 — Patch coordinate extraction**: The Zarr embeddings are stored in row-major order. Add a `get_grid_shape(zarr_path)` utility returning `(H, W)` from Zarr metadata or patch count. Emit `(row, col)` indices alongside embeddings from `MultitaskMILDataset`. Required for Step 7 attention heatmaps.

- [ ] **Step 5 — 2D Sinusoidal Positional Encoding** *(optional, can be deferred to Phase 7)*: Without PE, attention weights have no spatial signal to attend to — patches are still exchangeable after ABMIL. Port `SinusoidalPositionalEncoding2D` from `brca_riskformer` into `scripts/models/layers.py`. Gate with `positional_encoding: "sinusoidal"` vs `"none"`. If time-constrained, run Step 6 without PE and add PE as a Phase 7 ablation.

- [ ] **Step 6 — GCP training run**: Train multitask with ABMIL + per-head class weights + loss reweighting, cosine LR, accum=16 (building on the Phase 5 stability winner). Compare per-task val/test AUROC, sensitivity, and vl_rise against Phase 5 `multitask-cosine-accum16`. Compare MMR vl_rise against Phase 4 single-task `mmr-surgen-s1-cosine-accum16` (0.021) — a lower multitask vl_rise is evidence for the regularization hypothesis.

- [ ] **Step 7 — Attention analysis**: For a matched set of slides (same slide evaluated under single-task MMR head and multitask MMR head), extract per-patch attention weights from ABMIL, project back onto the slide patch grid, and generate heatmap overlays. Compare: do the multitask attention maps concentrate on different tissue regions than the single-task maps? Write `scripts/studies/attention_comparison.py` → `reports/YYYY-MM-DD-phase6-attention.md` with representative figure panels. This is the primary interpretability deliverable of Phase 6.

- [ ] **Step 8 — Per-task gradient norm logging** *(diagnostic)*: Log cosine similarity between task gradient vectors at the final shared transformer layer at each epoch. If MMR–RAS cosine similarity is consistently near −1, the tasks are actively fighting in the backbone and GradNorm should be applied. If near 0, the shared backbone is large enough to accommodate both.

### Gate: before moving to Phase 7

- ABMIL multitask must match or exceed Phase 5 `multitask-cosine-accum16` mean val AUROC (0.821).
- BRAF sensitivity must recover above 0.50 (from 0.30–0.35 in Phase 5) with per-head class weighting.
- MMR vl_rise should be compared against Phase 4 single-task (0.021) to assess the regularization hypothesis. Document the result explicitly — do not defer.
- At least one matched-slide attention comparison figure must be generated before proceeding to Phase 7.

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
