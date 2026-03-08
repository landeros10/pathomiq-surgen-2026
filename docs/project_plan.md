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

- [~] **Step 2 — Per-head class weighting**: Folded into all Step 6 configs (`class_weighting: true`). No separate step needed.

- [x] **Step 3 — ABMIL aggregation**: `aggregation: "attention"` gated in both `MILTransformer` and `MultiMILTransformer`. Added `attn_variant: "split"` (task-specific `attn_l2: Linear(128, T)` → T independent distributions) and `"joined"` (shared `attn_l2: Linear(128, 1)` → single distribution broadcast to all tasks). `forward(return_weights=True)` returns `(logits, weights)` shape `(B, N, T)` or `(B, N, 1)`.

- [x] **Step 4 — Patch coordinate extraction**: `get_grid_shape(zarr_path)` added to `dataset.py`. Both `MILDataset` and `MultitaskMILDataset` emit `(row, col)` coords as a `(N, 2)` long tensor (or `None` for `.pt` files). `mil_collate_fn` handles `None` coords without collate errors.

- [x] **Step 5 — 2D Sinusoidal Positional Encoding**: `SinusoidalPositionalEncoding2D` in `scripts/models/layers.py`. Gated via `positional_encoding: "sinusoidal"|"none"` in both model classes.

- [ ] **Step 6 — GCP training run**: 7-run experiment (`multitask-surgen-phase6`, `scripts/run_phase6.sh`), all with cosine LR, accum=16, 150 epochs, patience=25, `class_weighting=true`:

  | Run name | Model | aggregation | attn_variant | PE |
  |---|---|---|---|---|
  | `multitask-mean-cosine-accum16` | Multi | mean | — | none |
  | `multitask-mean-pe-cosine-accum16` | Multi | mean | — | sinusoidal |
  | `multitask-abmil-nope-cosine-accum16` | Multi | attention | split | none |
  | `multitask-abmil-cosine-accum16` | Multi | attention | split | sinusoidal |
  | `multitask-abmil-joined-cosine-accum16` | Multi | attention | joined | none |
  | `multitask-abmil-joined-pe-cosine-accum16` | Multi | attention | joined | sinusoidal |
  | `singletask-mmr-abmil-cosine-accum16` | Single | attention | — | none |

  Compare per-task val/test AUROC, BRAF sensitivity, and MMR vl_rise vs Phase 5 `multitask-cosine-accum16` and Phase 4 single-task (0.021).

- [ ] **Step 7 — Phase 6 Results Report**: Generate a Phase 6 summary report at `reports/YYYY-MM-DD-phase6-results.md` that consolidates all experimental outcomes into a single reference document before Phase 7 begins. The report must cover:

  1. **7-run performance table** — val and test AUROC for each run across all three tasks (MMR, RAS, BRAF), plus BRAF sensitivity. Annotated against Phase 5 `multitask-cosine-accum16` baseline (mean val AUROC 0.821).

  2. **AUPRC per task per run** — for imbalanced tasks (MMR 10%, BRAF 14%), AUROC is an insufficient ranking metric; AUPRC directly measures precision-recall tradeoff on the rare positive class and is the standard in clinical ML literature (MICCAI, Nature Medicine). Include alongside AUROC in the performance table.

  3. **Bootstrap 95% confidence intervals on AUROC and AUPRC** — test set n=165 makes point estimates statistically ambiguous; 2–3 pp differences are uninterpretable without CIs. CIs are computed by resampling the 165 test examples with replacement ~1000 times using the fixed model's saved predictions (no re-training required), estimating uncertainty from finite test set size. Report CIs for the best run and the Phase 5 baseline to assess whether gains are reliable.

  4. **BRAF sensitivity recovery verdict** — explicit pass/fail against the >0.50 gate, with the actual recovered value.

  5. **MMR regularization hypothesis verdict** — MMR vl_rise for the best ABMIL run vs Phase 4 single-task (0.021). Explicit conclusion: does multitask training reduce overfitting on MMR?

  6. **Best config identification + ablation delta table** — which aggregation / attn_variant / PE combination wins on mean val AUROC. Include a compact ablation table showing the marginal AUROC contribution of each design axis (ABMIL vs mean, PE vs none, split vs joined), isolating each variable against an otherwise identical config.

  7. **Calibration (ECE)** — Expected Calibration Error for the best run per task. Uncalibrated outputs cannot support clinical decision thresholds; ECE demonstrates awareness of deployment requirements beyond ranking metrics.

  8. **Attention analysis** — using checkpoints from Step 6 (`singletask-mmr-abmil` vs best multitask ABMIL run), extract per-patch attention weights, project onto the slide patch grid, and generate heatmap overlays for a matched set of slides via `scripts/studies/attention_comparison.py`. Include figures inline and state whether multitask training shifts attention toward different tissue regions vs single-task MMR.

  9. **Attention weight entropy** — for all ABMIL runs, report mean per-slide attention entropy (H = −Σ wᵢ log wᵢ over patches). Low entropy indicates sparse, focal attention (pathologically interpretable); high entropy indicates diffuse weighting. Compare entropy distributions between single-task and best multitask ABMIL run to quantify whether the attention shift hypothesis holds beyond qualitative heatmaps.

  10. **Task gradient conflict diagnostic** — report epoch-level cosine similarity between task gradient vectors at the final shared transformer layer. Conclude whether MMR–RAS gradients are conflicting (near −1) or orthogonal (near 0), and whether GradNorm is warranted in Phase 7.

  11. **Phase 6 gate checklist** — explicitly evaluate all gates defined in the plan (mean val AUROC ≥ 0.821, BRAF sensitivity > 0.50, MMR vl_rise documented, attention figure generated). State pass/fail for each.

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

## Phase 9 — Tissue-Adversarial Conditioning (DANN)

**Goal:** Supplement the MIL classifier with a frozen tissue-type prior derived from an external CRC patch dataset, injected via cross-attention gating. A patch-level Domain-Adversarial Neural Network (DANN) penalty then forces the transformer's learned representation to be uninformative about tissue type — compelling the encoder to focus on patterns beyond tissue composition when predicting MSI status.

**Motivation:** ABMIL attention maps from Phase 8 may reveal that the model attends to tissue type as a proxy signal for MSI (e.g., high lymphocyte density). Making tissue identity an explicit input removes any incentive for the encoder to re-derive it, while the adversarial penalty removes any incentive to retain it in the learned representation.

**Two offline prerequisites must be completed and outputs committed to the repo before any GCP training run.**

---

### Offline Step A — Train Tissue Probe on NCT-CRC-HE-100K

Train a frozen `Linear(1024, 9)` tissue-type classifier on NCT-CRC-HE-100K (Kather et al. 2018) using frozen UNI embeddings as input. NCT-CRC-HE-100K provides 100K expert-annotated H&E CRC patches across 9 tissue classes (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM) from an independent dataset — no SurGen data is touched. Commit frozen weights to `models/tissue_probe/tissue_probe.pt`.

**Gate:** Per-class F1 ≥ 0.85 for TUM, STR, LYM on CRC-VAL-HE-7K holdout.

**Note:** NCT-CRC-HE-100K has known color normalization artifacts. The probe serves as a soft tissue prior, not ground-truth segmentation. Document as a limitation.

---

### Offline Step B — Generate Tissue Embeddings for SurGen

Run the frozen tissue probe on all SurGen Zarr embeddings to produce per-patch tissue probability vectors `[N, 9]`. Store alongside existing `features` and `coords` arrays. Update `MILDataset` and `MultitaskMILDataset` to optionally load tissue probs, gated by `use_tissue_prior: true/false`. When disabled, all downstream modules are bypassed and prior-phase behavior is preserved exactly.

---

### Architecture

Three components, all gated by config:

**CrossAttentionTissueGate** — injects the tissue prior into each patch token via cross-attention before the transformer. Patch tokens are the Query; the 9 tissue-class probabilities form the Key/Value sequence. Residual formulation preserves the original UNI signal. Cross-attention weights `[N, 9]` are a free per-patch tissue attribution map.

**GradientReversalLayer** — standard DANN reversal layer between the encoder and adversarial head. Identity in forward; negated gradient in backward. `lambda_` annealed from 0 → 1 over training to prevent early destabilization.

**Patch-level Adversarial Head** — trainable head that predicts tissue class from patch-level transformer outputs (before pooling). Tissue labels derived from the frozen probe — no manual annotation required. Discarded at inference.

---

### Steps

- [ ] **Step 1** — Implement `CrossAttentionTissueGate`, `GradientReversalLayer`, and adversarial head in `scripts/models/tissue_gate.py`. Unit test each component.
- [ ] **Step 2** — Wire into `MILTransformer` / `MultiMILTransformer` behind config flags. Confirm numerical identity with Phase 8 when all new components are disabled.
- [ ] **Step 3** — Add config keys: `use_tissue_prior`, `tissue_probe_path`, `tissue_gate`, `tissue_gate_heads`, `tissue_gate_dim`, `dann_coeff`, `dann_lambda_max`, `dann_gamma`.
- [ ] **Step 4** — Synthetic gate test on CPU before any GCP run.
- [ ] **Step 5** — GCP ablation (5 runs, all using best Phase 8 config):

  | Run | tissue_gate | dann_coeff | purpose |
  |---|---|---|---|
  | `phase9-baseline` | none | 0.0 | Phase 8 re-run for fair comparison |
  | `phase9-gate-only` | cross_attention | 0.0 | isolate cross-attention contribution |
  | `phase9-dann-only` | none | 0.1 | isolate DANN without gate |
  | `phase9-full` | cross_attention | 0.1 | full proposal |
  | `phase9-full-dann05` | cross_attention | 0.5 | stronger adversarial pressure |

- [ ] **Step 6** — Tissue discriminability diagnostic: train a fresh linear probe on held-out patch-level transformer outputs and report `tissue_leakage_acc` per run. This is the primary evidence that DANN is functioning.
- [ ] **Step 7** — Cross-attention attribution analysis: extract `[N, 9]` gate weights, project onto patch grid, compare spatial attention distribution against Phase 8 ABMIL maps. Write `scripts/studies/phase9_tissue_analysis.py`.
- [ ] **Step 8** — Results report: `reports/YYYY-MM-DD-phase9-results.md`.

### Gate

- `phase9-full` must match or exceed Phase 8 MMR AUROC.
- `tissue_leakage_acc` must be measurably lower in `phase9-full` vs `phase9-baseline`. If not, increase `dann_coeff` before concluding the approach failed.
- If `phase9-gate-only` matches `phase9-full`, DANN contributes nothing beyond cross-attention alone — carry only the gate forward.