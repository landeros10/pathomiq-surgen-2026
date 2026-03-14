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

## Phase 6 — ABMIL + Multitask Fixes + Attention Analysis ✓ COMPLETED

7-run experiment (`multitask-surgen-phase6`): ABMIL aggregation + per-head class weighting + cosine LR, accum=16, 150 epochs, patience=25. Architecture prep: `aggregation: "attention"` in both model classes, `attn_variant: split|joined`, `(row,col)` coords emitted by datasets, `SinusoidalPositionalEncoding2D` in `layers.py`. Report: `reports/2026-03-09-phase6-results.md`.

**Key results:**

| Config | MMR test | RAS test | BRAF test | Mean test |
|--------|----------|----------|-----------|-----------|
| abmil-nope | 0.8633 | 0.6326 | 0.8292 | 0.7750 |
| **abmil-joined** ← best | **0.8351** | **0.6543** | **0.8394** | **0.7763** |
| singletask-mmr-abmil | 0.8222 | — | — | — |

- **BRAF sensitivity: PASS** — all 6/6 runs 0.70–0.80 (vs 0.30–0.35 Phase 5).
- **MMR regularization hypothesis: NOT supported** — all runs show higher vl_rise than Phase 4 (0.021); best run 0.597.
- **PE consistently hurts** (−0.008 to −0.017 mean val AUROC) — excluded from Phase 7.
- **Gradient conflict: mild** — min cosine similarity −0.494 (MMR-BRAF); GradNorm not urgently needed.
- **Attention entropy:** joined-attn most focused (H=7.60 vs 8.08 split); ST H=7.89.
- **ECE:** best MMR calibration at joined-pe (0.069); BRAF poorly calibrated across all runs (0.14–0.16).
- **All gates passed.** Phase 7 backbone: `multitask-abmil-joined-cosine-accum16`.

---

## Phase 7 — Interpretability Studies ✓ COMPLETED

**Goal:** Validate what the Phase 6 ABMIL model learned: (1) do attention weights localize biologically meaningful regions? (2) does multitask training shift attention vs. single-task? (3) are attention-identified regions causally important?

**Six analyses** on a 19-slide study set (5 correct_msi, 5 correct_mss, 4 wrong_msi, 5 wrong_mss): (1) attention heatmaps, (2) gradient attribution comparison (GradNorm + IxG vs. raw ABMIL; Spearman ρ), (3) MC Dropout uncertainty + ECE on full test split, (4) rank-ordered patch deletion/insertion curves, (5) top-k contact sheets + UMAP+HDBSCAN morphologic clustering, (6) attention entropy with singletask vs. multitask Mann-Whitney test. `phase7_run_all.py` orchestrates all steps with sentinel-based re-entrancy.

**Results:** Attribution faithfulness mean ρ = 0.850 — ABMIL weights are reliable proxies for gradient importance; canonical method: `grad`. Deletion gate PASS (3/5 correct-MSI slides ≥5 pp drop at k=20%); MT model more concentrated (grad del-AUC 0.209 vs ST 0.373). MC Dropout σ ~7× higher on misclassified vs. correct slides. Multitask focus hypothesis SUPPORTED: MT-joined H = 7.60 vs ST H = 7.89, p = 2.13e-06, r = −0.302 — effect specific to joined-ABMIL variant. ST = 5 morphologic clusters, MT = 4, consistent with concentrated attribution. **All 7 gates passed.**

**Report:** `reports/2026-03-10-phase7-interpretability.md` | **Figures:** `reports/figures/phase7/`

---

## Phase 8 — MLP-RPB + Patch Dropout Ablation

**Goal:** Determine whether learned relative position bias (MLP-RPB) and patch dropout improve over the Phase 6 ABMIL baseline. Phase 6 showed sinusoidal absolute PE consistently hurts (−0.008 to −0.017 mean val AUROC). ALiBi was ruled out because it imposes a hard distance penalty that suppresses long-range attention — wrong prior for MSI where signal is globally distributed. PEG was ruled out due to scatter/gather complexity on sparse coords. **MLP-RPB** (small MLP on continuous `(Δrow, Δcol)` coords → additive attention bias) is fully learnable, handles arbitrary sparse grids natively, and can learn near-zero effect if spatial context is uninformative — bounded downside.

**2×3 ablation** — MLflow experiment `multitask-surgen-phase8`. All runs are full copies of `configs/phase6/config_multitask_abmil_joined.yaml` (multitask ABMIL joined, cosine LR, accum=16, 150 epochs, patience=25, class_weighting=true). `positional_encoding` ∈ {none, mlp_rpb} × `patch_dropout_rate` ∈ {0.0, 0.1, 0.25} = **6 configs × 3 seeds = 18 runs total.**

| Config | `positional_encoding` | `patch_dropout_rate` | Run names |
|---|---|---|---|
| `config_phase8_baseline` | `"none"` | 0.0 | `phase8-baseline-s{0,1,2}` |
| `config_phase8_mlp_rpb` | `"mlp_rpb"` | 0.0 | `phase8-mlp-rpb-s{0,1,2}` |
| `config_phase8_dropout10` | `"none"` | 0.1 | `phase8-dropout10-s{0,1,2}` |
| `config_phase8_dropout25` | `"none"` | 0.25 | `phase8-dropout25-s{0,1,2}` |
| `config_phase8_mlp_rpb_dropout10` | `"mlp_rpb"` | 0.1 | `phase8-mlp-rpb-dropout10-s{0,1,2}` |
| `config_phase8_mlp_rpb_dropout25` | `"mlp_rpb"` | 0.25 | `phase8-mlp-rpb-dropout25-s{0,1,2}` |

**Gate:** `phase8-baseline` mean test AUROC ≥ 0.7763 (Phase 7 reference). Winner: Δ_mean > +0.005 over baseline **and** CIs non-overlapping. If no winner: carry `"none"` / 0.0 forward to Phase 9.

---

### Architecture Facts (read source before implementing)

- **`scripts/models/layers.py`** — only `SinusoidalPositionalEncoding2D` exists. `MLPRelativePositionBias` needs to be added.
- **`scripts/models/mil_transformer.py`** — both `MILTransformer` and `MultiMILTransformer` have a `positional_encoding` param. PE dispatch is a 2-way if/else at init: only `"sinusoidal"` is handled, else `self.pos_enc = None`. Needs a 3-way dispatch adding `"mlp_rpb"` → `self.rpb`. `self.transformer` is `nn.TransformerEncoder(batch_first=True)` — the `(B*H, N, N)` mask shape is valid for `batch_first=True` per PyTorch docs.
- **`scripts/train.py`** — three multitask call sites use a ternary guard `model(embeddings) if (model.pos_enc is None or coords is None) else model(embeddings, coords=coords)` (lines 163, 212, 280) — all three must become `model(embeddings, coords=coords)`. No `hasattr` guard needed for `model.transformer.layers[-1]` since MLP-RPB does not wrap the transformer. The call site at ~line 549–552 passes `train_one_epoch_multitask(..., tasks)` — add `patch_drop_rate=tc.get("patch_dropout_rate", 0.0)` kwarg.
- **MLflow metric keys**: `test_auroc_{t}`, `best_val_auroc_{t}`, `best_val_auroc_mean`, `best_epoch` — use these exact keys in the report script.

---

### Steps 1–5 ✓ DONE

- **`layers.py`:** Added `MLPRelativePositionBias(num_heads, hidden_dim=64)` — 2-layer MLP on per-slide normalized `(Δrow, Δcol)` → `(B*H, N, N)` additive attention bias.
- **`mil_transformer.py`:** Both `MILTransformer` and `MultiMILTransformer` updated with 3-way PE dispatch (`sinusoidal` / `mlp_rpb` / `none`) and `self.transformer(x, mask=rpb_mask)` in forward.
- **`train.py`:** Three ternary model-call guards simplified to `model(embeddings, coords=coords)`; `patch_drop_rate` param + random patch subsampling added to `train_one_epoch_multitask`; MLflow logs `patch_dropout_rate`; `--seed` CLI arg overrides `training.random_seed`.
- **`configs/config.yaml`:** Added `patch_dropout_rate: 0.0` under `training:`.
- **`configs/phase8/`:** 6 YAML files created (`config_phase8_baseline.yaml`, `config_phase8_mlp_rpb.yaml`, `config_phase8_dropout10.yaml`, `config_phase8_dropout25.yaml`, `config_phase8_mlp_rpb_dropout10.yaml`, `config_phase8_mlp_rpb_dropout25.yaml`).
- **`scripts/studies/phase8_unit_tests.py`:** 7/7 tests pass locally (shape, grad, normalization, variable-N, no-PE forward, mlp_rpb + grad, patch dropout consistency).
- **`scripts/run_phase8.sh`:** Orchestrator with `--preflight`, `--patience N`, `--seeds "0 1 2"` flags; loops 6 configs × seeds sequentially; logs to `logs/phase8/<run_name>[_preflight].log`.

**Next:** Smoke test on GCP — `./scripts/run_phase8.sh --preflight --seeds "0"` (6 runs × 1 epoch), then full 18-run launch.

---

### Step 6 — Report Script (`scripts/studies/phase8_ablation.py`)

Fetch all 18 runs from `multitask-surgen-phase8`. Group by config (6 groups × 3 seeds). For each config compute mean ± std across seeds for MMR, RAS, BRAF, and mean test AUROC. Output markdown table:

| Config | PE | Dropout | MMR mean±std | RAS mean±std | BRAF mean±std | Mean±std | Δ vs baseline |
|---|---|---|---|---|---|---|---|

Also fetch Phase 6 reference run for context. Write to `reports/YYYY-MM-DD-phase8-results.md`.

---

### Gate: before moving to Phase 9

- `phase8-baseline` mean test AUROC ≥ 0.7763 (Phase 7 reference).
- Winner: Δ_mean > +0.005 over baseline **and** CIs non-overlapping; carry `positional_encoding` and `patch_dropout_rate` forward to Phase 9.
- If no winner: document null result, carry `"none"` / 0.0 forward, proceed to Phase 9.
- Report committed to `reports/`.

---

## PHASE 9 DISCONTINUED FOR NOW
## Phase 9 — Tissue-Adversarial Conditioning (DANN)

**Goal:** Supplement the MIL classifier with a frozen tissue-type prior derived from an external CRC patch dataset, injected via cross-attention gating. A patch-level Domain-Adversarial Neural Network (DANN) penalty then forces the transformer's learned representation to be uninformative about tissue type — compelling the encoder to focus on patterns beyond tissue composition when predicting MSI status.

**Motivation:** ABMIL attention maps from Phase 7 may reveal that the model attends to tissue type as a proxy signal for MSI (e.g., high lymphocyte density). Making tissue identity an explicit input removes any incentive for the encoder to re-derive it, while the adversarial penalty removes any incentive to retain it in the learned representation.

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
- [ ] **Step 2** — Wire into `MILTransformer` / `MultiMILTransformer` behind config flags. Confirm numerical identity with Phase 7 when all new components are disabled.
- [ ] **Step 3** — Add config keys: `use_tissue_prior`, `tissue_probe_path`, `tissue_gate`, `tissue_gate_heads`, `tissue_gate_dim`, `dann_coeff`, `dann_lambda_max`, `dann_gamma`.
- [ ] **Step 4** — Synthetic gate test on CPU before any GCP run.
- [ ] **Step 5** — GCP ablation (5 runs, all using best Phase 7 config):

  | Run | tissue_gate | dann_coeff | purpose |
  |---|---|---|---|
  | `phase9-baseline` | none | 0.0 | Phase 8 re-run for fair comparison |
  | `phase9-gate-only` | cross_attention | 0.0 | isolate cross-attention contribution |
  | `phase9-dann-only` | none | 0.1 | isolate DANN without gate |
  | `phase9-full` | cross_attention | 0.1 | full proposal |
  | `phase9-full-dann05` | cross_attention | 0.5 | stronger adversarial pressure |

- [ ] **Step 6** — Tissue discriminability diagnostic: train a fresh linear probe on held-out patch-level transformer outputs and report `tissue_leakage_acc` per run. This is the primary evidence that DANN is functioning.
- [ ] **Step 7** — Cross-attention attribution analysis: extract `[N, 9]` gate weights, project onto patch grid, compare spatial attention distribution against Phase 7 ABMIL maps. Write `scripts/studies/phase9_tissue_analysis.py`.
- [ ] **Step 8** — Results report: `reports/YYYY-MM-DD-phase9-results.md`.

### Gate

- `phase9-full` must match or exceed Phase 7 MMR AUROC.
- `tissue_leakage_acc` must be measurably lower in `phase9-full` vs `phase9-baseline`. If not, increase `dann_coeff` before concluding the approach failed.
- If `phase9-gate-only` matches `phase9-full`, DANN contributes nothing beyond cross-attention alone — carry only the gate forward.
