# Phase 4 — Baseline Implementation

**Date:** 2026-11-15
**Status:** COMPLETE — M1–M4 done, Phase 5 execution approved (starts with the Kaggle upload)
**Commits:** `81f1ac0` (M1 pairs), `f8a54d6` (M2 baseline), `b564900` (checkpoint-load bug fix), `045fae4` (M3 + docs), `e61d687` (M4 draft), `25dab66` (M4 approvals)

## Scope (fixed by methodology §7.2, §9.1)

Phase 4 implements and proves the pipeline locally on CPU; real training is
Phase 5 on Kaggle (2× T4). Local runs <30 min per the compute policy.

- **Track used:** S2 Sim-ALSAT-2B ×4 — HR = 4-band 10m tile (B2/B3/B4/B8),
  LR = `ALSAT2BDegradation.forward_ms_only` (Wald protocol: Gaussian MTF
  σ=0.35 @10m, area ×4 downsample, Poisson-Gaussian noise SNR 20 dB,
  10-bit quantization; `configs/degradation/alsat2b.yaml`).
- **Deferred (documented):** the S2 ×2 native track (20m B5–B12 → 10m)
  needs the 20m bands, which were not acquired — approval-gated Phase 5
  download item. PROBA-V MISR pairs also deferred: `SRDataset` handles
  single-LR pairs only; a MISR multi-image dataset class is a Phase 5 item.

## M1 — Pair preparation (COMPLETE, verified on disk)

`scripts/prepare_pairs.py`:

- Tile selection: strictly `nan_fraction == 0.0` from the 24 tile-index
  CSVs; pools: **train 4,212 / 7,968 clean** (3,756 excluded), **val
  1,060 / 2,712 clean** (1,652 excluded). NaN pixels never enter the
  degradation pipeline (voids are excluded, not filled).
- Deterministic subsample, `rng(42)`, smoke scale **120 train / 30 val**
  (full-scale pair generation is a Phase 5 Kaggle job).
- Pairs saved FULL-TILE (no cropping — `SRDataset` does its own random
  patch crops at load time).

**Verified results** (`data/interim/pairs/s2_{train,val}/{lr,hr}/*.npy`,
manifest `s2_pairs_manifest.json`):

| Item | Value |
|---|---|
| Pairs | 119 train + 30 val (1 train tile skipped by the finite-check) |
| Shapes | LR [4, 256, 256] float32 / HR [4, 1024, 1024] float32 |
| Finiteness | all sampled pairs finite |
| Degradation sanity | degraded LR vs naive INTER_AREA downsample of HR: mean abs diff **0.0288** — MTF+noise+quantization demonstrably present, LR is not a plain resize |

## M2 — Bicubic baseline (COMPLETE, first real numbers)

`scripts/eval_bicubic_baseline.py`: bicubic ×4 upscale of the degraded LR
(`bicubic_upscale`, clipped to [0,1]) evaluated with the full metric suite
(`evaluate_sr`) against the HR tiles. Output:
`results/metrics/baseline_bicubic.json` (NOTE: `results/metrics/` is
gitignored — the JSON lives on disk only).

**Verified results (30 val pairs, 40.3 s):**

| Metric | mean | note |
|---|---|---|
| PSNR | **37.59 dB** | std 6.97, range 22.0–57.2 (high spread: some val tiles near-uniform) |
| SSIM | **0.899** | |
| SAM | 0.0204 | |
| ERGAS | 0.0311 | |
| wavelet HH energy | ~0.0000 | bicubic injects no high-frequency detail — the TUS baseline behavior |
| GLCM entropy | 6.29 | |

**Honest context:** PSNR is high because the degradation is the exactly
simulated Wald chain (paired, perfectly registered). Real or cross-sensor
numbers will be lower. The JSON stores mean/std/min/max/median per metric.

## M3 — EDSR-tiny CPU smoke train (COMPLETE, verified)

`scripts/train_smoke_edsr_tiny.py` + `configs/models/edsr_tiny.yaml`
(4 resblocks / 32 feats — smoke-only, NOT a thesis model; real EDSR-small
is 16/64 on Kaggle). Proves: SRDataset → SRTrainer (CPU, 2 epochs,
119-pair smoke set) → checkpoint saved → checkpoint reloaded
(`load_model_from_checkpoint`) → `evaluate_on_pairs` on full val tiles.

**Verified results (17.5 s total, well under the 30-min limit):**

| Item | Value |
|---|---|
| Model | EDSR-tiny, 0.160M params |
| Training | 2 epochs / 119 pairs, CPU, 12 s |
| Best val PSNR (48-px patches) | 24.06 dB |
| Roundtrip | checkpoint reloaded → full-tile eval on 3 val pairs: **23.32 dB**, finite |
| Checkpoints | `experiments/checkpoints/edsr_tiny_smoke/{best,epoch_1,epoch_2,last}.pt` |
| Result JSON | `results/metrics/smoke_edsr_tiny.json` |

**Real bug found and fixed by this run:** `load_model_from_checkpoint`
prioritized `checkpoint['config']` (which stores the *training* config,
lacking model keys) over the explicitly passed model config — silently
rebuilding a 16/64 EDSR under `strict=False` and crashing on size
mismatch. Fixed (explicit argument wins) + regression test; inference
suite 10 → 11 tests, all passing (commit `b564900`).

## M4 — Kaggle notebook (DRAFT COMPLETE, execution approved 2026-11-15)

`experiments/phase5/phase5_notebook_draft.py` (cell-by-cell draft) +
`experiments/phase5/kaggle_upload_checklist.md` (concrete upload plan).

**User approval received (2026-11-15):**
1. Upload train+val tiles (~72 GB) to a private Kaggle dataset — approved
2. EDSR-small trains first (GPU 0); GPU 1 idle for now — approved
3. S2 20m bands (B5–B12, ×2 native track) — DEFERRED until ×4 results exist
4. Timing-probe cell is mandatory before committing to 300 epochs
   (honest estimates only after measurement)

Phase 5 execution may start once the user completes the browser upload
(checklist has realistic transfer-time estimates).
