# Phase 5 — Super-Resolution Experiments (EDSR-small)

**Date:** 2026-11-18
**Status:** COMPLETE — EDSR-small trained on Kaggle 2× T4; full-tile eval +3.03 dB vs bicubic
**Commits:** `2c079be` (docs), `a36fa10` (status table cleanup)

## Scope

Phase 5 trains the first real SR model (EDSR-small) on the full-scale S2
Sim-ALSAT-2B ×4 dataset and validates it against the bicubic baseline.
Training runs on Kaggle (2× T4, GPU 0 only; GPU 1 idle per approved plan).
No local GPU (Intel Iris Xe) — all training/eval in the cloud.

## Dataset on Kaggle

- `aymencharafedn/s2-tiles-trainval` (v18, current): 10,697 files / 78.16 GB
  - 10,680 train+val TIFs (4-band float32, 10m, 1024×1024) across 8 AOIs
  - 16 tile-index CSVs + 1 normalization_stats.json
  - All 16 AOI×split group counts verified against local indexes (verify_kaggle_files.py)
  - Test tiles (4,068) intentionally NOT uploaded — kept local for Phase 6
- `aymencharafedn/master2-a1-code` (80 KB zip): src/, configs/, scripts/prepare_pairs.py

## Pipeline (Kaggle notebook, API-pushed as script kernel)

1. **Timing probe (mandatory):** EDSR-small 1.52M params, 61 ms/step, 0.25 s/tile
   degradation → variant A (precompute LR), fits 12 h session (≈1.3 h for 300 epochs).
2. **Pair precompute (variant A):** 4,154 train + 995 val LR .npy (Wald degradation,
   seed 42; 123 tiles excluded — nan_fraction==0 in index but non-finite on read).
   HR streamed at train time via windowed rasterio reads (192×192 patch region).
3. **Training:** EDSR-small (16 resblocks / 64 feats / scale 4), 300 epochs,
   batch 16, patch 48, L1 loss, fp16 AMP, AdamW lr=1e-4 (cosine), seed 42.
   78,020 steps, 6.82 h. Best val-patch PSNR 25.169 dB (epoch 239).
4. **Evaluation:** full-tile (LR 256→SR 1024) on 30 val pairs, bicubic recomputed
   on the same pairs for fairness.

## Results

| Metric | EDSR-small | Bicubic ×4 |
|---|---|---|
| PSNR (dB) | **24.86** | 21.83 |
| SSIM | 0.771 | 0.629 |
| MSE | 0.00329 | 0.00659 |
| MAE | 0.0370 | 0.0552 |
| ERGAS | 0.0514 | 0.1032 |
| SAM | 0.0203 | 0.0364 |

**VERDICT: EDSR-small beats bicubic by +3.03 dB PSNR.** ✅

## Infrastructure lessons (for reproducibility)

- Kaggle input layout: `/kaggle/input/datasets/<owner>/<slug>` (not `/kaggle/input/<slug>`).
- `glob(..., recursive=True)` is unreliable on Kaggle's virtual mount (saw 204/10,680 tifs);
  `os.walk` sees all of them.
- Dataset attachment in kernel-metadata uses `dataset_sources` (not `dataset_data` — the
  latter is silently dropped).
- PyTorch 2.6 changed `torch.load` default to `weights_only=True`; loading our
  checkpoints requires `weights_only=False` (trusted source).
- Kaggle interactive sessions die ~20 min after the tab closes; headless
  (`kernels push`) is the reliable cloud path for long training.
- Kaggle auto-unzips uploaded zips; very large file counts per dataset are not
  fully visible in the notebook mount unless walked.

## Artifacts

- `experiments/checkpoints/edsr_small_phase5_best.pt` (18.3 MB, epoch 239)
- `results/metrics/phase5_edsr_small_eval.json` (full metric suite, 30 pairs)
- `results/metrics/phase5_train_summary.json` (best_val_psnr, epochs, elapsed)
- Kaggle kernels: `phase-5-edsr-small-x4-training-s2-sim-alsat-2b` (v2, training),
  `phase-5-edsr-small-eval-vs-bicubic` (v2, eval)

## Next

Phase 6: hallucination detection on the 4,068 test tiles (kept local).
