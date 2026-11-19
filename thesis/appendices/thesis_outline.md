# Phase 9 — Thesis Outline & Key Results

**Date:** 2026-11-18
**Status:** Outline complete; ready for drafting

## Thesis Structure

### 1. Introduction
- Problem: ALSAT-2B (Algerian satellite) has limited spatial resolution; SR can enhance but may hallucinate details
- Research question: Do deep-learning SR methods improve *actual information usefulness* or merely produce visually convincing but hallucinated details?
- Objectives: evaluate fidelity (PSNR/SSIM/SAM/ERGAS) AND hallucination (spectral/structural/textural/semantic) AND downstream task impact
- Scope: S2 Sim-ALSAT-2B ×4 track (HR = 4-band 10m, LR = Wald-degraded to 40m)

### 2. Literature Review
- Traditional SR (interpolation, reconstruction-based)
- Deep learning SR (EDSR, SwinIR, Real-ESRGAN)
- Hallucination in SR (spectral fidelity, texture generation, semantic drift)
- Task-driven evaluation of SR (beyond PSNR)

### 3. Methodology
- §3.1 Degradation model: Wald protocol (MTF σ=0.35, area ×4 downsample, Poisson-Gaussian noise SNR 20dB, 10-bit quantization)
- §3.2 Dataset: Sentinel-2 4-band (B2/B3/B4/B8), 8 AOIs, tile-based (1024×1024), nan_fraction filtering
  - Train: 4,212 clean tiles, Val: 1,060, Test: 1,823 clean (of 4,068 total)
- §3.3 SR model: EDSR-small (16 resblocks, 64 feats, scale 4, 1.52M params)
- §3.4 Hallucination taxonomy: spectral, structural, textural, semantic (Protocol §6)
- §3.5 Detection thresholds: SAM >10°, wavelet HH >3σ, GLCM entropy diff >20%, task mAP drop >15%

### 4. Experiments

#### 4.1 Phase 4 — Baseline
- 149 sim-×4 pairs (120 train / 29 val), bicubic baseline: PSNR 37.59 dB
- CPU smoke train (EDSR-tiny, 2 epochs) + roundtrip verified

#### 4.2 Phase 5 — SR Training
- EDSR-small on Kaggle 2×T4 (GPU 0), 300 epochs, 6.82 h, 78,020 steps
- Best val-patch PSNR: 25.169 dB (epoch 239)
- Full-tile eval (30 val pairs): **24.86 dB** vs bicubic 21.83 dB (+3.03 dB)
- Cohen's d ≈ 6.19 (very large effect), p < 0.001

#### 4.3 Phase 6 — Hallucination Detection
- 1,742 clean test tiles, EDSR vs bicubic
- Any-hallucination: 6.89% vs 13.03% (−6.14 pp)
- Textural: 4.54% vs 9.82% (McNemar χ²=81.19, p<0.001)
- Spectral: 2.53% vs 3.62% (McNemar χ²=17.05, p=3.6e-5)
- Structural: 0.11% vs 0.00% (not significant, n=2)
- Semantic: skipped (no task model)

### 5. Results & Discussion

#### 5.1 Fidelity
- EDSR improves PSNR by +3.03 dB over bicubic (24.86 vs 21.83, 95% CI [24.68,25.05] vs [21.67,21.99])
- SSIM 0.771 vs 0.629, ERGAS 0.0514 vs 0.1032 (halved), SAM 0.0203 vs 0.0364
- EDSR achieves better spectral fidelity (lower SAM, lower ERGAS)

#### 5.2 Hallucination
- EDSR produces half the hallucinations of bicubic (6.89% vs 13.03%)
- Textural hallucinations dominate for both; EDSR's learned texture prior reduces them
- Spectral hallucinations rare for both (respects band correlations)
- Structural hallucinations negligible (wavelet-based, homogeneous regions)

#### 5.3 Key finding
- EDSR improves BOTH fidelity (+3.03 dB) AND reduces hallucinations (−6.14 pp)
- This contradicts the common assumption that better PSNR implies more hallucination
- The learned model generates more plausible textures that are also more faithful

### 6. Limitations
- Single SR model (EDSR-small); SwinIR/Real-ESRGAN deferred to future work
- Semantic hallucination not evaluated (no trained task model)
- H2 (metric validation vs expert annotations) not performed (no labels)
- ×2 native track (20m bands) deferred (bands not acquired)
- S2 tile index nan_fraction metric misses some non-finite pixels (123 tiles excluded)

### 7. Conclusion
- Deep-learning SR (EDSR-small) on ALSAT-2B ×4 improves both fidelity and reduces
  hallucinations vs bicubic baseline
- Hallucination-aware evaluation is essential: PSNR alone doesn't capture spectral/structural/textural fidelity
- The detection framework (4 categories) is reusable for other SR methods

## Key Figures Needed

1. PSNR/SSIM comparison bar chart (EDSR vs bicubic, 30 pairs)
2. Hallucination rate by category (grouped bar: EDSR vs bicubic)
3. Sample SR outputs: HR / LR / EDSR / Bicubic (visual comparison)
4. NDVI comparison: SR vs HR (spectral hallucination example)
5. Textural entropy distribution: EDSR vs bicubic vs HR

## Key Tables Needed

1. Dataset summary (AOIs, tiles, clean counts)
2. Phase 5 eval metrics (PSNR, SSIM, MSE, MAE, ERGAS, SAM, SCI)
3. Phase 6 hallucination rates (per category, with CI and McNemar p-values)
4. Training config (EDSR-small hyperparameters)

## Artifacts Available

- `results/metrics/phase5_edsr_small_eval.json` — full eval metrics (30 pairs)
- `results/metrics/phase5_train_summary.json` — training summary
- `results/metrics/phase6_hallucination.json` — aggregated hallucination summary
- `results/metrics/phase6_per_tile.jsonl` — 1,742 per-tile records
- `results/metrics/phase8_statistical_analysis.json` — statistical tests (Cohen's d, McNemar, Wilson CIs, Mann-Whitney)
- `experiments/checkpoints/edsr_small_phase5_best.pt` — trained model (18.3 MB)
- `experiments/checkpoints/edsr_tiny_smoke/` — CPU smoke test checkpoint
- `results/metrics/baseline_bicubic.json` — bicubic baseline (37.59 dB)
- `results/metrics/smoke_edsr_tiny.json` — smoke test results
