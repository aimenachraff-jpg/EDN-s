# Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery

**Master 2 Thesis — Computer Science / Artificial Intelligence, Algeria**

**Author:** [Student Name]
**Supervisor:** [Supervisor Name]
**Date:** November 2026

---

## Abstract

Super-resolution (SR) methods for satellite imagery are typically evaluated using
fidelity metrics such as PSNR and SSIM, which measure pixel-level similarity
but cannot distinguish genuine information recovery from visually plausible
hallucination. This thesis investigates whether deep-learning SR improves
*actual information usefulness* for ALSAT-2B satellite imagery, or merely
produces visually convincing but hallucinated details that degrade downstream
task performance.

We implement a hallucination-aware evaluation framework comprising four
categories (spectral, structural, textural, semantic) and apply it to EDSR-small
trained on a Sentinel-2 simulated ALSAT-2B ×4 dataset. Our results show that
EDSR-small achieves **24.86 dB PSNR vs 21.83 dB for bicubic (+3.03 dB)** on
full-tile evaluation, while simultaneously reducing the hallucination rate from
**13.03% (bicubic) to 6.89% (EDSR-small)** — a 6.14 percentage-point reduction.
Textural hallucinations, the dominant category, are halved (9.82% → 4.54%).
These findings demonstrate that learned SR can improve both fidelity and
faithfulness, challenging the assumption that higher PSNR implies more
hallucination.

**Keywords:** super-resolution, hallucination detection, satellite imagery,
ALSAT-2B, Sentinel-2, EDSR, spectral fidelity, task-driven evaluation

---

## 1. Introduction

### 1.1 Motivation

ALSAT-2B is an Algerian Earth-observation satellite providing multispectral
imagery at 10 m spatial resolution. For many applications — land-cover
classification, building detection, change detection — this resolution is
insufficient to resolve fine-scale features. Super-resolution (SR) offers a
post-processing path to enhance spatial detail without new hardware.

However, deep-learning SR methods are trained to minimize pixel-wise loss
against high-resolution references. This objective can be satisfied by
generating plausible-looking details that were not present in the original
scene — a phenomenon termed **hallucination**. Hallucinated details may
improve PSNR while degrading the spectral/structural fidelity that downstream
tasks depend on.

### 1.2 Research Questions

- **RQ1:** Do deep-learning SR methods improve fidelity metrics (PSNR, SSIM,
  SAM, ERGAS) over bicubic interpolation on ALSAT-2B imagery?
- **RQ2:** Do SR methods introduce hallucinations (spectral, structural,
  textural, semantic), and at what rate?
- **RQ3:** Does SR reduce or increase hallucination compared to the
  non-learned bicubic baseline?

### 1.3 Contributions

1. A hallucination taxonomy and detection framework with four categories
   (spectral, structural, textural, semantic) and configurable thresholds.
2. A simulated ALSAT-2B ×4 dataset derived from Sentinel-2 (8 AOIs, 4 bands,
   ~10,680 train+val tiles).
3. Empirical evidence that EDSR-small improves fidelity (+3.03 dB PSNR) AND
   reduces hallucination rate (6.89% vs 13.03%) compared to bicubic.
4. An open, reproducible pipeline (code, configs, scripts) for SR evaluation
   on satellite imagery.

### 1.4 Thesis Structure

Chapter 2 reviews related work. Chapter 3 describes the methodology.
Chapter 4 presents baseline implementation. Chapter 5 reports SR experiments.
Chapter 6 presents hallucination-aware evaluation. Chapter 7 discusses results.
Chapter 8 concludes.

---

## 2. Literature Review

### 2.1 Super-Resolution for Remote Sensing

Traditional SR methods (bicubic, Lanczos) are fast but cannot recover
high-frequency detail. Deep learning SR — EDSR (Lim et al., 2017), SwinIR
(Liang et al., 2021), Real-ESRGAN (Wang et al., 2021) — learns a mapping
from LR to HR using convolutional or transformer architectures, achieving
state-of-the-art fidelity on natural images.

For remote sensing, SR has been applied to Sentinel-2, Landsat, and
commercial imagery. However, most evaluations rely solely on PSNR/SSIM,
which do not capture spectral fidelity (critical for vegetation indices like
NDVI) or structural plausibility.

### 2.2 Hallucination in Super-Resolution

Hallucination refers to generated details that are plausible but not grounded
in the input. In satellite imagery, this includes:
- **Spectral:** implausible band correlations, NDVI out of physical range
- **Structural:** spurious high-frequency content in homogeneous regions
- **Textural:** mismatched GLCM entropy / texture statistics
- **Semantic:** task-performance degradation vs HR reference

Existing work (e.g., Blau & Michaeli, 2018; Ledig et al., 2017) discusses
the perception-distortion tradeoff, but systematic hallucination detection for
satellite SR remains underexplored.

### 2.3 Task-Driven Evaluation

Fixed-backbone evaluation (train a task model on HR, freeze, evaluate on SR
outputs) provides a task-level metric normalized to HR performance. This
directly measures whether SR preserves information useful for downstream tasks,
beyond pixel fidelity.

---

## 3. Methodology

### 3.1 Degradation Model (Wald Protocol)

HR = 4-band 10 m Sentinel-2 tile (B2/B3/B4/B8). LR = `ALSAT2BDegradation.forward_ms_only`:

1. Gaussian MTF blur (σ = 0.35 at 10 m)
2. Area ×4 downsample (10 m → 40 m)
3. Poisson-Gaussian noise (SNR 20 dB)
4. 10-bit quantization

Config: `configs/degradation/alsat2b.yaml`. Deterministic with seed 42.

### 3.2 Dataset

| Split | Tiles (all) | Clean (nan==0) | Used |
|---|---|---|---|
| Train | 7,968 | 4,212 | 4,154 (58 excluded for non-finite) |
| Val | 2,712 | 1,060 | 995 (65 excluded) |
| Test | 4,068 | 1,823 | 1,742 (81 skipped) |

- 8 AOIs: BR_01, CL_01, DZ_01–04, FR_01, US_01
- Tile size: 1024×1024 (10 m), 4 bands (B2/B3/B4/B8)
- NaN pixels excluded (never filled); `nan_fraction` in tile-index CSVs
- Normalization: `normalization_stats.json` (per-band mean/std)

### 3.3 SR Model: EDSR-small

| Parameter | Value |
|---|---|
| Architecture | EDSR (Lim et al., 2017) |
| Residual blocks | 16 |
| Features | 64 |
| Scale | ×4 |
| Channels | 4 (B2/B3/B4/B8) |
| Parameters | 1.52M |
| Loss | L1 |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | CosineAnnealing (T_max=300, η_min=1e-6) |
| Batch / Patch | 16 / 48×48 |
| Epochs | 300 |
| AMP | fp16 |

Config: `configs/models/edsr_small.yaml`. Training on Kaggle 2× T4 (GPU 0).

### 3.4 Hallucination Detection Framework

Four categories (Protocol §6):

| Category | Metric | Threshold | Detection |
|---|---|---|---|
| Spectral | Mean SAM, NDVI range, correlation deviation | SAM > 10° (0.175 rad) | SR spectral angle exceeds threshold |
| Structural | Wavelet HH energy in homogeneous regions | > 3σ local mean | Spurious high-freq in smooth regions |
| Textural | GLCM entropy mismatch | > 20% diff | Texture statistics diverge from HR |
| Semantic | Task mAP drop | > 15% vs HR | Downstream task degradation |

Implementation: `src/hallucination/detection.py` (`HallucinationDetector.detect_all`).

### 3.5 Evaluation Protocol

- **Fidelity:** PSNR, SSIM, MSE, MAE, ERGAS, SAM, SCI, wavelet HH energy, GLCM entropy
  (via `src/evaluation/metrics.py::evaluate_sr`)
- **Hallucination:** per-category detection rates + confidence (via `detect_all`)
- **Baseline:** bicubic ×4 recompute on the same pairs for fairness

---

## 4. Baseline Implementation (Phase 4)

- **M1 (Pairs):** `scripts/prepare_pairs.py` — 120 train / 30 val smoke pairs
  (seed 42, nan_fraction==0.0 strictly). HR = 4-band 10m tile, LR = Wald-degraded.
- **M2 (Bicubic baseline):** `scripts/eval_bicubic_baseline.py` — 37.59 dB PSNR
  on 30 val pairs (full metric suite).
- **M3 (Smoke train):** EDSR-tiny (4 resblocks/32 feats), 2 epochs, CPU, 119 pairs.
  Checkpoint saved → reloaded → evaluate_on_pairs roundtrip verified.
- **M4 (Kaggle upload):** single-version upload (33 blobs → v18, 78.16 GB, 10,697 files
  verified). Code dataset `master2-a1-code` (80 KB).

---

## 5. Super-Resolution Experiments (Phase 5)

### 5.1 Training

EDSR-small trained on Kaggle 2× T4 (GPU 0 only):
- 300 epochs, 78,020 steps, 6.82 h
- Best val-patch PSNR: **25.169 dB** (epoch 239)
- Final: val-patch PSNR ~24.95 dB, val SSIM ~0.76, LR ~1e-6
- Checkpoint: `experiments/checkpoints/edsr_small_phase5_best.pt` (18.3 MB)

### 5.2 Evaluation (full-tile, 30 val pairs)

| Metric | EDSR-small | Bicubic ×4 | Δ |
|---|---|---|---|
| PSNR (dB) | **24.86** | 21.83 | +3.03 |
| SSIM | 0.771 | 0.629 | +0.142 |
| MSE | 0.00329 | 0.00659 | −0.00330 |
| MAE | 0.0370 | 0.0552 | −0.0182 |
| ERGAS | 0.0514 | 0.1032 | −0.0518 |
| SAM | 0.0203 | 0.0364 | −0.0161 |

Cohen's d ≈ 6.19 (very large effect). 95% CI: EDSR [24.68, 25.05] vs
bicubic [21.67, 21.99] (non-overlapping). p < 0.001.

### 5.3 Infrastructure Notes

- Kaggle input layout: `/kaggle/input/datasets/<owner>/<slug>` (not `/kaggle/input/<slug>`)
- `glob(..., recursive=True)` unreliable on Kaggle mount (204/10,680 tifs); use `os.walk`
- Dataset attachment in kernel-metadata uses `dataset_sources` (not `dataset_data`)
- PyTorch 2.6: `torch.load` default `weights_only=True` breaks checkpoint loading;
  use `weights_only=False` for trusted sources
- Training run "error" status was only the eval step (torch.load incompatibility);
  training itself completed fully (epoch_300.pt + last.pt saved)

---

## 6. Hallucination-Aware Evaluation (Phase 6)

### 6.1 Method

Applied `HallucinationDetector.detect_all` (spectral/structural/textural) to
EDSR-small and bicubic SR outputs vs HR reference on 1,742 clean test tiles.
Semantic detection skipped (no task model). Resume-safe per-tile JSONL.

### 6.2 Results

| Category | EDSR-small | Bicubic ×4 | McNemar p |
|---|---|---|---|
| Spectral | 2.53% (44) | 3.62% (63) | 3.6e-5 |
| Structural | 0.11% (2) | 0.00% (0) | 0.48 (n.s.) |
| Textural | 4.54% (79) | 9.82% (171) | <0.001 |
| **Any** | **6.89% (120)** | **13.03% (226)** | <0.001 |

- EDSR produces **half** the hallucination rate of bicubic (−6.14 pp)
- Textural hallucinations dominate and show the largest gap (−5.28 pp)
- Spectral hallucinations rare for both (respects band correlations)
- Structural negligible for both

### 6.3 Key Finding

EDSR-small improves **both** fidelity (+3.03 dB PSNR) **and** reduces
hallucinations (6.89% vs 13.03%). This contradicts the common assumption that
higher PSNR implies more hallucination — the learned model generates textures
that are more plausible AND more faithful to the HR reference.

---

## 7. Discussion

### 7.1 Fidelity
- EDSR achieves +3.03 dB PSNR, +0.142 SSIM, halved ERGAS (0.0514 vs 0.1032),
  and lower SAM (0.0203 vs 0.0364) vs bicubic.
- The improvement is consistent across all fidelity metrics, not just PSNR.

### 7.2 Hallucination
- Textural hallucinations dominate for both models (bicubic 9.82%, EDSR 4.54%).
  Bicubic's smoothing/overshoot creates texture mismatches; EDSR's learned
  texture prior reduces these.
- Spectral hallucinations are rare (2.53% vs 3.62%), indicating both methods
  respect band correlations (NDVI in range, correlation matrix stable).
- Structural hallucinations are negligible — wavelet-based detection in
  homogeneous regions finds almost no spurious high-frequency content.

### 7.3 Limitations
- Single SR model (EDSR-small); SwinIR/Real-ESRGAN deferred
- Semantic detection not evaluated (no trained task model)
- H2 (metric validation vs expert annotations) not performed
- ×2 native track (20m bands) deferred (bands not acquired)
- S2 tile index `nan_fraction` metric misses some non-finite pixels (123 tiles excluded)
- 81 of 1,823 clean test tiles skipped (missing or non-finite on read)

### 7.4 Downstream Task Evaluation (Deferred)
- `FixedBackboneEvaluator` implemented in `src/downstream/tasks.py`
- Requires: labeled task data + trained task model on HR
- Documented as future work if task-level impact is needed

---

## 8. Conclusion

This thesis demonstrates that deep-learning super-resolution (EDSR-small) on
ALSAT-2B ×4 satellite imagery improves both fidelity (+3.03 dB PSNR) and
reduces hallucination rates (6.89% vs 13.03%) compared to bicubic
interpolation. The hallucination-aware evaluation framework (spectral,
structural, textural, semantic) provides a more complete picture than
PSNR alone, revealing that:

1. Learned SR generates textures that are more faithful, not just more plausible.
2. Spectral fidelity is preserved (low SAM, rare spectral hallucinations).
3. The perception-distortion tradeoff does not necessarily hold: EDSR achieves
   both lower distortion AND fewer hallucinations.

The detection framework and evaluator are reusable for other SR methods and
datasets. Future work includes evaluating SwinIR/Real-ESRGAN, training task
models for semantic hallucination detection, and validating metrics against
expert annotations (H2).

---

## References

- Lim, B., et al. (2017). Enhanced Deep Residual Networks for Single Image
  Super-Resolution. CVPR.
- Liang, J., et al. (2021). SwinIR: Image Restoration Using Swin Transformer.
  ICCV.
- Wang, X., et al. (2021). Real-ESRGAN: Training Real-World Blind
  Super-Resolution with Pure Synthetic Data. ICCV.
- Blau, Y., & Michaeli, T. (2018). The Perception-Distortion Tradeoff.
  CVPR.
- Wald, L., et al. (1997). Fusion of satellite images of different spatial
  resolutions. ISPRS.
- Ledig, C., et al. (2017). Photo-Realistic Single Image Super-Resolution
  Using a Generative Adversarial Network. CVPR.

---

## Appendix: Artifact Inventory

| Artifact | Path | Size |
|---|---|---|
| EDSR-small checkpoint (best) | `experiments/checkpoints/edsr_small_phase5_best.pt` | 18.3 MB |
| Phase 5 eval metrics | `results/metrics/phase5_edsr_small_eval.json` | 5.4 KB |
| Phase 5 train summary | `results/metrics/phase5_train_summary.json` | 87 B |
| Phase 6 per-tile records | `results/metrics/phase6_per_tile.jsonl` | 1.6 MB |
| Phase 6 aggregated summary | `results/metrics/phase6_hallucination.json` | 1.6 KB |
| Phase 8 statistical analysis | `results/metrics/phase8_statistical_analysis.json` | ~15 KB |
| Bicubic baseline | `results/metrics/baseline_bicubic.json` | 2.5 KB |
| Smoke test results | `results/metrics/smoke_edsr_tiny.json` | 2.4 KB |
| Training run output | Kaggle kernel `phase-5-edsr-small-x4-training-s2-sim-alsat-2b` (v2) | — |
| Eval run output | Kaggle kernel `phase-5-edsr-small-eval-vs-bicubic` (v2) | — |
<|end|> Tile dataset (verified) | Kaggle `s2-tiles-trainval` (v18, 78.16 GB, 10,697 files) | — |
| Code dataset | Kaggle `master2-a1-code` (80 KB) | — |
