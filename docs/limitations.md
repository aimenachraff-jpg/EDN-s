# Limitations

This document records the limits of what EDN-S is and what its results support. It is
intended to be read alongside the README results section. Nothing here is an apology for
the work; it is the boundary of the claims.

## 1. Data

| Limit | Consequence |
|---|---|
| No real ALSAT-2B imagery was obtained | No result describes real ALSAT-2B scenes. All experiments use Sentinel-2 with simulated degradation. |
| Sentinel-2 is not ALSAT-2B | The sensor geometry, spectral response, noise and radiometry differ. The simulation approximates the setting, not the instrument. |
| The degradation is a model | MTF, noise and spectral-response parameters are documented estimates. They do not reproduce the full ALSAT-2B acquisition chain. |
| Only four 10 m bands are used | The 20 m and 60 m Sentinel-2 bands were not acquired for this track. No conclusion is drawn about them. |
| PROBA-V was acquired but not used | It is present for provenance and future work, not for the reported results. |

## 2. Model and training

| Limit | Consequence |
|---|---|
| The model is EDSR-small | It is a compact baseline, not a new architecture. Larger models may behave differently. |
| Single training run | One seed (42) was used. Run-to-run variance was not measured. |
| Fixed hyperparameters | Batch 16, patch 48, L1 loss, AdamW 1e-4, 300 epochs. No hyperparameter search was performed. |
| 300 epochs / 6.82 h | A longer schedule may improve the result; this was the compute budget available. |
| One GPU | Kaggle 2× T4 was allocated but only GPU 0 was used. |
| No multi-scale training | The model is trained and evaluated at ×4 only. |

## 3. Evaluation

| Limit | Consequence |
|---|---|
| Quality metrics use 30 validation pairs | The sample is small; confidence intervals are correspondingly wide. |
| Per-pair PSNR was not saved | The PSNR significance test uses summary statistics and is a conservative approximation, not an exact paired test. |
| Hallucination detection is reference-based | It requires an aligned HR reference. It cannot certify an image without one. |
| Semantic hallucination was not evaluated | The interface exists, but no downstream task model was trained. |
| Downstream task evaluation is a framework only | No classification, segmentation, detection or change-detection results are claimed. |
| Structural hallucination favours bicubic | 2 tiles vs 0 in the reported run (0.11% vs 0.00%). This is reported, not hidden. |
| No human annotation study | `AnnotationProtocol` is implemented but was not run with experts. |
| Thresholds are fixed | The detector uses protocol thresholds (e.g., SAM > 0.175 rad, 20% GLCM entropy mismatch). Sensitivity to those thresholds was not analysed. |

## 4. Software

| Limit | Consequence |
|---|---|
| Two code trees | `src/edns/` (package) and `src/models`, `src/training`, … (research) overlap and can drift. The packaged tree is the supported API. |
| No trained weights on PyPI | The package ships code only. Inference requires a checkpoint supplied by the user. |
| CPU inference is the tested local path | GPU inference works through PyTorch but was not benchmarked. |
| Large-image performance is not characterised | Tiled inference is implemented; timing across image sizes was not systematically measured. |
| TUS is a heuristic | The task-utility score is not a validated perceptual or task metric. |
| Checkpoint compatibility | Loading uses `weights_only=False`; checkpoints are trusted local files. |

## 5. Scope and academic status

EDN-S is an independently developed implementation associated with a research direction
that was **proposed but not officially confirmed or approved** as a Master’s thesis or
university project. It is not presented as an approved thesis, and no university,
supervisor or agency adoption is implied.

The project is **not affiliated with, endorsed by, or conducted in partnership with** the
Algerian Space Agency (ASAL). ALSAT-2B is referenced only as the application context that
motivated the degradation model.

## 6. What the results do support

Within the simulated setting, the reported experiments support these statements:

- EDSR-small outperforms bicubic ×4 on PSNR, SSIM, SAM and ERGAS over 30 validation pairs.
- The overall reference-based hallucination rate is lower for EDSR-small than for bicubic
  over 1,742 clean test tiles (6.89% vs 13.03%).
- The textural and spectral differences are statistically significant under McNemar’s
  test; the structural difference is not.

They do **not** support claims about real ALSAT-2B performance, semantic hallucination,
downstream task utility, or the superiority of EDSR-small over other SR architectures.
