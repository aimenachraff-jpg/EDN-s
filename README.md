# EDN-S: Enhanced Deep Network for Satellite Super-Resolution

## Overview
EDN-S (Enhanced Deep Network for Satellite Super-Resolution) is a lightweight PyTorch implementation of EDSR-small for satellite image super-resolution. Trained on simulated ALSAT-2B ×4 Sentinel-2 imagery (149 pairs, 119 train / 30 val), it delivers **+3.03 dB PSNR over bicubic** on the validation set and is deployable as a pip-installable package with CLI and Python API.

**Thesis:** "Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery"
**Degree:** Master 2 Computer Science / Artificial Intelligence — Algeria

---

## Project Status — All Phases Complete ✅

| Phase | Status | Key Results |
|-------|--------|-------------|
| **0** Infrastructure & Environment | ✅ 2026-11-12 | CPU-only PyTorch, FreeLLMAPI, OpenCode |
| **1** Literature Review | ✅ 2026-11-13 | 47 papers, research gap identified |
| **2** Research Questions & Methodology | ✅ 2026-11-12/13 | Protocol finalized, code implemented, 38 tests passing |
| **3** Dataset Acquisition & Preparation | ✅ 2026-11-15 | PROBA-V (1,450 sets), Sentinel-2 (87 scenes, 14,748 tiles), void-fill bug fixed |
| **4** Baseline Implementation | ✅ 2026-11-15 | 149 Sim-ALSAT-2B pairs, bicubic baseline, EDSR-tiny smoke test |
| **5** Super-Resolution Experiments | ✅ 2026-11-18 | **EDSR-small: 24.86 dB vs bicubic 21.83 dB (+3.03 dB)** on Kaggle 2×T4 |
| **6** Hallucination-Aware Evaluation | ✅ 2026-11-18 | 1,742 test tiles; **EDSR hallucination 6.89% vs bicubic 13.03%** |
| **7** Downstream Task Evaluation | ⏳ Optional | Framework implemented (`src/downstream/tasks.py`), training deferred |
| **8** Statistical Analysis & Discussion | ✅ 2026-11-18 | Cohen's d 6.19, McNemar p<0.001 (textural/spectral) |
| **9** Thesis Writing & Final Validation | ✅ 2026-11-19 | 24-page LaTeX thesis compiled with all chapters, bibliography, appendices |

---

## Quick Start

### Install from source
```bash
git clone https://github.com/aimenachraff-jpg/MASTER2-A1.git
cd MASTER2-A1
pip install -e .
```

### Run super-resolution
```python
from edns.inference import run_sr

result_path = run_sr(
    input_path="data/raw/sentinel2/test.tif",
    output_path="output_sr.tif"
)
```

### CLI
```bash
edns-sr input.tif [output.tif]
```

Environment variables: `EDNS_CKPT`, `EDNS_CONFIG`

---

## Phase 2 Deliverables Summary

### Research Questions (P0 — Primary)
| ID | Question | Hypothesis |
|----|----------|------------|
| RQ1 | Degradation model fidelity — can Wald-protocol produce synthetic LR indistinguishable from real ALSAT-2B? | H1 |
| RQ2 | Can RS-SR hallucinations cluster into 4 measurable categories (spectral, structural, textural, semantic)? | H2 |
| RQ3 | Does SCI correlate with task utility (ρ > 0.7) better than PSNR/SSIM/LPIPS? | H3 |
| RQ4 | Which SR family maximizes Task Utility / Hallucination Index ratio at ×4 on Sentinel-2? | H4 |
| RQ5 | Does cross-sensor domain gap follow Diffusion > GAN > Transformer > CNN ordering? | H5 |
| RQ6 | Can tiny models (>90% task utility) run at <10% compute on CPU (<1s/patch)? | H6 |

### Hypotheses — Falsifiable Claims
| ID | Hypothesis | Falsification Condition |
|----|------------|------------------------|
| H1 | Wald degradation with estimated NAOMI params → synthetic LR ≈ real ALSAT-2B LR | FID ≥ 10 OR PSNR gap ≥ 0.5 dB vs real data |
| H2 | RS-SR hallucinations form 4 distinct measurable clusters | Silhouette ≤ 0.3 OR Fleiss' κ ≤ 0.4 on 100+ patches |
| H3 | SCI ρ > 0.7 with task utility; outperforms PSNR/SSIM/LPIPS | SCI ρ ≤ 0.5 on all 3 tasks OR any baseline beats SCI on ≥2 |
| H4 | SwinIR-base has highest Task Utility / Hallucination Index ratio | Not top-1 after bootstrap CI (1000 resamples) |
| H5 | Domain gap severity: Diffusion > GAN > Transformer > CNN (monotonic) | Ordering violated on ≥2/3 target domains |
| H6 | Tiny models (EDSR-small 1.5M, TinySwinIR ~150K) reach >90% large-model utility | Ratio < 0.85 OR CPU time ≥ 1s per patch |

### Implemented Modules (Phase 2 Code)
| Module | File | Tests |
|--------|------|-------|
| Data Pipeline | `src/data/sentinel2_downloader.py`, `probav_loader.py` | 12 tests ✅ |
| Degradation Engine | `src/degradation/alsat2b_degradation.py` | 26 tests ✅ |
| Evaluation Metrics | `src/evaluation/metrics.py` | PSNR, SSIM, LPIPS, SAM, ERGAS, SCI, HF Energy |
| Hallucination Detection | `src/hallucination/detection.py` | Spectral, Structural, Textural, Semantic + AnnotationProtocol |
| Downstream Tasks | `src/downstream/tasks.py` | UNet classification, YOLO detection, DeepLab segmentation, Siamese change-detect |
| Models | `src/models/edsr.py`, `swinir.py`, `realesrgan.py` | CPU smoke tests ✅ |
| Training | `src/training/trainer.py` | MLflow tracking |
| Inference | `src/inference/inference.py` | Tiled processing |

### Config Files
`configs/data/sentinel2.yaml`, `configs/data/probav.yaml`, `configs/degradation/alsat2b.yaml`, `configs/models/*.yaml`, `configs/training/trainer_config.yaml`

### Key Design Decisions (from `docs/methodology_decisions.md`)
- RCAN removed (redundant with EDSR); diffusion conditional on compute
- Full Wald protocol adopted (SRF → MTF → noise → quantization → shift → downsample)
- MTF σ, SNR labeled **ESTIMATED** (not documented — conservative bounds applied)
- Fixed-backbone evaluation protocol (no fine-tuning on SR outputs)
- Minimum 2 random seeds (42, 123) for statistical robustness
- Quantile normalization on train split only; NaN excluded; percentile 2–98 + mean/std

---

## Phase 3 — Dataset Acquisition (Quick Reference)

Sentinel-2 and PROBA-V datasets acquired and verified:

```
PROBA-V MISR       ✅  74 regions, 1,450 imgsets (1,160 train / 290 test)
                    ✅  MD5: 7101101c7c91eeb5979d472861c1b784
Sentinel-2         ✅  87/88 scenes downloaded (31SDB_20201223 rejected, degenerate granule)
                    ✅  8 AOIs across Algeria + global coverage
                    ✅  14,748 tiles generated (7,968 train / 2,712 val / 4,068 test)
                    ✅  1024×1024, overlap 128px, float32, ~100 GB
                    ⚠️  Void-fill bug found & fixed (NaN silent exact-0 fill in rotated granules)
Normalization stats computed on train split only (NaN-excluded, percentile 2–98):
                    ✅  data/metadata/sentinel2/normalization_stats.json
Split verification   ✅  Hash-verified train/val/test manifests (zero leakage)
```

### Accessing the Data
```python
from src.data.sentinel2_downloader import Sentinel2Downloader
from src.data.probav_loader import PROBAVLoader

# Download Sentinel-2 tiles
downloader = Sentinel2Downloader()
tiles = downloader.download(aoi="DZ_01", year_range=(2018, 2024))

# Load PROBA-V MISR samples
loader = PROBAVLoader()
samples = loader.load(region_id="CL_01")
```

---

## Key Features
- **Lightweight**: 1.52M parameters, designed for CPU inference
- **Satellite-optimized**: 4-channel input (B2, B3, B4, B8) for Sentinel-2 imagery
- **High-quality**: +3.03 dB PSNR improvement over bicubic on Sentinel-2 validation set
- **Reproducible**: Full training and evaluation pipeline with metrics
- **Deployable**: CLI tool and Python API for easy integration
- **Hallucination-aware**: Detection framework covering Spectral, Structural, Textural, Semantic categories

## Model Architecture
- **Base**: EDSR-small (16 residual blocks, 64 features)
- **Scale**: ×4 super-resolution
- **Input**: 4 channels (B2, B3, B4, B8) — Sentinel-2 surface reflectance
- **Output**: 4 channels at 4× higher resolution

## Verified Results Summary

| Metric | EDSR-small | Bicubic | Δ |
|--------|-----------|---------|---|
| PSNR (dB) | **24.86** | 21.83 | **+3.03** |
| SSIM | 0.72 | 0.61 | +0.11 |
| SAM (rad) | 0.058 | 0.082 | -0.024 |
| ERGAS | 2.14 | 4.31 | -2.17 |
| Hallucination rate | **6.89%** | 13.03% | **-6.14%** |

Statistical significance: Cohen's d = 6.19 (very large), McNemar p < 0.001 for textural and spectral hallucination categories.

## License
MIT License — see [LICENSE](LICENSE) file

## Acknowledgments
This work was supported by the MASTER2-A1 project. Special thanks to the Kaggle community for dataset access and the OpenCode framework for enabling this development.

## Contact
For questions or collaboration, contact: aimenachraff@example.com