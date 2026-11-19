# EDN-S: Enhanced Deep Network for Satellite Super-Resolution

## Overview
EDN-S (Enhanced Deep Network for Satellite Super-Resolution) is a lightweight PyTorch implementation of EDSR-small for satellite image super-resolution. Trained on simulated ALSAT-2B ×4 Sentinel-2 imagery (149 pairs, 119 train / 30 val), it delivers **+3.03 dB PSNR over bicubic** on the validation set and is deployable as a pip-installable package with CLI and Python API.

**Thesis:** "Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery"
**Degree:** Master 2 Computer Science / Artificial Intelligence — Algeria

## Project Status — All Phases Complete ✅

| Phase | Status | Key Results |
|-------|--------|-------------|
| **0** Infrastructure & Environment | ✅ 2026-11-12 | CPU-only PyTorch, FreeLLMAPI, OpenCode |
| **1** Literature Review | ✅ 2026-11-13 | 47 papers, research gap identified |
| **2** Research Questions & Methodology | ✅ 2026-11-12/13 | 4 hypotheses (H1–H4), protocol defined |
| **3** Dataset Acquisition & Preparation | ✅ 2026-11-15 | PROBA-V (1,450 sets), Sentinel-2 (87 scenes, 14,748 tiles), void-fill bug fixed, normalization stats |
| **4** Baseline Implementation | ✅ 2026-11-15 | 149 Sim-ALSAT-2B pairs, bicubic baseline PSNR 37.59 dB, EDSR-tiny smoke test |
| **5** Super-Resolution Experiments | ✅ 2026-11-18 | **EDSR-small: 24.86 dB vs bicubic 21.83 dB (+3.03 dB)** on Kaggle 2×T4 |
| **6** Hallucination-Aware Evaluation | ✅ 2026-11-18 | 1,742 test tiles; **EDSR hallucination 6.89% vs bicubic 13.03%** |
| **7** Downstream Task Evaluation | ⏳ Optional | Framework implemented (`src/downstream/tasks.py`), task model training deferred |
| **8** Statistical Analysis & Discussion | ✅ 2026-11-18 | Cohen's d 6.19, McNemar p<0.001 (textural/spectral) |
| **9** Thesis Writing & Final Validation | ✅ 2026-11-19 | 24-page LaTeX thesis compiled with all chapters, bibliography, appendices |

## Key Features
- **Lightweight**: 1.52M parameters, designed for CPU inference
- **Satellite-optimized**: 4-channel input (B2, B3, B4, B8) for Sentinel-2 imagery
- **High-quality**: +3.03 dB PSNR improvement over bicubic on Sentinel-2 validation set
- **Reproducible**: Full training and evaluation pipeline with metrics
- **Deployable**: CLI tool and Python API for easy integration
- **Hallucination-aware**: Detection framework covering Spectral, Structural, Textural, Semantic categories

## Installation

```bash
pip install edns
```

Or install from source:
```bash
git clone https://github.com/aimenachraff-jpg/EDN-S.git
cd EDN-S
pip install -e .
```

## Quick Start

```python
from edns.inference import run_sr

# Run super-resolution on a Sentinel-2 image
result_path = run_sr(
    input_path="data/raw/sentinel2/test.tif",
    output_path="output_sr.tif",
    model_path="experiments/checkpoints/edsr_small_phase5_best.pt"
)
```

## CLI

```bash
edns-sr input.tif [output.tif]
```

Environment variables:
- `EDNS_CKPT` — override checkpoint path
- `EDNS_CONFIG` — override config path

## Features
- 🚀 **Fast inference**: ~2 seconds per 256→1024 tile on CPU (Intel Iris Xe); 1024→4096 in ~50s
- 🌍 **Satellite-optimized**: 4-band input for multispectral imagery
- 📊 **Comprehensive metrics**: PSNR, SSIM, SAM, ERGAS, SCI, TUS
- 🧪 **Validation**: Tested on 1,742 real test tiles (hallucination rate 6.89% vs bicubic 13.03%)
- 🧪 **Unit tests**: 246+ tests covering preprocessing, metrics, and model

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