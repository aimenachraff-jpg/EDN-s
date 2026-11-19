# EDN-S: Enhanced Deep Network for Satellite Super-Resolution

[![PyPI version](https://badge.fury.io/py/edns.svg)](https://pypi.org/project/edns/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/aimenachraff-jpg/EDN-s?style=social)](https://github.com/aimenachraff-jpg/EDN-s)

## Overview
EDN-S (Enhanced Deep Network for Satellite Super-Resolution) is a lightweight PyTorch implementation of EDSR-small for satellite image super-resolution, deployable as a pip-installable package with CLI and Python API.

**Author:** Chekalil Aimene Charef Eddine  
**Project type:** Independent research startup (self-directed, completed without institutional affiliation)  
**Thesis:** "Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery"  
**Institution:** Master 2 Computer Science / AI — Algeria  
**Related:** ASAL (Algerian Space Agency) — ALSAT-2B satellite imagery research

---

## All Phases Complete ✅

| # | Phase | Status | Date | Result |
|---|-------|--------|------|--------|
| 0 | Infrastructure & Environment | ✅ | 2026-11-12 | CPU-only PyTorch, FreeLLMAPI |
| 1 | Literature Review | ✅ | 2026-11-13 | 47 papers, research gap identified |
| 2 | Research Questions & Methodology | ✅ | 2026-11-12/13 | 6 RQs (H1–H6), protocol finalized |
| 3 | Dataset Acquisition & Preparation | ✅ | 2026-11-15 | PROBA-V (1,450 imgsets), Sentinel-2 (87 scenes, 14,748 tiles) |
| 4 | Baseline Implementation | ✅ | 2026-11-15 | 149 Sim-ALSAT-2B pairs, bicubic baseline, EDSR-tiny smoke test |
| 5 | Super-Resolution Experiments | ✅ | 2026-11-18 | **EDSR-small: 24.86 dB vs bicubic 21.83 dB (+3.03 dB)** on Kaggle 2×T4 |
| 6 | Hallucination-Aware Evaluation | ✅ | 2026-11-18 | 1,742 test tiles; **EDSR hallucination 6.89% vs bicubic 13.03%** |
| 7 | Downstream Task Evaluation | ⏳ Optional | — | Framework implemented (`src/downstream/tasks.py`), deferred |
| 8 | Statistical Analysis & Discussion | ✅ | 2026-11-18 | Cohen's d 6.19, McNemar p<0.001 (textural/spectral) |
| 9 | Thesis Writing & Final Validation | ✅ | 2026-11-19 | 24-page LaTeX thesis compiled |

---

## Installation

```bash
pip install edns
```

Or from source:
```bash
git clone https://github.com/aimenachraff-jpg/EDN-s.git
cd EDN-s
pip install -e .
```

---

## Quick Start

### Python API
```python
from edns.inference import run_sr

# Basic super-resolution
result = run_sr(
    input_path="data/raw/sentinel2/test.tif",
    output_path="output_sr.tif"
)

# With hallucination detection (requires HR reference)
result = run_sr(
    input_path="input.tif",
    output_path="output_sr.tif",
    compute_hallucination=True,
    hr_reference_path="reference_hr.tif",
    hallucination_report_path="hallucination_report.json"
)
print(result["hallucination_report"]["summary"])
```

### CLI
```bash
# Basic super-resolution
edns-sr input.tif [output.tif]

# With hallucination detection
edns-sr input.tif --hallucination --hr reference_hr.tif

# With hallucination report saved to JSON
edns-sr input.tif -o output.tif --hallucination --hr reference.tif --report report.json

# Verbose output
edns-sr input.tif -v --hallucination --hr reference.tif
```

Environment variables: `EDNS_CKPT`, `EDNS_CONFIG`

---

## Key Features
- 🚀 **Lightweight**: 1.52M parameters, fast on CPU (~50s for 1024→4096 tile on Intel Iris Xe)
- 🌍 **Satellite-optimized**: 4-channel input (B2, B3, B4, B8) for Sentinel-2 imagery
- 📊 **Comprehensive metrics**: PSNR, SSIM, SAM, ERGAS, SCI, TUS
- 🧪 **Hallucination detection**: Spectral, Structural, Textural, Semantic categories
- 🛠️ **Deployable**: CLI + Python API, pip-installable package (`edns`)

---

## Hallucination Detection

The package includes a built-in hallucination detector that evaluates super-resolved outputs against a ground-truth HR reference across four categories:

| Category | What it detects |
|----------|-----------------|
| **Spectral** | Unnatural spectral signatures, band correlation violations |
| **Structural** | Geometric distortions, edge artifacts, ringing |
| **Textural** | Invented textures, GLCM/entropy anomalies |
| **Semantic** | Downstream task performance degradation (requires task model) |

### CLI Output Example
```bash
$ edns-sr input.tif --hallucination --hr reference.tif
Processing input.tif -> input_sr.tif
Hallucination detection: ON (HR reference: reference.tif)
Done in 52.34s. Saved to input_sr.tif

=== Hallucination Detection Report ===
  Spectral     : ✓  Clean (confidence: 0.0123)
  Structural   : ✓  Clean (confidence: 0.0001)
  Textural     : ⚠️  DETECTED (confidence: 0.0456)
  Semantic     : —  Skipped (no task model)
  Any          : ⚠️  YES
======================================
```

### Python API Report Structure
```python
result = run_sr(
    input_path="input.tif",
    output_path="output.tif",
    compute_hallucination=True,
    hr_reference_path="reference.tif",
    hallucination_report_path="report.json"
)

# result["hallucination_report"] contains:
{
    "input_path": "input.tif",
    "output_path": "output.tif",
    "hr_reference_path": "reference.tif",
    "hallucination": {
        "spectral": {"detected": False, "confidence": 0.0123, ...},
        "structural": {"detected": False, "confidence": 0.0001, ...},
        "textural": {"detected": True, "confidence": 0.0456, ...},
        "semantic": {"detected": False, "confidence": 0.0, "skipped": True},
        "any_hallucination": True
    },
    "summary": {
        "any_hallucination": True,
        "detected_categories": ["textural"],
        "max_confidence": 0.0456
    }
}
```

## Verified Results

| Metric | EDSR-small | Bicubic | Δ |
|--------|-----------|---------|---|
| PSNR (dB) | **24.86** | 21.83 | **+3.03** |
| SSIM | 0.72 | 0.61 | +0.11 |
| SAM (rad) | 0.058 | 0.082 | −0.024 |
| ERGAS | 2.14 | 4.31 | −2.17 |
| Hallucination rate | **6.89%** | 13.03% | **−6.14%** |

Statistical significance: Cohen's d = 6.19 (very large), McNemar p < 0.001 for textural and spectral hallucination categories.

---

## Model Architecture
- **Base**: EDSR-small (16 residual blocks, 64 features)
- **Scale**: ×4 super-resolution
- **Input**: 4 channels (B2, B3, B4, B8) — Sentinel-2 surface reflectance
- **Output**: 4 channels at 4× higher resolution

---

## Package Structure (`edns`)
```
src/edns/
├── models/          EDSR implementation
├── inference/       SRInference class + CLI entry point (edns-sr)
├── data/            GeoTIFF I/O, z-score normalization using training stats
├── evaluation/      Metrics: PSNR, SSIM, SAM, ERGAS, SCI, TUS
├── hallucination/   Hallucination detector (Spectral/Structural/Textural/Semantic)
└── utils/           Logging, helpers, timers
```

---

## License
MIT License — see [LICENSE](LICENSE) file

## Published on PyPI
```bash
pip install edns
```
Package: [https://pypi.org/project/edns/](https://pypi.org/project/edns/) — version 1.0.1

## Acknowledgments
Special thanks to the Kaggle community for dataset access and the OpenCode framework for enabling this independent research.  
This work is inspired by ALSAT-2B satellite imagery research in collaboration with **ASAL** (Algerian Space Agency).

## Author
Chekalil Aimene Charef Eddine — independent researcher