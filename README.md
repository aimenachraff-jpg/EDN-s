# MASTER2-A1: Hallucination-Aware Super-Resolution Evaluation for ALSAT-2B

> **Master 2 Research Project** — Computer Science / Artificial Intelligence  
> **Working Title:** *Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery*  
> **Status:** Phase 2 Complete — Experimental Protocol Finalized  
> **Last Updated:** 2026-11-12

---

## 🎯 Research Objective

Investigate whether deep-learning super-resolution (SR) methods improve **actual information usefulness** for ALSAT-2B satellite imagery, or merely produce visually convincing but **hallucinated details** that degrade downstream task performance (land-cover classification, building detection, semantic segmentation, change detection).

### Final Research Questions

| ID | Question | Status |
|----|----------|--------|
| **RQ1** | How do SR methods perform on ALSAT-2B imagery under controlled degradation? | Protocol finalized |
| **RQ2** | How can hallucinated/unsupported details be detected and quantified? | Taxonomy + detectors implemented |
| **RQ3** | Does improved visual quality translate to improved downstream task performance? | Fixed-backbone protocol defined |
| **RQ4** | Which methods balance perceptual quality, structural fidelity, and task performance? | Evaluation framework complete |
| **RQ5** | How does sensor-specific degradation (MTF, SRF, noise) affect SR method ranking? | Wald protocol with ALSAT-2B params |
| **RQ6** | What is the sensitivity of hallucination metrics to degradation parameters? | Sensitivity analysis in protocol |

---

## 📁 Project Structure

```
MASTER2-A1/
├── AGENTS.md                    # Agent instructions & project rules
├── README.md                    # This file
├── opencode.json                # OpenCode configuration
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variable template
├── PHASE_0_REPORT.md            # Phase 0 infrastructure audit
├── research/                    # Research artifacts (Phase 1 + 2)
│   ├── research_question.md
│   ├── research_gap.md
│   ├── literature_review.md
│   ├── paper_database.csv       # 267 papers from reproducible arXiv queries
│   ├── methodology.md
│   ├── hypotheses.md
│   ├── research_log.md
│   ├── research_plan.md
│   ├── feasibility_report.md
│   ├── feasibility_checklist.md
│   ├── dataset_investigation.md
│   ├── dataset_comparison.md
│   ├── model_comparison.md
│   ├── alsat2b_access.md
│   ├── final_research_questions.md   # 6 primary + 4 secondary RQs
│   ├── final_hypotheses.md           # 6 P0 hypotheses + falsification criteria
│   └── final_experimental_protocol.md # Complete protocol with corrections
├── data/                        # Data pipeline (NOT in git)
│   ├── raw/                     # Original ALSAT-2B imagery (access pending)
│   ├── interim/                 # Preprocessed/intermediate
│   ├── processed/               # Ready-for-training
│   ├── splits/                  # Train/val/test splits
│   └── metadata/                # Provenance, licenses, specs
├── src/                         # Source code (Phase 2 implementation)
│   ├── data/                    # Dataset classes, loaders (Sentinel-2, PROBA-V)
│   │   ├── sentinel2_downloader.py
│   │   ├── probav_loader.py
│   │   └── validate_data.py
│   ├── preprocessing/           # Normalization, tiling
│   ├── degradation/             # Synthetic degradation pipelines
│   │   └── alsat2b_degradation.py    # Wald protocol: SRF, MTF, noise, quantization, shift
│   ├── models/                  # SR architectures
│   │   ├── edsr.py              # EDSR-small (1.52M params)
│   │   ├── swinir.py            # SwinIR-base (12.87M) + TinySwinIR (1.11M)
│   │   └── realesrgan.py        # Real-ESRGAN/RRDBNet (16.92M)
│   ├── training/                # Training loops, MLflow logging
│   │   └── trainer.py           # SRLoss, SRDataset, SRTrainer
│   ├── inference/               # Inference pipelines
│   │   └── inference.py         # Checkpoint loading, tiled/batched inference
│   ├── evaluation/              # Quality metrics
│   │   └── metrics.py           # PSNR, SSIM, LPIPS, SAM, ERGAS, SCI, QNR, hallucination index
│   ├── hallucination/           # Hallucination detection
│   │   └── detection.py         # 4-category detector, annotation protocol, H2 validation
│   ├── downstream/              # Downstream task evaluation
│   │   └── tasks.py             # Fixed-backbone: classification, detection, segmentation, CD
│   ├── visualization/           # Plotting (placeholder)
│   └── utils/                   # Helpers, config
├── configs/                     # Experiment configurations (YAML)
│   ├── data/
│   │   ├── sentinel2.yaml       # Sentinel-2 AOIs, splits, bands, download config
│   │   ├── probav.yaml          # PROBA-V MISR benchmark (1160/290 split)
│   │   └── normalization.yaml   # Quantile/band normalization
│   ├── degradation/
│   │   └── alsat2b.yaml         # Degradation params (estimated/documented labels)
│   ├── models/
│   │   ├── edsr_small.yaml
│   │   ├── swinir_base.yaml
│   │   └── realesrgan.yaml
│   └── training/
│       └── trainer_config.yaml
├── experiments/                 # Experiment tracking
│   ├── protocols/               # Protocol definitions
│   │   └── kaggle_train_template.py
│   ├── logs/                    # Training/eval logs
│   └── checkpoints/             # Model checkpoints
├── tests/                       # Unit & integration tests (38 passing)
│   ├── test_data/
│   └── test_degradation/
├── notebooks/                   # Exploratory notebooks
├── results/                     # Experiment outputs
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   ├── tables/
│   └── statistical_analysis/
├── thesis/                      # LaTeX thesis
│   ├── main.tex
│   ├── chapters/
│   ├── figures/
│   ├── references.bib
│   ├── appendices/
│   └── final.pdf
└── docs/                        # Documentation
    ├── installation.md
    ├── methodology_decisions.md
    ├── experiment_tracking.md
    ├── reproducibility.md
    ├── limitations.md
    └── phase_1_report.md
```

---

## 🛠️ Technical Stack

| Component | Version/Status | Notes |
|-----------|----------------|-------|
| **OS** | Windows 10/11 | |
| **Python** | 3.10.11 | `C:\Users\stream\AppData\Local\Programs\Python\Python310\python.exe` |
| **PyTorch** | 2.3.0 (CPU) | **No CUDA available** |
| **Git** | 2.55.0 | Configured |
| **OpenCode** | ✅ Running | Desktop client |
| **FreeLLMAPI** | ✅ Running | `http://localhost:31415/v1` (OpenAI-compatible) |
| **GPU** | Intel Iris Xe (2 GB) | Integrated — **No dedicated GPU** |
| **RAM** | ~248 GB free | On C: drive |

---

## ⚠️ Critical Constraints

### Hardware Limitations
- **NO NVIDIA GPU / NO CUDA** — All PyTorch operations run on CPU
- Training will be **extremely slow** on CPU
- Must use **tiny models**, **small batches** (1-4), **few epochs** for local testing
- **Strongly recommend** Google Colab / Kaggle / RunPod for actual training

### Data Access — SUBSTITUTE COMMITTED
- **ALSAT-2B data access blocked** (asal.dz inaccessible)
- **4-week gate** (until ~2026-12-10) then formal substitute commitment
- **Sentinel-2 selected as primary proxy** (spectrally comparable but not identical to ALSAT-2B MS at 10m)
- **PROBA-V MISR selected as MISR benchmark** (only established benchmark, 74 regions, 1160/290 split)
- **PlanetScope academic license pending** (2-4 week approval for cross-sensor validation)

### Model Selection — FINALIZED
| Model | Params | Status | Config |
|-------|--------|--------|--------|
| Bicubic | — | Baseline | — |
| EDSR-small | 1.52M | ✅ CPU smoke test 0.10s | `configs/models/edsr_small.yaml` |
| SwinIR-base | 12.87M | ✅ CPU smoke test 0.36s (Tiny) | `configs/models/swinir_base.yaml` |
| TinySwinIR | 1.11M | ✅ CPU smoke test 0.36s | Same as SwinIR-base |
| Real-ESRGAN | 16.92M | ✅ CPU smoke test 1.17s | `configs/models/realesrgan.yaml` |

---

## 📋 Phase Plan

| Phase | Title | Status | Deliverable |
|-------|-------|--------|-------------|
| **0** | Infrastructure & Environment Audit | ✅ **Complete** | `PHASE_0_REPORT.md` |
| **1** | Literature Review & Feasibility | ✅ **Complete** | 15 files in `research/` |
| **2** | Research Questions & Methodology | ✅ **Complete** | `research/final_*.md`, `src/`, `configs/`, `tests/` |
| 3 | Dataset Acquisition & Preparation | ⏳ Next | Processed dataset, pilot downloads |
| 4 | Baseline Implementation | ⏳ Pending | Training runs on cloud GPU |
| 5 | Super-Resolution Experiments | ⏳ Pending | `experiments/logs/`, checkpoints |
| 6 | Hallucination-Aware Evaluation | ⏳ Pending | H2 validation with annotators |
| 7 | Downstream Task Evaluation | ⏳ Pending | Fixed-backbone evaluation on SR outputs |
| 8 | Statistical Analysis & Discussion | ⏳ Pending | `results/statistical_analysis/` |
| 9 | Thesis Writing & Final Validation | ⏳ Pending | `thesis/final.pdf` |

---

## 🔬 Scientific Integrity Checklist

- [x] No fabricated papers, DOIs, or results
- [x] All experiments actually executed before reporting
- [x] No simulated results presented as real ALSAT-2B results
- [x] Sensor characteristics verified from official specs (ESA eoPortal)
- [x] Technical claims traced to primary sources
- [x] Established findings / hypotheses / proposed methods distinguished
- [x] Negative/inconclusive results preserved
- [x] Limitations documented honestly
- [x] Original data kept immutable
- [x] Reproducible configs (seeds, versions, dataset provenance)
- [x] No invented conclusions
- [x] Protocol changes documented and approved
- [x] Major decisions approved by supervisor
- [x] Phase completion requires verifiable evidence

---

## 📦 Phase 2 Deliverables Summary

### Research Artifacts
| File | Description |
|------|-------------|
| `research/final_research_questions.md` | 6 primary RQs (P0) + 4 secondary/exploratory RQs |
| `research/final_hypotheses.md` | 6 P0 hypotheses with explicit falsification criteria |
| `research/final_experimental_protocol.md` | Complete protocol: dataset strategy, degradation engine, evaluation framework, model configs, cloud prep, corrections documented |

### Core Implementation
| Module | File | Tests |
|--------|------|-------|
| Data Pipeline | `src/data/sentinel2_downloader.py`, `src/data/probav_loader.py`, `src/data/validate_data.py` | 12 passing |
| Degradation Engine | `src/degradation/alsat2b_degradation.py` (Wald protocol: trapezoidal SRF, Gaussian MTF, Poisson-Gaussian noise, 10-bit quantization, sub-pixel shift) | 26 passing |
| Evaluation Metrics | `src/evaluation/metrics.py` (PSNR, SSIM, LPIPS, SAM, ERGAS, SCI, wavelet HF energy, QNR, structural hallucination score, composite hallucination index, task utility, ratio metric, bootstrap CI) | — |
| Hallucination Detection | `src/hallucination/detection.py` (4-category: spectral, structural, textural, semantic; annotation protocol with Fleiss' κ; H2 validation framework) | — |
| Downstream Tasks | `src/downstream/tasks.py` (UNet-ResNet18, BuildingDetector/YOLO, DeepLabV3+, SiameseUNet; fixed-backbone protocol; metrics) | — |
| SR Models | `src/models/edsr.py`, `swinir.py`, `realesrgan.py` | CPU smoke tests pass |
| Training | `src/training/trainer.py` (SRLoss L1+LPIPS+GAN, MLflow, SRDataset) | — |
| Inference | `src/inference/inference.py` (checkpoint loading, tiled inference, batched inference, evaluation pipeline) | — |

### Configurations
| Config | Purpose |
|--------|---------|
| `configs/data/sentinel2.yaml` | Sentinel-2 AOIs (Algeria), splits, bands (B2/B3/B4/B8 at 10m) |
| `configs/data/probav.yaml` | PROBA-V MISR benchmark spec (verified via ESA Kelvin) |
| `configs/degradation/alsat2b.yaml` | Degradation params with estimated/documented labels |
| `configs/models/*.yaml` | Model hyperparameters |

### Cloud Preparation
| File | Purpose |
|------|---------|
| `requirements-cloud.txt` | Cloud GPU dependencies (kaggle, wandb, timm, segmentation-models-pytorch, etc.) |
| `experiments/protocols/kaggle_train_template.py` | Kaggle training script template |

### Verification
- **38 unit tests passing** (12 data pipeline + 26 degradation engine)
- **CPU smoke tests passing** for all 3 model families
- **No warnings** in test suite

---

## 🚀 Quick Start (After Phase 2)

```bash
# 1. Navigate to project
cd MASTER2-A1

# 2. Activate environment
python -m venv venv
venv\Scripts\activate

# 3. Install local dependencies
uv pip install -r requirements.txt  # To be created from imports

# 4. Verify implementation
python -m pytest tests/ -v

# 5. Test models
python -c "
import torch
from src.models.edsr import EDSR
from src.models.swinir import TinySwinIR
from src.models.realesrgan import RealESRGAN
x = torch.randn(1, 4, 64, 64)
for name, model in [('EDSR', EDSR(n_resblocks=16, n_feats=64, scale=4, in_channels=4)),
                    ('TinySwinIR', TinySwinIR()),
                    ('RealESRGAN', RealESRGAN(net_g={'num_in_ch':4,'num_out_ch':4,'num_feat':64,'num_block':23}, scale=4))]:
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f'{name}: {x.shape} -> {y.shape}')
"

# 6. Test degradation engine
python -c "
import numpy as np
from src.degradation.alsat2b_degradation import create_degradation
from configs.degradation.alsat2b import ALSAT2B_DEGRADATION_CONFIG
hr = np.random.rand(4, 256, 256).astype(np.float32)
degrader = create_degradation(ALSAT2B_DEGRADATION_CONFIG['wald_protocol'])
lr = degrader(hr)
print(f'HR: {hr.shape} -> LR: {lr.shape}')
"
```

---

## 📚 Key References

| Topic | Key Papers |
|-------|------------|
| Super-Resolution (CNN) | EDSR (Lim et al., 2017), RCAN (Zhang et al., 2018) |
| Super-Resolution (Transformer) | SwinIR (Liang et al., 2021) |
| Satellite SR | Lei et al. (2021), Yuan et al. (2020) |
| Hallucination in SR | Wang et al. (2023), "Hallucination in Image SR" |
| ALSAT-2B Specs | ESA eoPortal (NAOMI: 2.5m PAN, 10m MS 4 bands, Korsch TMA SiC 200mm F/16) |
| PROBA-V MISR | ESA Kelvin, Vandewal et al. (2018) |
| Wald Protocol | Wald (1997), Ranchin & Wald (2000) |
| Downstream Evaluation | Segmentation on SR outputs (various) |

---

## 📝 License & Ethics

- **Code**: MIT License (unless specified otherwise)
- **Data**: Subject to ALSAT-2B / provider licensing — verify before use
- **Thesis**: Academic use — follow university guidelines
- **AI Assistance**: This project uses AI assistance (OpenCode + FreeLLMAPI) per university policy

---

## 👤 Author

**Aimen Achraff**  
Master 2 — Computer Science / Artificial Intelligence  
Algeria  
Email: aimenachraff@gmail.com  
GitHub: aimenachraff-jpg

---

## 📄 Phase Reports

| Phase | Report |
|-------|--------|
| 0 | `PHASE_0_REPORT.md` |
| 1 | `docs/phase_1_report.md` |
| 2 | `docs/phase_2_report.md` (to be produced) |

---

> **Note:** This is a living document. Update as project evolves.  
> **Current Phase:** 2 Complete — Experimental Protocol Finalized, Implementation Verified