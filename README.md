# MASTER2-A1: Hallucination-Aware Super-Resolution Evaluation for ALSAT-2B

> **Master 2 Research Project** — Computer Science / Artificial Intelligence  
> **Working Title:** *Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery*  
> **Status:** Phase 0 — Infrastructure Audit  
> **Last Updated:** 2026-11-12

---

## 🎯 Research Objective

Investigate whether deep-learning super-resolution (SR) methods improve **actual information usefulness** for ALSAT-2B satellite imagery, or merely produce visually convincing but **hallucinated details** that degrade downstream task performance (land-cover classification, building detection, road extraction, etc.).

### Research Questions (Provisional)

| ID | Question |
|----|----------|
| RQ1 | How do selected SR methods perform on ALSAT-2B imagery under controlled degradation? |
| RQ2 | How can hallucinated/unsupported details be detected and quantified? |
| RQ3 | Does improved visual quality translate to improved downstream task performance? |
| RQ4 | Which methods balance perceptual quality, structural fidelity, and task performance? |

---

## 📁 Project Structure

```
MASTER2-A1/
├── AGENTS.md                    # Agent instructions & project rules
├── README.md                    # This file
├── opencode.json                # OpenCode configuration
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variable template
├── research/                    # Research artifacts
│   ├── research_question.md
│   ├── research_gap.md
│   ├── literature_review.md
│   ├── paper_database.csv
│   ├── methodology.md
│   ├── hypotheses.md
│   ├── research_log.md
│   ├── research_plan.md
│   └── feasibility_report.md
├── data/                        # Data pipeline (NOT in git)
│   ├── raw/                     # Original ALSAT-2B imagery
│   ├── interim/                 # Preprocessed/intermediate
│   ├── processed/               # Ready-for-training
│   ├── splits/                  # Train/val/test splits
│   └── metadata/                # Provenance, licenses, specs
├── src/                         # Source code
│   ├── data/                    # Dataset classes, loaders
│   ├── preprocessing/           # Normalization, tiling, registration
│   ├── degradation/             # Synthetic degradation pipelines
│   ├── models/                  # SR architectures (EDSR, RCAN, SwinIR, etc.)
│   ├── training/                # Training loops, schedulers, logging
│   ├── inference/               # Inference pipelines
│   ├── evaluation/              # Quality metrics (PSNR, SSIM, LPIPS, etc.)
│   ├── hallucination/           # Hallucination detection metrics
│   ├── downstream/              # Downstream task evaluation
│   ├── visualization/           # Plotting, figure generation
│   └── utils/                   # Helpers, config, logging
├── configs/                     # Experiment configurations (YAML)
├── experiments/                 # Experiment tracking
│   ├── protocols/               # Protocol definitions
│   ├── logs/                    # Training/eval logs
│   └── checkpoints/             # Model checkpoints
├── tests/                       # Unit & integration tests
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
    └── limitations.md
```

---

## 🛠️ Technical Stack

| Component | Version/Status | Notes |
|-----------|----------------|-------|
| **OS** | Windows 10/11 | |
| **Python** | 3.10.11 | `C:\Users\stream\AppData\Local\Programs\Python\Python310\python.exe` |
| **PyTorch** | 2.3.0 (CPU) | **No CUDA available** |
| **Git** | 2.55.0 | Configured (aimenachraff@gmail.com) |
| **OpenCode** | ❌ Not installed | Needs setup |
| **FreeLLMAPI** | ❌ Not configured | Needs investigation |
| **GPU** | Intel Iris Xe (2 GB) | Integrated — **No dedicated GPU** |
| **RAM** | ~248 GB free | On C: drive |
| **Key Packages** | rasterio, geopandas, albumentations, timm, segmentation-models-pytorch, torchmetrics, torchvision | Full list in `pip list` |

---

## ⚠️ Critical Constraints

### Hardware Limitations
- **NO NVIDIA GPU / NO CUDA** — All PyTorch operations run on CPU
- Training will be **extremely slow** on CPU
- Must use **tiny models**, **small batches** (1-4), **few epochs** for local testing
- **Strongly recommend** Google Colab / Kaggle / RunPod for actual training

### Data Access — UNKNOWN
- ALSAT-2B data availability **not yet verified**
- Must investigate: ASAL (Algerian Space Agency), CNES, ESA, commercial providers
- Academic access may require formal request
- **If unavailable**: Need scientifically defensible substitute (PROBA-V, Sentinel-2, WorldView, etc.)

### LLM Provider
- FreeLLMAPI GitHub repository **not yet identified**
- No API keys configured
- Must verify: tool-calling support, context limits, rate limits, model availability

---

## 📋 Phase Plan

| Phase | Title | Status | Deliverable |
|-------|-------|--------|-------------|
| **0** | Infrastructure & Environment Audit | 🔄 In Progress | `PHASE_0_REPORT.md` |
| 1 | Literature Review & Feasibility | ⏳ Pending | `literature_review.md`, `feasibility_report.md` |
| 2 | Research Questions & Methodology | ⏳ Pending | `methodology.md`, `hypotheses.md` |
| 3 | Dataset Acquisition & Preparation | ⏳ Pending | Processed dataset, `data/metadata/` |
| 4 | Baseline Implementation | ⏳ Pending | `src/models/`, training scripts |
| 5 | Super-Resolution Experiments | ⏳ Pending | `experiments/logs/`, checkpoints |
| 6 | Hallucination-Aware Evaluation | ⏳ Pending | `src/hallucination/`, metrics |
| 7 | Downstream Task Evaluation | ⏳ Pending | `src/downstream/`, task metrics |
| 8 | Statistical Analysis & Discussion | ⏳ Pending | `results/statistical_analysis/` |
| 9 | Thesis Writing & Final Validation | ⏳ Pending | `thesis/final.pdf` |

---

## 🔬 Scientific Integrity Checklist

- [ ] No fabricated papers, DOIs, or results
- [ ] All experiments actually executed before reporting
- [ ] No simulated results presented as real ALSAT-2B results
- [ ] Sensor characteristics verified from official specs
- [ ] Technical claims traced to primary sources
- [ ] Established findings / hypotheses / proposed methods distinguished
- [ ] Negative/inconclusive results preserved
- [ ] Limitations documented honestly
- [ ] Original data kept immutable
- [ ] Reproducible configs (seeds, versions, dataset provenance)
- [ ] No invented conclusions
- [ ] Protocol changes documented and approved
- [ ] Major decisions approved by supervisor
- [ ] Phase completion requires verifiable evidence

---

## 🚀 Quick Start (After Phase 0)

```bash
# 1. Activate environment
cd MASTER2-A1
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt  # To be created in Phase 1

# 3. Configure environment
cp .env.example .env
# Edit .env with API keys

# 4. Verify setup
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# 5. Run Phase 1 literature search
python src/utils/literature_search.py  # To be created
```

---

## 📚 Key References (To Be Expanded)

| Topic | Key Papers |
|-------|------------|
| Super-Resolution (CNN) | EDSR (Lim et al., 2017), RCAN (Zhang et al., 2018) |
| Super-Resolution (Transformer) | SwinIR (Liang et al., 2021) |
| Satellite SR | Lei et al. (2021), Yuan et al. (2020) |
| Hallucination in SR | Wang et al. (2023), "Hallucination in Image SR" |
| ALSAT-2B Specs | ASAL documentation, CNES catalog |
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

> **Note:** This is a living document. Update as project evolves.  
> **Current Phase:** 0 — See `PHASE_0_REPORT.md` for audit results.