# MASTER2-A1 Research Plan

## Project Overview
- **Thesis Title**: Hallucination-Aware and Task-Driven Evaluation of Super-Resolution Methods for ALSAT-2B Satellite Imagery
- **Degree**: Master 2 Computer Science / AI
- **Duration**: ~6 months (typical M2 thesis)
- **Supervisor**: [To be confirmed]
- **Start Date**: 2026-11-12

---

## Phase Timeline

| Phase | Duration | Start | End | Status |
|-------|----------|-------|-----|--------|
| 0 | 1 week | 2026-11-12 | 2026-11-19 | 🔄 In Progress |
| 1 | 3 weeks | 2026-11-19 | 2026-12-10 | ⏳ Pending |
| 2 | 2 weeks | 2026-12-10 | 2026-12-24 | ⏳ Pending |
| 3 | 3 weeks | 2026-12-24 | 2027-01-14 | ⏳ Pending |
| 4 | 3 weeks | 2027-01-14 | 2027-02-04 | ⏳ Pending |
| 5 | 4 weeks | 2027-02-04 | 2027-03-04 | ⏳ Pending |
| 6 | 3 weeks | 2027-03-04 | 2027-03-25 | ⏳ Pending |
| 7 | 3 weeks | 2027-03-25 | 2027-04-15 | ⏳ Pending |
| 8 | 2 weeks | 2027-04-15 | 2027-04-29 | ⏳ Pending |
| 9 | 3 weeks | 2027-04-29 | 2027-05-20 | ⏳ Pending |
| **Buffer** | 2 weeks | 2027-05-20 | 2027-06-03 | ⏳ Pending |

**Total**: ~26 weeks (~6 months) + buffer

---

## Phase 0: Infrastructure & Environment Audit (Week 1)

### Objectives
- [ ] Inspect working directory and existing files
- [ ] Audit Windows, Python, Git, OpenCode environments
- [ ] Check GPU/CUDA, RAM, storage
- [ ] Investigate FreeLLMAPI installation and documentation
- [ ] Identify available models and test connectivity
- [ ] Verify tool-calling support
- [ ] Identify missing dependencies and risks
- [ ] Create project directory structure
- [ ] Create AGENTS.md, README.md, research_plan.md
- [ ] Create preliminary feasibility checklist
- [ ] Produce PHASE_0_REPORT.md

### Deliverables
- `PHASE_0_REPORT.md` — Complete environment audit
- Project structure initialized
- `AGENTS.md` — Agent instructions
- `README.md` — Project overview
- `research_plan.md` — This file

### Risks
- OpenCode not installed → Need to install/configure
- FreeLLMAPI not configured → Need GitHub URL and API key
- No CUDA → CPU-only training (very slow)
- ALSAT-2B data access unknown → May need substitute dataset

---

## Phase 1: Literature Review & Research Feasibility (Weeks 2-4)

### Objectives
- [ ] Systematic literature search on satellite image super-resolution
- [ ] Review hallucination detection in SR
- [ ] Review downstream task evaluation methodologies
- [ ] Investigate ALSAT-2B specifications and data access
- [ ] Identify methodological gaps
- [ ] Assess feasibility of ALSAT-2B data acquisition
- [ ] Create verified bibliography (BibTeX)
- [ ] Document research gap

### Key Search Topics
1. **Satellite SR Methods**: EDSR, RCAN, SwinIR, ESRGAN, Real-ESRGAN, diffusion-based SR
2. **Satellite-specific SR**: Pan-sharpening, multi-spectral SR, SAR-optical SR
3. **Hallucination in SR**: Detection metrics, uncertainty estimation, fidelity evaluation
4. **Downstream Evaluation**: Classification/detection on SR outputs, task-aware SR
5. **ALSAT-2B**: Sensor specs (2.5m pan, 10m MS), data policy, access procedures
6. **Evaluation Metrics**: PSNR, SSIM, LPIPS, SAM, ERGAS, UIQI, FID, hallucination-specific

### Deliverables
- `research/literature_review.md` — Structured review with citations
- `research/paper_database.csv` — Verified papers (title, authors, year, venue, DOI, key findings, relevance)
- `research/research_gap.md` — Identified gaps this thesis addresses
- `research/feasibility_report.md` — Data access, compute, timeline feasibility
- `thesis/references.bib` — BibTeX bibliography (growing)

### Approval Required
- [ ] Final research questions (RQ1-RQ4 or revised)
- [ ] Selected baseline methods
- [ ] Feasibility of ALSAT-2B data (if unavailable, approved substitute)

---

## Phase 2: Research Questions & Methodology (Weeks 5-6)

### Objectives
- [ ] Finalize research questions and hypotheses
- [ ] Define methodology for each RQ
- [ ] Design experimental protocols
- [ ] Define evaluation metrics with mathematical formulations
- [ ] Design hallucination detection framework
- [ ] Plan statistical analysis approach
- [ ] Document methodology decisions

### Research Questions (Current Draft)
| RQ | Question | Hypothesis |
|----|----------|------------|
| RQ1 | SR performance on ALSAT-2B under controlled degradation | Transformer-based (SwinIR) > CNN-based (EDSR/RCAN) > bicubic |
| RQ2 | Hallucination detection and quantification | Reference-based consistency metrics + no-reference uncertainty can detect hallucinations |
| RQ3 | Visual quality → downstream task improvement | Improved PSNR/SSIM ≠ improved downstream mIoU/F1; hallucinations degrade tasks |
| RQ4 | Best perceptual/fidelity/task balance | Methods with explicit fidelity constraints (e.g., perceptual + adversarial + task loss) |

### Deliverables
- `research/research_question.md` — Finalized RQs
- `research/hypotheses.md` — Testable hypotheses per RQ
- `research/methodology.md` — Complete methodology
- `docs/methodology_decisions.md` — Decision log with rationale

### Approval Required
- [ ] Final methodology
- [ ] Selected metrics (with formulas)
- [ ] Hallucination detection approach
- [ ] Downstream task selection

---

## Phase 3: Dataset Acquisition & Preparation (Weeks 7-9)

### Objectives
- [ ] Acquire ALSAT-2B imagery (or approved substitute)
- [ ] Verify sensor specs, formats, licensing
- [ ] Develop preprocessing pipeline
- [ ] Create degradation simulation (controlled)
- [ ] Create spatially separated train/val/test splits
- [ ] Document dataset provenance

### ALSAT-2B Specifications (To Verify)
| Parameter | Value | Source |
|-----------|-------|--------|
| Panchromatic Resolution | 2.5 m | ASAL/CNES |
| Multispectral Resolution | 10 m | ASAL/CNES |
| Spectral Bands | B1: Blue (0.45-0.52µm), B2: Green (0.53-0.60µm), B3: Red (0.62-0.69µm), B4: NIR (0.77-0.89µm) | ASAL |
| Swath Width | ~17.5 km | ASAL |
| Revisit Time | ~4 days | ASAL |
| Data Format | GeoTIFF, JPEG2000 | ASAL |
| Radiometric Resolution | 10-bit / 12-bit | ASAL |

### Preprocessing Pipeline
1. **Radiometric calibration** → TOA reflectance
2. **Geometric correction** → Orthorectification
3. **Co-registration** → Pan-MS alignment (sub-pixel)
4. **Tiling** → 256×256 or 512×512 patches with overlap
5. **Normalization** → Per-band statistics
6. **Quality filtering** → Cloud/snow/shadow masks

### Degradation Simulation (Controlled)
| Degradation | Parameters | Justification |
|-------------|------------|---------------|
| Downsampling | Bicubic, factor 4× (10m→2.5m) | Simulate MS→Pan resolution gap |
| Gaussian Blur | σ ∈ {0.5, 1.0, 1.5} | Sensor MTF approximation |
| Gaussian Noise | σ ∈ {5, 10, 20} (DN) | Sensor noise model |
| JPEG Compression | Q ∈ {75, 90, 95} | Transmission compression |

### Dataset Splits (Critical: No Leakage)
- **Spatial separation**: Different geographic regions per split
- **Scene-level splits**: Entire scenes assigned to one split
- **Target**: 70/15/15 or 60/20/20 train/val/test

### Deliverables
- `data/raw/` — Original imagery (not in git)
- `data/processed/` — Preprocessed patches (not in git)
- `data/splits/` — Train/val/test indices (JSON)
- `data/metadata/` — Provenance, licenses, preprocessing logs
- `src/data/dataset.py` — PyTorch Dataset class
- `src/preprocessing/` — Pipeline scripts
- `src/degradation/` — Degradation simulation
- `docs/reproducibility.md` — Dataset creation reproducibility

### Approval Required
- [ ] Data acquisition confirmed (or substitute approved)
- [ ] Preprocessing pipeline
- [ ] Split strategy (spatial separation verified)

---

## Phase 4: Baseline Implementation (Weeks 10-12)

### Objectives
- [ ] Implement baseline models
- [ ] Create training infrastructure
- [ ] Implement evaluation metrics
- [ ] Set up experiment tracking

### Selected Baselines (Tentative - Pending Literature Review)
| Model | Type | Parameters | Source |
|-------|------|------------|--------|
| Bicubic | Interpolation | — | Baseline |
| EDSR | CNN (Residual) | ~1.5M | Lim et al. 2017 |
| RCAN | CNN (Channel Attention) | ~15M | Zhang et al. 2018 |
| SwinIR | Transformer (Swin) | ~12M | Liang et al. 2021 |
| **Lightweight variants** | For CPU training | <1M each | Adapted |

### Training Infrastructure
- **Config system**: YAML → Hydra/OmegaConf
- **Logging**: TensorBoard + CSV + JSON
- **Checkpointing**: Best val loss + last + periodic
- **Reproducibility**: Seed everything, deterministic ops
- **Mixed precision**: Not applicable (CPU)

### Deliverables
- `src/models/` — Model implementations
- `src/training/` — Train loop, schedulers, losses
- `src/evaluation/` — Metric implementations
- `configs/` — Experiment configs
- `experiments/protocols/` — Protocol definitions

### Approval Required
- [ ] Final model selection (based on hardware constraints)
- [ ] Training protocol (epochs, LR, batch size, augmentation)

---

## Phase 5: Super-Resolution Experiments (Weeks 13-16)

### Objectives
- [ ] Train all baselines on ALSAT-2B (or substitute)
- [ ] Run inference on test set
- [ ] Compute quality metrics (PSNR, SSIM, LPIPS, SAM, ERGAS)
- [ ] Generate visual comparisons
- [ ] Document all experiments

### Experiment Matrix
| Model | Degradation | Scale | Metrics | Status |
|-------|-------------|-------|---------|--------|
| Bicubic | All | 4× | All | ⏳ |
| EDSR | All | 4× | All | ⏳ |
| RCAN | All | 4× | All | ⏳ |
| SwinIR | All | 4× | All | ⏳ |

### Compute Strategy (CPU-only)
- **Batch size**: 1-2
- **Epochs**: 50-100 (with early stopping)
- **Patch size**: 64×64 or 128×128
- **Augmentation**: Flip, rotate, noise
- **Consider**: Pre-train on larger dataset (DIV2K, satellite datasets) → fine-tune

### Deliverables
- `experiments/logs/` — Training logs
- `experiments/checkpoints/` — Model weights
- `results/metrics/` — CSV/JSON metric tables
- `results/figures/` — Visual comparisons
- `results/tables/` — Summary tables

### Approval Required
- [ ] Training start (long-running)
- [ ] Any protocol changes mid-training

---

## Phase 6: Hallucination-Aware Evaluation (Weeks 17-19)

### Objectives
- [ ] Implement hallucination detection metrics
- [ ] Evaluate all SR outputs for hallucinations
- [ ] Correlate hallucination metrics with quality metrics
- [ ] Analyze spatial distribution of hallucinations

### Candidate Hallucination Metrics
| Metric | Type | Reference | Formula |
|--------|------|-----------|---------|
| **Spectral Consistency** | Ref-based | SAM, spectral angle | `arccos(⟨x, y⟩ / (‖x‖‖y‖))` |
| **Edge Consistency** | Ref-based | Gradient correlation | `corr(∇SR, ∇HR)` |
| **Feature Correspondence** | Ref-based | Deep feature distance | `‖φ(SR) - φ(HR)‖` |
| **Local Uncertainty** | No-ref | Monte Carlo dropout / ensemble | `Var({M_i(SR)})` |
| **Frequency Analysis** | No-ref | High-frequency energy ratio | `‖HP(SR)‖ / ‖HP(LR)‖` |
| **Semantic Consistency** | Task-based | Downstream feature match | `‖F_down(SR) - F_down(HR)‖` |

### Validation Strategy
- Synthetic hallucination insertion (known false details)
- Human expert annotation (if feasible)
- Correlation with downstream task degradation

### Deliverables
- `src/hallucination/` — Detection implementations
- `results/metrics/hallucination/` — Hallucination scores
- `results/figures/hallucination/` — Heatmaps, distributions
- `research/methodology.md` — Updated with hallucination framework

### Approval Required
- [ ] Final hallucination metric selection
- [ ] Validation approach

---

## Phase 7: Downstream Task Evaluation (Weeks 20-22)

### Objectives
- [ ] Select feasible downstream task
- [ ] Acquire/prepare task labels
- [ ] Train/evaluate task model on HR, LR, SR outputs
- [ ] Measure task performance improvement/degradation

### Candidate Tasks (Priority Order)
| Task | Labels Available? | Difficulty | Metrics |
|------|-------------------|------------|---------|
| **Land-cover classification** | CORINE, ESA WorldCover, custom | Medium | mIoU, F1, Acc |
| Building detection | OpenStreetMap, custom | Medium | AP, F1 |
| Road extraction | OpenStreetMap, custom | Medium | AP, F1 |
| Vegetation/agriculture | NDVI threshold, custom | Low | F1, IoU |

### Experimental Design
1. Train task model on **HR (ground truth)** → upper bound
2. Train task model on **LR (degraded)** → lower bound
3. Apply SR → run task model on **SR outputs**
4. Compare: Does SR close gap to HR? Or widen it?

### Deliverables
- `src/downstream/` — Task models, evaluation
- `results/metrics/downstream/` — Task metrics per SR method
- `results/tables/downstream_comparison.csv` — Summary
- `results/figures/downstream/` — Qualitative results

### Approval Required
- [ ] Final task selection (labels must exist)
- [ ] Task model architecture

---

## Phase 8: Statistical Analysis & Discussion (Weeks 23-24)

### Objectives
- [ ] Statistical analysis of all results
- [ ] Ablation studies
- [ ] Error analysis
- [ ] Limitations documentation
- [ ] Scientific discussion

### Statistical Methods
- **Descriptive**: Mean, std, median, IQR across test scenes
- **Inferential**: Paired t-test / Wilcoxon (SR vs bicubic per scene)
- **Correlation**: Hallucination metrics vs. downstream metrics
- **Confidence intervals**: Bootstrap (if n ≥ 30 scenes)
- **Effect sizes**: Cohen's d

### Ablation Studies
- Degradation components (blur only, noise only, combined)
- Loss function components (L1, perceptual, adversarial)
- Model capacity (layers, channels, attention heads)

### Deliverables
- `results/statistical_analysis/` — Stats outputs
- `results/tables/statistical_summary.csv`
- `docs/limitations.md` — Honest limitations
- `research/discussion.md` — Scientific discussion notes

---

## Phase 9: Thesis Writing & Final Validation (Weeks 25-27)

### Objectives
- [ ] Write complete thesis in LaTeX
- [ ] Generate all figures from actual results
- [ ] Verify all citations
- [ ] Internal review (self-review + supervisor)
- [ ] Final formatting and submission

### Thesis Structure
| Chapter | Content | Source |
|---------|---------|--------|
| 1. Introduction | Context, problem, objectives, contributions | `research/` |
| 2. Background | SR theory, satellite imaging, evaluation | `research/literature_review.md` |
| 3. Literature Review | Organized by theme | `research/literature_review.md` |
| 4. Problem Statement | Gap, RQs, hypotheses | `research/research_question.md`, `hypotheses.md` |
| 5. Methodology | Data, models, metrics, protocols | `research/methodology.md` |
| 6. Dataset | ALSAT-2B, preprocessing, splits | `data/metadata/`, `docs/reproducibility.md` |
| 7. Experimental Setup | Training, compute, baselines | `configs/`, `experiments/protocols/` |
| 8. Results | Quality, hallucination, downstream | `results/` |
| 9. Discussion | Interpretation, limitations, implications | `research/discussion.md`, `docs/limitations.md` |
| 10. Conclusion | Summary, contributions, future work | Synthesis |
| References | BibTeX | `thesis/references.bib` |
| Appendices | Configs, extra tables, code snippets | `thesis/appendices/` |

### Deliverables
- `thesis/main.tex` — Main document
- `thesis/chapters/` — Chapter files
- `thesis/figures/` — All figures (vector where possible)
- `thesis/references.bib` — Complete bibliography
- `thesis/final.pdf` — Compiled thesis

### Approval Required
- [ ] Thesis draft review by supervisor
- [ ] Final submission

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ALSAT-2B data unavailable | High | Critical | Identify substitutes early (Sentinel-2, PROBA-V, WorldView); Phase 1 feasibility |
| No GPU / CPU too slow | Certain | High | Use Colab/Kaggle/RunPod for training; local for dev only |
| FreeLLMAPI not working | Medium | Medium | Fallback: direct API (OpenAI, Anthropic) or local LLM (Ollama) |
| Hallucination metrics not validated | Medium | High | Synthetic validation; focus on reference-based metrics |
| Downstream labels unavailable | Medium | High | Use open labels (OSM, WorldCover); simplify task |
| Timeline overrun | High | Medium | Buffer weeks; incremental delivery; regular checkpoints |
| Supervisor feedback delays | Medium | Medium | Early drafts; scheduled reviews |

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Literature review | ≥50 verified papers, organized by theme |
| ALSAT-2B data | Acquired OR substitute justified & acquired |
| Baselines implemented | ≥3 SR models + bicubic, trained & evaluated |
| Quality metrics | PSNR, SSIM, LPIPS, SAM, ERGAS on test set |
| Hallucination framework | ≥3 metrics implemented, validated |
| Downstream task | 1 task with labels, evaluated on HR/LR/SR |
| Statistical rigor | Per-scene stats, significance tests, CIs |
| Thesis | Complete, compiles, all claims evidenced |
| Reproducibility | All configs, seeds, data provenance documented |

---

## Approval Gates Summary

| Gate | Phase | Decision Required |
|------|-------|-------------------|
| G1 | 0→1 | Environment ready; proceed to lit review |
| G2 | 1→2 | RQs finalized; baselines selected; data feasible |
| G3 | 2→3 | Methodology approved; metrics defined |
| G4 | 3→4 | Data acquired; splits created; no leakage |
| G5 | 4→5 | Models implemented; training protocol approved |
| G6 | 5→6 | SR experiments complete; results logged |
| G7 | 6→7 | Hallucination framework validated |
| G8 | 7→8 | Downstream evaluation complete |
| G9 | 8→9 | Analysis complete; thesis outline approved |
| G10 | 9→Submit | Thesis draft reviewed; final PDF approved |

---

## Next Immediate Actions (Phase 0 Completion)

1. [ ] Install OpenCode
2. [ ] Identify FreeLLMAPI GitHub repository
3. [ ] Configure FreeLLMAPI API key
4. [ ] Test model connectivity and tool-calling
5. [ ] Complete PHASE_0_REPORT.md
6. [ ] Request approval to proceed to Phase 1

---

*Document version: 1.0*  
*Created: 2026-11-12*  
*Last updated: 2026-11-12*