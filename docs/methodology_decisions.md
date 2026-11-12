# Methodology Decisions Log

**Project**: MASTER2-A1 — Hallucination-Aware SR Evaluation for ALSAT-2B  
**Purpose**: Record all methodological decisions with rationale and alternatives considered  
**Status**: Phase 2 Complete — All Major Decisions Documented  
**Last Updated**: 2026-11-12

---

## 1. Decision Template

Each decision follows this format:
- **Decision ID**: `MD-XXX`
- **Date**: YYYY-MM-DD
- **Status**: `PROPOSED` | `APPROVED` | `SUPERSEDED` | `REJECTED`
- **Context**: Why this decision was needed
- **Options Considered**: Alternatives with pros/cons
- **Decision**: What was chosen
- **Rationale**: Technical/scientific justification
- **Implications**: Downstream effects
- **Approval**: Who approved / when

---

## 2. Phase 0: Infrastructure Decisions

### MD-001: Local Development Environment
- **Date**: 2026-11-10
- **Status**: APPROVED
- **Context**: Need reproducible local environment for CPU-only development
- **Options**:
  - A: Conda environment
  - B: Python venv + pip (chosen)
  - C: Docker container
- **Decision**: Python 3.10.11 venv with `requirements.txt` / `requirements-cloud.txt`
- **Rationale**: Lightweight, no Docker overhead on Windows, matches cloud Python version
- **Implications**: Must maintain two requirements files (local vs cloud)

### MD-002: LLM Provider — FreeLLMAPI
- **Date**: 2026-11-10
- **Status**: APPROVED
- **Context**: Need local LLM for agent assistance without API costs
- **Options**:
  - A: Ollama (local models)
  - B: FreeLLMAPI (OpenAI-compatible, chosen)
  - C: Direct API calls to commercial providers
- **Decision**: FreeLLMAPI at `http://localhost:31415/v1`
- **Rationale**: OpenAI-compatible API, runs locally, supports tool-calling, no cost
- **Implications**: API key embedded in OpenCode desktop app; need to extract for scripts

### MD-003: Version Control — Git with Conventional Commits
- **Date**: 2026-11-10
- **Status**: APPROVED
- **Context**: Need structured history for thesis reproducibility
- **Options**:
  - A: Basic Git
  - B: Git + Conventional Commits (chosen)
  - C: Git + GitFlow
- **Decision**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- **Rationale**: Automatic changelog generation, clear semantic history, thesis integration

---

## 3. Phase 1: Dataset & Substitute Decisions

### MD-101: ALSAT-2B Real Data Access Strategy
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: ALSAT-2B data access unknown; asal.dz inaccessible
- **Options**:
  - A: Wait indefinitely for ASAL response
  - B: 4-week gate then formal substitute commitment (chosen)
  - C: Use only simulated data
- **Decision**: 4-week investigation period (until ~2026-12-10), then formal substitute declaration
- **Rationale**: Thesis timeline requires progress; substitute strategy scientifically defensible
- **Implications**: All protocols designed for substitutes; real data would be bonus validation

### MD-102: Primary Substitute Dataset — Sentinel-2
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: Need spectrally similar, openly accessible alternative
- **Options**:
  - A: Landsat 8/9 (30m, poor match)
  - B: Sentinel-2 (10m, 4-band match, chosen)
  - C: WorldView (commercial, high-res)
  - D: PlanetScope (3-5m, academic license needed)
- **Decision**: Sentinel-2 L2A (B2/B3/B4/B8 at 10m) as primary proxy
- **Rationale**: Near-perfect spectral match to ALSAT-2B MS bands; open access; extensive SR literature
- **Implications**: Band edge differences documented (protocol Section 3.2); domain gap studied in H5

### MD-103: MISR Benchmark — PROBA-V MISR
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: Need established multi-image SR benchmark for comparison
- **Options**:
  - A: Custom multi-temporal Sentinel-2
  - B: PROBA-V MISR (official challenge, chosen)
  - C: Simulated multi-angle
- **Decision**: PROBA-V MISR Challenge data (74 regions, 1160/290 split)
- **Rationale**: Only established MISR benchmark; 50+ papers for comparison; official scoring (cPSNR)
- **Implications**: Registration NOT provided (sub-pixel shifts); LR=300m multi-angle, HR=100m

### MD-104: Cross-Sensor Validation Datasets
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: Test domain generalization (H5)
- **Options**:
  - A: NAIP only (US, open)
  - B: PlanetScope only (academic license)
  - C: Both + Simulated ALSAT-2B (chosen)
- **Decision**: NAIP (open, US), PlanetScope (pending license), Simulated ALSAT-2B (controlled)
- **Rationale**: Maximum diversity; NAIP immediate, PlanetScope aspirational, SimALSAT controlled
- **Implications**: H5 tested on available domains; PlanetScope delay = NAIP + SimALSAT primary

---

## 4. Phase 1: Model Selection Decisions

### MD-105: Core Model Families (3)
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: Need representative architectures for CNN, Transformer, GAN
- **Options**:
  - A: EDSR, RCAN, SwinIR, Real-ESRGAN (4)
  - B: EDSR-small, SwinIR-base, Real-ESRGAN (3, chosen)
  - C: EDSR, SwinIR, StableSR (diffusion)
- **Decision**: EDSR-small (CNN), SwinIR-base (Transformer), Real-ESRGAN (GAN)
- **Rationale**: 
  - EDSR-small: Canonical CNN SR, 1.52M params, well-understood
  - SwinIR-base: SOTA Transformer SR, 12.87M params, window attention
  - Real-ESRGAN: GAN-based, perceptual quality, 16.92M params
- **Implications**: Bicubic as baseline; diffusion conditional on compute

### MD-106: Tiny Models for Local CPU Development
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: CPU-only development needs fast iteration
- **Options**:
  - A: FSRCNN only
  - B: Tiny variants of core models (chosen)
- **Decision**: TinySwinIR (depths=2, embed_dim=60, 1.11M params) + EDSR-small
- **Rationale**: Same architecture family as cloud models; enables architecture-specific debugging
- **Implications**: Real-ESRGAN too large for CPU (16.92M, 1.17s); only CNN/Transformer tiny variants

### MD-107: Diffusion Models — Conditional
- **Date**: 2026-11-11
- **Status**: APPROVED
- **Context**: Diffusion SR emerging but compute-intensive
- **Options**:
  - A: Include as core (StableSR, S2DiffSR)
  - B: Conditional on Kaggle compute budget (chosen)
  - C: Exclude entirely
- **Decision**: Conditional — train if compute allows after core 3 families
- **Rationale**: 4-8× compute of GAN; scientific value high but risk to core timeline
- **Implications**: H5 cross-family includes diffusion as 4th family if trained

---

## 5. Phase 2: Degradation Protocol Decisions

### MD-201: Wald Protocol for Synthetic Degradation
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need physically-plausible degradation matching ALSAT-2B NAOMI
- **Options**:
  - A: Bicubic downsampling only
  - B: Bicubic + Gaussian noise
  - C: Full Wald protocol (chosen): SRF → MTF → Noise → Quantization → Shift → Downsample
- **Decision**: Wald protocol with 6 stages in correct physical order
- **Rationale**: Only physically-correct order; matches satellite imaging chain; enables sensitivity analysis
- **Implications**: Parameters estimated from specs (documented); sensitivity analysis in H5/H6

### MD-202: NAOMI Parameter Estimation
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: ALSAT-2B NAOMI MTF/SRF/noise not publicly specified
- **Options**:
  - A: Use Sentinel-2 MSI parameters
  - B: Estimate from NAOMI specs + typical values (chosen)
  - C: Treat all as hyperparameters
- **Decision**: 
  - MTF: Gaussian σ=0.35px (from MTF@Nyquist>0.15)
  - SRF: Trapezoidal from documented band edges
  - SNR: 20dB conservative (spec >100)
  - Quantization: 12→10 bit exact
  - Shift: <0.3px uniform exact
- **Rationale**: Conservative estimates bias against SR methods (harder degradation); all labeled ESTIMATED/DOCUMENTED
- **Implications**: H1 tests if this produces realistic LR; sensitivity analysis varies each parameter

### MD-203: Degradation Config Variants
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need ablation configs for sensitivity analysis
- **Decision**: Four configs in `configs/degradation/alsat2b.yaml`:
  1. `wald_protocol` — Full estimated NAOMI
  2. `bicubic_only` — Ideal downsampling (upper bound)
  3. `bicubic_noise` — Bicubic + noise only
  4. `ideal_optics` — SRF + MTF only (no noise/quantization)
- **Rationale**: Enables isolation of each degradation component effect

---

## 6. Phase 2: Evaluation Framework Decisions

### MD-204: Hallucination Taxonomy — 4 Categories
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: No established RS-SR hallucination taxonomy
- **Options**:
  - A: Binary (hallucinated / not)
  - B: 3 categories (spectral, spatial, semantic)
  - C: 4 categories (chosen): Spectral, Structural, Textural, Semantic
- **Decision**: 
  - **Spectral**: Band-wise statistics, SAM, NDVI, spectral angle
  - **Structural**: Wavelet energy, edge consistency, GLCM
  - **Textural**: Periodicity (FFT), entropy, local binary patterns
  - **Semantic**: Downstream task performance drop
- **Rationale**: Covers physics-based (spectral), structural (wavelet), textural (FFT), functional (task) aspects
- **Implications**: H2 tests if hallucinations cluster into these 4; human annotation protocol designed

### MD-205: Primary Hallucination Metric — Composite Index
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need single metric for model ranking
- **Options**:
  - A: Spectral Consistency Index (SCI) only
  - B: Structural hallucination score only
  - C: Weighted composite (chosen): 0.4×Spectral + 0.3×Structural + 0.3×Task
- **Decision**: Composite Hallucination Index with task-utility normalization
- **Rationale**: Single metric cannot capture all aspects; composite balances; task weight ensures utility focus
- **Implications**: H4 uses Task Utility / Hallucination Index ratio

### MD-206: Spectral Consistency Index (SCI)
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need physics-informed spectral metric for H3
- **Decision**: SCI = 1 - mean(|corr(SR_band_i, HR_band_i) - 1|) across bands
- **Rationale**: Directly measures spectral relationship preservation; correlates with vegetation indices
- **Implications**: H3 tests SCI vs PSNR/SSIM/LPIPS correlation with task utility (ρ > 0.7 target)

### MD-207: Fixed-Backbone Downstream Protocol
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: How to evaluate SR on downstream tasks
- **Options**:
  - A: Retrain task models on SR outputs
  - B: Fixed-backbone: train on HR, evaluate on SR (chosen)
  - C: Fine-tune on SR
- **Decision**: Fixed-backbone — task models trained on HR, frozen, evaluated on SR outputs
- **Rationale**: Isolates SR quality effect; standard in SR literature; prevents task model compensation
- **Implications**: 4 tasks: Classification (EuroSAT), Detection (OSM), Segmentation (LandCover.ai), CD (OSCD)

### MD-208: Downstream Task Datasets
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need standard datasets for each task
- **Decision**:
  - Classification: EuroSAT (10 classes) / DynamicWorld (9 classes)
  - Detection: OSM building footprints / SpaceNet
  - Segmentation: LandCover.ai / custom Sentinel-2 labels
  - Change Detection: OSCD (Sentinel-2, 2 dates)
- **Rationale**: Established benchmarks; compatible with Sentinel-2 10m; open access

---

## 7. Phase 2: Statistical Rigor Decisions

### MD-209: Bootstrap Confidence Intervals
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need uncertainty quantification for all metrics
- **Decision**: 10,000 bootstrap resamples, 95% CI, fixed seed (42)
- **Rationale**: Non-parametric; works for any metric distribution; reproducible with fixed seed
- **Implications**: All reported metrics include CI; pairwise comparisons use CI overlap

### MD-210: Multiple Comparison Correction
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Many model/metric/task combinations tested
- **Decision**: Benjamini-Hochberg FDR (q=0.05) for all pairwise comparisons
- **Rationale**: More powerful than Bonferroni; controls false discovery rate; standard in ML

### MD-211: Correlation Testing — Spearman ρ
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Test metric correlations (H3: SCI vs task utility)
- **Decision**: Spearman rank correlation (non-parametric, robust to outliers)
- **Rationale**: Metric distributions non-Gaussian; monotonic relationship expected

### MD-212: Seed Replication
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Single seed insufficient for stochastic training
- **Decision**: 2 seeds minimum (42, 123) for all training runs; 3 for final results
- **Rationale**: Detects seed sensitivity; enables paired statistical tests

---

## 8. Phase 2: Cloud Compute Decisions

### MD-213: Primary Cloud Platform — Kaggle
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need free GPU for training
- **Options**:
  - A: Google Colab Pro ($10/mo)
  - B: Kaggle (free, 2×T4, 30h/week, chosen)
  - C: RunPod (pay-per-use)
  - D: Lambda Labs (pay-per-use)
- **Decision**: Kaggle free tier primary; Colab Pro / RunPod as backup
- **Rationale**: 60 GPU-hours/week > estimated 40-54 needed; persistent storage; MLflow compatible
- **Implications**: Training scripts must be Kaggle-compatible (template in `experiments/protocols/`)

### MD-214: MLflow for Experiment Tracking
- **Date**: 2026-11-12
- **Status**: APPROVED
- **Context**: Need reproducible experiment logging
- **Options**:
  - A: TensorBoard only
  - B: Weights & Biases
  - C: MLflow (chosen) — local + cloud
- **Decision**: MLflow with local server + cloud artifact store
- **Rationale**: Open source; model registry; Git integration; no vendor lock-in; runs on Kaggle

---

## 9. Superseded / Rejected Decisions

| ID | Original Decision | Superseded By | Reason |
|----|-------------------|---------------|--------|
| MD-105-alt | Include RCAN as 4th core model | MD-105 | RCAN similar to EDSR (channel attention); compute budget |
| MD-107-alt | Exclude diffusion entirely | MD-107 | Conditional inclusion preserves option |
| MD-201-alt | Bicubic + noise as primary | MD-201 | Not physically plausible; Wald protocol standard in RS |
| MD-207-alt | Fine-tune task models on SR | MD-207 | Confounds SR quality with task adaptation |

---

## 10. Decision Approval Authority

| Decision Type | Approval Required |
|---------------|-------------------|
| Dataset substitution | Supervisor + Author |
| Model family addition/removal | Supervisor |
| Protocol modification (degradation, evaluation) | Supervisor + Author |
| Statistical method change | Author |
| Compute platform change | Author |
| Timeline/budget change | Supervisor |

---

## 11. Traceability Matrix

| Decision | Protocol Section | Config File | Code Module | Test |
|----------|------------------|-------------|-------------|------|
| MD-101 | 3.1 | — | `alsat2b_access.md` | — |
| MD-102 | 3.2 | `sentinel2.yaml` | `sentinel2_downloader.py` | `test_data_pipeline.py` |
| MD-103 | 3.3 | `probav.yaml` | `probav_loader.py` | `test_data_pipeline.py` |
| MD-104 | 3.4 | — | — | — |
| MD-105 | 4.1 | `edsr_small.yaml`, `swinir_base.yaml`, `realesrgan.yaml` | `edsr.py`, `swinir.py`, `realesrgan.py` | CPU smoke |
| MD-106 | 4.2 | Same as MD-105 | `swinir.py` (TinySwinIR) | CPU smoke |
| MD-201 | 4.3 | `alsat2b.yaml` | `alsat2b_degradation.py` | `test_alsat2b_degradation.py` |
| MD-202 | 4.3.1-4.3.6 | `alsat2b.yaml` | `alsat2b_degradation.py` | `test_alsat2b_degradation.py` |
| MD-203 | 4.3 | `alsat2b.yaml` | `alsat2b_degradation.py` | `test_alsat2b_degradation.py` |
| MD-204 | 4.4 | — | `detection.py` | — |
| MD-205 | 4.4 | — | `metrics.py` (`hallucination_index`) | — |
| MD-206 | 4.4 | — | `metrics.py` (`spectral_consistency_index`) | — |
| MD-207 | 5.3 | — | `tasks.py` | — |
| MD-208 | 5.3 | — | `tasks.py` | — |
| MD-209 | 5.4 | — | `metrics.py` (`bootstrap_ci`) | — |
| MD-210 | 5.4 | — | `metrics.py` | — |
| MD-211 | 5.4 | — | `metrics.py` | — |
| MD-212 | 5.1 | `trainer_config.yaml` | `trainer.py` | — |
| MD-213 | 6 | `kaggle_train_template.py` | `trainer.py` | — |
| MD-214 | 6 | `trainer_config.yaml` | `trainer.py` (MLflow) | — |

---

## 12. Change Log

| Date | Decision(s) | Author |
|------|-------------|--------|
| 2026-11-10 | MD-001, MD-002, MD-003 | A. Achraff |
| 2026-11-11 | MD-101 through MD-107 | A. Achraff |
| 2026-11-12 | MD-201 through MD-214 | A. Achraff |

---

> **Note**: This log is append-only. For modifications, add new decision with `SUPERSEDES: MD-XXX` reference.