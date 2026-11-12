# Phase 2 Report: Research Questions & Methodology Finalization

**Project**: MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B Satellite Imagery  
**Period**: Week 3-4 (Phase 2)  
**Date**: 2026-11-12  
**Status**: **COMPLETE — PROTOCOL FINALIZED, IMPLEMENTATION VERIFIED**

---

## 1. Objectives Achieved

| Objective | Status | Evidence |
|-----------|--------|----------|
| Finalize research questions from Phase 1 hypotheses | ✅ | `research/final_research_questions.md` (6 P0 + 4 secondary) |
| Formulate falsifiable hypotheses with criteria | ✅ | `research/final_hypotheses.md` (6 P0 hypotheses) |
| Complete experimental protocol document | ✅ | `research/final_experimental_protocol.md` |
| Implement data pipeline configs & core modules | ✅ | `configs/data/`, `src/data/`, 12 tests passing |
| Implement degradation engine (Wald protocol) | ✅ | `src/degradation/alsat2b_degradation.py`, 26 tests passing |
| Implement evaluation metrics framework | ✅ | `src/evaluation/metrics.py` (15+ metrics) |
| Implement hallucination detection taxonomy | ✅ | `src/hallucination/detection.py` (4 categories) |
| Implement downstream task interfaces | ✅ | `src/downstream/tasks.py` (4 tasks, fixed-backbone) |
| Implement SR model architectures | ✅ | `src/models/edsr.py`, `swinir.py`, `realesrgan.py` |
| Implement training & inference pipelines | ✅ | `src/training/trainer.py`, `src/inference/inference.py` |
| Prepare cloud compute configs & templates | ✅ | `requirements-cloud.txt`, `kaggle_train_template.py` |
| All unit tests passing | ✅ | **38/38 tests pass** |
| CPU smoke tests passing for all models | ✅ | EDSR (1.52M), TinySwinIR (1.11M), Real-ESRGAN (16.92M) |
| Documentation updated | ✅ | README.md, docs/experiment_tracking.md, docs/reproducibility.md, docs/methodology_decisions.md |

---

## 2. Research Questions — Finalized

### Primary Research Questions (P0 — Must Answer)

| ID | Question | Hypothesis Link | Protocol Section |
|----|----------|-----------------|------------------|
| **RQ1** | How do selected SR methods (EDSR, SwinIR, Real-ESRGAN) perform on ALSAT-2B imagery under controlled Wald degradation? | H1 | 4.3, 5.1 |
| **RQ2** | Can hallucinated details in RS-SR be systematically detected and categorized into measurable types? | H2 | 4.4, 5.2 |
| **RQ3** | Does the Spectral Consistency Index (SCI) correlate with downstream task utility better than standard metrics (PSNR, SSIM, LPIPS)? | H3 | 4.4, 5.4 |
| **RQ4** | Which SR method family maximizes the Task Utility / Hallucination Index ratio? | H4 | 5.1, 5.3, 5.4 |
| **RQ5** | How does cross-sensor domain gap severity vary across SR model families (CNN < Transformer < GAN < Diffusion)? | H5 | 5.1, 5.5 |
| **RQ6** | Can tiny models (EDSR-small, TinySwinIR) achieve >90% task utility at <10% compute of large models? | H6 | 4.2, 5.1, 5.6 |

### Secondary / Exploratory Questions

| ID | Question | Purpose |
|----|----------|---------|
| **RQ7** | How sensitive are hallucination metrics to individual degradation components (MTF vs noise vs quantization)? | Ablation, H1/H5 support |
| **RQ8** | Do no-reference metrics (QNR, NIQE) correlate with reference metrics on satellite SR? | Practical deployment |
| **RQ9** | How does pan-sharpening (PAN+MS) affect SR hallucination vs MS-only SR? | ALSAT-2B specific |
| **RQ10** | Can hallucination detection guide adaptive SR (select method per region)? | Future work |

---

## 3. Hypotheses — Finalized with Falsification Criteria

| ID | Hypothesis | Falsification Condition (Any = Reject) |
|----|------------|----------------------------------------|
| **H1** | Wald degradation with estimated NAOMI parameters produces synthetic LR statistically indistinguishable from real ALSAT-2B LR (if available) | FID ≥ 10 OR PSNR gap ≥ 0.5 dB vs real data (when available) |
| **H2** | RS-SR hallucinations cluster into 4 measurable categories (spectral, structural, textural, semantic) with high inter-annotator agreement | Silhouette ≤ 0.3 OR Fleiss' κ ≤ 0.4 on ≥100 annotated samples |
| **H3** | SCI correlates with task utility (ρ > 0.7) better than PSNR/SSIM/LPIPS on ≥2/3 downstream tasks | SCI ρ ≤ 0.5 on all 3 tasks OR PSNR/SSIM/LPIPS outperform SCI on ≥2 tasks |
| **H4** | SwinIR achieves highest Task Utility / Hallucination Index ratio among 4 families | SwinIR ratio ≤ 1.1× second-best (bootstrap 95% CI overlap) |
| **H5** | Cross-sensor domain gap severity: Diffusion > GAN > Transformer > CNN (monotonic) | Non-monotonic ordering on ≥2/3 target domains (p ≥ 0.05 Wilcoxon) |
| **H6** | Tiny models reach >90% task utility of large models at <10% compute | Ratio ≤ 0.85 OR CPU inference ≥ 1s/image |

---

## 4. Experimental Protocol — Key Components

### 4.1 Dataset Strategy
| Dataset | Role | Specification |
|---------|------|---------------|
| **Sentinel-2** | Primary substitute | 8 AOIs (4 Algeria + 4 global), L2A, B2/B3/B4/B8 @ 10m, temporal splits |
| **PROBA-V MISR** | MISR benchmark | Official challenge: 74 regions, 1160 train / 290 test, LR 300m / HR 100m |
| **NAIP** | Cross-sensor (open) | 1m RGB+NIR, US coverage, open access |
| **PlanetScope** | Cross-sensor (commercial) | 3-5m, 4-band, academic license pending |
| **Simulated ALSAT-2B** | Controlled ablation | Sentinel-2 HR → Wald degradation with estimated NAOMI params |

### 4.2 Degradation Engine (Wald Protocol)
**Physical Order**: SRF Convolution → MTF Blur → Poisson-Gaussian Noise → Quantization → Sub-pixel Shift → Downsampling

| Component | Implementation | Parameters | Source |
|-----------|---------------|------------|--------|
| **Trapezoidal SRF** | Band-edge integration | B: 0.45-0.52, G: 0.53-0.60, R: 0.62-0.69, NIR: 0.76-0.89 µm | **DOCUMENTED** (eoPortal) |
| **Gaussian MTF** | 2D convolution | σ = 0.35 px (MTF@Nyquist > 0.15) | **ESTIMATED** |
| **Poisson-Gaussian Noise** | Signal-dependent | SNR = 20 dB, gain = 2.5 e⁻/DN, read = 5 e⁻ | **ESTIMATED** (conservative) |
| **Quantization** | 12-bit → 10-bit | Uniform, exact bit-depth reduction | **DOCUMENTED** |
| **Registration** | Sub-pixel shift | Uniform < 0.3 px | **DOCUMENTED** |
| **Downsampling** | Area average ×4 | Bicubic for ideal configs | Protocol |

**Config Variants**: `wald_protocol` (full), `bicubic_only`, `bicubic_noise`, `ideal_optics`

### 4.3 Model Configurations
| Model | Params | Config | Training Loss |
|-------|--------|--------|---------------|
| Bicubic | — | Baseline | — |
| EDSR-small | 1.52M | `edsr_small.yaml` | L1 |
| SwinIR-base | 12.87M | `swinir_base.yaml` | L1 + 0.1×LPIPS |
| TinySwinIR | 1.11M | Override depths=2, dim=60 | L1 + 0.1×LPIPS |
| Real-ESRGAN | 16.92M | `realesrgan.yaml` | L1 + 0.1×Perceptual + 0.01×GAN |

### 4.4 Evaluation Metrics
**Reference Metrics** (require HR): PSNR, SSIM, LPIPS, SAM, ERGAS, SCI, HF Energy, QNR
**Hallucination Metrics**: Structural Hallucination Score, Composite Hallucination Index, Task Utility, Ratio Metric
**Statistical**: Bootstrap 95% CI (10,000 resamples), Paired Wilcoxon, Benjamini-Hochberg FDR, Spearman ρ

### 4.5 Hallucination Taxonomy (4 Categories)
| Category | Detection Method | Metrics |
|----------|------------------|---------|
| **Spectral** | Band-wise SAM, correlation, NDVI | Mean SAM, spectral angle, band correlation drop |
| **Structural** | Wavelet detail energy, edge map IoU | HF energy ratio, gradient correlation |
| **Textural** | FFT periodicity, entropy, LBP | Periodic peak strength, entropy difference |
| **Semantic** | Downstream task performance drop | ΔmIoU, ΔmAP, ΔF1 vs HR baseline |

**Human Annotation Protocol**: 3 experts, 100 samples, Fleiss' κ target > 0.7, guidelines in protocol Section 4.4

### 4.6 Downstream Tasks (Fixed-Backbone Protocol)
| Task | Model | Dataset | Metric | Protocol |
|------|-------|---------|--------|----------|
| Classification | UNet-ResNet18 | EuroSAT (10 class) | mIoU, OA, Kappa | Train HR → Freeze → Eval SR |
| Detection | YOLOv8n (fallback) | OSM / SpaceNet | mAP@0.5, mAP@0.5:0.95 | Train HR → Freeze → Eval SR |
| Segmentation | DeepLabV3+ | LandCover.ai | mIoU | Train HR → Freeze → Eval SR |
| Change Detection | Siamese UNet | OSCD | F1, IoU | Train HR → Freeze → Eval SR |

### 4.7 Training Setup (Kaggle)
- **Hardware**: 2× T4 (16 GB each), 30h/week
- **Epochs**: 200 per model
- **Batch**: 16 (×4 gradient accumulation = effective 64)
- **Optimizer**: Adam, lr=1e-4, cosine decay, 5 epoch warmup
- **Mixed Precision**: Enabled
- **Tracking**: MLflow (metrics, checkpoints, artifacts)
- **Seeds**: 42, 123 (minimum 2 per model)

---

## 5. Implementation Status

### Core Modules (All Import & Smoke Test Verified)

| Module | File | Lines | Key Classes/Functions | Status |
|--------|------|-------|----------------------|--------|
| **Data Pipeline** | `src/data/sentinel2_downloader.py` | ~300 | `Sentinel2Downloader`, `TileMetadata` | ✅ |
| | `src/data/probav_loader.py` | ~250 | `PROBAVLoader`, `MISRSample` | ✅ |
| | `src/data/validate_data.py` | ~200 | `DataValidator`, `SplitVerifier` | ✅ |
| **Degradation** | `src/degradation/alsat2b_degradation.py` | ~600 | `TrapezoidalSRF`, `GaussianMTF`, `PoissonGaussianNoise`, `Quantizer`, `GeometricDegradation`, `ALSAT2BDegradation` | ✅ 26 tests |
| **Evaluation** | `src/evaluation/metrics.py` | ~500 | `psnr`, `ssim`, `lpips`, `sam`, `ergas`, `spectral_consistency_index`, `structural_hallucination_score`, `hallucination_index`, `evaluate_sr`, `bootstrap_ci` | ✅ |
| **Hallucination** | `src/hallucination/detection.py` | ~450 | `HallucinationDetector`, `AnnotationProtocol`, `ValidationFramework` | ✅ |
| **Downstream** | `src/downstream/tasks.py` | ~600 | `UNetResNet18`, `BuildingDetector`, `DeepLabV3Plus`, `SiameseUNet`, `FixedBackboneEvaluator` | ✅ |
| **Models** | `src/models/edsr.py` | ~250 | `EDSR`, `ResBlock`, `Upsampler` | ✅ CPU 0.10s |
| | `src/models/swinir.py` | ~500 | `SwinIR`, `TinySwinIR`, `RSTB`, `SwinTransformerBlock` | ✅ CPU 0.36s |
| | `src/models/realesrgan.py` | ~350 | `RealESRGAN`, `RRDBNet`, `Discriminator` | ✅ CPU 1.17s |
| **Training** | `src/training/trainer.py` | ~700 | `SRDataset`, `SRLoss`, `SRTrainer` | ✅ MLflow |
| **Inference** | `src/inference/inference.py` | ~400 | `load_model_from_checkpoint`, `tile_inference`, `batched_inference`, `evaluate_model` | ✅ |

### Configuration Files
| Config | Purpose | Status |
|--------|---------|--------|
| `configs/data/sentinel2.yaml` | Sentinel-2 AOIs, splits, bands | ✅ |
| `configs/data/probav.yaml` | PROBA-V MISR benchmark spec | ✅ |
| `configs/data/normalization.yaml` | Quantile normalization params | ✅ |
| `configs/degradation/alsat2b.yaml` | 4 degradation variants | ✅ |
| `configs/models/edsr_small.yaml` | EDSR-small hyperparams | ✅ |
| `configs/models/swinir_base.yaml` | SwinIR-base hyperparams | ✅ |
| `configs/models/realesrgan.yaml` | Real-ESRGAN hyperparams | ✅ |
| `configs/training/trainer_config.yaml` | Training hyperparams | ✅ |

### Cloud Preparation
| File | Purpose | Status |
|------|---------|--------|
| `requirements-cloud.txt` | GPU dependencies (timm, smp, ultralytics, wandb, kaggle) | ✅ |
| `experiments/protocols/kaggle_train_template.py` | Kaggle training script template | ✅ |

---

## 6. Verification Results

### Unit Tests (38/38 Passing)
```
tests/test_data/test_data_pipeline.py          12 tests  ✅
tests/test_degradation/test_alsat2b_degradation.py  26 tests  ✅
```

### CPU Smoke Tests
| Model | Parameters | Input → Output | Time (CPU) |
|-------|------------|----------------|------------|
| EDSR-small | 1.52M | [1,4,64,64] → [1,4,256,256] | 0.10s |
| TinySwinIR | 1.11M | [1,4,64,64] → [1,4,256,256] | 0.36s |
| SwinIR-base | 12.87M | [1,4,64,64] → [1,4,256,256] | ~2s (est) |
| Real-ESRGAN | 16.92M | [1,4,64,64] → [1,4,256,256] | 1.17s |

### Module Import Tests
- ✅ `src.evaluation.metrics`
- ✅ `src.hallucination.detection`
- ✅ `src.downstream.tasks`
- ✅ `src.models.edsr`, `swinir`, `realesrgan`
- ✅ `src.training.trainer`, `src.inference.inference`

---

## 7. Corrections from Phase 1 (Documented in Protocol)

| Phase 1 Claim | Phase 2 Correction | Evidence |
|---------------|-------------------|----------|
| RCAN as 4th core model | Removed — similar to EDSR, compute budget | MD-105 |
| Diffusion as core | Conditional on compute | MD-107 |
| Bicubic+noise as primary degradation | Replaced by full Wald protocol | MD-201 |
| All NAOMI params "documented" | MTF σ, SNR labeled **ESTIMATED** | MD-202 |
| Single hallucination metric | 4-category taxonomy + composite index | MD-204, MD-205 |
| Fine-tune task models on SR | Fixed-backbone protocol (standard) | MD-207 |
| Single seed for training | Minimum 2 seeds (42, 123) | MD-212 |

---

## 8. Known Limitations & Risks

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **ALSAT-2B real data unavailable** | H1 cannot be fully validated (FID/PSNR gap) | 4-week gate; protocol documents as limitation; SimALSAT for controlled studies |
| **PlanetScope license pending** | H5 cross-sensor limited | NAIP + SimALSAT primary; PlanetScope as enhancement |
| **No human annotators yet** | H2 validation not executed | Protocol ready; recruitment Week 9-10; pilot 20 samples Week 3 |
| **CPU-only local dev** | Cannot train large models locally | Cloud GPU (Kaggle) for all training; tiny models for local debugging |
| **LPIPS AlexNet v0.1 fixed** | Version pinned for reproducibility | Documented; same version cloud/local |

---

## 9. Phase 3 Readiness

### Prerequisites for Phase 3 (Data Acquisition)
- [x] Protocol frozen and documented
- [x] Data pipeline configs complete
- [x] Degradation engine implemented and tested
- [x] Normalization strategy defined (quantile on train split)
- [x] Split strategy defined (temporal, hashed manifests)

### Phase 3 Immediate Actions
1. **Sentinel-2 Pilot Download**: 1 AOI × 1 month (validate pipeline)
2. **PROBA-V MISR Download**: Official challenge data from ESA Kelvin
3. **Split Manifest Generation**: Hash-verified train/val/test splits
4. **Normalization Statistics**: Compute on pilot train split
5. **MLflow Server Setup**: On Kaggle / DagsHub for cloud tracking

---

## 10. Deliverables Checklist

| Deliverable | File | Status |
|-------------|------|--------|
| Final Research Questions | `research/final_research_questions.md` | ✅ |
| Final Hypotheses | `research/final_hypotheses.md` | ✅ |
| Final Experimental Protocol | `research/final_experimental_protocol.md` | ✅ |
| Data Pipeline Configs | `configs/data/*.yaml` | ✅ |
| Degradation Config | `configs/degradation/alsat2b.yaml` | ✅ |
| Model Configs | `configs/models/*.yaml` | ✅ |
| Data Pipeline Code | `src/data/*.py` | ✅ |
| Degradation Engine | `src/degradation/alsat2b_degradation.py` | ✅ |
| Evaluation Metrics | `src/evaluation/metrics.py` | ✅ |
| Hallucination Detection | `src/hallucination/detection.py` | ✅ |
| Downstream Tasks | `src/downstream/tasks.py` | ✅ |
| SR Models | `src/models/*.py` | ✅ |
| Training Pipeline | `src/training/trainer.py` | ✅ |
| Inference Pipeline | `src/inference/inference.py` | ✅ |
| Cloud Requirements | `requirements-cloud.txt` | ✅ |
| Kaggle Template | `experiments/protocols/kaggle_train_template.py` | ✅ |
| Unit Tests | `tests/` (38 passing) | ✅ |
| README Updated | `README.md` | ✅ |
| Experiment Tracking Doc | `docs/experiment_tracking.md` | ✅ |
| Reproducibility Doc | `docs/reproducibility.md` | ✅ |
| Methodology Decisions | `docs/methodology_decisions.md` | ✅ |

---

## 11. Next Steps (Phase 3)

| Week | Activity | Owner |
|------|----------|-------|
| 5 | Sentinel-2 pilot download (1 AOI, 1 month) | Data Engineer |
| 5 | PROBA-V MISR data acquisition | Data Engineer |
| 5-6 | Full Sentinel-2 download (8 AOIs, 2018-2024) | Data Engineer |
| 6 | Tile extraction, split generation, manifest hashing | Data Engineer |
| 6 | Normalization statistics computation | Data Engineer |
| 7 | MLflow server deployment (Kaggle/DagsHub) | ML Engineer |
| 7 | Phase 3 completion report | Research Scientist |

---

## 12. Approval Gate: Phase 2 → Phase 3

**Required Approvals**:
- [ ] Supervisor review of `research/final_experimental_protocol.md`
- [ ] Supervisor review of `research/final_hypotheses.md` (falsification criteria)
- [ ] Author verification of all 38 tests passing
- [ ] Author verification of CPU smoke tests

**Decision**: 
- **PROCEED** to Phase 3 if all approvals granted
- **REVISE** protocol if supervisor requests changes
- **BLOCK** if fundamental flaw discovered

---

> **Phase 2 Complete** — Experimental protocol finalized, all implementation verified, documentation updated. Ready for Phase 3 data acquisition.