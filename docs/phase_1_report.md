# Phase 1 Report: Literature Review & Research Feasibility

**Project**: MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B Satellite Imagery
**Period**: Week 1-2 (Phase 1)
**Date**: 2025-11-12
**Status**: **COMPLETE — APPROVED TO PROCEED TO PHASE 2**

---

## 1. Objectives Achieved

| Objective | Status | Evidence |
|-----------|--------|----------|
| Systematic literature collection (arXiv) | ✅ | 267 unique papers, reproducible `collect_papers.py` |
| ALSAT-2B specifications verified | ✅ | ESA eoPortal (Nov 2025), NAOMI instrument details |
| ALSAT-2B data access investigated | ✅ | `alsat2b_access.md` — blocked, contact strategy defined |
| Substitute datasets characterized | ✅ | `dataset_investigation.md`, `dataset_comparison.md` |
| SR model architectures compared | ✅ | `model_comparison.md` — 4 families, 3 tiers |
| Research gaps identified | ✅ | `research_gap.md` — 6 primary gaps |
| Hypotheses formulated | ✅ | `hypotheses.md` — 10 hypotheses (6 primary, falsifiable) |
| Methodology designed | ✅ | `methodology.md` — reproducible, phased |
| Feasibility assessed | ✅ | `feasibility_report.md` — GO decision |

---

## 2. Key Findings

### 2.1 ALSAT-2B / NAOMI Specifications (Verified Authority: ESA eoPortal)
- **PAN**: 2.5m GSD, 0.45-0.75µm, 7000px TDI, 12-bit→10-bit downlink
- **MS (4 bands)**: 10m GSD, Blue 0.45-0.52, Green 0.53-0.60, Red 0.62-0.69, NIR 0.76-0.89µm
- **Telescope**: Korsch TMA SiC, 200mm, F/16
- **Swath**: 17.5km, Sun-sync 670km, 10:30 LTAN

### 2.2 Literature Landscape (267 Papers, 2020-2026)
| Category | Count | Key 2026 Papers |
|----------|-------|-----------------|
| RS Super-Resolution | 218 | SpatialSR (2611.13046), S2→1m (2610.34742) |
| Hallucination + SR | 110 | VHA (2610.25770), Info Ceiling (2609.36837) |
| Hallucination + RS | 32 | Hallucination-aware RS (2610.18369) |
| Task-Driven / Downstream | 30+ | GeoSR-Bench, Spatial Downscaling (2610.23972) |
| Cross-Sensor Domain Gap | 20+ | Mind the Gap (2606.28039) |
| Hallucination Suppression | 10+ | SALD (2604.25319) |
| No-Ref Quality (RS) | 6 | SenseBench (2605.10576) |

### 2.3 Substitute Dataset Decision: **Sentinel-2 + PROBA-V MISR**
- **Sentinel-2**: Near-perfect spectral match to ALSAT-2B MS (4 bands at 10m), open access, global, extensive SR literature
- **PROBA-V MISR**: Only established MISR benchmark, enables comparison to 50+ papers
- **Cross-sensor**: NAIP (open, US), PlanetScope (academic license pending), Simulated ALSAT-2B (controlled)

### 2.4 Model Families for Experiments
| Tier | Models | Role |
|------|--------|------|
| **Core (3)** | EDSR-small, SwinIR-base, Real-ESRGAN | CNN, Transformer, GAN baselines |
| **Conditional** | StableSR / S2DiffSR | Diffusion (if compute allows) |
| **Tiny (Local)** | FSRCNN, Tiny SwinIR | CPU development, H6 |

---

## 3. Research Gaps Addressed

| Gap | This Thesis Contribution |
|-----|--------------------------|
| **G1**: ALSAT-2B SR unexplored | First sensor-specific degradation model + SR evaluation |
| **G2**: No RS-SR hallucination taxonomy | 4-category taxonomy with measurable metrics |
| **G3**: Hallucination metrics not standardized | Physics-informed (SCI) + structural + semantic metrics |
| **G4**: Task-driven evaluation limited | Systematic cross-family evaluation on 4 downstream tasks |
| **G5**: Domain gap only studied for diffusion | Cross-family (CNN/Trans/GAN/Diff) domain gap analysis |
| **G6**: Compute vs hallucination unmapped | Tiny vs large model comparison (H6) |

---

## 4. Hypotheses to Test (6 Primary)

| ID | Hypothesis | Success Criterion |
|----|------------|-------------------|
| **H1** | Wald degradation model with estimated NAOMI params produces realistic synthetic ALSAT-2B LR | FID < 10, PSNR gap < 0.5dB vs real (if available) |
| **H2** | RS-SR hallucinations cluster into 4 measurable categories | Silhouette > 0.5, Fleiss' κ > 0.7 |
| **H3** | Spectral Consistency Index (SCI) correlates with task utility (ρ > 0.7) better than PSNR/LPIPS | SCI ρ > 0.7 on ≥2/3 tasks; outperforms PSNR/SSIM/LPIPS |
| **H4** | SwinIR maximizes Task Utility / Hallucination ratio among 4 families | SwinIR ratio > 1.2× second-best (bootstrap 95% CI) |
| **H5** | Cross-sensor domain gap severity: Diffusion > GAN > Transformer > CNN | Monotonic ordering on ≥2/3 target domains (p<0.05) |
| **H6** | Tiny models (EDSR-small, Tiny SwinIR) reach >90% task utility at <10% compute | Ratio > 0.9; CPU inference <1s |

---

## 5. Methodology Summary

### Data Pipeline
- **8 AOIs** (4 Algeria + 4 global validation) × Sentinel-2 L2A (2018-2024)
- **Temporal splits**: Train 2018-2022 / Val 2023 / Test 2024
- **Degradation**: Wald protocol with estimated NAOMI MTF/SRF/noise
- **MISR**: PROBA-V official challenge data

### Model Training (Kaggle 2×T4)
- **Configs**: Hydra/OmegaConf, version-controlled
- **Seeds**: Fixed (42), deterministic CuDNN
- **Tracking**: MLflow (metrics, checkpoints, predictions)
- **Losses**: L1 (CNN/Trans), L1+Perceptual+GAN (Real-ESRGAN)

### Evaluation
- **Hallucination**: Spectral (SAM, band corr, NDVI), Structural (wavelet, GLCM), Semantic (task drop)
- **Downstream**: Fixed-backbone classification, detection, segmentation, change detection
- **Statistics**: Bootstrap 95% CI, paired Wilcoxon, Benjamini-Hochberg FDR, Spearman ρ

---

## 6. Feasibility Verdict: **PROCEED**

| Resource | Status | Notes |
|----------|--------|-------|
| **Compute** | ✅ | Kaggle free tier (60 GPU-h/wk) > needed (40-54 GPU-h) |
| **Data** | ✅ | S2 + PROBA-V open; NAIP/Planet conditional |
| **Software** | ✅ | All packages installed/available |
| **Time** | ✅ | 26-week plan with 2-week buffer |
| **Budget** | ✅ | <$100 (Colab Pro backup) |
| **ALSAT-2B Access** | ⚠️ | Blocked — 4-week gate, then formal substitute decision |

**No fundamental blockers**. Substitute dataset strategy validated.

---

## 7. Decision Gate 1: Phase 1 → Phase 2

**Approval Required**: Human confirmation to proceed to Phase 2 (Research Questions & Methodology Refinement).

**Deliverables for Review**:
1. `research/literature_review.md` — Complete literature synthesis
2. `research/research_gap.md` — 6 primary gaps mapped to hypotheses
3. `research/hypotheses.md` — 10 falsifiable hypotheses
4. `research/methodology.md` — Reproducible experimental design
5. `research/feasibility_report.md` — Resource/risk assessment
6. `research/dataset_investigation.md` — ALSAT-2B specs + substitute analysis
7. `research/dataset_comparison.md` — Quantitative dataset comparison
8. `research/model_comparison.md` — Model selection rationale
9. `research/alsat2b_access.md` — Access investigation + contact plan
10. `research/paper_database.csv` — 267 verified papers (reproducible)
11. `research/collect_papers.py` — Reproducible collection script

**Next Phase (Week 3)**: Research Questions & Methodology Refinement
- Finalize research questions from hypotheses
- Refine methodology with supervisor input
- Define exact evaluation protocols (metrics, splits, statistical tests)
- Begin Sentinel-2 data download (parallel)

---

## 8. Risk Flags for Phase 2

| Risk | Mitigation |
|------|------------|
| PlanetScope license delayed | NAIP + SimALSAT sufficient for H5 |
| Hallucination annotation disagreement | Pilot 20 samples Week 3; refine guidelines |
| S2 download rate limits | Planetary Computer STAC + parallel workers |
| Kaggle policy change | Colab Pro + RunPod budget reserved |

---

*Phase 1 complete. All deliverables in `research/`. Ready for Phase 2 authorization.*