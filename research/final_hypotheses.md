# Final Hypotheses — Prioritized and Falsifiable

**Project**: MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B Satellite Imagery  
**Phase**: 2 — Research Questions & Methodology Finalization  
**Date**: 2025-11-12  
**Status**: FROZEN FOR APPROVAL (with `final_research_questions.md`)

---

## 1. Hypothesis Design Principles

Each hypothesis adheres to:
- **Specific**: Single testable claim with defined measurement protocol
- **Falsifiable**: Explicit rejection criteria (pre-registered)
- **Feasible**: Executable within Master 2 compute/data constraints
- **Independent**: Minimal overlap with other hypotheses
- **Contributory**: Each addresses a distinct research gap from Phase 1

Priority tiers:
- **P0 (Must)**: Core thesis claims; failure invalidates central contribution
- **P1 (Should)**: Significant contributions; strengthen thesis if confirmed
- **P2 (Could)**: Mechanistic insights; ablation studies
- **P3 (Exploratory)**: Nice to have; not required for thesis defense

---

## 2. Primary Hypotheses (P0 — Must Test)

### H1: ALSAT-2B Degradation Model Fidelity
> **Claim**: A Wald-protocol degradation model using estimated NAOMI MTF (Gaussian, σ≈0.35 pixels at Nyquist), trapezoidal SRFs from published band edges, Poisson-Gaussian noise (SNR≈20 dB conservative), 10-bit quantization, and <0.3px inter-band registration produces synthetic LR data statistically indistinguishable from real ALSAT-2B data.

**Measurement Protocol**:
1. Generate synthetic LR from Sentinel-2 10m HR (Algeria AOIs) using degradation model
2. Compare to *any available* ALSAT-2B samples (even n<10):
   - Radial power spectral density (PSD) per band
   - 4×4 band-to-band correlation matrices
   - Noise variance vs. mean radiance curves
   - Fréchet Inception Distance (FID) on 64×64 patches (Inception-v3 features)
3. Train EDSR-small on synthetic LR→HR pairs; evaluate PSNR on held-out real ALSAT-2B (if available)

**Variables**:
- Independent: MTF σ (0.30-0.40), SNR (15-25 dB), registration error (0-0.5px)
- Dependent: KS statistic (PSD), Frobenius norm (correlation), FID, PSNR gap

**Success Criteria** (ALL must hold):
- FID < 10 between synthetic and real patch distributions
- PSNR gap < 0.5 dB (synthetic-trained vs. real-trained on held-out real)
- KS statistic < 0.15 for all 4 bands' PSD
- Correlation matrix Frobenius difference < 0.1

**Falsification Criteria** (ANY triggers rejection):
- FID > 20 with any real ALSAT-2B data
- PSNR gap > 1.5 dB
- KS statistic > 0.25 for ≥2 bands
- Correlation matrix difference > 0.2

**Required Data**: Sentinel-2 HR (Algeria), estimated NAOMI params, any ALSAT-2B samples  
**Required Models**: EDSR-small (lightweight proxy)  
**Compute**: CPU (degradation) + Kaggle GPU (EDSR-small training)  
**Timeline**: Phase 3-4  

**Scientific Contribution**: First validated sensor-specific degradation model for ALSAT-2B/NAOMI; establishes credibility of all simulated experiments in thesis

---

### H2: RS-SR Hallucination Taxonomy Validity
> **Claim**: Hallucinations in remote sensing super-resolution cluster into four distinct, measurable categories with unique spectral/spatial signatures: (1) Spectral — violates physical band correlations, (2) Structural — spurious high-frequency energy in homogeneous regions, (3) Textural — plausible but ungrounded periodic patterns, (4) Semantic — downstream task confidence on false structures.

**Measurement Protocol**:
1. Generate SR outputs from 4 model families (EDSR-small, SwinIR-base, Real-ESRGAN, StableSR*) on Sentinel-2 test set (8 AOIs, 50 patches each = 200 patches/model)
2. Expert annotation (3 remote sensing experts):
   - Label each patch: {None, Spectral, Structural, Textural, Semantic, Multiple}
   - Guidelines: Spectral→SAM>10°/NDVI out-of-range; Structural→wavelet HH>3σ in homogeneous; Textural→GLCM entropy mismatch>20%; Semantic→mAP drop>15% vs HR
3. Compute proposed metrics for each patch
4. Cluster analysis (UMAP + HDBSCAN) on 4D metric vectors
5. Compare clusters to expert labels

**Variables**:
- Independent: Model family, scale (×2, ×4), degradation type
- Dependent: SAM, band correlation deviation, NDVI consistency, wavelet HH energy, GLCM entropy mismatch, task performance drop

**Success Criteria** (ALL must hold):
- Silhouette score > 0.5 for 4-cluster solution on metric vectors
- Fleiss' κ > 0.7 for expert agreement on 4-category labels
- Each cluster maps predominantly (>70%) to one expert category

**Falsification Criteria** (ANY triggers rejection):
- Silhouette score < 0.3
- Fleiss' κ < 0.5
- No cluster maps to >50% single expert category

**Required Data**: Sentinel-2 test set, SR outputs from 4 families  
**Required Models**: 4 core families (StableSR conditional on compute)  
**Compute**: Kaggle GPU (inference) + CPU (metrics)  
**Timeline**: Phase 6  
**Dependency**: H1 (degradation model for test data)

**Scientific Contribution**: First RS-specific hallucination taxonomy with operationalized, validated metrics; foundation for H3, H4

---

### H3: Physics-Informed Metrics Correlate with Task Utility
> **Claim**: The Spectral Consistency Index (SCI) — mean SAM across all band pairs weighted by vegetation/water sensitivity (NDVI/MNDWI bands weighted 2×) — correlates with downstream task performance (Spearman ρ > 0.7) better than PSNR (ρ < 0.4), SSIM (ρ < 0.5), or LPIPS (ρ < 0.5).

**Measurement Protocol**:
1. Generate 500+ diverse SR outputs (4 families × 2 scales × multiple degradations × checkpoints)
2. Compute on each: SCI, PSNR, SSIM, LPIPS, NIQE, BRISQUE
3. Evaluate on 3 downstream tasks with fixed-backbone protocol:
   - Land cover classification (mIoU, EuroSAT labels on S2)
   - Building detection (mAP@0.5, OSM footprints on S2)
   - Semantic segmentation (mIoU, LandCover.ai on S2)
4. Compute Spearman ρ between each IQA metric and each task metric

**Variables**:
- Independent: SR model output quality (diverse)
- Dependent: SCI, PSNR, SSIM, LPIPS, NIQE, BRISQUE vs. task metrics

**SCI Definition**:
```
SCI = (1/6) * Σᵢ<ⱼ wᵢⱼ * SAM(bᵢ, bⱼ)
where wᵢⱼ = 2.0 if (i,j) ∈ {(Red,NIR), (Green,NIR), (Red,SWIR)} else 1.0
```
(Bands: Blue=0, Green=1, Red=2, NIR=3; vegetation/water sensitive pairs weighted)

**Success Criteria** (ALL must hold):
- SCI ρ > 0.7 with ≥2 of 3 primary task metrics
- SCI ρ > PSNR ρ, SCI ρ > SSIM ρ, SCI ρ > LPIPS ρ on ALL 3 tasks
- Permutation test p < 0.01 for SCI vs. best baseline metric

**Falsification Criteria** (ANY triggers rejection):
- SCI ρ < 0.5 with any primary task metric
- PSNR ρ > SCI ρ on ≥2 tasks
- LPIPS ρ > SCI ρ on ≥2 tasks

**Required Data**: 500+ SR outputs, downstream task models (fixed backbone)  
**Required Models**: All 4 SR families + 3 fixed-backbone task models  
**Compute**: Kaggle GPU (tasks)  
**Timeline**: Phase 7  
**Dependency**: H2 (taxonomy validates SCI components), H4 (models trained)

**Scientific Contribution**: Validated physics-informed metric for RS-SR model selection without downstream training; practical tool for practitioners

---

### H4: Transformer Maximizes Task Utility / Hallucination Ratio
> **Claim**: For RS-SR at ×4 scale on Sentinel-2 MS data, SwinIR-base achieves the highest downstream task utility per unit of hallucination among CNN (EDSR-small), Transformer (SwinIR-base), GAN (Real-ESRGAN), and Diffusion (StableSR*) families.

**Measurement Protocol**:
1. Train all 4 model families identically on Sentinel-2 train split (Algeria AOIs, 2018-2022, ×4 MS 10m→2.5m sim)
2. Evaluate on held-out test (2024, 4 Algeria AOIs + 4 global):
   - Reconstruction: PSNR, SSIM, LPIPS
   - Hallucination: Spectral (SCI), Structural (wavelet HH), Semantic (task drop)
   - Downstream: Fixed-backbone classification mIoU, detection mAP, segmentation mIoU
3. Compute composite scores:
   - **Task Utility Score** = mean(mIoU_class/HR, mAP_det/HR, mIoU_seg/HR)  [normalized to HR=1]
   - **Hallucination Index** = 0.3×SCI + 0.3×Structural_Score + 0.4×Semantic_Score  [lower=better]
   - **Ratio** = Task Utility / (1 + Hallucination Index)

**Variables**:
- Independent: Model family (4), training seed (3 seeds for CI)
- Dependent: Ratio, Task Utility, Hallucination Index

**Success Criteria** (ALL must hold):
- SwinIR ratio > 1.2× second-best model family
- Bootstrap 95% CI for ratio difference excludes 1.0 (1000 resamples)
- Effect size (Cohen's d) > 0.8 for SwinIR vs. second-best

**Falsification Criteria** (ANY triggers rejection):
- SwinIR not top-1 in ratio
- GAN or Diffusion ratio > SwinIR with statistical significance (p<0.05)
- Ratio difference not significant after Benjamini-Hochberg FDR correction

**Required Data**: Sentinel-2 train/val/test (temporal/spatial splits)  
**Required Models**: 4 core families (StableSR conditional)  
**Compute**: Kaggle GPU (4 families × 3 seeds ≈ 12 training runs)  
**Timeline**: Phase 5-6  
**Dependency**: H1, H2

**Scientific Contribution**: Evidence-based model family recommendation balancing reconstruction quality, hallucination, and downstream utility

---

### H5: Cross-Sensor Domain Gap Severity Ordering
> **Claim**: When trained on Sentinel-2 (source) and tested on target sensors (NAIP 1m, PlanetScope 3m, Simulated ALSAT-2B 2.5m PAN), the relative performance drop (ΔPSNR, ΔTask Utility) follows the ordering: Diffusion > GAN > Transformer > CNN.

**Measurement Protocol**:
1. Train each model family on Sentinel-2 only (same as H4)
2. Test on 3 target domains (no fine-tuning):
   - NAIP: S2 10m → NAIP 1m (×10), US AOIs, RGBN bands
   - PlanetScope: S2 10m → Planet 3m (×3), academic license pending
   - Sim ALSAT-2B: S2 10m → ALSAT-2B PAN 2.5m (×4), degradation model (H1)
3. Measure:
   - ΔPSNR = PSNR_S2_test - PSNR_target
   - ΔTask Utility = TaskUtility_S2_test - TaskUtility_target
   - FID (ResNet-50 features) between S2 and target SR outputs
   - Spectral shift: mean band correlation deviation

**Variables**:
- Independent: Target domain (3), model family (4)
- Dependent: ΔPSNR, ΔTask Utility, FID, spectral shift

**Success Criteria** (ALL must hold):
- Monotonic ordering (Diffusion > GAN > Transformer > CNN) holds for ΔPSNR on ≥2/3 target domains (p<0.05 paired t-test)
- Same ordering holds for ΔTask Utility on ≥2/3 domains
- Effect size (Cohen's d) > 0.5 for adjacent pairs in ordering

**Falsification Criteria** (ANY triggers rejection):
- Ordering violated on ≥2 domains for both ΔPSNR and ΔTask Utility
- CNN shows smallest gap (best generalization) on ≥2 domains
- No statistically significant differences between adjacent families

**Required Data**: Sentinel-2 (source), NAIP (US AOIs), PlanetScope (if license), Sim ALSAT-2B  
**Required Models**: 4 families trained on S2 only  
**Compute**: Kaggle GPU (inference on target domains)  
**Timeline**: Phase 6  
**Dependency**: H1, H4

**Scientific Contribution**: Cross-sensor generalization ranking to guide sensor-agnostic SR model deployment

---

### H6: Tiny Models Reach >90% Task Utility at <10% Compute
> **Claim**: EDSR-small (1.5M params) and Tiny SwinIR (depth=2, ~150K params) achieve >90% of the Task Utility Score of their large counterparts (EDSR-baseline 43M params, SwinIR-base 11.9M params) while using <10% FLOPs and enabling CPU inference <1s per 256² patch on Intel Iris Xe.

**Measurement Protocol**:
1. Train small/large pairs for CNN and Transformer on identical Sentinel-2 data:
   - EDSR-small (16 blocks, 64 feats) vs EDSR-baseline (32 blocks, 256 feats)
   - Tiny SwinIR (depths=[2,2,2,2,2,2], embed_dim=60) vs SwinIR-base (depths=[6,6,6,6,6,6], embed_dim=180)
2. Measure:
   - Task Utility Score (same as H4)
   - FLOPs via `thop` on 1×4×256×256 input
   - CPU inference time (Intel Iris Xe, PyTorch CPU, batch=1, 100 warmup + 100 timed runs)
   - Parameter count
3. Compute ratios: Task_Utility_small / Task_Utility_large, FLOPs_small / FLOPs_large

**Variables**:
- Independent: Model size variant (small vs. large) × architecture (CNN, Transformer)
- Dependent: Task Utility ratio, FLOPs ratio, CPU time

**Success Criteria** (ALL must hold):
- Task Utility ratio > 0.90 for BOTH CNN and Transformer pairs
- FLOPs ratio < 0.10 for BOTH pairs
- CPU inference time < 1.0s for BOTH small models

**Falsification Criteria** (ANY triggers rejection):
- Task Utility ratio < 0.85 for either pair
- FLOPs ratio > 0.15 for either pair
- CPU time > 2.0s for either small model

**Required Data**: Sentinel-2 train/test (same as H4)  
**Required Models**: 4 model variants (2 small, 2 large)  
**Compute**: Local CPU (inference timing) + Kaggle GPU (training)  
**Timeline**: Phase 4-5  
**Dependency**: H4 (Task Utility definition)

**Scientific Contribution**: Practical deployment guidelines for resource-constrained RS-SR; validates tiny models for operational use

---

## 3. Secondary Hypotheses (P1 — Should Test)

### H7: Hallucination Type Dominance by SR Task
> **Claim**: In MS→MS super-resolution (S2 20m→10m), spectral hallucination (SAM, band correlation) is the primary failure mode (highest metric value). In PAN-sharpening simulation (S2 10m→2.5m PAN), structural hallucination (wavelet HH energy in homogeneous regions) dominates.

**Measurement**: Compare H2 metric values across two SR tasks on same model families  
**Falsification**: Spectral not dominant in MS→MS; structural not dominant in PAN-sharpening  
**Compute**: Additional inference runs; no new training  
**Timeline**: Phase 6 (parallel with H2/H4)

---

### H8: Physics-Informed Loss Suppresses Hallucination
> **Claim**: Adding Spectral Consistency Loss (L_spectral = mean SAM between SR and HR band pairs, λ=0.1) to SwinIR training reduces SCI by ≥20% and improves classification mIoU by ≥3% without PSNR degradation (> -0.2 dB).

**Measurement**: Ablation study — SwinIR trained with L1 vs L1+λ·L_spectral  
**Falsification**: SCI reduction < 10% OR mIoU gain < 1% OR PSNR drop > 0.5 dB  
**Compute**: 2 additional SwinIR training runs  
**Timeline**: Phase 5 (if compute allows)

---

### H9: MISR Reduces Hallucination vs. SISR
> **Claim**: For SwinIR/EDSR, multi-image SR (PROBA-V, 5-9 frames) reduces structural hallucination (wavelet metric ↓ 30%) compared to single-image SR at same scale, with proportional task utility gain.

**Measurement**: Train on PROBA-V MISR (multi-frame) vs SISR (single frame); compare H2 metrics  
**Falsification**: MISR structural hallucination ≥ SISR OR task utility not improved  
**Compute**: PROBA-V training runs (additional)  
**Timeline**: Phase 6 (parallel)

---

## 4. Exploratory Hypotheses (P2/P3 — Conditional)

### H10: No-Reference Proxy for Model Selection
> **Claim**: Linear combination of SenseBench LVLM score + SCI + NIQE predicts Task Utility Score with R² > 0.8.

**Measurement**: Regression on H3/H4 data; requires SenseBench access  
**Falsification**: R² < 0.6  
**Compute**: Minimal (post-hoc analysis)  
**Timeline**: Phase 7-8 (if SenseBench available)

---

## 5. Hypothesis Testing Matrix Summary

| ID | Hypothesis | Primary Experiment | Compute | Data Required | Phase | Priority |
|----|------------|-------------------|---------|---------------|-------|----------|
| H1 | Degradation Fidelity | Degradation + EDSR-small | CPU + GPU | S2 HR, ALSAT specs, any real ALSAT | 3-4 | **P0** |
| H2 | Hallucination Taxonomy | Annotation + Clustering | GPU (inference) | S2 test, 4 models | 6 | **P0** |
| H3 | SCI Correlation | Metric vs. Task Study | GPU (tasks) | 500 SR outputs, 3 tasks | 7 | **P0** |
| H4 | Task/Hallucination Ratio | 4-Family Comparison | GPU (4× training) | S2 train/test | 5-6 | **P0** |
| H5 | Domain Gap Ordering | Cross-Sensor Generalization | GPU (inference) | S2, NAIP, Planet, Sim ALSAT | 6 | **P1** |
| H6 | Tiny Model Efficiency | Scaling Study | CPU + GPU | S2 train/test | 4-5 | **P0** |
| H7 | Task-Dependent Hallucination | MS vs PAN Comparison | GPU (inference) | S2 (both tasks) | 6 | **P1** |
| H8 | Physics-Informed Loss | Ablation | GPU (2× training) | S2 train | 5 | **P2** |
| H9 | MISR vs SISR | PROBA-V Ablation | GPU (training) | PROBA-V MISR | 6 | **P2** |
| H10 | NR Proxy | Regression | CPU | H3/H4 data, SenseBench | 7-8 | **P3** |

---

## 6. Falsification Tracking (Pre-Registered)

| Hypothesis | Falsified If | Status |
|------------|--------------|--------|
| H1 | FID > 20 or PSNR gap > 1.5 dB or KS > 0.25 (≥2 bands) | Pending |
| H2 | Silhouette < 0.3 or κ < 0.5 or no cluster >50% purity | Pending |
| H3 | SCI ρ < 0.5 or PSNR ρ > SCI ρ (≥2 tasks) | Pending |
| H4 | SwinIR not top-1 ratio; GAN/Diffusion wins (p<0.05) | Pending |
| H5 | Ordering violated (≥2 domains); CNN best | Pending |
| H6 | Ratio < 0.85 or FLOPs ratio > 0.15 or CPU > 2s | Pending |
| H7 | Spectral not dominant in MS→MS; Structural not in PAN | Pending |
| H8 | SCI ↓ < 10% or mIoU ↑ < 1% or PSNR ↓ > 0.5 dB | Pending |
| H9 | MISR structural hallucination ≥ SISR | Pending |
| H10 | R² < 0.6 | Pending |

---

## 7. Approval Required

This document defines the finalized, prioritized hypotheses for Phase 2. **Human approval is required** before proceeding to freeze the experimental protocol.

**Please confirm**:
- [ ] P0 hypotheses (H1-H6) are the core thesis claims
- [ ] Falsification criteria are explicit, measurable, and acceptable
- [ ] Priority ranking reflects feasibility and scientific value
- [ ] P1/P2/P3 hypotheses are correctly scoped as conditional
- [ ] Ready to proceed to `final_experimental_protocol.md`