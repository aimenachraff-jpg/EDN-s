# Final Research Questions

**Project**: MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B Satellite Imagery  
**Phase**: 2 — Research Questions & Methodology Finalization  
**Date**: 2025-11-12  
**Status**: FROZEN FOR APPROVAL

---

## 1. Overarching Research Question

> **RQ0**: *How can super-resolution methods for ALSAT-2B satellite imagery be evaluated to ensure that reconstructed high-resolution outputs are both scientifically faithful (minimal hallucination) and practically useful for downstream Earth observation tasks?*

This question integrates two critical dimensions often treated separately in remote sensing super-resolution (RS-SR):
- **Hallucination-awareness**: Detecting and quantifying physically implausible or ungrounded details in SR outputs
- **Task-driven evaluation**: Measuring SR quality by its impact on real downstream applications (classification, detection, segmentation, change detection)

---

## 2. Primary Research Questions (P0 — Must Answer)

### RQ1: Degradation Model Fidelity
> **RQ1**: *Can a Wald-protocol degradation model using estimated NAOMI instrument parameters (MTF, SRF, noise, quantization) produce synthetic low-resolution data that is statistically indistinguishable from real ALSAT-2B imagery?*

**Motivation**: ALSAT-2B data access is blocked. A validated degradation model enables controlled, reproducible experiments simulating ALSAT-2B's NAOMI sensor characteristics without requiring real data. This is foundational for H1.

**Variables**:
- **Independent**: MTF kernel (Gaussian σ), SRF shape (trapezoidal edges), noise model (Poisson-Gaussian SNR), quantization bits, inter-band registration error
- **Dependent**: Power spectral density (PSD) match, band-to-band correlation matrix match, noise statistics match, Fréchet Inception Distance (FID) between synthetic and real patch distributions, downstream SR model PSNR gap

**Required Data**: Sentinel-2 10m HR (Algeria AOIs), estimated NAOMI parameters, any available ALSAT-2B samples for validation (even few)

**Required Models**: EDSR-small (lightweight proxy for degradation validation)

**Evaluation Metrics**: Radial PSD comparison, 4×4 band correlation matrices, noise vs. radiance curves, FID (Inception-v3), PSNR on held-out real data

**Statistical Test**: Two-sample Kolmogorov-Smirnov test for PSD distributions; Frobenius norm of correlation matrix difference; bootstrap 95% CI for FID

**Falsification**: FID > 20 or PSNR gap > 1.5 dB with any real ALSAT-2B data

**Expected Contribution**: First validated ALSAT-2B/NAOMI degradation model; establishes credibility of all subsequent simulated experiments

---

### RQ2: RS-SR Hallucination Taxonomy
> **RQ2**: *Do hallucinations in remote sensing super-resolution cluster into four distinct, measurable categories (Spectral, Structural, Textural, Semantic), and can each be quantified with validated metrics?*

**Motivation**: Current hallucination literature in RS-SR borrows terminology from natural images (texture, structure) without RS-specific types (spectral, cross-sensor). A unified taxonomy enables systematic comparison across models, datasets, and tasks.

**Variables**:
- **Independent**: Model family (CNN, Transformer, GAN, Diffusion), scale factor (×2, ×4), degradation type, training data
- **Dependent**: Spectral Angle Mapper (SAM), band correlation deviation, NDVI consistency, wavelet high-frequency energy in homogeneous regions, GLCM entropy mismatch, downstream task performance drop

**Required Data**: Sentinel-2 test set (8 AOIs), SR outputs from 4 model families

**Required Models**: EDSR-small, SwinIR-base, Real-ESRGAN, StableSR (if compute allows)

**Evaluation Metrics**: 
- Spectral: SAM (mean), band correlation deviation, NDVI consistency
- Structural: Wavelet HH-band energy in homogeneous regions
- Textural: GLCM entropy difference
- Semantic: Task performance drop (classification mIoU, detection mAP, segmentation mIoU)

**Statistical Test**: Silhouette score for 4-cluster separation; Fleiss' κ for expert annotation agreement (3 annotators, 200 patches × 4 models)

**Falsification**: Silhouette < 0.3 or κ < 0.5 — categories not distinct or measurable

**Expected Contribution**: First RS-specific hallucination taxonomy with operationalized, validated metrics; enables H3, H4

---

### RQ3: Physics-Informed Metrics vs. Task Utility
> **RQ3**: *Does the proposed Spectral Consistency Index (SCI) — a physics-informed metric combining band-wise SAM weighted by vegetation/water sensitivity — correlate with downstream task performance (ρ > 0.7) better than standard metrics (PSNR, SSIM, LPIPS)?*

**Motivation**: PSNR/SSIM correlate poorly with semantic quality in RS-SR. LPIPS is trained on natural images. A physics-informed metric designed for multispectral data should better predict task utility.

**Variables**:
- **Independent**: SR model outputs spanning diverse quality levels (4 families × scales × degradations)
- **Dependent**: SCI, PSNR, SSIM, LPIPS, NIQE, BRISQUE vs. downstream task metrics (classification mIoU, detection mAP, segmentation mIoU, change detection F1)

**Required Data**: 500+ SR outputs with corresponding HR references, downstream task models trained on HR

**Required Models**: Fixed-backbone task models (UNet for classification, RT-DETR for detection, DeepLabV3+ for segmentation, Siamese UNet for change detection)

**Evaluation Metrics**: Spearman ρ between each IQA metric and each task metric

**Statistical Test**: Spearman correlation with permutation test (n=10,000); Fisher's z-test for comparing dependent correlations

**Falsification**: SCI ρ < 0.5 or PSNR ρ > SCI ρ on ≥2 of 3 primary tasks

**Expected Contribution**: Validated physics-informed metric for RS-SR model selection without downstream training

---

### RQ4: Model Family Comparison — Task Utility / Hallucination Ratio
> **RQ4**: *Which super-resolution model family (CNN, Transformer, GAN, Diffusion) maximizes downstream task utility per unit of hallucination for Sentinel-2 MS super-resolution at ×4 scale?*

**Motivation**: Model selection in RS-SR typically uses PSNR/SSIM. This ignores hallucination, which may degrade downstream tasks differently across model families. A ratio metric balances utility and faithfulness.

**Variables**:
- **Independent**: Model family (EDSR-small, SwinIR-base, Real-ESRGAN, StableSR), training configuration
- **Dependent**: Task Utility Score (mean normalized mIoU_class, mAP_det, mIoU_seg), Hallucination Index (weighted composite of H2 metrics), Ratio = Task Utility / (1 + Hallucination Index)

**Required Data**: Sentinel-2 train/val/test (Algeria AOIs, temporal splits)

**Required Models**: All 4 core model families trained identically

**Evaluation Metrics**: Task Utility Score, Hallucination Index, Ratio; bootstrap 95% CI for ratio differences

**Statistical Test**: Paired bootstrap (1000 resamples) for ratio comparison; Benjamini-Hochberg FDR correction across model pairs

**Falsification**: SwinIR not top-1 in ratio; GAN or Diffusion wins with statistical significance

**Expected Contribution**: Evidence-based model family recommendation for RS-SR practitioners balancing quality and faithfulness

---

### RQ5: Cross-Sensor Domain Gap Severity by Model Family
> **RQ5**: *When trained on Sentinel-2 and tested on target sensors (NAIP, PlanetScope, simulated ALSAT-2B), does the relative performance drop follow the ordering: Diffusion > GAN > Transformer > CNN?*

**Motivation**: Domain generalization is critical for operational RS-SR. Prior work shows diffusion models suffer larger domain gaps, but systematic cross-family comparison on satellite data is lacking.

**Variables**:
- **Independent**: Source domain (Sentinel-2), target domain (NAIP 1m, PlanetScope 3m, Simulated ALSAT-2B 2.5m PAN), model family
- **Dependent**: ΔPSNR (source - target), ΔTask Utility (source - target), FID between source/target feature distributions, spectral shift (band correlation deviation)

**Required Data**: Sentinel-2 (source training), NAIP (US AOIs), PlanetScope (academic license pending), Simulated ALSAT-2B (degradation model)

**Required Models**: All 4 families trained on Sentinel-2 only

**Evaluation Metrics**: ΔPSNR, ΔTask Utility, FID, spectral shift

**Statistical Test**: Paired t-test for monotonic ordering across ≥2/3 target domains; effect sizes (Cohen's d)

**Falsification**: Ordering violated on ≥2 domains; CNN worst performer

**Expected Contribution**: Cross-sensor generalization ranking to guide sensor-agnostic SR model deployment

---

### RQ6: Tiny Model Efficiency for Operational Deployment
> **RQ6**: *Can tiny models (EDSR-small 1.5M params, Tiny SwinIR ~150K params) achieve >90% of the Task Utility Score of their large counterparts (EDSR-baseline 43M, SwinIR-base 11.9M) at <10% FLOPs with CPU inference <1s per 256² patch?*

**Motivation**: Operational RS-SR often runs on edge/CPU infrastructure. Understanding the utility/compute trade-off informs practical deployment decisions.

**Variables**:
- **Independent**: Model size variant (small vs. large) for CNN and Transformer families
- **Dependent**: Task Utility Score (from RQ4), FLOPs (thop), CPU inference time (Intel Iris Xe), parameter count

**Required Data**: Sentinel-2 train/test (same as RQ4)

**Required Models**: EDSR-small vs EDSR-baseline; Tiny SwinIR vs SwinIR-base

**Evaluation Metrics**: Task Utility ratio (small/large), FLOPs ratio, CPU time

**Statistical Test**: Bootstrap 95% CI for Task Utility ratio; direct measurement of FLOPs and CPU time

**Falsification**: Ratio < 0.85 for either pair; CPU time > 2s

**Expected Contribution**: Practical deployment guidelines for resource-constrained RS-SR

---

## 3. Secondary Research Questions (P1 — Should Answer)

### RQ7: Hallucination Type by SR Task
> **RQ7**: *In MS→MS super-resolution (S2 20m→10m), is spectral hallucination the dominant failure mode, while in PAN-sharpening simulation (S2 10m→2.5m), does structural hallucination dominate?*

**Motivation**: Different SR tasks have different physical constraints; understanding dominant failure modes guides targeted mitigation.

**Variables**: SR task type (MS upscaling vs. PAN-sharpening), model family

**Expected Contribution**: Task-specific hallucination profiles

---

### RQ8: Physics-Informed Loss for Hallucination Suppression
> **RQ8**: *Does adding a Spectral Consistency Loss (mean SAM between SR and HR band pairs) to SwinIR training reduce spectral hallucination (SCI ↓ 20%) and improve classification mIoU (↑ 3-5%) without PSNR degradation?*

**Motivation**: If physics-informed losses can suppress hallucination while improving task utility, this provides a practical training strategy.

**Variables**: Loss function (L1 vs. L1 + λ·L_spectral)

**Expected Contribution**: Training recipe for hallucination-aware RS-SR

---

### RQ9: MISR vs. SISR Hallucination Reduction
> **RQ9**: *For SwinIR/EDSR, does multi-image SR (PROBA-V style, 5-9 frames) reduce structural hallucination (wavelet metric ↓ 30%) compared to single-image SR at the same scale, with proportional task utility gain?*

**Motivation**: Multi-image SR has more information; quantifying its hallucination reduction vs. SISR informs data acquisition strategy.

**Variables**: Number of input frames (1 vs. 5-9), model family

**Expected Contribution**: Evidence for MISR adoption in operational RS-SR

---

## 4. Exploratory Research Questions (P2/P3 — Could Answer)

### RQ10: No-Reference Proxy for Model Selection
> **RQ10**: *Can a linear combination of SenseBench LVLM score + SCI + NIQE predict Task Utility Score with R² > 0.8, enabling model selection without downstream task training?*

**Motivation**: Downstream task evaluation is expensive. A no-reference proxy would enable rapid model screening.

**Variables**: NR metrics (SenseBench, SCI, NIQE, BRISQUE, etc.)

**Expected Contribution**: Practical model selection tool for RS-SR practitioners

---

## 5. Summary: Hypothesis Mapping

| Research Question | Primary Hypothesis | Priority | Feasibility |
|-------------------|-------------------|----------|-------------|
| RQ1 | H1: Degradation Model Fidelity | P0 | ✅ High (CPU + simulated) |
| RQ2 | H2: Hallucination Taxonomy | P0 | ✅ High (needs annotation) |
| RQ3 | H3: SCI Correlation | P0 | ✅ High (compute-intensive) |
| RQ4 | H4: Task Utility / Hallucination Ratio | P0 | ✅ High (core experiment) |
| RQ5 | H5: Domain Gap Ordering | P1 | ✅ High (needs NAIP/Planet) |
| RQ6 | H6: Tiny Model Efficiency | P0 | ✅ High (local CPU validation) |
| RQ7 | H7: Task-Dependent Hallucination | P1 | ⚠️ Medium (additional exps) |
| RQ8 | H8: Physics-Informed Loss | P2 | ⚠️ Medium (ablation) |
| RQ9 | H9: MISR Hallucination Reduction | P2 | ✅ High (PROBA-V available) |
| RQ10 | H10: NR Proxy | P3 | ⚠️ Low (SenseBench access) |

---

## 6. Scope Boundaries (Explicit Exclusions)

The following are **NOT** in scope for this Master 2 thesis:

1. **Real ALSAT-2B data acquisition** — blocked; substitute strategy validated
2. **Training diffusion models (StableSR)** — conditional on compute; not required for core claims
3. **Novel model architecture design** — using established architectures only
4. **PlanetScope data** — only if academic license granted in time; NAIP + SimALSAT sufficient
4. **Annotator recruitment beyond 3 experts** — pilot validates feasibility
5. **Operational deployment system** — research evaluation only
6. **Cross-sensor training (domain adaptation)** — only cross-sensor *testing* (generalization)
7. **Spectral super-resolution beyond 4 bands** — ALSAT-2B MS is 4 bands
8. **Video/temporal super-resolution** — single-date SR only
9. **Comprehensive hyperparameter optimization** — standard configs from literature
10. **Comparison to >4 model families** — 4 families sufficient for cross-family claims

---

## 7. Approval Required

This document defines the finalized research questions for Phase 2. **Human approval is required** before proceeding to freeze the experimental protocol.

**Please confirm**: 
- [ ] Research questions are appropriately scoped for Master 2
- [ ] Priority ranking reflects feasibility and impact
- [ ] Falsification criteria are explicit and acceptable
- [ ] Exclusions are correctly identified
- [ ] Ready to proceed to `final_hypotheses.md` and `final_experimental_protocol.md`