# Research Gap Analysis

## 1. Gap Synthesis from Literature Review

Based on 267 papers (2020-2026) and the specific focus on **ALSAT-2B / NAOMI** super-resolution with **hallucination-aware** and **task-driven** evaluation, the following gaps are identified:

---

## 2. Primary Research Gaps

### Gap 1: ALSAT-2B Super-Resolution is Unexplored
**Evidence**: Zero papers on ALSAT-2B SR in 267-paper collection. ALSAT-2B mentioned only in mission descriptions.
**Impact**: No baseline methods, no degradation models, no evaluation protocols for this sensor.
**Opportunity**: First systematic SR study for NAOMI instrument — defines the sensor-specific problem.

### Gap 2: No Unified Hallucination Taxonomy for Remote Sensing SR
**Evidence**: 
- 32 papers on "hallucination + remote sensing" — mostly MLLM reasoning, not SR
- 110 papers on "hallucination + super-resolution" — natural image terminology (texture, structure)
- RS-specific types (spectral, cross-sensor) absent from taxonomies
**Impact**: Cannot systematically compare hallucination across models/datasets/tasks.
**Opportunity**: Define RS-SR hallucination taxonomy with measurable categories.

### Gap 3: Hallucination Detection Metrics Not Standardized for RS
**Evidence**:
- VHA (2610.25770): VQA for generative SR — natural images
- SenseBench (2605.10576): LVLM perception benchmark — not a metric
- Spectral metrics (SAM, ERGAS) used for pansharpening quality, not hallucination
- No "hallucination score" adopted in RS-SR papers
**Impact**: Hallucination reported anecdotally ("visual inspection"), not quantified.
**Opportunity**: Develop physics-informed + task-driven hallucination metrics.

### Gap 4: Task-Driven Evaluation Limited in Scope
**Evidence**:
- GeoSR-Bench (2026): 4 tasks, limited sensors, no hallucination correlation
- Spatial Downscaling (2610.23972): Coherence/structure/utility but no model family comparison
- Most RS-SR papers (200+) still report only PSNR/SSIM/SAM
**Impact**: No evidence on which SR models *actually* help/harm downstream tasks.
**Opportunity**: Systematic task-driven evaluation across model families on same data.

### Gap 5: Cross-Sensor Domain Gap Studied Only for Diffusion
**Evidence**:
- "Mind the Gap" (2606.28039): Diffusion cross-sensor domain gap only
- Real-ESRGAN uses synthetic degradation but no systematic cross-sensor study
- No comparison: CNN vs Transformer vs GAN vs Diffusion on same cross-sensor task
**Impact**: Cannot choose model family for operational cross-sensor SR.
**Opportunity**: Cross-family domain gap analysis with ALSAT-2B simulation.

### Gap 6: Computational Budget vs. Hallucination Trade-off Unmapped
**Evidence**: No paper compares tiny (CPU) vs large (GPU) models on hallucination profile.
**Impact**: Operational deployment (edge/onboard) lacks guidance.
**Opportunity**: Map hallucination vs compute for model selection.

---

## 3. Secondary Gaps (Supporting)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Spectral hallucination metric** | No standard metric for band-correlation violation | High |
| **Temporal hallucination in MISR** | Hallucination across time series unstudied | Medium |
| **Uncertainty quantification for RS-SR** | Bayesian/ensemble methods rare in RS-SR | Medium |
| **Onboard/edge SR constraints** | Compute/power limits not addressed | Low (future) |
| **SAR-optical SR hallucination** | Cross-modality hallucination | Low (out of scope) |

---

## 4. Gap-to-Research-Question Mapping

| Gap | Research Question | Hypothesis |
|-----|-------------------|------------|
| G1 | How to model ALSAT-2B degradation for SR? | Wald protocol with estimated MTF/SRF yields realistic synthetic LR |
| G2 | What hallucination types occur in RS-SR? | Spectral & structural hallucination dominate; semantic is task-dependent |
| G3 | How to detect/quantify RS-SR hallucination? | Physics-informed metrics (spectral consistency) + task performance drop correlate |
| G4 | Which SR models maximize downstream task utility? | **Transformer (SwinIR) > CNN (EDSR) > GAN (Real-ESRGAN) > Diffusion** for task utility |
| G5 | How does cross-sensor domain gap vary by model family? | Diffusion > GAN > Transformer > CNN for domain gap severity |
| G6 | Can tiny models achieve acceptable task utility with low hallucination? | EDSR-small / Tiny SwinIR reach >90% task utility of large models with <10% compute |

---

## 5. Feasibility Assessment per Gap

| Gap | Data Available | Methods Available | Compute Needed | Risk |
|-----|----------------|-------------------|----------------|------|
| G1 (ALSAT-2B model) | S2 as proxy + sensor specs | Wald protocol, MTF estimation | Low (CPU) | Low |
| G2 (Taxonomy) | S2, PROBA-V, NAIP | Visual analysis + spectral stats | Low | Low |
| G3 (Metrics) | Same + downstream tasks | Custom metrics + existing (SAM, LPIPS) | Medium (GPU for tasks) | Medium |
| G4 (Task eval) | GeoSR-Bench tasks + custom | Fixed-backbone protocol | High (GPU) | Medium |
| G5 (Domain gap) | S2→NAIP, S2→Planet, sim ALSAT | Cross-sensor training | High (GPU) | High |
| G6 (Compute trade-off) | All models at multiple sizes | Model scaling | Medium | Low |

---

## 6. Scope Decision (What This Thesis Will NOT Address)

| Excluded | Reason |
|----------|--------|
| Real ALSAT-2B data acquisition | Access blocked; 4-week decision gate |
| Onboard/edge deployment | Out of scope (Master 2 research focus) |
| SAR-optical fusion | Different modality |
| Video SR / temporal SR | MISR only, not video |
| Hyperspectral SR | ALSAT-2B is multispectral (4 bands) |
| Training diffusion models from scratch | Compute budget; use pretrained if at all |

---

## 7. Conclusion: Research Focus Confirmed

**Core Contribution**: First hallucination-aware, task-driven evaluation framework for ALSAT-2B super-resolution, validated on Sentinel-2 proxy with cross-sensor generalization analysis.

**Three Pillars**:
1. **Sensor-Specific Degradation Modeling** (ALSAT-2B/NAOMI simulation)
2. **Hallucination Taxonomy + Detection Metrics** (RS-specific, physics-informed)
3. **Task-Driven Model Selection** (Downstream utility > PSNR/SSIM)

This addresses Gaps 1-4 directly; Gaps 5-6 as secondary analyses.