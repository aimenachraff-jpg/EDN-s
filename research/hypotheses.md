# Research Hypotheses

## 1. Hypothesis Structure

Each hypothesis follows: **H{id}: [Specific, testable claim] — [Measurement protocol] — [Success criterion]**

---

## 2. Primary Hypotheses

### H1: ALSAT-2B Degradation Model Fidelity
> **H1**: A Wald-protocol degradation model using estimated NAOMI MTF (Gaussian, σ≈0.35 pixels at Nyquist) and trapezoidal SRFs from published band edges produces synthetic LR data that is statistically indistinguishable from real ALSAT-2B data in terms of: (a) power spectral density, (b) band-to-band correlation structure, (c) noise statistics (SNR>100, 10-bit quantization).
>
> **Measurement**: 
> - Generate synthetic LR from Sentinel-2 10m HR (Algeria AOIs)
> - Compare PSD, correlation matrices, noise distributions to *any available* ALSAT-2B samples (even few)
> - Fréchet Inception Distance (FID) on patch distributions
> - Train EDSR-small on synthetic vs. real (if real available) — compare PSNR on held-out real
>
> **Success Criterion**: FID < 10 between synthetic and real patch distributions; PSNR gap < 0.5 dB.

### H2: RS-SR Hallucination Taxonomy Validity
> **H2**: Hallucinations in RS-SR cluster into **four measurable categories** with distinct spectral/spatial signatures:
> 1. **Spectral Hallucination** — Violates physical band correlations (SAM > 10°, NDVI out-of-range)
> 2. **Structural Hallucination** — High-frequency energy in homogeneous regions (wavelet HH band > 3σ above local mean)
> 3. **Textural Hallucination** — Plausible but ungrounded periodic patterns (GLCM entropy mismatch > 20%)
> 4. **Semantic Hallucination** — Downstream classifier/detector confidence on false structures (mAP drop > 15% vs. HR)
>
> **Measurement**:
> - Generate SR outputs from 4 model families (CNN, Transformer, GAN, Diffusion) on S2 test set
> - Annotate 200 patches per model with hallucination type (expert labeling)
> - Compute proposed metrics for each type
> - Cluster analysis (t-SNE/UMAP) on metric vectors
>
> **Success Criterion**: Silhouette score > 0.5 for 4 clusters; expert agreement (Fleiss' κ) > 0.7.

### H3: Physics-Informed Metrics Correlate with Task Utility
> **H3**: The proposed **Spectral Consistency Index (SCI)** — mean SAM across band pairs weighted by vegetation/water sensitivity — correlates with downstream task performance (ρ > 0.7) better than PSNR (ρ < 0.4) or LPIPS (ρ < 0.5).
>
> **Measurement**:
> - Compute SCI, PSNR, SSIM, LPIPS, NIQE, BRISQUE on 500 SR outputs (varying models, scales, degradations)
> - Evaluate on 3 downstream tasks: land cover classification (mIoU), building detection (mAP), change detection (F1)
> - Spearman correlation between each metric and each task metric
>
> **Success Criterion**: SCI ρ > 0.7 with at least 2/3 tasks; SCI outperforms PSNR/SSIM/LPIPS on all 3.

### H4: Transformer Models Maximize Task Utility / Hallucination Ratio
> **H4**: For RS-SR at ×4 scale on Sentinel-2 data, **SwinIR (base) achieves the highest downstream task utility per unit of hallucination** among CNN (EDSR-small), Transformer (SwinIR), GAN (Real-ESRGAN), and Diffusion (StableSR) families.
>
> **Measurement**:
> - Train all 4 model families on identical S2 training data (Algeria AOIs, ×4, MS 20m→10m)
> - Evaluate on held-out test: PSNR, SSIM, LPIPS, SCI (H3), Structural Hallucination Score (H2)
> - Downstream: Fixed-backbone classification (mIoU), detection (mAP), segmentation (mIoU)
> - Compute **Task Utility Score** = mean(mIoU_class, mAP_det, mIoU_seg)
> - Compute **Hallucination Index** = weighted sum of H2 metrics
> - Ratio = Task Utility / (1 + Hallucination Index)
>
> **Success Criterion**: SwinIR ratio > 1.2× second-best; statistical significance (bootstrap 95% CI).

### H5: Cross-Sensor Domain Gap Severity: Diffusion > GAN > Transformer > CNN
> **H5**: When trained on Sensor A (Sentinel-2) and tested on Sensor B (NAIP/PlanetScope/Simulated ALSAT-2B), the **relative performance drop** (ΔPSNR, ΔTask Utility) follows: Diffusion > GAN > Transformer > CNN.
>
> **Measurement**:
> - Train each model family on S2 (source)
> - Test on: (a) NAIP 1m (cross-sensor, ×10), (b) PlanetScope 3m (×3), (c) Simulated ALSAT-2B (×4)
> - Measure ΔPSNR = PSNR_on_source_domain - PSNR_on_target_domain
> - Measure ΔTask Utility on target domain
>
> **Success Criterion**: Monotonic ordering holds for ≥2/3 target domains; p < 0.05 (paired t-test).

### H6: Tiny Models Reach >90% Task Utility of Large Models at <10% Compute
> **H6**: EDSR-small (1.5M params) and Tiny SwinIR (depth=2, ~150K params) achieve >90% of the **Task Utility Score** (H4) of their large counterparts (EDSR-baseline 43M, SwinIR-base 11.9M) while using <10% FLOPs and enabling CPU inference (<1s per 256² patch).
>
> **Measurement**:
> - Train small/large pairs for CNN and Transformer
> - Measure: Task Utility Score (H4), FLOPs (thop), CPU inference time (Intel Iris Xe)
> - Compute ratio: Task_Utility_small / Task_Utility_large
>
> **Success Criterion**: Ratio > 0.9 for both pairs; CPU time < 1s.

---

## 3. Secondary / Exploratory Hypotheses

### H7: Spectral Hallucination Dominates in MS SR; Structural in PAN-Sharpening
> **H7**: In MS→MS super-resolution (S2 20m→10m), **spectral hallucination** (SAM, band correlation) is the primary failure mode. In PAN-sharpening simulation (S2 10m→2.5m PAN), **structural hallucination** (wavelet high-freq in smooth areas) dominates.

### H8: Hallucination Suppression via Physics-Informed Loss Improves Task Utility
> **H8**: Adding a **Spectral Consistency Loss** (L_spectral = mean SAM between SR and HR band pairs) to SwinIR training reduces spectral hallucination (SCI ↓ 20%) and improves classification mIoU (↑ 3-5%) without PSNR degradation.

### H9: MISR Reduces Hallucination vs. SISR for Same Model
> **H9**: For SwinIR/EDSR, multi-image SR (PROBA-V style, 5-9 frames) reduces structural hallucination (wavelet metric ↓ 30%) compared to single-image SR at same scale, with proportional task utility gain.

### H10: No-Reference Metrics Can Proxy Task Utility for Model Selection
> **H10**: A linear combination of **SenseBench LVLM score** + **SCI** + **NIQE** predicts Task Utility Score (H4) with R² > 0.8, enabling model selection without downstream task training.

---

## 4. Hypothesis Testing Matrix

| Hypothesis | Primary Experiment | Compute | Data Required | Timeline |
|------------|-------------------|---------|---------------|----------|
| H1 | Degradation modeling + stats | CPU | S2 HR, ALSAT specs | Phase 3 |
| H2 | Hallucination annotation + clustering | CPU + GPU (inference) | S2 test, 4 models | Phase 6 |
| H3 | Metric correlation study | GPU (tasks) | S2 test, downstream tasks | Phase 7 |
| H4 | Model family comparison | GPU (4× training) | S2 train/test | Phase 5-6 |
| H5 | Cross-sensor generalization | GPU (4× training) | S2, NAIP, Planet, sim ALSAT | Phase 6 |
| H6 | Model scaling study | CPU + GPU | S2 train/test | Phase 4-5 |
| H7 | MS vs PAN-sharpening comparison | GPU | S2 (both tasks) | Phase 6 |
| H8 | Physics-informed loss ablation | GPU | S2 train | Phase 5 |
| H9 | MISR vs SISR ablation | GPU | PROBA-V MISR | Phase 6 |
| H10 | NR metric proxy validation | GPU (tasks) | S2 test, SenseBench | Phase 7 |

---

## 5. Falsification Criteria (Critical)

| Hypothesis | Falsified If |
|------------|--------------|
| H1 | FID > 20 or PSNR gap > 1.5 dB with any real ALSAT-2B data |
| H2 | Silhouette < 0.3 or κ < 0.5 — categories not distinct |
| H3 | SCI ρ < 0.5 or PSNR ρ > SCI ρ on ≥2 tasks |
| H4 | SwinIR not top-1 in ratio; GAN or Diffusion wins |
| H5 | Ordering violated on ≥2 domains; CNN worst |
| H6 | Ratio < 0.85 or CPU time > 2s |
| H7 | Spectral hallucination not dominant in MS SR |
| H8 | SCI loss degrades PSNR > 0.5 dB or no mIoU gain |
| H9 | MISR hallucination ≥ SISR |
| H10 | R² < 0.6 for NR proxy model |

---

## 6. Priority Ranking

| Priority | Hypotheses | Rationale |
|----------|------------|-----------|
| **P0 (Must)** | H1, H4, H6 | Core thesis claims; feasible; high impact |
| **P1 (Should)** | H2, H3, H5 | Hallucination taxonomy + task correlation + domain gap |
| **P2 (Could)** | H7, H8, H9 | Mechanistic insights; ablation studies |
| **P3 (Exploratory)** | H10 | NR proxy — nice to have |

---

*All hypotheses are testable with the proposed dataset stack (Sentinel-2 primary, PROBA-V MISR, NAIP/Planet cross-sensor, simulated ALSAT-2B) and model families (EDSR, SwinIR, Real-ESRGAN, StableSR). Falsification criteria are explicit to maintain scientific integrity.*