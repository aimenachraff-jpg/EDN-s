# Methodology: Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B

## 1. Overall Research Design

### 1.1 Approach
**Empirical, experimental, reproducible** — all claims backed by executed code and measured results.

### 1.2 Phases Mapping to Methodology
| Phase | Methodology Component | Deliverable |
|-------|----------------------|-------------|
| 3 | Dataset acquisition, preprocessing, degradation modeling | `src/data/`, `data/processed/`, `configs/degradation.yaml` |
| 4 | Baseline implementation (EDSR-small, Tiny SwinIR) | `src/models/`, `src/training/`, `experiments/phase4/` |
| 5 | Full model training (4 families × scales) | `experiments/phase5/`, trained checkpoints |
| 6 | Hallucination evaluation (taxonomy, metrics, detection) | `src/hallucination/`, `results/metrics/hallucination/` |
| 7 | Downstream task evaluation (classification, detection, segmentation) | `src/downstream/`, `results/metrics/downstream/` |
| 8 | Statistical analysis, cross-sensor generalization | `results/statistical_analysis/`, `docs/methodology_decisions.md` |

### 1.3 Reproducibility Pillars
1. **Version-controlled configs** (Hydra/OmegaConf) for all experiments
2. **Fixed seeds** (42) + deterministic CuDNN
3. **Containerized environment** (Docker/requirements.txt + git commit hashes)
4. **Artifact tracking** (MLflow local) for metrics, models, predictions
5. **Public code release** at thesis completion

---

## 2. Dataset Methodology

### 2.1 Primary Dataset: Sentinel-2 (Proxy for ALSAT-2B MS)

#### AOI Selection (Stratified by Land Cover)
| AOI ID | Location | Land Cover | Size | Purpose |
|--------|----------|------------|------|---------|
| `DZ_01` | Algiers, Algeria | Urban / Coastal | 100×100 km | Primary (target country) |
| `DZ_02` | Tell Atlas, Algeria | Agricultural / Forest | 100×100 km | Vegetation spectral fidelity |
| `DZ_03` | Sahara (Ghardaïa), Algeria | Desert / Bare soil | 100×100 km | Low-signal, high-reflectance |
| `DZ_04` | Oran Coast, Algeria | Coastal / Wetland | 100×100 km | Water/land boundary |
| `FR_01` | Paris, France | Dense Urban | 50×50 km | Validation: urban structures |
| `US_01` | Iowa, USA | Intensive Agriculture | 50×50 km | Validation: crop patterns |
| `CL_01` | Atacama, Chile | Hyper-arid Desert | 50×50 km | Validation: extreme spectra |
| `BR_01` | Amazon Mouth, Brazil | Tropical Coastal | 50×50 km | Validation: high biomass |

#### Temporal Sampling
- **Train**: 2018-2022 (S2 L2A mature processing baseline)
- **Val**: 2023
- **Test**: 2024 (held-out year)
- **Cloud Filter**: <20% for train/val (Sen2Cor SCL mask); **no filter for test** (realistic)

#### Band Selection
| Role | Bands | Resolution | Use |
|------|-------|------------|-----|
| **HR Target** | B2, B3, B4, B8 (Blue, Green, Red, NIR) | 10m | MS super-resolution target |
| **LR Input (×2)** | B5, B6, B7, B8a, B11, B12 | 20m | Native S2 MS upscaling |
| **LR Input (×4 sim)** | B2, B3, B4, B8 degraded to 40m | 40m | Simulated ALSAT-2B-like |
| **PAN Sim (×4)** | Synthetic PAN from B2-B8 (weighted sum) | 2.5m | Pansharpening simulation |

#### Data Format
- **Storage**: Cloud Optimized GeoTIFF (COG) per band per tile
- **Tiling**: 1024×1024 px (10m) / 512×512 px (20m) with 128px overlap
- **Normalization**: Per-band percentiles (2nd, 98th) → [0,1]; stored in `configs/normalization.yaml`
- **Metadata**: STAC catalog + CSV index (tile_id, date, cloud%, AOI, bounds)

### 2.2 Secondary Dataset: PROBA-V MISR

#### Source
- ESA Kelvin platform: `kelvins.esa.int/proba-v-super-resolution/`
- Proba-V-ref repurposed dataset (with HR references)

#### Processing
- LR stacks: 9-15 images per location (multi-angle, multi-temporal)
- HR reference: 100m (where available)
- Split: Official challenge split (train/val/test)
- Alignment: Sub-pixel registration verified

### 2.3 Cross-Sensor Datasets

| Dataset | Role | Acquisition |
|---------|------|-------------|
| **NAIP** | HR target for S2×10 | AWS S3 sync (US AOIs only) |
| **PlanetScope** | HR target for S2×3 | Academic license application |
| **Simulated ALSAT-2B** | Target domain for H5 | Degradation model (Sec 3) |

### 2.4 Data Splits & Leakage Prevention
- **Spatial**: No overlapping tiles between train/val/test
- **Temporal**: Year-based splits (no same-date leakage)
- **Scene**: No two tiles from same S2 granule in different splits
- **Verification**: Automated overlap check in `src/data/verify_splits.py`

---

## 3. Degradation Modeling (ALSAT-2B Simulation)

### 3.1 Wald Protocol Implementation
```python
# src/degradation/alsat2b_degradation.py
class ALSAT2BDegradation:
    def __init__(self, config):
        # MTF: Gaussian kernel estimated from spec (MTF@Nyquist > 0.15)
        self.mtf_pan = GaussianKernel(sigma=0.35)   # 2.5m PAN
        self.mtf_ms = GaussianKernel(sigma=0.35)    # 10m MS (same optics)
        
        # SRF: Trapezoidal from band edges (Table 1, dataset_investigation.md)
        self.srf = TrapezoidalSRF(band_edges=NAOMI_BAND_EDGES)
        
        # Noise: Poisson-Gaussian matching SNR>100 at reference radiance
        self.noise = PoissonGaussianNoise(snr_db=20)  # conservative
        
        # Quantization: 12-bit → 10-bit (downlink)
        self.quantize = Quantize(bits=10)
        
        # Geometric: <0.3 pixel inter-band registration error
        self.geometric = RandomShift(max_shift=0.3)
    
    def forward(self, hr_cube):  # hr_cube: [C, H, W] at 2.5m
        # 1. Spectral degradation (if HR has more bands)
        ms = self.srf.convolve(hr_cube)  # [4, H, W]
        
        # 2. Spatial degradation
        ms_lr = downsample(ms, scale=4, kernel=self.mtf_ms)  # 10m
        pan_lr = downsample(hr_cube[pan_idx], scale=1, kernel=self.mtf_pan)  # 2.5m
        
        # 3. Radiometric degradation
        ms_lr = self.noise(ms_lr)
        ms_lr = self.quantize(ms_lr)
        pan_lr = self.noise(pan_lr)
        pan_lr = self.quantize(pan_lr)
        
        # 4. Geometric
        ms_lr, pan_lr = self.geometric(ms_lr, pan_lr)
        
        return ms_lr, pan_lr
```

### 3.2 Parameter Estimation (Since No Real Data)
| Parameter | Source | Value | Uncertainty |
|-----------|--------|-------|-------------|
| MTF σ (PAN) | Spec: MTF@Nyquist>0.15, Gaussian fit | 0.35 px | ±0.05 |
| MTF σ (MS) | Same optics, larger pixel | 0.35 px | ±0.05 |
| SRF shape | Trapezoidal from band edges | Table 1 | Edges exact |
| SNR | Spec: >100 at reference | 20 dB (conservative) | Factor of 2 |
| Quantization | 12-bit → 10-bit downlink | 10 bits | Exact |
| Registration | Spec: <0.3 pixel | Uniform [-0.3, 0.3] | Exact |

### 3.3 Validation of Degradation Model (H1)
- **PSD Comparison**: Radial PSD of synthetic LR vs. any real ALSAT-2B
- **Band Correlation**: Correlation matrices (4×4) — synthetic vs. real
- **Noise Statistics**: Mean/variance vs. radiance curves
- **FID**: Patch-level distribution (Inception-v3 features)

---

## 4. Model Implementation Methodology

### 4.1 Unified Interface (`src/models/base.py`)
```python
class SRModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.scale = config.scale
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
    
    def forward(self, lr: Tensor) -> Tensor:  # lr: [B, C, H, W]
        raise NotImplementedError
    
    def flops(self, input_shape) -> int:  # for compute analysis
        raise NotImplementedError
```

### 4.2 Model Configurations (Hydra/OmegaConf)

#### EDSR-small (CNN Baseline)
```yaml
# configs/model/edsr_small.yaml
model:
  _target_: src.models.edsr.EDSR
  n_resblocks: 16
  n_feats: 64
  scale: 4
  in_channels: 4  # MS bands
  out_channels: 4
  res_scale: 1.0
  rgb_range: 1.0  # normalized [0,1]
```

#### SwinIR-base (Transformer Baseline)
```yaml
# configs/model/swinir_base.yaml
model:
  _target_: src.models.swinir.SwinIR
  upscale: 4
  img_size: 64
  window_size: 8
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  in_chans: 4
  out_chans: 4
```

#### Real-ESRGAN (GAN Baseline)
```yaml
# configs/model/realesrgan.yaml
model:
  _target_: src.models.realesrgan.RealESRGAN
  net_g:
    _target_: basicsr.archs.rrdbnet_arch.RRDBNet
    num_in_ch: 4
    num_out_ch: 4
    num_feat: 64
    num_block: 23
  scale: 4
```

### 4.3 Training Pipeline (`src/training/trainer.py`)
- **Framework**: PyTorch Lightning (or custom loop with Hydra)
- **Mixed Precision**: AMP (fp16) — mandatory for GPU memory
- **Gradient Accumulation**: Effective batch size 16-32
- **Checkpointing**: Best val PSNR + last + every N epochs
- **Logging**: MLflow (metrics, params, artifacts)

### 4.4 Loss Functions
| Model | Loss | Config |
|-------|------|--------|
| EDSR, SwinIR | L1 | `loss: l1` |
| SwinIR+Perceptual | L1 + 0.1×LPIPS | `loss: l1_lpips` |
| Real-ESRGAN | L1 + 1.0×Perceptual(VGG) + 0.1×GAN(Ra) | `loss: realesrgan` |
| StableSR | LDM loss (latent + pixel) | `loss: stablesr` |

---

## 5. Hallucination Evaluation Methodology

### 5.1 Taxonomy Operationalization (H2)

#### Spectral Hallucination Metrics
```python
# src/hallucination/spectral.py
def spectral_angle_mapper(sr, hr):  # [B, C, H, W]
    # SAM per pixel, mean over valid pixels
    dot = (sr * hr).sum(dim=1)
    norm_sr = sr.norm(dim=1)
    norm_hr = hr.norm(dim=1)
    sam = torch.acos(torch.clamp(dot / (norm_sr * norm_hr + 1e-8), -1, 1))
    return sam.mean().item()  # radians

def band_correlation_deviation(sr, hr):
    # Flatten spatial: [C, N]
    sr_flat = sr.flatten(1)
    hr_flat = hr.flatten(1)
    corr_sr = torch.corrcoef(sr_flat)
    corr_hr = torch.corrcoef(hr_flat)
    return (corr_sr - corr_hr).abs().mean().item()

def ndvi_consistency(sr, hr, red_idx=2, nir_idx=3):
    ndvi_sr = (sr[:, nir_idx] - sr[:, red_idx]) / (sr[:, nir_idx] + sr[:, red_idx] + 1e-8)
    ndvi_hr = (hr[:, nir_idx] - hr[:, red_idx]) / (hr[:, nir_idx] + hr[:, red_idx] + 1e-8)
    return (ndvi_sr - ndvi_hr).abs().mean().item()
```

#### Structural Hallucination Metrics
```python
# src/hallucination/structural.py
def wavelet_highfreq_energy(img, wavelet='db1', level=2):
    # DWT on each channel, mean |HH| energy in homogeneous regions
    # Homogeneous: local variance (3x3) < threshold
    pass

def glcm_entropy_mismatch(sr, hr):
    # Gray-Level Co-occurrence Matrix entropy difference
    pass
```

#### Semantic Hallucination Metrics (Downstream)
```python
# src/hallucination/semantic.py
def task_performance_drop(sr, hr, task_model, task_metric):
    # Fixed backbone protocol
    pred_sr = task_model(sr)
    pred_hr = task_model(hr)
    return task_metric(pred_hr, gt) - task_metric(pred_sr, gt)
```

### 5.2 Hallucination Index (Composite)
```python
# Weighted composite for H4 ratio
HALLUCINATION_INDEX = (
    0.3 * spectral_consistency_index +      # SCI (lower=better)
    0.3 * structural_hallucination_score +  # Wavelet/GLCM (lower=better)
    0.4 * semantic_hallucination_score      # Task drop (lower=better)
)
```

### 5.3 Annotation Protocol (For H2 Validation)
- **Annotators**: 3 domain experts (remote sensing)
- **Samples**: 200 patches × 4 models = 800 patches
- **Labels**: {None, Spectral, Structural, Textural, Semantic, Multiple}
- **Tool**: Custom labeling UI (Label Studio or CVAT)
- **Agreement**: Fleiss' κ

---

## 6. Downstream Task Evaluation Methodology

### 6.1 Tasks & Datasets

| Task | Dataset (HR) | Model | Metric | Protocol |
|------|--------------|-------|--------|----------|
| **Land Cover Classification** | S2 + EuroSAT labels / DynamicWorld | UNet (ResNet18 encoder) frozen | mIoU, OA, Kappa | Fixed backbone: train head on HR, eval on SR |
| **Building Detection** | S2 + OSM footprints / SpaceNet | RT-DETR / YOLOv8 | mAP@0.5, mAP@0.5:0.95 | Fixed backbone |
| **Semantic Segmentation** | S2 + LandCover.ai / custom | DeepLabV3+ (ResNet50) | mIoU | Fixed backbone |
| **Change Detection** | S2 time pairs + OSCD / custom | Siamese UNet | F1, IoU | Fixed backbone |

### 6.2 Evaluation Protocols

#### Protocol A: Fixed Backbone (Primary)
1. Train task model on **HR reference data** (S2 10m) → freeze weights
2. Evaluate on: HR (upper bound), SR outputs from each model
3. Measures: *Inherent SR quality for task* — no adaptation

#### Protocol B: Fine-Tune on SR (Secondary)
1. Train task model on **SR outputs** (from each SR model)
2. Evaluate on HR test set
3. Measures: *Adaptability* — can task model learn to ignore hallucinations?

#### Protocol C: Joint Training (Exploratory)
1. End-to-end SR + Task (gradient flows through SR to task loss)
2. Compare to two-stage

### 6.3 Task Utility Score (H4)
```python
TASK_UTILITY = np.mean([
    classification_miou / classification_miou_hr,  # normalized to HR=1
    detection_map / detection_map_hr,
    segmentation_miou / segmentation_miou_hr
])
```

---

## 7. Cross-Sensor Generalization Methodology (H5)

### 7.1 Source Domain: Sentinel-2 (Algeria AOIs)
- Train all 4 model families on S2 MS 20m→10m (×2) and 10m→2.5m PAN sim (×4)

### 7.2 Target Domains
| Target | Scale | Acquisition | Bands |
|--------|-------|-------------|-------|
| **NAIP** | ×10 (10m→1m) | AWS S3 (US AOIs) | RGBN |
| **PlanetScope** | ×3 (10m→3m) | Academic API | RGBN (4-band) |
| **Sim ALSAT-2B** | ×4 (10m→2.5m PAN) | Degradation model | PAN + 4 MS |

### 7.3 Domain Gap Metrics
- **ΔPSNR**: PSNR_on_S2_test - PSNR_on_target
- **ΔTask Utility**: Task_Utility_on_S2_test - Task_Utility_on_target
- **FID**: Feature distribution shift (Inception/ResNet features)
- **Spectral Shift**: Mean band correlation deviation

---

## 8. Statistical Analysis Methodology

### 8.1 Uncertainty Quantification
- **Bootstrap**: 1000 resamples of test patches → 95% CI for all metrics
- **Paired Tests**: Wilcoxon signed-rank (non-parametric) for model comparisons
- **Effect Size**: Cohen's d for pairwise differences

### 8.2 Multiple Comparison Correction
- **Benjamini-Hochberg** FDR control (q=0.05) across all model/metric combinations

### 8.3 Correlation Analysis (H3)
- **Spearman ρ** (non-linear monotonic relationships)
- **Permutation test** for significance (n=10000)

### 8.4 Reporting Standards
- Mean ± 95% CI (bootstrap)
- Median [IQR] for skewed distributions
- Effect sizes with CIs
- Exact p-values (not just <0.05)

---

## 9. Compute & Infrastructure

### 9.1 Local Development (CPU)
- **Device**: Intel Iris Xe (integrated)
- **Use**: Code development, tiny model smoke tests, data preprocessing, evaluation
- **Models**: EDSR-small, Tiny SwinIR (inference only)
- **Time Limit**: <30 min per run

### 9.2 Cloud Training (Kaggle Primary)
- **Platform**: Kaggle Notebooks (free, 30h/week, 2×T4 16GB)
- **Models**: All 4 families × scales
- **Parallelization**: 2 GPUs → 2 models simultaneously
- **Storage**: Kaggle datasets (S2 tiles), output to GCS or download

### 9.3 Experiment Tracking
```python
# MLflow local (Docker) or Kaggle-native
mlflow.set_experiment("master2-a1-phase5")
with mlflow.start_run(run_name="swinir_x4_s2_algeria"):
    mlflow.log_params(omegaconf_to_dict(cfg))
    mlflow.log_metrics({"val_psnr": val_psnr, "val_ssim": val_ssim})
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifacts("predictions/val")
```

---

## 10. Deliverables & Timeline Mapping

| Week | Phase | Key Deliverable |
|------|-------|-----------------|
| 1-2 | 3 | `src/data/` pipeline, S2 tiles downloaded, PROBA-V loaded, degradation model |
| 3-4 | 4 | `src/models/`, `src/training/`, EDSR-small + Tiny SwinIR trained & validated |
| 5-8 | 5 | All 4 model families trained (Kaggle), checkpoints saved |
| 9-10 | 6 | Hallucination taxonomy implemented, metrics computed, annotation done |
| 11-12 | 7 | Downstream tasks trained (HR), evaluated on all SR outputs |
| 13-14 | 8 | Statistical analysis, cross-sensor eval, figures/tables generated |
| 15-18 | 9 | Thesis writing (LaTeX), code cleanup, reproducibility package |

---

## 11. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ALSAT-2B data never accessible | High | Medium | Proxy validated; decision gate at Week 4 |
| Kaggle GPU quota exhausted | Medium | High | Colab Pro backup; RunPod budget |
| Hallucination annotation low agreement | Medium | Medium | Clear guidelines; pilot 20 samples first |
| Downstream task training unstable | Low | Medium | Fixed backbone protocol; pretrained weights |
| Compute budget overrun | Medium | Medium | Phase 4 tiny models first; scale up incrementally |

---

*This methodology is designed for full reproducibility. All configs, code, and random seeds are version-controlled. Cloud training uses free tiers with documented fallbacks.*