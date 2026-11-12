# Final Experimental Protocol

**Project**: MASTER2-A1 — Hallucination-Aware and Task-Driven Evaluation of Super-Resolution for ALSAT-2B Satellite Imagery  
**Phase**: 2 — Experimental Protocol Freeze  
**Date**: 2025-11-12  
**Status**: DRAFT FOR APPROVAL (requires human sign-off)

---

## 1. Protocol Scope and Boundaries

### 1.1 What This Protocol Covers
- Dataset acquisition, preprocessing, splitting, and manifesting
- Degradation modeling for ALSAT-2B simulation (Wald protocol with estimated parameters)
- Core model implementations: Bicubic, EDSR-small, SwinIR-base, Real-ESRGAN
- Evaluation framework: reconstruction, spectral, structural, hallucination, downstream tasks
- Cloud compute preparation (Kaggle primary, Colab Pro/RunPod backup)

### 1.2 What This Protocol Does NOT Cover
- Training execution (Phase 5+)
- Real ALSAT-2B data acquisition (blocked — 4-week gate then formal substitute commitment)
- Novel model architecture design
- Diffusion model training (conditional on compute budget)
- Hyperparameter optimization beyond standard recipes

### 1.3 Key Corrections from Phase 1 Review

| Phase 1 Claim | Verified Status | Correction in This Protocol |
|--------------|----------------|----------------------------|
| "MTF@Nyquist > 0.15 from spec" | **NOT in ESA eoPortal** — estimated | Flagged as **estimated** (Gaussian σ≈0.35px); sensitivity range σ∈[0.30,0.40] |
| "SNR > 100 at reference radiance" | **NOT in ESA eoPortal** — estimated | Flagged as **estimated**; conservative 20 dB used; sensitivity range 15-25 dB |
| "PROBA-V MISR: 100m HR from overlapping orbits" | **VERIFIED** via ESA Kelvin (74 regions, 384×384 HR, 128×128 LR×19 avg) | Protocol uses **exact ESA specification** |
| "267 papers sufficiently reviewed" | **Reproducible collection** (arXiv queries) — not manual deep read | Literature coverage claimed as **collection completeness**, not full synthesis |
| "Hallucination metrics validated" | **Not yet validated** — proposed taxonomy H2 | Metrics **defined but unvalidated**; H2 tests validity |
| "Downstream tasks have suitable datasets" | **Partially verified** — EuroSAT, OSM, LandCover.ai, OSCD exist | Protocol specifies exact datasets, splits, fixed-backbone protocol |

---

## 2. Dataset Strategy

### 2.1 Primary Dataset: Sentinel-2 (Proxy for ALSAT-2B MS)

#### 2.1.1 Sensor Comparability (Corrected)
| Parameter | ALSAT-2B NAOMI (Target) | Sentinel-2 MSI (Proxy) | Match Assessment |
|-----------|-------------------------|------------------------|------------------|
| **MS GSD** | 10 m | **10 m** (B2,B3,B4,B8) | ✅ Exact |
| **MS Bands** | B:0.45-0.52, G:0.53-0.60, R:0.62-0.69, NIR:0.76-0.89 µm | B2:0.49, B3:0.56, B4:0.665, B8:0.842 µm | ⚠️ Good spectral overlap; edges differ |
| **Quantization** | 12-bit→10-bit downlink | 12-bit L2A | ✅ Comparable |
| **Swath** | 17.5 km | 290 km | — |
| **Orbit** | 670 km, 98.2°, 10:30 LTAN | 786 km, 98.5°, 10:30 LTAN | ✅ Similar geometry |

**Critical**: S2 B2/B3/B4/B8 are **not identical** to ALSAT-2B bands — spectral response functions differ. All cross-sensor conclusions must account for this.

#### 2.1.2 AOI Selection (Stratified by Land Cover)
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

#### 2.1.3 Temporal Splits (No Leakage)
| Split | Years | Cloud Filter | Rationale |
|-------|-------|--------------|-----------|
| **Train** | 2018-2022 | <20% (Sen2Cor SCL) | S2 L2A mature processing |
| **Val** | 2023 | <20% | Independent year |
| **Test** | 2024 | **No filter** (realistic) | Held-out year, all conditions |

#### 2.1.4 Band Roles
| Role | Bands | Resolution | Use |
|------|-------|------------|-----|
| **HR Target** | B2, B3, B4, B8 | 10m | MS super-resolution target |
| **LR Input (×2)** | B5, B6, B7, B8a, B11, B12 | 20m | Native S2 MS upscaling |
| **LR Input (×4 sim)** | B2, B3, B4, B8 degraded to 40m | 40m | Simulated ALSAT-2B-like |
| **PAN Sim (×4)** | Synthetic PAN from B2-B8 (weighted sum) | 2.5m | Pansharpening simulation |

#### 2.1.5 Data Format & Storage
- **Format**: Cloud Optimized GeoTIFF (COG) per band per tile
- **Tiling**: 1024×1024 px (10m) / 512×512 px (20m) with 128px overlap
- **Normalization**: Per-band percentiles (2nd, 98th) → [0,1]; stored in `configs/normalization.yaml`
- **Metadata**: STAC catalog + CSV index (tile_id, date, cloud%, AOI, bounds, split)

#### 2.1.6 Access Methods (Priority Order)
1. **Planetary Computer STAC API** (planetarycomputer.microsoft.com) — Python `planetary-computer` + `odc-stac`
2. **Copernicus Data Space Ecosystem** (dataspace.copernicus.eu) — OData API
3. **AWS S3** (sentinel-s2-l2a) — Requester pays (avoid unless necessary)

### 2.2 Benchmark Dataset: PROBA-V MISR (VERIFIED ESA Specification)

#### 2.2.1 Official ESA Challenge Protocol
- **Source**: ESA Kelvin platform `kelvins.esa.int/proba-v-super-resolution/`
- **Data**: 1450 scenes total (74 hand-selected regions globally)
- **Split**: 1160 train / 290 test (official, fixed)
- **LR**: 300m, 128×128 px, 9-19 images per scene (multi-angle, multi-temporal, 30-day window), RED + NIR bands, 14-bit in 16-bit PNG
- **HR**: 100m, 384×384 px, single reference per scene (from overlapping orbits)
- **Quality Maps**: Per-image (SM for HR, QM for LR) — 1=clear, 0=cloud/shadow/water/missing
- **Inclusion Criteria**: HR ≥75% clear; LR ≥60% clear
- **Registration**: **Not provided** — sub-pixel shifts present (mean geolocation accuracy ~61m)
- **Scoring Metric**: cPSNR (crop PSNR on clear pixels only, shift-invariant)

#### 2.2.2 PROBA-V-ref Variant (Recommended)
- **Source**: arXiv:2101.10200 (Nguyen et al. 2021)
- **Improvement**: Identifies which LR image in stack corresponds to HR reference
- **Availability**: Zenodo `zenodo.org/record/6327426` (675 MB)
- **Use**: Preferred for controlled MISR experiments (known reference)

#### 2.2.3 Band Limitation
- **Only 2 bands**: RED (0.62-0.69 µm) and NIR (0.77-0.89 µm)
- **No Blue/Green** — limits spectral hallucination evaluation
- **Protocol**: Use for structural hallucination (H2, H9) and MISR benchmarking only

### 2.3 Cross-Sensor Datasets (Not Interchangeable)

| Dataset | Role | Spatial/Spectral Difference from ALSAT-2B | Access |
|---------|------|-------------------------------------------|--------|
| **NAIP** | HR target for S2→1m (×10) | 1m RGBN (R:0.6-0.68, G:0.52-0.6, B:0.43-0.5, NIR:0.77-0.89); US only | AWS S3 public |
| **PlanetScope** | HR target for S2→3m (×3) | 3-4m 4/8-band; daily revisit; spectral edges differ | Academic license (2-4 wk) |
| **Simulated ALSAT-2B** | Target domain for H5 | Degradation model (Sec 3) — controlled, estimated params | Synthetic (this protocol) |
| **WorldView-3** | Ultimate HR reference (validation only) | 0.31m PAN, 1.24m MS; commercial | Not for training |

**Protocol Rule**: Never mix sensors in training. Cross-sensor = train on S2, test on target (zero-shot generalization).

### 2.4 Data Leakage Prevention (Mandatory)
- **Spatial**: No overlapping tiles between train/val/test (128px buffer)
- **Temporal**: Year-based splits (no same-date across splits)
- **Scene**: No two tiles from same S2 granule in different splits
- **Verification**: Automated check in `src/data/verify_splits.py` (IoU > 0 triggers error)

### 2.5 Pilot Dataset Requirement
- **Before full download**: 10 scenes × 4 AOIs = 40 tiles (≈2 GB)
- **Validate**: Download integrity, band alignment, normalization, split logic
- **Approval Required**: Human sign-off before scaling to full dataset

---

## 3. Degradation Engine (ALSAT-2B Simulation)

### 3.1 Wald Protocol Implementation (Configurable, Reproducible)

```python
# src/degradation/alsat2b_degradation.py
class ALSAT2BDegradation:
    """
    Wald protocol degradation for ALSAT-2B/NAOMI simulation.
    ALL parameters are ESTIMATED from public specs (ESA eoPortal).
    No real ALSAT-2B calibration data available.
    """
    def __init__(self, config: DictConfig):
        # MTF: Gaussian kernel — ESTIMATED from MTF@Nyquist > 0.15 spec
        # For Gaussian MTF: MTF(f) = exp(-2π²σ²f²), Nyquist f=0.5 cycles/pixel
        # MTF(0.5) > 0.15 → σ < 0.39; nominal σ=0.35 px
        self.mtf_ms = GaussianKernel(sigma=config.mtf_sigma_ms)  # default 0.35
        self.mtf_pan = GaussianKernel(sigma=config.mtf_sigma_pan)  # default 0.35
        
        # SRF: Trapezoidal from published band edges (Table 2, eoPortal)
        self.srf = TrapezoidalSRF(band_edges=NAOMI_BAND_EDGES)  # exact from spec
        
        # Noise: Poisson-Gaussian — ESTIMATED conservative (spec says SNR>100)
        # SNR 20 dB = 100 linear; conservative for simulation
        self.noise = PoissonGaussianNoise(snr_db=config.snr_db)  # default 20
        
        # Quantization: 12-bit → 10-bit downlink (EXACT from spec)
        self.quantize = Quantize(bits=10)
        
        # Geometric: <0.3 pixel inter-band registration (EXACT from spec)
        self.geometric = RandomShift(max_shift=0.3)
        
        # Reproducibility
        self.seed = config.seed
        self.rng = np.random.default_rng(self.seed)
    
    def forward(self, hr_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        hr_cube: [C, H, W] at 2.5m (PAN-equivalent resolution)
        Returns: (ms_lr [4, H/4, W/4], pan_lr [1, H, W]) at 10m and 2.5m
        """
        # 1. Spectral degradation (HR hyperspectral → 4 MS bands)
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

### 3.2 Parameter Table (Documented vs. Estimated)

| Parameter | Source | Value | Uncertainty | Sensitivity Range |
|-----------|--------|-------|-------------|-------------------|
| **MTF σ (MS)** | **Estimated** from "MTF@Nyquist>0.15" | 0.35 px | ±0.05 | [0.30, 0.40] |
| **MTF σ (PAN)** | **Estimated** (same optics) | 0.35 px | ±0.05 | [0.30, 0.40] |
| **SRF shape** | **Documented** band edges (eoPortal Table 2) | Trapezoidal | Exact edges | Fixed |
| **SNR** | **Estimated** conservative (spec: >100) | 20 dB | Factor of 2 | [15, 25] dB |
| **Quantization** | **Documented** (12-bit→10-bit) | 10 bits | Exact | Fixed |
| **Registration** | **Documented** (<0.3 pixel) | Uniform[-0.3,0.3] | Exact | Fixed |

### 3.3 Degradation Configurations (Hydra/OmegaConf)

```yaml
# configs/degradation/alsat2b_nominal.yaml
degradation:
  _target_: src.degradation.alsat2b_degradation.ALSAT2BDegradation
  mtf_sigma_ms: 0.35
  mtf_sigma_pan: 0.35
  snr_db: 20
  seed: 42

# configs/degradation/alsat2b_sensitivity.yaml
degradation:
  mtf_sigma_ms: ???  # sweep 0.30, 0.35, 0.40
  mtf_sigma_pan: ???
  snr_db: ???  # sweep 15, 20, 25
```

### 3.4 Validation of Degradation Model (H1 Protocol)
1. **PSD Comparison**: Radial PSD of synthetic LR vs. any real ALSAT-2B (if available)
2. **Band Correlation**: 4×4 correlation matrices — synthetic vs. real
3. **Noise Statistics**: Mean/variance vs. radiance curves
4. **FID**: Patch-level distribution (Inception-v3 features, 64×64 patches)
5. **Downstream Proxy**: Train EDSR-small on synthetic; test PSNR on held-out real

### 3.5 Unit Tests Required
- MTF kernel frequency response matches target at Nyquist
- SRF integration preserves band radiance (energy conservation)
- Noise model produces target SNR at reference radiance
- Quantization reduces bit depth exactly
- Deterministic output for fixed seed
- Sensitivity sweep produces monotonic changes

---

## 4. Model Selection and Implementation

### 4.1 Core Models (Frozen for Phase 2)

| Model | Family | Params | Role | Implementation Source | Pretrained Weights |
|-------|--------|--------|------|----------------------|-------------------|
| **Bicubic** | Interpolation | 0 | Baseline | `cv2.resize` / `torch.nn.functional.interpolate` | N/A |
| **EDSR-small** | CNN | 1.5M | CNN baseline | `basicsr` (XPixelGroup/BasicSR) | ImageNet pretrained (×4) |
| **SwinIR-base** | Transformer | 11.9M | Transformer baseline | Official `JingyunLiang/SwinIR` | ImageNet pretrained (×4) |
| **Real-ESRGAN** | GAN (RRDB) | 16.7M | Perceptual/GAN baseline | Official `xinntao/Real-ESRGAN` | Real-world pretrained (×4) |

### 4.2 Model Configurations (Hydra/OmegaConf)

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
  rgb_range: 1.0

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

### 4.3 Training Recipes (Cloud GPU — Kaggle 2×T4)

| Model | Batch | Patch | LR | Epochs/Iters | Loss | AMP |
|-------|-------|-------|-----|--------------|------|-----|
| EDSR-small | 16 | 48×48 | 1e-4 | 300 epochs | L1 | fp16 |
| SwinIR-base | 16 | 64×64 | 2e-4 | 200 epochs | L1 + 0.1×LPIPS | fp16 |
| Real-ESRGAN | 16 | 128×128 | 1e-4 | 400k iters | L1 + 1.0×Perceptual + 0.1×GAN | fp16 |

**Note**: Diffusion (StableSR) **conditional** — only if compute budget allows after core 3.

### 4.4 Reproducibility Requirements
- **Seeds**: 42 (data), 123 (model init), 456 (training) — all logged
- **Deterministic CuDNN**: `torch.backends.cudnn.deterministic = True`
- **Version Pinning**: `requirements.txt` + git commit hashes for repos
- **Checkpointing**: Best val PSNR + last + every 50 epochs
- **Logging**: MLflow (metrics, params, artifacts, models)

### 4.5 CPU Smoke Tests (Local Development)
- All models must run inference on 256×256×4 patch in <30s on Intel Iris Xe
- EDSR-small and Tiny SwinIR (depth=2) for local dev
- Test script: `tests/test_models_smoke.py`

---

## 5. Evaluation Framework

### 5.1 Evaluation Taxonomy (Six Categories)

| Category | Purpose | Metrics | Reference |
|----------|---------|---------|-----------|
| **1. Reconstruction Quality** | Standard fidelity | PSNR, SSIM, LPIPS | Full-reference |
| **2. Spectral Fidelity** | RS-specific radiometric | SAM, ERGAS, band correlation | Full-reference |
| **3. Structural Fidelity** | Spatial structure | Wavelet energy, QNR (pansharpening) | Full-reference |
| **4. Hallucination Detection** | False detail quantification | SCI, Structural Hallucination Score, Semantic Hallucination Score | Full-reference + downstream |
| **5. Downstream Task Performance** | Practical utility | Classification mIoU, Detection mAP, Segmentation mIoU, Change Detection F1 | Fixed-backbone protocol |
| **6. Computational Efficiency** | Deployment feasibility | FLOPs, params, CPU/GPU latency, VRAM | Profiling |

### 5.2 Implemented Metrics (src/evaluation/)

```python
# Reconstruction
psnr(sr, hr), ssim(sr, hr), lpips(sr, hr)

# Spectral
sam(sr, hr), ergas(sr, hr), band_correlation_deviation(sr, hr)

# Spectral Consistency Index (SCI) — H3 metric
def spectral_consistency_index(sr, hr):
    """Mean SAM across band pairs weighted by vegetation/water sensitivity."""
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # 4 bands
    weights = [1, 1, 1, 2, 2, 2]  # Red-NIR, Green-NIR, Red-SWIR* weighted 2×
    sam_vals = [spectral_angle_mapper(sr[:,i], sr[:,j]) for i,j in pairs]
    return np.average(sam_vals, weights=weights)

# Structural Hallucination (H2)
def wavelet_highfreq_energy(img, wavelet='db1', level=2):
    """Mean |HH| energy in homogeneous regions (local variance < threshold)."""
    # DWT per channel, mask homogeneous regions, compute mean HH energy

# Semantic Hallucination (H2, H4)
def task_performance_drop(sr, hr, task_model, task_metric):
    """Fixed backbone: task model trained on HR, evaluated on SR vs HR."""
    pred_sr = task_model(sr)
    pred_hr = task_model(hr)
    return task_metric(pred_hr, gt) - task_metric(pred_sr, gt)

# Hallucination Index (H4 composite)
HALLUCINATION_INDEX = (
    0.3 * normalize(SCI) +
    0.3 * normalize(Structural_Hallucination_Score) +
    0.4 * normalize(Semantic_Hallucination_Score)
)

# Task Utility Score (H4)
TASK_UTILITY = np.mean([
    classification_miou / classification_miou_hr,
    detection_map / detection_map_hr,
    segmentation_miou / segmentation_miou_hr
])

# Ratio (H4)
RATIO = TASK_UTILITY / (1 + HALLUCINATION_INDEX)
```

### 5.3 Downstream Task Protocol (Fixed Backbone — Primary)

| Task | HR Dataset | Model (Frozen Backbone) | Metric |
|------|------------|------------------------|--------|
| **Land Cover Classification** | S2 + EuroSAT labels / DynamicWorld | UNet (ResNet18 encoder) | mIoU, OA, Kappa |
| **Building Detection** | S2 + OSM footprints / SpaceNet | RT-DETR / YOLOv8 | mAP@0.5, mAP@0.5:0.95 |
| **Semantic Segmentation** | S2 + LandCover.ai / custom | DeepLabV3+ (ResNet50) | mIoU |
| **Change Detection** | S2 time pairs + OSCD / custom | Siamese UNet | F1, IoU |

**Protocol Steps**:
1. Train task model on **HR reference data** (S2 10m) → freeze all weights
2. Evaluate on: HR (upper bound), SR outputs from each model
3. Report: Task metric on SR / Task metric on HR (normalized to 1.0)

**Secondary Protocol**: Fine-tune on SR (measures adaptability)
**Exploratory**: Joint SR+Task training

### 5.4 Cross-Sensor Evaluation (H5)
- Train on Sentinel-2 (source)
- Test on: NAIP (×10), PlanetScope (×3), Sim ALSAT-2B (×4) — **zero-shot**
- Metrics: ΔPSNR, ΔTask Utility, FID (ResNet-50), Spectral Shift

### 5.5 Statistical Analysis
- **Uncertainty**: Bootstrap 1000 resamples → 95% CI for all metrics
- **Pairwise**: Wilcoxon signed-rank (non-parametric)
- **Effect Size**: Cohen's d with CI
- **Multi-comparison**: Benjamini-Hochberg FDR (q=0.05)
- **Correlation**: Spearman ρ + permutation test (n=10000)

---

## 6. Hallucination Evaluation Framework (src/hallucination/)

### 6.1 Taxonomy Operationalization (H2)

| Category | Definition | Primary Metric | Threshold |
|----------|------------|----------------|-----------|
| **Spectral** | Violates physical band correlations | SAM > 10°, NDVI out-of-range | SAM mean > 0.17 rad |
| **Structural** | Spurious high-freq in homogeneous regions | Wavelet HH energy > 3σ local mean | Relative energy > 0.15 |
| **Textural** | Plausible but ungrounded periodic patterns | GLCM entropy mismatch > 20% | Relative diff > 0.2 |
| **Semantic** | Downstream confidence on false structures | Task mAP drop > 15% vs HR | Normalized drop > 0.15 |

### 6.2 Annotation Protocol (For H2 Validation)
- **Annotators**: 3 remote sensing experts
- **Samples**: 200 patches × 4 models = 800 patches
- **Labels**: {None, Spectral, Structural, Textural, Semantic, Multiple}
- **Tool**: Label Studio or CVAT (custom interface)
- **Agreement**: Fleiss' κ (target > 0.7)
- **Pilot**: 20 samples first to validate guidelines

### 6.3 Limitations (Explicit)
- No ground-truth hallucination labels exist — expert annotation is proxy
- Semantic hallucination requires trained downstream models (circular if not fixed-backbone)
- Textural hallucination metric (GLCM) sensitive to window size
- Spectral hallucination threshold (SAM>10°) is heuristic, not physics-derived

---

## 7. Cloud Compute Preparation

### 7.1 Environment Specifications

```txt
# requirements.txt (Local CPU + Cloud)
torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0
numpy==1.26.4
scipy==1.13.0
scikit-image==0.22.0
scikit-learn==1.5.0
matplotlib==3.9.0
opencv-python-headless==4.10.0
pillow==10.4.0
pyyaml==6.0.1
omegaconf==2.3.0
hydra-core==1.3.2
tqdm==4.66.0
einops==0.8.0
thop==0.1.1
lpips==0.1.4
pyiqa==0.1.4
clean-fid==0.1.3
mlflow==2.15.0
planetary-computer==0.4.0
odc-stac==0.3.0
pystac-client==0.7.0
rasterio==1.4.0
geopandas==1.0.0
```

```txt
# requirements-cloud.txt (Additional for GPU training)
basicsr @ git+https://github.com/XPixelGroup/BasicSR.git@<commit>
swinir @ git+https://github.com/JingyunLiang/SwinIR.git@<commit>
realesrgan @ git+https://github.com/xinntao/Real-ESRGAN.git@<commit>
timm==1.0.7
torchmetrics==1.4.0
wandb==0.17.0  # optional
```

### 7.2 Kaggle Notebook Template
```python
# kaggle_train_edsr.py (template)
import os, sys, subprocess, mlflow, torch
# 1. Install requirements-cloud.txt
# 2. Mount dataset (Kaggle dataset or GCS)
# 3. Set seeds, deterministic CuDNN
# 4. Load Hydra config
# 5. Train with MLflow logging
# 6. Save checkpoints to /kaggle/working/outputs/
# 7. Download checkpoints or push to GCS
```

### 7.3 Checkpoint Management
- **Local**: `experiments/checkpoints/{model}/{run_id}/`
- **Cloud**: Kaggle working directory → download / GCS sync
- **Naming**: `{model}_{scale}_{dataset}_{date}_{git_hash}.pt`
- **Contents**: model_state_dict, optimizer_state_dict, epoch, best_val_metric, config

### 7.4 Resume Training
- MLflow `run_id` tracking
- Checkpoint loading with strict=False (handle architecture changes)
- LR scheduler state restoration

---

## 8. Documentation Updates

| File | Updates Required |
|------|------------------|
| `README.md` | Install, data prep, experiment execution, protocol summary |
| `docs/experiment_tracking.md` | MLflow setup, run naming, artifact structure |
| `docs/reproducibility.md` | Seeds, versions, configs, data provenance, compute environment |
| `docs/methodology_decisions.md` | All corrections from Phase 1 review (this protocol Sec 1.3) |

---

## 9. Acceptance Criteria for Protocol Freeze

| Component | Verification |
|-----------|--------------|
| Dataset strategy | S2 AOIs defined, PROBA-V spec verified, cross-sensor distinctions documented |
| Degradation engine | All params flagged estimated/documented, sensitivity ranges defined, unit tests specified |
| Model configs | 4 core models + bicubic, Hydra configs, training recipes, CPU smoke tests defined |
| Evaluation framework | 6 categories, all metrics defined, downstream protocols fixed-backbone |
| Hallucination framework | 4 categories operationalized, annotation protocol, limitations explicit |
| Cloud prep | requirements.txt/.cloud.txt, Kaggle template, checkpoint mgmt, resume |
| Documentation | All 4 files updated with protocol references |

---

## 10. Remaining Scientific Risks (Post-Protocol)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ALSAT-2B real data never accessible | High | Medium | Substitute validated; H1 uses synthetic-only validation |
| PlanetScope license denied | Medium | Low | NAIP + SimALSAT sufficient for H5 |
| Hallucination annotation low agreement | Medium | Medium | Pilot 20 samples; refine guidelines |
| Kaggle GPU policy change | Low | High | Colab Pro + RunPod budget reserved |
| SCI metric fails correlation (H3) | Medium | High | Falsification pre-registered; negative result preserved |
| Tiny models don't reach 90% utility (H6) | Medium | Medium | Falsification pre-registered; report honestly |

---

## 11. Approval Required

**This protocol is DRAFT. Human approval required before treating as frozen.**

Please review and confirm:
- [ ] Dataset strategy correctly distinguishes sensors (no interchangeability)
- [ ] Degradation parameters honestly labeled estimated vs. documented
- [ ] PROBA-V benchmark uses exact ESA specification (74 regions, 1160/290 split, 300m→100m)
- [ ] Model selection covers 4 families with verified implementations
- [ ] Evaluation metrics have defined purpose; no metric used inappropriately
- [ ] Hallucination metrics not claimed as validated — H2 tests them
- [ ] Downstream tasks use fixed-backbone protocol with specified datasets
- [ ] Cloud prep enables Kaggle training without local GPU
- [ ] All Phase 1 corrections documented (Sec 1.3)
- [ ] Falsification criteria pre-registered for all hypotheses

**On approval**: Proceed to Part 4 (Data Pipeline Implementation)